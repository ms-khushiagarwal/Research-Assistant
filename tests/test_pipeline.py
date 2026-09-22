"""Run: .venv/Scripts/python.exe -m unittest discover -s tests -v"""
import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from time import perf_counter

import chromadb
import numpy as np

from utilities.Database import index_document
from utilities.LLM import generate_response
from utilities.Retrieval import hybrid_retrieve, bm25_scores
from utilities.Reranking import rerank_results
from utilities.Diagnostics import build_retrieval_trace, render_retrieval_trace


def loader_functions():
    # Execute the real function bodies without loading Docling or embedding models.
    tree = ast.parse(Path("utilities/DataLoader.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name != 'get_text_embedding']
    namespace = dict(np=np, os=os, Path=Path, uuid=uuid,
                     get_text_embedding=lambda text: [1.0, 0.0])
    exec(compile(tree, "utilities/DataLoader.py", "exec"), namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    def test_trace_includes_excluded_chunks_and_preserves_scores(self):
        candidates = [{'id': str(i), 'content': f'full passage {i}', 'metadata': {'page_number': i},
                       'score': 0.02, 'cosine_similarity': 0.7, 'bm25_score': i}
                      for i in range(3)]
        ranked = [{**candidates[i], 'rerank_score': score} for i, score in [(2, 8), (0, 2), (1, -1)]]
        trace = build_retrieval_trace('question', candidates, ranked, ranked[:1], {}, {})
        self.assertEqual(len(trace['candidates']), 3)
        self.assertEqual(trace['final_context_ids'], ['2'])
        self.assertEqual([row['rerank_score'] for row in trace['candidates']], [2, -1, 8])
        self.assertEqual(trace['candidates'][2]['citation'], 'S1')
        self.assertFalse(trace['candidates'][0]['sent_to_model'])
        candidates[0]['metadata']['page_number'] = 99
        self.assertEqual(trace['candidates'][0]['metadata']['page_number'], 0)
        fallback = build_retrieval_trace('q', candidates, [], candidates[:1], {}, {}, 'offline')
        self.assertIsNone(fallback['candidates'][0]['rerank_score'])
        self.assertEqual(fallback['reranking_error'], 'offline')
        from unittest.mock import MagicMock
        import sys
        st = MagicMock()
        with patch.dict(sys.modules, {'streamlit': st}):
            render_retrieval_trace(trace)
        self.assertFalse(st.expander.call_args.kwargs['expanded'])
        self.assertEqual(st.text.call_count, 3)

    def test_reranking_scores_all_candidates_before_truncation(self):
        candidates = [{"id": str(i), "content": f"passage {i}",
                       "metadata": {"page_number": i}, "score": 1 / (i + 1)}
                      for i in range(8)]
        original = copy.deepcopy(candidates)
        model = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([0, 0, 0, 0, 0, 1, 3, 2]))
        with patch.object(model, 'predict', wraps=model.predict) as predict:
            results = rerank_results('specific question', candidates, top_k=2, model=model)
        self.assertEqual([r['id'] for r in results], ['6', '7'])
        self.assertEqual(predict.call_args.args[0], [('specific question', c['content']) for c in candidates])
        self.assertEqual(results[0]['metadata'], {'page_number': 6})
        self.assertEqual(results[0]['hybrid_rank'], 7)
        self.assertEqual(results[0]['rerank_score'], 3)
        self.assertEqual(candidates, original)

    def test_reranking_ties_empty_and_invalid_scores(self):
        candidates = [{'id': str(i), 'content': 'text'} for i in range(2)]
        model = SimpleNamespace(predict=lambda *a, **kw: [[1], [1]])
        self.assertEqual([r['id'] for r in rerank_results('q', candidates, model=model)], ['0', '1'])
        with patch('utilities.Reranking.load_reranker') as load:
            self.assertEqual(rerank_results('q', []), [])
            self.assertEqual(rerank_results(' ', candidates), [])
            load.assert_not_called()
        for scores in ([1], [float('nan'), 0], [1, float('inf')], [[1, 2], [3, 4]]):
            model = SimpleNamespace(predict=lambda *a, **kw: scores)
            with self.assertRaises(ValueError):
                rerank_results('q', candidates, model=model)

    def test_app_passes_reranked_context_and_falls_back(self):
        tree = ast.parse(Path('Homepage.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'get_text_only_response')
        from unittest.mock import MagicMock
        candidates = [{'id': str(i), 'content': 'text'} for i in range(20)]
        selected = list(reversed(candidates))[:5]
        for fail in (False, True):
            st = MagicMock()
            st.session_state.messages = []
            st.session_state.retrieval_traces = {}
            image = MagicMock()
            image.count.return_value = 0
            ns = dict(st=st, os=os, Path=Path, text_collection=object(), image_collection=image,
                      perf_counter=perf_counter, DEFAULT_MODEL='test-reranker',
                      build_retrieval_trace=build_retrieval_trace, render_retrieval_trace=MagicMock(),
                      get_text_embedding=object(), hybrid_retrieve=MagicMock(return_value=candidates),
                      rerank_results=MagicMock(return_value=selected),
                      generate_response=MagicMock(return_value='answer'))
            if fail:
                ns['rerank_results'].side_effect = RuntimeError('model unavailable')
            exec(compile(ast.Module(body=[function], type_ignores=[]), 'Homepage.py', 'exec'), ns)
            with patch.dict(os.environ, {'RETRIEVAL_TOP_K': '5', 'RERANK_CANDIDATES': '20'}):
                # Avoid rendering fake source rows; inspect answer-generation arguments.
                ns['generate_response'].side_effect = lambda *a: 'answer'
                # Supply fields needed by the real rendering code.
                for row in candidates:
                    row.update(metadata={}, cosine_similarity=0.1, bm25_score=1.0)
                ns['get_text_only_response']('question')
            self.assertEqual(ns['generate_response'].call_args.args[1], candidates[:5] if fail else selected)
            self.assertEqual(ns['hybrid_retrieve'].call_args_list[0].kwargs['top_k'], 20)
            self.assertEqual(ns['hybrid_retrieve'].call_count, 1)
            self.assertEqual(st.warning.called, fail)
            self.assertEqual(ns['rerank_results'].call_args.kwargs['top_k'], 20)
            self.assertEqual(len(st.session_state.retrieval_traces[1]['candidates']), 20)
            self.assertEqual(set(st.session_state.messages[1]), {'role', 'content'})

    def test_parser_reads_beyond_first_image_and_text_only(self):
        ns = loader_functions()
        class Text:
            def __init__(self, text):
                self.text = text
                self.prov = [SimpleNamespace(page_no=1)]
        class Picture:
            prov = []
            def get_image(self, document):
                return SimpleNamespace(save=lambda path: None)
        ns.update(TextItem=Text, PictureItem=Picture)
        for items in ([Text("before"), Picture(), Text("after")], [Text("text only")], []):
            document = SimpleNamespace(iterate_items=lambda: [(item, 0) for item in items])
            ns['converter'] = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))
            with tempfile.TemporaryDirectory() as folder:
                text, images = ns['parsepdf'](Path("paper.pdf"), folder)
            joined = " ".join(c['content'] for c in text)
            self.assertEqual(joined, " ".join(i.text for i in items if isinstance(i, Text)))
            self.assertEqual(len(images), sum(isinstance(i, Picture) for i in items))

    def test_chunk_bounds_and_empty_contract(self):
        chunk = loader_functions()['semantic_chunking']
        self.assertEqual(chunk({'extracted': []}), ([], []))
        words = [f'w{i}' for i in range(950)]
        document = {'paper_path': 'paper.pdf', 'extracted': [
            {'type': 'text', 'sentence': ' '.join(words), 'page': 1},
            {'type': 'text', 'sentence': 'final page', 'page': 2},
        ]}
        chunks, _ = chunk(document, max_chunk_words=400)
        self.assertTrue(all(len(c['content'].split()) <= 400 for c in chunks))
        self.assertEqual(' '.join(c['content'] for c in chunks).split(), words + ['final', 'page'])
        self.assertEqual(chunks[-1]['page_end'], 2)

    def test_chunker_uses_injected_embedding_function(self):
        from unittest.mock import Mock
        chunk = loader_functions()['semantic_chunking']
        embed = Mock(return_value=[1.0, 0.0])
        text, _ = chunk({'paper_path': 'p', 'extracted': [
            {'type': 'text', 'sentence': 'test sentence', 'page': 1}]}, embedding_fn=embed)
        embed.assert_called_once_with('test sentence')
        self.assertEqual(text[0]['content'], 'test sentence')

    def test_answer_contains_current_question_and_bounded_history(self):
        history = [{'role': 'user', 'content': str(i)} for i in range(20)]
        results = [{'content': 'The learning rate is 0.01.', 'metadata': {'page_number': 8}}]
        with patch('utilities.LLM.chat', return_value=SimpleNamespace(
                message=SimpleNamespace(content='0.01 [S1]'))) as chat:
            answer = generate_response('What is the learning rate?', results, history)
        self.assertIn('[S1]', answer)
        messages = chat.call_args.kwargs['messages']
        self.assertIn('What is the learning rate?', messages[-1]['content'])
        self.assertEqual(messages[1:-1], history[-10:])
        self.assertEqual(len(history), 20)
        with patch('utilities.LLM.chat') as chat:
            generate_response('Question', [], [])
            chat.assert_not_called()

    def test_bm25_rare_term_and_empty(self):
        scores = bm25_scores('ZXQ42', ['general research paper', 'ZXQ42 optimizer', 'other method'])
        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertEqual(bm25_scores('unknown', ['text']).tolist(), [0])
        self.assertEqual(bm25_scores('query', []).tolist(), [])

    def test_real_chroma_hybrid_and_reingestion(self):
        # Ephemeral Chroma, explicit vectors: no model downloads or production DB writes.
        client = chromadb.EphemeralClient()
        name = uuid.uuid4().hex
        text = client.create_collection('text_' + name)
        image = client.create_collection('image_' + name)
        try:
            text.add(ids=['dense', 'keyword', 'noise'],
                     documents=['semantic relevant passage', 'ZXQ42 optimizer', 'unrelated background'],
                     embeddings=[[10., 0.], [0.8, 0.6], [-1., 0.]],
                     metadatas=[{'paper_path': 'paper.pdf', 'page_number': i} for i in range(3)])
            dense = hybrid_retrieve(text, 'ZXQ42', lambda q: [1., 0.], dense_weight=1)
            lexical = hybrid_retrieve(text, 'ZXQ42', lambda q: [1., 0.], dense_weight=0)
            hybrid = hybrid_retrieve(text, 'ZXQ42', lambda q: [1., 0.])
            self.assertEqual(dense[0]['id'], 'dense')  # cosine, not L2 distance
            self.assertEqual(lexical[0]['id'], 'keyword')
            self.assertEqual(hybrid[0]['id'], 'keyword')
            text.add(ids=['duplicate'], documents=['ZXQ42 optimizer'], embeddings=[[0.8, 0.6]],
                     metadatas=[{'paper_path': 'paper.pdf', 'page_number': 1}])
            self.assertEqual(len(hybrid_retrieve(text, 'ZXQ42', lambda q: [1., 0.])), 3)
            records = [{'content': 'new evidence', 'page_number': None}]
            for _ in range(2):
                index_document(text, image, 'paper.pdf', records, [], lambda s: [1., 0.], lambda s: [1., 0.])
            self.assertEqual(text.count(), 1)
            self.assertEqual(image.count(), 0)
            self.assertEqual(text.get()['documents'], ['new evidence'])
            with self.assertRaisesRegex(RuntimeError, 'embedding failed'):
                index_document(text, image, 'paper.pdf', records, [],
                               lambda s: (_ for _ in ()).throw(RuntimeError('embedding failed')),
                               lambda s: [1., 0.])
            self.assertEqual(text.count(), 1)
        finally:
            client.delete_collection(text.name)
            client.delete_collection(image.name)

    def test_upload_precedes_question(self):
        tree = ast.parse(Path('Homepage.py').read_text(encoding='utf-8'))
        final_if = tree.body[-1]
        calls = []
        ns = {'prompt': {'files': ['paper.pdf'], 'text': 'question'},
              'get_file_response': lambda files: calls.append('ingest'),
              'get_text_only_response': lambda text: calls.append('answer')}
        exec(compile(ast.Module(body=[final_if], type_ignores=[]), 'Homepage.py', 'exec'), ns)
        self.assertEqual(calls, ['ingest', 'answer'])


if __name__ == '__main__':
    unittest.main()
