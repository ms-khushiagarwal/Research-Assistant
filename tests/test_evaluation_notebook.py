"""Test notebook logic without cloud calls or model downloads."""
import ast
import copy
import json
from pathlib import Path
import random
from functools import lru_cache
import unittest
from unittest.mock import Mock

import pandas as pd


def notebook_functions():
    notebook = json.loads(Path('answer_evaluation.ipynb').read_text(encoding='utf-8'))
    namespace = {'pd': pd, 'json': json, 'random': random, 'lru_cache': lru_cache,
                 'CANDIDATE_COUNT': 20, 'ANSWER_MODEL': 'test'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     or isinstance(node, ast.ClassDef) and node.name == 'JudgmentValidationError']
        exec(compile(tree, 'answer_evaluation.ipynb', 'exec'), namespace)
    return namespace


class NotebookTests(unittest.TestCase):
    def test_judge_json_wrappers_and_retry_delay(self):
        ns = notebook_functions()
        parse = ns['parse_judge_json']
        for text in ('{"score":1}', '```json\n{"score":1}\n```', 'Result: {"score":1}'):
            self.assertEqual(parse(text), {'score': 1})
        for text in ('', 'No answer', '[1]', '{"outer":{"score":1}', '{"a":1} {"b":2}'):
            with self.assertRaises(ValueError):
                parse(text)
        self.assertEqual(ns['judge_retry_delay'](0, '5'), 5)
        self.assertEqual(ns['judge_retry_delay'](0, '900'), 30)
        self.assertEqual(ns['judge_retry_delay'](1, None), 4)

    def test_resume_reuses_answers_and_preserves_successful_systems(self):
        ns = notebook_functions()
        context = [{'id': 'gold', 'content': 'evidence'}]
        ns['hybrid_retrieve'] = Mock(return_value=context)
        qa = {'id': 'Q1', 'question': 'q', 'reference_answer': 'gold', 'relevant_chunk_ids': ['gold']}
        judge = Mock()
        judge.answer.side_effect = ValueError('bad JSON')
        answer = Mock(return_value='saved answer')
        first = ns['evaluate_question'](qa, None, None, judge, answer, Mock(return_value=context))
        self.assertTrue(all(r['stage'] == 'judging' and r['answer'] == 'saved answer' for r in first))
        first[0]['status'] = 'ok'
        judge.answer.side_effect = None
        judge.answer.return_value = {'faithfulness': 90, 'citation_correctness': 80, 'reason': 'ok'}
        judge.answer.reset_mock()
        answer.reset_mock()
        ns['hybrid_retrieve'].reset_mock()
        checkpoint = Mock()
        resumed = ns['evaluate_question'](qa, None, None, judge, answer, Mock(), first, checkpoint)
        self.assertTrue(all(r['status'] == 'ok' for r in resumed))
        answer.assert_not_called()
        ns['hybrid_retrieve'].assert_not_called()
        self.assertEqual(judge.answer.call_count, 2)
        self.assertIs(resumed[0], first[0])
        self.assertEqual(checkpoint.call_count, 4)

    def test_source_grounding_and_label_validation(self):
        ns = notebook_functions()
        source = {'id': 'chunk', 'content': ' '.join(['evidence'] * 40), 'metadata': {'paper_path': 'p'}}
        judge = Mock()
        def ask(instruction, payload, schema, validate):
            qa = {'question': 'What is described?', 'reference_answer': 'evidence', 'evidence_id': 'E1'}
            validate(qa)
            return qa
        judge.ask.side_effect = ask
        dataset = ns['generate_ground_truth']({'chunk': source}, judge, 1, 42)
        ns['validate_dataset'](dataset, {'chunk': source})
        self.assertEqual(dataset[0]['relevant_chunk_ids'], ['chunk'])
        self.assertFalse(dataset[0]['reviewed'])
        bad = copy.deepcopy(dataset)
        bad[0]['evidence'][0]['quote'] = 'invented quote'
        with self.assertRaises(ValueError):
            ns['validate_dataset'](bad, {'chunk': source})
        with self.assertRaises(ValueError):
            ns['validate_qa']({'question': 'q', 'reference_answer': 'a', 'evidence_quote': 'invented'}, source)

    def test_evidence_selection_preserves_pdf_text_and_rejects_unknown_ids(self):
        ns = notebook_functions()
        text = 'The \ufb01nding\u00a0was 95.2%\u2014not 99%.\nA second line follows.'
        spans = ns['evidence_spans'](text, max_words=5)
        self.assertTrue(all(value in text for value in spans.values()))
        self.assertIn('\ufb01', spans['E1'])
        with self.assertRaises(ValueError):
            ns['validate_qa_selection']({'question': 'q', 'reference_answer': 'a', 'evidence_id': 'E999'}, spans)

    def test_generation_skips_exhausted_validation_but_not_network_errors(self):
        ns = notebook_functions()
        corpus = {str(i): {'id': str(i), 'content': ' '.join(['evidence'] * 40),
                           'metadata': {'paper_path': 'p'}} for i in range(2)}
        judge = Mock()
        judge.ask.side_effect = [ns['JudgmentValidationError']('invalid evidence ID'),
                                 {'question': 'q', 'reference_answer': 'a', 'evidence_id': 'E1'}]
        failures = []
        dataset = ns['generate_ground_truth'](corpus, judge, 1, 42, failures)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(failures), 1)
        self.assertNotEqual(dataset[0]['relevant_chunk_ids'][0], failures[0]['chunk_id'])
        ns['validate_dataset'](dataset, corpus)
        judge.ask.side_effect = RuntimeError('authentication failure')
        with self.assertRaises(RuntimeError):
            ns['generate_ground_truth'](corpus, judge, 1, 42)

    def test_metrics_generation_has_no_gold_and_failed_reranker_not_substituted(self):
        ns = notebook_functions()
        context = [{'id': 'other', 'content': 'other'}, {'id': 'gold', 'content': 'evidence'}]
        self.assertEqual(ns['retrieval_metrics'](context, ['gold', 'missing']), (0.5, 0.5))
        ns['hybrid_retrieve'] = Mock(return_value=context)
        judge = Mock()
        judge.answer.return_value = {'faithfulness': 80, 'citation_correctness': None, 'reason': 'test'}
        answer = Mock(return_value='answer')
        qa = {'id': 'Q1', 'question': 'question', 'reference_answer': 'secret gold answer', 'relevant_chunk_ids': ['gold']}
        records = ns['evaluate_question'](qa, None, None, judge, answer, Mock(side_effect=RuntimeError('offline')))
        self.assertEqual([r['status'] for r in records], ['ok', 'ok', 'error'])
        self.assertEqual(answer.call_count, 2)
        self.assertEqual(answer.call_args.args, ('question', context, []))
        self.assertEqual(judge.answer.call_args.args[-1], 'secret gold answer')
        summary, complete = ns['summarize_results'](records)
        self.assertEqual(complete, [])
        self.assertTrue(summary.isna().all().all())

    def test_app_has_no_evaluation_dependencies(self):
        source = Path('Homepage.py').read_text(encoding='utf-8')
        self.assertNotIn('render_evaluation', source)
        self.assertNotIn('utilities.Evaluation', source)
        self.assertNotIn('vector_context', source)
        self.assertFalse(Path('utilities/Evaluation.py').exists())

    def test_notebook_uses_input_pdfs_not_app_database(self):
        n = json.loads(Path('answer_evaluation.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')
        self.assertIn("INPUT_DIR = PROJECT_ROOT / 'input'", source)
        self.assertIn("INPUT_DIR.glob('*.pdf')", source)
        self.assertIn('embedding_fn=get_text_embedding', source)
        self.assertNotIn('get_chroma_client', source)
        self.assertNotIn('from utilities.Embed', source)


if __name__ == '__main__':
    unittest.main()
