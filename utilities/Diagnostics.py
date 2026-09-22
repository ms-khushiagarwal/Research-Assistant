"""Per-answer retrieval traces, kept separate from LLM conversation history."""
from copy import deepcopy


def build_retrieval_trace(query, candidates, ranked, selected, settings, timings, error=None):
    reranked = {row['id']: (rank, row.get('rerank_score'))
                for rank, row in enumerate(ranked, 1)}
    citations = {row['id']: f'S{rank}' for rank, row in enumerate(selected, 1)}
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        rerank_rank, score = reranked.get(candidate['id'], (None, None))
        rows.append({
            'chunk_id': candidate['id'],
            'hybrid_rank': rank,
            'cosine_similarity': candidate.get('cosine_similarity'),
            'bm25_score': candidate.get('bm25_score'),
            'hybrid_rrf_score': candidate.get('score'),
            'rerank_rank': rerank_rank,
            'rerank_score': score,
            'sent_to_model': candidate['id'] in citations,
            'citation': citations.get(candidate['id']),
            'metadata': deepcopy(candidate.get('metadata') or {}),
            'content': candidate['content'],
        })
    return {'query': query, 'settings': dict(settings), 'timings_seconds': dict(timings),
            'reranking_error': error, 'candidates': rows,
            'final_context_ids': [row['id'] for row in selected]}


def render_retrieval_trace(trace):
    import streamlit as st

    with st.expander('Retrieval details — chunks and reranking scores', expanded=False):
        st.write('Query:', trace['query'])
        st.json({'settings': trace['settings'], 'timings_seconds': trace['timings_seconds'],
                 'final_context_ids': trace['final_context_ids']})
        if trace['reranking_error']:
            st.warning('Reranking failed; hybrid ordering was used. ' + trace['reranking_error'])
        rows = trace['candidates']
        if not rows:
            st.info('No text chunks were retrieved.')
        else:
            st.caption('All hybrid candidates, including chunks excluded from the final answer context. '
                       'Reranking scores are relevance signals, not probabilities.')
            st.dataframe([{key: value for key, value in row.items()
                           if key not in ('metadata', 'content')} for row in rows],
                         hide_index=True)
            for row in rows:
                st.markdown(f"**Hybrid #{row['hybrid_rank']} · "
                            + (f"[{row['citation']}] Sent to model**" if row['sent_to_model']
                               else 'Not sent to model**'))
                st.json({key: value for key, value in row.items() if key != 'content'})
                st.text(row['content'])
        if trace.get('figures'):
            st.write('Related figure retrieval (not used as answer evidence):')
            st.json(trace['figures'])
