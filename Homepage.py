from pathlib import Path
import os
from time import perf_counter

import streamlit as st
from chromadb.utils.data_loaders import ImageLoader
from utilities.Embed import (
    TextEmbeddingFunction, ImageEmbeddingFunction, get_text_embedding,
    get_image_embedding, get_image_query_embedding,
)
from utilities.Database import get_chroma_client, index_document
from utilities.DataLoader import parsepdf
from utilities.LLM import generate_response
from utilities.Retrieval import hybrid_retrieve
from utilities.Reranking import rerank_results, DEFAULT_MODEL
from utilities.Diagnostics import build_retrieval_trace, render_retrieval_trace

st.set_page_config(page_title="Research Paper Assistant", page_icon=":page_facing_up:",
                   layout="wide", initial_sidebar_state="expanded")
st.title("Research Paper Assistant")

@st.cache_resource
def get_collections():
    client = get_chroma_client()
    return (
        client.get_or_create_collection(name="text_embeddings",
                                       embedding_function=TextEmbeddingFunction()),
        client.get_or_create_collection(name="image_embeddings",
                                       embedding_function=ImageEmbeddingFunction(),
                                       data_loader=ImageLoader()),
    )

text_collection, image_collection = get_collections()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_traces" not in st.session_state:
    st.session_state.retrieval_traces = {}
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message_index in st.session_state.retrieval_traces:
            render_retrieval_trace(st.session_state.retrieval_traces[message_index])


def get_text_only_response(prompt):
    st.chat_message("user").write(prompt)
    with st.spinner("Finding relevant passages..."):
        # Use the actual question; broad generated query variants diluted relevance.
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        candidate_count = int(os.getenv("RERANK_CANDIDATES", "20"))
        if top_k < 1 or candidate_count < top_k:
            raise ValueError("Require 1 <= RETRIEVAL_TOP_K <= RERANK_CANDIDATES.")
        started = perf_counter()
        candidates = hybrid_retrieve(text_collection, prompt, get_text_embedding,
                                     top_k=candidate_count, candidate_k=max(30, candidate_count))
        timings = {"hybrid_retrieval": perf_counter() - started}
        started = perf_counter()
        reranking_error = None
        ranked = []
        try:
            # Retain every candidate's score for diagnostics; truncate only for generation.
            ranked = rerank_results(prompt, candidates, top_k=max(1, len(candidates)))
            results = ranked[:top_k]
        except Exception as exc:
            reranking_error = str(exc)
            st.warning(f"Reranking unavailable; using hybrid search results. {exc}")
            results = candidates[:top_k]
        timings["reranking"] = perf_counter() - started
        started = perf_counter()
        response = generate_response(prompt, results, st.session_state.messages)
        timings["answer_generation"] = perf_counter() - started
        trace = build_retrieval_trace(prompt, candidates, ranked, results, {
            "requested_candidates": candidate_count, "final_top_k": top_k,
            "reranker_model": os.getenv("RERANKER_MODEL", DEFAULT_MODEL),
            "answer_model": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        }, timings, reranking_error)
    with st.chat_message("assistant"):
        st.write(response)
        if results:
            with st.expander("Retrieved sources"):
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    st.markdown(f"**[S{i}] {Path(metadata.get('paper_path', 'Unknown')).name}, "
                                f"page {metadata.get('page_number', 'Unknown')}**")
                    st.write(result["content"])
                    st.caption(f"Cosine: {result['cosine_similarity']:.3f} | "
                               f"BM25: {result['bm25_score']:.3f}")
                    if "rerank_score" in result:
                        st.caption(f"Relevance score: {result['rerank_score']:.3f} | "
                                   f"Original hybrid rank: {result['hybrid_rank']}")
        # Image retrieval is optional and cannot prevent a text answer.
        if image_collection.count():
            try:
                images = image_collection.query(
                    query_embeddings=[get_image_query_embedding(prompt)],
                    n_results=min(3, image_collection.count()), include=["metadatas", "uris"],
                )
                trace['figures'] = images
                paths, captions = [], []
                for uri, metadata in zip((images.get("uris") or [[]])[0],
                                         (images.get("metadatas") or [[]])[0]):
                    if uri and Path(uri).is_file():
                        paths.append(uri)
                        captions.append(f"Page {(metadata or {}).get('page_number', 'Unknown')}")
                if paths:
                    with st.expander("Related figures (not used as answer evidence)"):
                        st.image(paths, caption=captions)
            except Exception as exc:
                trace['figures'] = {'error': str(exc)}
                st.warning(f"Figure retrieval unavailable: {exc}")
        render_retrieval_trace(trace)
    # Store diagnostics outside messages so verbose chunks never enter future LLM history.
    st.session_state.retrieval_traces[len(st.session_state.messages) + 1] = trace
    st.session_state.messages.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ])
    return response


def get_file_response(files):
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if Path(file.name).suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file: {file.name}. Upload a PDF.")
        file_path = upload_dir / Path(file.name).name
        file_path.write_bytes(file.getbuffer())
        with st.spinner(f"Indexing {file.name}..."):
            chunks, image_chunks = parsepdf(str(file_path))
            if not chunks:
                raise ValueError(f"No text extracted from {file.name}. Scanned PDFs may require OCR.")
            index_document(text_collection, image_collection, str(file_path), chunks, image_chunks,
                           get_text_embedding, get_image_embedding)
        st.success(f"Indexed {len(chunks)} text chunks and {len(image_chunks)} figures from {file.name}.")


prompt = st.chat_input("Upload a PDF or ask a question...", accept_file="multiple", file_type=["pdf"])
if prompt:
    try:
        # Newly attached documents must be available before answering the question.
        if prompt["files"]:
            get_file_response(prompt["files"])
        if prompt["text"]:
            get_text_only_response(prompt["text"])
    except Exception as exc:
        st.error(f"Could not complete the request: {exc}")
else:
    st.info("Upload a PDF, then ask a specific question about it.")
