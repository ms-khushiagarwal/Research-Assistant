import chromadb
from dotenv import load_dotenv
import os
import hashlib
load_dotenv()

def get_chroma_client():
    """
    Initializes and returns a Chroma client.
    """
    return chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "chroma_db"))


def index_document(text_collection, image_collection, paper_path, chunks, image_chunks,
                   embed_text, embed_image):
    """Replace a document's chunks; retries do not create random duplicate IDs.

    Prepare embeddings before writes and remove stale/legacy IDs only after both
    collections have accepted the new records. Chroma has no cross-collection transaction.
    """
    paper_path = str(paper_path)
    batches = []
    for collection, records, kind, embedding_fn in (
        (text_collection, chunks, "text", embed_text),
        (image_collection, image_chunks, "image", embed_image),
    ):
        records = [record for record in records if record["content"].strip()]
        ids = [hashlib.sha256(f"{paper_path}\0{kind}\0{i}\0{record['content']}".encode()).hexdigest()
               for i, record in enumerate(records)]
        metadata = [{"paper_path": paper_path,
                     "page_number": record.get("page_number") or -1,
                     "page_end": record.get("page_end") or record.get("page_number") or -1,
                     "chunk_index": i, "ingestion_version": 2}
                    for i, record in enumerate(records)]
        embeddings = [embedding_fn(record["content"]) for record in records]
        old_ids = collection.get(where={"paper_path": paper_path}, include=[])["ids"]
        batches.append((collection, records, kind, ids, metadata, embeddings, old_ids))
    for collection, records, kind, ids, metadata, embeddings, _ in batches:
        for start in range(0, len(ids), 100):
            stop = start + 100
            values = [row["content"] for row in records[start:stop]]
            collection.upsert(ids=ids[start:stop], metadatas=metadata[start:stop],
                              embeddings=embeddings[start:stop],
                              **({"documents": values} if kind == "text" else {"uris": values}))
    for collection, _, _, ids, _, _, old_ids in batches:
        current = set(ids)
        stale = [chunk_id for chunk_id in old_ids if chunk_id not in current]
        for start in range(0, len(stale), 100):
            collection.delete(ids=stale[start:start + 100])
