from docling_core.types.doc import (
    TextItem,
    PictureItem,
)
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import uuid
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()


def get_text_embedding(text):
    # Load application models only when the default embedding implementation is used.
    from utilities.Embed import get_text_embedding as embed_text
    return embed_text(text)

# Configure Docling
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
# This parser extracts TextItem and PictureItem only, not table structures.
# Avoid downloading TableFormer for a feature that is not used.
pipeline_options.do_table_structure = False

# Enable image extraction
pipeline_options.generate_picture_images = True

# Higher image resolution improves OCR on small text
pipeline_options.images_scale = 2.0

# Initialize converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

def parsepdf(pdf, output_dir="extracted_images", embedding_fn=None):
    pdf = str(pdf)
    result = converter.convert(pdf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    content = dict()
    content['paper_id'] = Path(pdf).stem
    content['paper_path'] = pdf
    content['extracted'] = list()
    document = result.document

    for item, level in document.iterate_items():

        # Get the page number
        page_no = item.prov[0].page_no if item.prov else None

        if isinstance(item, TextItem):

            content["extracted"].append({
                "page": page_no,
                "sentence": item.text,
                "type": "text"
            })

        elif isinstance(item, PictureItem):
            image = item.get_image(document)

            if image is None:
                continue

            image_id = uuid.uuid4().hex
            image_filename = f"{image_id}.png"
            image_path = output_dir / image_filename

            content["extracted"].append({
                "page": page_no,
                "image_path": str(image_path),
                "image_id": image_id,
                "type": "image"
            })

            image.save(image_path)

    # Chunk only after all pages and images have been extracted.
    return semantic_chunking(
        documents=content,
        buffer_size=int(os.getenv("BUFFER_SIZE", 1)),
        breakpoint_percentile_threshold=int(os.getenv("BREAKPOINT_PERCENTILE_THRESHOLD", 80)),
        max_chunk_words=int(os.getenv("MAX_CHUNK_WORDS", 400)),
        embedding_fn=embedding_fn,
    )

def semantic_chunking(
    documents: dict,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 80,
    max_chunk_words: int = 400,
    embedding_fn=None,
) -> tuple[list, list]:

    if buffer_size < 0 or not 0 <= breakpoint_percentile_threshold <= 100:
        raise ValueError("Invalid semantic chunking parameters.")
    if max_chunk_words < 1:
        raise ValueError("max_chunk_words must be positive.")

    extracted = documents["extracted"]
    chunks = []
    image_chunks = []

    if not extracted:
        return chunks, image_chunks

    embed_text = embedding_fn if embedding_fn is not None else get_text_embedding

    # Step 1: Create contextual sentence buffers.
    for i, item in enumerate(extracted):

        if item["type"] != "text":
            continue

        combined_sentences = []

        for j in range(
            max(0, i - buffer_size),
            min(len(extracted), i + buffer_size + 1),
        ):
            if extracted[j]["type"] == "text":
                combined_sentences.append(
                    extracted[j]["sentence"]
                )

        item["combined_sentences"] = " ".join(combined_sentences)

    # Step 2: Generate embeddings.
    for item in extracted:

        if item["type"] == "text":

            item["combined_sentence_embedding"] = (
                embed_text(item["combined_sentences"])
            )

    # Step 3: Calculate distances between consecutive text elements.
    distances = []

    for i in range(len(extracted) - 1):

        current = extracted[i]
        next_item = extracted[i + 1]

        if (
            current["type"] == "text"
            and next_item["type"] == "text"
        ):

            embedding1 = np.asarray(
                current["combined_sentence_embedding"]
            )

            embedding2 = np.asarray(
                next_item["combined_sentence_embedding"]
            )

            denominator = (
                np.linalg.norm(embedding1)
                * np.linalg.norm(embedding2)
            )

            distance = (
                1 - np.dot(embedding1, embedding2) / denominator
                if denominator != 0
                else 0.0
            )

            distances.append(distance)

            current["semantic_distance_to_next"] = distance

        else:

            distances.append(None)

            current["semantic_distance_to_next"] = None

    # Step 4: Find semantic breakpoints.
    valid_distances = [
        d for d in distances
        if d is not None
    ]

    if valid_distances:

        breakpoint_threshold = np.percentile(
            valid_distances,
            breakpoint_percentile_threshold,
        )

        indices_above_threshold = {
            i
            for i, distance in enumerate(distances)
            if distance is not None
            and distance > breakpoint_threshold
        }

    else:
        indices_above_threshold = set()

    # Step 5: Construct text and image chunks.
    current_text = []
    current_page = None
    current_words = 0
    current_end_page = None

    def flush_text_chunk():

        nonlocal current_text, current_page, current_words, current_end_page

        if current_text:

            chunks.append({
                "paper_path": documents["paper_path"],
                "page_number": current_page,
                "page_end": current_end_page,
                "content": " ".join(current_text),
                "type": "text",
            })

            current_text = []
            current_page = None
            current_words = 0
            current_end_page = None

    for i, item in enumerate(extracted):

        # Images are separate chunks.
        if item["type"] == "image":

            flush_text_chunk()

            image_chunks.append({
                "paper_path": documents["paper_path"],
                "page_number": item["page"],
                "content": item["image_path"],
                "type": "image",
            })

            continue

        if item["type"] != "text":
            continue

        # Bound context even when the percentile produces no semantic splits.
        # Long extracted elements are divided at word boundaries without dropping text.
        words = item["sentence"].split()
        if current_text and current_words + len(words) > max_chunk_words:
            flush_text_chunk()
        for start in range(0, len(words), max_chunk_words):
            part = words[start:start + max_chunk_words]
            if not current_text:
                current_page = item["page"]
            current_text.append(" ".join(part))
            current_words += len(part)
            current_end_page = item["page"]
            if current_words >= max_chunk_words:
                flush_text_chunk()

        # End the chunk at a semantic breakpoint.
        if i in indices_above_threshold:
            flush_text_chunk()

    # Add any remaining text.
    flush_text_chunk()

    return chunks, image_chunks
