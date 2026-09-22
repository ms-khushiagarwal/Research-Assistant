from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils.data_loaders import ImageLoader
import hashlib
import numpy as np
from pathlib import Path
from filelock import FileLock
from huggingface_hub import get_token, login

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_saved_model(model_id, model_type):
    """Authenticate once and keep a complete, offline-loadable model on disk."""
    if not model_id:
        raise ValueError(f"{model_type.upper()}_EMBEDDING_MODEL must be set")

    cache_root = Path(__file__).resolve().parents[1] / "models"
    cache_root.mkdir(parents=True, exist_ok=True)
    model_key = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
    model_path = cache_root / f"{model_type}-{model_key}"
    ready = model_path / ".complete"

    # Prevent concurrent app starts from reading a partially saved model.
    with FileLock(str(model_path) + ".lock"):
        saved = ready.is_file()
        source = str(model_path) if saved else model_id
        local = saved or Path(model_id).is_dir()
        if not local and get_token() is None:
            # Opens the Hugging Face browser/device-code login flow and saves
            # credentials in the standard Hugging Face credential cache.
            login()

        if model_type == "text":
            loaded = (SentenceTransformer(source, local_files_only=local),)
        else:
            loaded = (
                AutoModel.from_pretrained(source, local_files_only=local),
                AutoProcessor.from_pretrained(source, local_files_only=local),
            )

        if not saved:
            model_path.mkdir(parents=True, exist_ok=True)
            for component in loaded:
                component.save_pretrained(str(model_path))
            # Only mark the cache usable after every component is saved.
            ready.touch()
        return loaded


text_model, = _load_saved_model(os.getenv("TEXT_EMBEDDING_MODEL"), "text")
model = text_model
image_model, image_processor = _load_saved_model(
    os.getenv("IMAGE_EMBEDDING_MODEL"), "image"
)
# Reuse the tokenizer saved with the image processor instead of fetching it again.
image_tokenizer = image_processor.tokenizer

image_model = image_model.to(device)
image_model.eval()


def get_text_embedding(text):
    embedding = text_model.encode(text, convert_to_tensor=True)
    return embedding.cpu().tolist()

def get_image_embedding(image_input):

    if isinstance(image_input, (str, bytes)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = Image.fromarray(
            np.asarray(image_input).astype("uint8")
        ).convert("RGB")

    inputs = image_processor(
        images=[image],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embedding = image_model.get_image_features(
            **inputs,
            return_dict=True
        )

    if not isinstance(embedding, torch.Tensor):
        embedding = embedding.pooler_output

    embedding = F.normalize(
        embedding,
        p=2,
        dim=-1
    )

    return embedding.squeeze(0).cpu().tolist()

def get_image_query_embedding(text: str) -> list:
    """
    Encode a text query into the same embedding space
    as images stored in image_collection.
    """

    inputs = image_tokenizer(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embedding = image_model.get_text_features(
            **inputs
        )

    if not isinstance(embedding, torch.Tensor):
        embedding = embedding.pooler_output

    embedding = F.normalize(
        embedding,
        p=2,
        dim=-1
    )

    return embedding.squeeze(0).cpu().tolist()

class TextEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        return [
            get_text_embedding(text)
            for text in input
        ]

    @staticmethod
    def name() -> str:
        return "my-text-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return TextEmbeddingFunction()


class ImageEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input) -> Embeddings:
        return [
            get_image_embedding(image)
            for image in input
        ]

    @staticmethod
    def name() -> str:
        return "my-image-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return ImageEmbeddingFunction()

def get_sha512_hash(text: str) -> str:
    """Return a SHA-512 hash of *text* used for unique document IDs."""

    return hashlib.sha512(text.encode("utf-8")).hexdigest()
