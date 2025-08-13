from typing import List
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from .config import AppConfig


_sbert_model: SentenceTransformer | None = None
_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_sbert_loaded = False
_clip_loaded = False


def _ensure_sbert() -> SentenceTransformer:
	global _sbert_model, _sbert_loaded
	if _sbert_loaded and _sbert_model is not None:
		return _sbert_model
	try:
		_sbert_model = SentenceTransformer(AppConfig.SBERT_MODEL_NAME, device=str(_device))
		_sbert_loaded = True
		return _sbert_model
	except Exception as e:
		print(f"Failed to load SBERT model: {e}")
		raise


def _ensure_clip() -> tuple[CLIPModel, CLIPProcessor]:
	global _clip_model, _clip_processor, _clip_loaded
	if _clip_loaded and _clip_model is not None and _clip_processor is not None:
		return _clip_model, _clip_processor
	try:
		_clip_model = CLIPModel.from_pretrained(AppConfig.CLIP_MODEL_NAME).to(_device)
		_clip_processor = CLIPProcessor.from_pretrained(AppConfig.CLIP_MODEL_NAME)
		_clip_loaded = True
		return _clip_model, _clip_processor
	except Exception as e:
		print(f"Failed to load CLIP model: {e}")
		raise


def get_text_embedding_sbert(texts: List[str]) -> np.ndarray:
	model = _ensure_sbert()
	emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
	return emb


def get_clip_text_embedding(texts: List[str]) -> np.ndarray:
	model, processor = _ensure_clip()
	inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(_device)
	with torch.no_grad():
		text_embeds = model.get_text_features(**inputs)
	text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
	return text_embeds.cpu().numpy()


def get_clip_image_embedding(image_paths: List[Path]) -> np.ndarray:
	model, processor = _ensure_clip()
	images = [Image.open(p).convert("RGB") for p in image_paths]
	inputs = processor(images=images, return_tensors="pt", padding=True).to(_device)
	with torch.no_grad():
		image_embeds = model.get_image_features(**inputs)
	image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
	return image_embeds.cpu().numpy()
