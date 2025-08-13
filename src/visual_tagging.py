from __future__ import annotations
from typing import List
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTForImageClassification, ViTImageProcessor

from .config import AppConfig


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_blip_model: BlipForConditionalGeneration | None = None
_blip_processor: BlipProcessor | None = None
_vit_model: ViTForImageClassification | None = None
_vit_processor: ViTImageProcessor | None = None
_blip_loaded = False
_vit_loaded = False


def _ensure_blip() -> tuple[BlipForConditionalGeneration, BlipProcessor]:
	global _blip_model, _blip_processor, _blip_loaded
	if _blip_loaded and _blip_model is not None and _blip_processor is not None:
		return _blip_model, _blip_processor
	try:
		_blip_model = BlipForConditionalGeneration.from_pretrained(AppConfig.BLIP_MODEL_NAME).to(_device)
		_blip_processor = BlipProcessor.from_pretrained(AppConfig.BLIP_MODEL_NAME)
		_blip_loaded = True
		return _blip_model, _blip_processor
	except Exception as e:
		print(f"Failed to load BLIP model: {e}")
		raise


def _ensure_vit() -> tuple[ViTForImageClassification, ViTImageProcessor]:
	global _vit_model, _vit_processor, _vit_loaded
	if _vit_loaded and _vit_model is not None and _vit_processor is not None:
		return _vit_model, _vit_processor
	try:
		_vit_model = ViTForImageClassification.from_pretrained(AppConfig.VIT_CLS_MODEL_NAME).to(_device)
		_vit_processor = ViTImageProcessor.from_pretrained(AppConfig.VIT_CLS_MODEL_NAME)
		_vit_loaded = True
		return _vit_model, _vit_processor
	except Exception as e:
		print(f"Failed to load ViT model: {e}")
		raise


def caption_image(image_path: Path) -> str:
	try:
		model, processor = _ensure_blip()
		image = Image.open(image_path).convert("RGB")
		inputs = processor(images=image, return_tensors="pt").to(_device)
		with torch.no_grad():
			out_ids = model.generate(**inputs, max_new_tokens=32)
		caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
		return caption.strip()
	except Exception as e:
		print(f"Error captioning image {image_path}: {e}")
		return "Image caption failed"


def classify_image(image_path: Path, top_k: int = 5) -> List[str]:
	try:
		model, processor = _ensure_vit()
		image = Image.open(image_path).convert("RGB")
		inputs = processor(images=image, return_tensors="pt").to(_device)
		with torch.no_grad():
			logits = model(**inputs).logits
		probs = torch.nn.functional.softmax(logits, dim=-1)
		values, indices = probs.topk(top_k, dim=-1)
		labels = [model.config.id2label[idx.item()] for idx in indices[0]]
		return labels
	except Exception as e:
		print(f"Error classifying image {image_path}: {e}")
		return ["classification failed"]
