from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_vqa_model: Optional[ViltForQuestionAnswering] = None
_vqa_proc: Optional[ViltProcessor] = None
_vqa_loaded: bool = False


def _ensure_vqa():
	global _vqa_model, _vqa_proc, _vqa_loaded
	if _vqa_loaded and _vqa_model is not None and _vqa_proc is not None:
		return _vqa_model, _vqa_proc
	try:
		# Small and widely available VQA model
		_vqa_proc = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
		_vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(_device)
		_vqa_loaded = True
		return _vqa_model, _vqa_proc
	except Exception as e:
		print(f"Failed to load VQA model: {e}")
		_vqa_loaded = False
		return None, None


def answer_visual_question(image_path: Path, question: str) -> str:
	model, proc = _ensure_vqa()
	if model is None or proc is None:
		return ""
	try:
		image = Image.open(image_path).convert("RGB")
		inputs = proc(image, question, return_tensors="pt").to(_device)
		with torch.no_grad():
			outputs = model(**inputs)
		logits = outputs.logits
		idx = logits.argmax(-1).item()
		answer = model.config.id2label[idx]
		return str(answer)
	except Exception as e:
		print(f"VQA failed on {image_path}: {e}")
		return ""
