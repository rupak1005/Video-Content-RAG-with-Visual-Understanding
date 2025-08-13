from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import AppConfig
from .vqa import answer_visual_question


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: AutoModelForSeq2SeqLM | None = None
_tok: AutoTokenizer | None = None
_model_loaded = False


def _ensure_model():
	global _model, _tok, _model_loaded
	if _model_loaded and _model is not None and _tok is not None:
		return _model, _tok
	
	try:
		_tok = AutoTokenizer.from_pretrained(AppConfig.GENERATION_MODEL_NAME)
		_model = AutoModelForSeq2SeqLM.from_pretrained(AppConfig.GENERATION_MODEL_NAME).to(_device)
		_model_loaded = True
		return _model, _tok
	except Exception as e:
		print(f"Failed to load generation model: {e}")
		# Try fallback to smaller model
		try:
			fallback_model = "google/flan-t5-small"
			_tok = AutoTokenizer.from_pretrained(fallback_model)
			_model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model).to(_device)
			_model_loaded = True
			return _model, _tok
		except Exception as e2:
			print(f"Failed to load fallback model too: {e2}")
			_model_loaded = False
			return None, None


def _simple_summary(results: List[Dict[str, Any]]) -> str:
	"""Fallback summary when generation model fails"""
	transcript_parts = []
	frame_parts = []
	
	for r in results[:5]:  # Top 5 results
		meta = r.get("metadata", {})
		ts = meta.get("timestamp", 0)
		time_str = f"{int(ts//60)}:{int(ts%60):02d}"
		
		if meta.get("type") == "transcript":
			text = r.get("document", "").strip()
			if text:
				transcript_parts.append(f"[{time_str}] {text}")
		elif meta.get("type") == "frame":
			caption = meta.get("caption", "").strip()
			labels = meta.get("labels", [])
			if isinstance(labels, list):
				labels = ", ".join(labels)
			if caption or labels:
				frame_parts.append(f"[{time_str}] {caption} (tags: {labels if isinstance(labels, str) else ''})")
	
	summary = []
	if transcript_parts:
		summary.append("Key transcript moments:")
		summary.extend(transcript_parts[:3])
	if frame_parts:
		summary.append("\nKey visual moments:")
		summary.extend(frame_parts[:3])
	
	return "\n".join(summary) if summary else "No relevant content found."


def _likely_visual_query(query: str) -> bool:
	q = query.lower()
	visual_keywords = [
		"color", "colour", "wearing", "look like", "object", "what is in",
		"what is shown", "shape", "number of", "how many", "logo", "brand",
		"text on", "read on", "shirt", "dress", "car", "animal", "scene"
	]
	return any(k in q for k in visual_keywords)


def _answer_with_vqa(query: str, results: List[Dict[str, Any]]) -> str:
	"""Use top frame results to answer a visual question via VQA."""
	for r in results:
		meta = r.get("metadata", {})
		if meta.get("type") == "frame":
			img_path = meta.get("image_path")
			if img_path:
				ans = answer_visual_question(Path(img_path), query)
				if ans:
					return ans
	return ""


def generate_answer(query: str, results: List[Dict[str, Any]], max_new_tokens: int = 128) -> str:
	# If question is visual, try VQA first
	if _likely_visual_query(query):
		vqa_answer = _answer_with_vqa(query, results)
		if vqa_answer:
			return vqa_answer

	model, tok = _ensure_model()
	if model is None or tok is None:
		return _simple_summary(results)
	
	try:
		context_chunks: List[str] = []
		for r in results[:8]:
			meta = r.get("metadata", {})
			prefix = "FRAME:" if meta.get("type") == "frame" else "TRANSCRIPT:"
			if meta.get("type") == "frame":
				context_chunks.append(f"{prefix} {meta.get('caption','')} labels={meta.get('labels','')} t={meta.get('timestamp')}")
			else:
				context_chunks.append(f"{prefix} {r.get('document', '')} t={meta.get('timestamp')}")
		context = "\n".join(context_chunks)
		prompt = f"Answer the question using the provided context. Include time references when helpful.\nQuestion: {query}\nContext:\n{context}\nAnswer:"
		inputs = tok(prompt, return_tensors="pt", truncation=True).to(_device)
		with torch.no_grad():
			out = model.generate(**inputs, max_new_tokens=max_new_tokens)
		text = tok.decode(out[0], skip_special_tokens=True)
		return text.strip()
	except Exception as e:
		return _simple_summary(results)
