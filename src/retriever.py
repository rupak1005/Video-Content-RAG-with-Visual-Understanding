from __future__ import annotations
from typing import List, Dict, Any

from .embeddings import get_text_embedding_sbert, get_clip_text_embedding
from .indexer import VideoIndexer


class MultimodalRetriever:
	def __init__(self, indexer: VideoIndexer) -> None:
		self.indexer = indexer

	def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
		# Query transcripts (SBERT)
		q_emb_sbert = get_text_embedding_sbert([query])[0]
		transcript_results = self.indexer.transcripts.query(
			query_embeddings=[q_emb_sbert.tolist()],
			n_results=top_k,
			include=["documents", "metadatas", "distances"],
		)

		# Query frames (CLIP text to image)
		q_emb_clip = get_clip_text_embedding([query])[0]
		frame_results = self.indexer.frames.query(
			query_embeddings=[q_emb_clip.tolist()],
			n_results=top_k,
			include=["documents", "metadatas", "distances"],
		)

		# Normalize distances into scores (cosine space; smaller distance → better)
		items: List[Dict[str, Any]] = []
		for docs, metas, dists in zip(
			transcript_results.get("documents", [[]]),
			transcript_results.get("metadatas", [[]]),
			transcript_results.get("distances", [[]]),
		):
			for doc, meta, dist in zip(docs, metas, dists):
				items.append({"score": float(1.0 - dist), "document": doc, "metadata": meta})

		for docs, metas, dists in zip(
			frame_results.get("documents", [[]]),
			frame_results.get("metadatas", [[]]),
			frame_results.get("distances", [[]]),
		):
			for doc, meta, dist in zip(docs, metas, dists):
				items.append({"score": float(1.0 - dist), "document": doc, "metadata": meta})

		items.sort(key=lambda x: x["score"], reverse=True)
		seen = set()
		unique: List[Dict[str, Any]] = []
		for it in items:
			meta = it["metadata"]
			uid = (meta.get("type"), meta.get("video_id"), meta.get("timestamp"), meta.get("image_path", ""))
			if uid in seen:
				continue
			seen.add(uid)
			unique.append(it)

		return unique[:top_k]
