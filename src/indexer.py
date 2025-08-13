from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import AppConfig
from .schemas import TranscriptChunk, Keyframe
from .embeddings import get_text_embedding_sbert, get_clip_image_embedding


class VideoIndexer:
	def __init__(self) -> None:
		# Prefer persistent DB; fall back to in-memory if sqlite is unavailable
		try:
			self.client = chromadb.PersistentClient(
				path=str(AppConfig.CHROMA_PERSIST_DIR),
				settings=Settings(anonymized_telemetry=False),
			)
		except Exception as e:
			print(f"Chroma persistent init failed ({e}); falling back to in-memory client.")
			self.client = chromadb.Client(Settings(anonymized_telemetry=False))
		self.transcripts = self._get_or_create_collection(AppConfig.COLLECTION_TRANSCRIPTS)
		self.frames = self._get_or_create_collection(AppConfig.COLLECTION_FRAMES)

	def _get_or_create_collection(self, name: str):
		try:
			return self.client.get_collection(name=name)
		except Exception:
			return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

	def _delete_by_video(self, collection, video_id: str) -> None:
		try:
			existing = collection.get(where={"video_id": video_id}, include=["ids"])  # type: ignore
			ids = existing.get("ids", []) if isinstance(existing, dict) else []
			if ids:
				collection.delete(ids=ids)
		except Exception:
			pass

	def upsert_transcript_chunks(self, video_id: str, video_path: Path, chunks: List[TranscriptChunk]) -> None:
		self._delete_by_video(self.transcripts, video_id)
		if not chunks:
			return
		ids = [f"{video_id}:transcript:{i}:{int(c.start*1000)}" for i, c in enumerate(chunks)]
		documents = [c.text for c in chunks]
		metadatas = [
			{
				"type": "transcript",
				"video_id": video_id,
				"video_path": str(video_path),
				"timestamp": float(c.start),
				"start": float(c.start),
				"end": float(c.end),
				"segment_count": int(c.segment_count),
			}
			for c in chunks
		]
		embeddings = get_text_embedding_sbert(documents)
		self.transcripts.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)

	def upsert_keyframes(self, video_id: str, video_path: Path, frames: List[Keyframe], captions: List[str], labels_per_frame: List[List[str]]) -> None:
		self._delete_by_video(self.frames, video_id)
		if not frames:
			return
		ids = [f"{video_id}:frame:{int(fr.timestamp*1000)}" for fr in frames]
		image_paths = [fr.image_path for fr in frames]
		embeddings = get_clip_image_embedding(image_paths)
		documents = captions
		metadatas = []
		for fr, caption, labels in zip(frames, captions, labels_per_frame):
			labels_str = ", ".join(labels) if isinstance(labels, list) else str(labels)
			metadatas.append(
				{
					"type": "frame",
					"video_id": video_id,
					"video_path": str(video_path),
					"timestamp": float(fr.timestamp),
					"image_path": str(fr.image_path),
					"caption": caption,
					"labels": labels_str,
				}
			)
		self.frames.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)
