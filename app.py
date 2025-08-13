import os
import io
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from src.config import AppConfig, ensure_directories
from src.utils import seconds_to_hhmmss, slugify
from src.video_processing import extract_keyframes, get_video_duration
from src.audio_transcription import extract_audio_to_wav, transcribe_audio, chunk_transcript
from src.visual_tagging import caption_image, classify_image
from src.embeddings import get_clip_text_embedding
from src.indexer import VideoIndexer
from src.retriever import MultimodalRetriever
from src.generator import generate_answer


def _inject_css() -> None:
	st.markdown(
		"""
		<style>
			/* Tighter containers and nicer cards */
			.block-container {padding-top: 2rem; padding-bottom: 2rem;}
			.result-card {border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; margin-bottom: 14px; background: rgba(255,255,255,0.02);} 
			.result-title {font-weight: 700; margin-bottom: 8px;}
			.kpi {text-align:center; padding: 10px; border-radius: 10px; background: rgba(124,77,255,0.1); border:1px solid rgba(124,77,255,0.25);}
			.kpi h3 {margin: 0; font-size: 1.2rem;}
			.kpi p {margin: 0; opacity: 0.8;}
		</style>
		""",
		unsafe_allow_html=True,
	)


def init_state() -> None:
	if "indexer" not in st.session_state:
		ensure_directories()
		st.session_state.indexer = VideoIndexer()
	if "retriever" not in st.session_state:
		st.session_state.retriever = MultimodalRetriever(st.session_state.indexer)


def process_and_index_video(video_path: Path) -> Dict[str, Any]:
	video_id = slugify(video_path.stem)
	st.write(f"Indexing video: {video_path.name} (id: {video_id})")

	with st.status("Extracting audio and transcribing...", expanded=False) as status_audio:
		wav_path = extract_audio_to_wav(video_path)
		segments = transcribe_audio(wav_path)
		chunks = chunk_transcript(segments, chunk_seconds=AppConfig.TRANSCRIPT_CHUNK_SECONDS)
		status_audio.update(label=f"Transcribed {len(segments)} segments → {len(chunks)} chunks", state="complete")

	with st.status("Extracting keyframes and visual tags...", expanded=False) as status_frames:
		frames = extract_keyframes(
			video_path=video_path,
			min_scene_length_seconds=AppConfig.MIN_SCENE_LENGTH_SECONDS,
			frame_interval_seconds=AppConfig.FRAME_INTERVAL_SECONDS,
			histogram_threshold=AppConfig.HISTOGRAM_DIFF_THRESHOLD,
		)
		captions = []
		labels = []
		for frame in frames:
			captions.append(caption_image(frame.image_path))
			labels.append(classify_image(frame.image_path, top_k=5))
		status_frames.update(label=f"Extracted {len(frames)} keyframes", state="complete")

	with st.status("Indexing into vector database...", expanded=False) as status_index:
		st.session_state.indexer.upsert_transcript_chunks(video_id, video_path, chunks)
		st.session_state.indexer.upsert_keyframes(video_id, video_path, frames, captions, labels)
		status_index.update(label="Indexing complete", state="complete")

	return dict(video_id=video_id, chunk_count=len(chunks), frame_count=len(frames))


def main() -> None:
	st.set_page_config(page_title="Video RAG (Visual + Audio)", layout="wide")
	_inject_css()
	st.title("Video Content RAG with Visual Understanding")
	init_state()

	# Top KPIs row
	k1, k2, k3 = st.columns(3)
	with k1:
		st.markdown('<div class="kpi"><h3>Local Models</h3><p>No API Keys</p></div>', unsafe_allow_html=True)
	with k2:
		st.markdown('<div class="kpi"><h3>Multimodal</h3><p>Audio + Visual</p></div>', unsafe_allow_html=True)
	with k3:
		st.markdown('<div class="kpi"><h3>Timecoded</h3><p>Timestamps</p></div>', unsafe_allow_html=True)

	with st.sidebar:
		st.header("Ingest Videos")
		uploaded_files = st.file_uploader("Upload video files", type=["mp4", "mov", "mkv", "avi"], accept_multiple_files=True)
		if uploaded_files:
			for uploaded in uploaded_files:
				video_bytes = uploaded.read()
				video_path = Path(AppConfig.DATA_VIDEOS_DIR) / uploaded.name
				video_path.parent.mkdir(parents=True, exist_ok=True)
				with open(video_path, "wb") as f:
					f.write(video_bytes)
				with st.expander(f"Process {uploaded.name}", expanded=False):
					stats = process_and_index_video(video_path)
					st.success(f"Indexed: {stats['frame_count']} frames, {stats['chunk_count']} transcript chunks")

	st.divider()

	# Tabs for Search and Library
	tab_search, tab_library = st.tabs(["Search", "Library"])
	with tab_search:
		st.subheader("Search Library")
		query = st.text_input("Enter your query (text)", "Find the moment where the person shows a chart and talks about revenue")
		col_q1, col_q2, col_q3 = st.columns([1,1,1])
		with col_q1:
			top_k = st.slider("Results", min_value=3, max_value=20, value=8, step=1)
		with col_q2:
			generate = st.toggle("Generate answer from retrieved context", value=False)
		with col_q3:
			st.caption("Search both transcripts and frames")

		if st.button("Search", type="primary") and query.strip():
			with st.spinner("Retrieving..."):
				results = st.session_state.retriever.search(query, top_k=top_k)
				st.write(f"Found {len(results)} results")

				if generate:
					answer = generate_answer(query, results)
					st.subheader("Answer")
					st.success(answer)

				st.subheader("Results")
				for i, item in enumerate(results, start=1):
					meta = item["metadata"]
					result_type = meta.get("type")
					video_path = Path(meta.get("video_path"))
					video_name = video_path.name
					ts = meta.get("timestamp")
					start_ts = seconds_to_hhmmss(ts) if ts is not None else "0:00"
					score = item.get("score", 0.0)

					st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
					st.markdown(f"<div class='result-title'>{i}. {result_type.upper()} — {video_name} — t={start_ts} — score={score:.3f}</div>", unsafe_allow_html=True)
					col1, col2 = st.columns([1,2])
					with col1:
						if result_type == "frame" and meta.get("image_path") and Path(meta.get("image_path")).exists():
							st.image(meta.get("image_path"), caption=meta.get("caption", ""), use_container_width=True)
						else:
							st.text_area("Transcript", item.get("document", ""), height=150)
					with col2:
						if video_path.exists():
							st.video(str(video_path))
							st.caption("Use the timecode above to seek.")
					st.markdown("</div>", unsafe_allow_html=True)

	with tab_library:
		st.subheader("Indexed Library")
		st.caption("Upload new videos from the sidebar to add to the library.")
		st.info("This view can be extended to list all videos and stats.")

	st.caption("Tip: Upload multiple videos to build a small library. Queries search both visual (CLIP) and audio (transcript) spaces.")


if __name__ == "__main__":
	main()
