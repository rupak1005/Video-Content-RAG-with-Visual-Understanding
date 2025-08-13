# Video Content RAG with Visual Understanding

A multimodal RAG system that processes videos, extracts keyframes, transcribes audio, tags visual content, and enables unified search over visual and audio elements with timestamped references.

## Demo

![App screenshot](assets/ui_demo.png)

> Place your screenshot at `assets/ui_demo.png` so it renders on GitHub.

## Features
- Video processing and keyframe extraction (OpenCV + FFmpeg fallback)
- Audio transcription with timestamps (faster-whisper)
- Visual captions and tags (BLIP captioning + ViT classification)
- Visual question answering for frame-level questions (ViLT VQA)
- Unified multimodal retrieval (CLIP images + SBERT text) using ChromaDB
- Streamlit UI for upload, indexing, and querying with temporal references
- Optional answer generation over retrieved context (FLAN‑T5, with graceful fallback)

## Tech Stack
- Python, Streamlit
- OpenCV, FFmpeg
- faster-whisper
- Transformers (CLIP, BLIP, ViT, ViLT, FLAN‑T5), Sentence-Transformers
- ChromaDB (local persistent)

## Quickstart

1) Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ensure FFmpeg is installed (Arch):
```bash
sudo pacman -S ffmpeg
```

3) Run the app:
```bash
streamlit run app.py
```

4) Upload one or more videos in the sidebar. The app will:
- Extract keyframes (scene-change + periodic fallback)
- Extract audio and transcribe to timestamped segments
- Caption frames and classify with ViT
- Build embeddings and index into ChromaDB

5) Search the library using natural language. Toggle “Generate answer” to synthesize an answer using retrieved context.

## How it works
- Ingestion
  - Saves the video file under `data/videos/`
  - `audio_transcription.py` extracts mono 16kHz WAV (FFmpeg) and transcribes (faster-whisper)
  - `video_processing.py` extracts keyframes using histogram deltas, with periodic FFmpeg fallback
  - `visual_tagging.py` generates BLIP captions and ViT top‑k labels
- Embeddings & Index
  - `embeddings.py` encodes transcript chunks (SBERT) and keyframe images (CLIP)
  - `indexer.py` writes to Chroma: `transcripts` and `frames` collections
- Retrieval & QA
  - `retriever.py` queries by text against both spaces and merges results
  - `generator.py` optionally composes an answer with FLAN‑T5
  - Visual questions (e.g., “what color is the shirt?”) first try ViLT VQA over top frames

## Project Structure
```
assign/
  app.py
  requirements.txt
  README.md
  .gitignore
  .streamlit/
    config.toml
  src/
    __init__.py
    config.py
    schemas.py
    utils.py
    embeddings.py
    indexer.py
    video_processing.py
    audio_transcription.py
    visual_tagging.py
    retriever.py
    generator.py
    vqa.py
  data/
    .gitkeep
  db/
    .gitkeep
  logs/
    .gitkeep
  assets/
    ui_demo.png  # add your screenshot file here
```

## Configuration
You can tune behavior via environment variables:
```bash
export TRANSCRIPT_CHUNK_SECONDS=20
export FRAME_INTERVAL_SECONDS=1.0
export MIN_SCENE_LENGTH_SECONDS=2.0
export HISTOGRAM_DIFF_THRESHOLD=0.4
export WHISPER_MODEL_SIZE="small"            # tiny, base, small, medium, large-v3
export SBERT_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
export BLIP_MODEL_NAME="Salesforce/blip-image-captioning-base"
export VIT_CLS_MODEL_NAME="google/vit-base-patch16-224"
export GENERATION_MODEL_NAME="google/flan-t5-base"  # or google/flan-t5-small
```

## Evaluation
Basic checks you can run manually:
- Retrieval sanity: ask for a specific phrase from the transcript and verify timecode
- Visual sanity: ask about a visible object/color; verify a keyframe appears with that content
- Latency: time indexing and query in the Streamlit logs
- Optional: integrate RAGAS for text segments or report@k on known queries

## Deployment (HuggingFace Spaces)
- Create a new Space (Streamlit)
- Push this repo; ensure `requirements.txt` is present
- Set Space Hardware to CPU (works) or GPU (faster for models)

## Troubleshooting
- Model download fails (no internet): answers fall back to a simple summary; try smaller models (`flan-t5-small`, `WHISPER_MODEL_SIZE=tiny`).
- No keyframes: FFmpeg fallback extracts periodic frames; ensure `ffmpeg` is installed.
- Chroma metadata error: labels are stored as a string to satisfy primitive-only metadata constraints.
- CUDA OOM: switch to CPU or smaller models; reduce batch sizes by default.

## Security & Privacy
- All processing is local. No API keys required. No external data is uploaded.

## License
MIT
