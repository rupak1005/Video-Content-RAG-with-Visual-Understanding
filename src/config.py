from pathlib import Path
import os


class AppConfig:
	BASE_DIR: Path = Path(__file__).resolve().parent.parent
	DATA_DIR: Path = BASE_DIR / "data"
	DATA_VIDEOS_DIR: Path = DATA_DIR / "videos"
	DATA_AUDIO_DIR: Path = DATA_DIR / "audio"
	DATA_FRAMES_DIR: Path = DATA_DIR / "keyframes"
	LOGS_DIR: Path = BASE_DIR / "logs"
	DB_DIR: Path = BASE_DIR / "db"

	# Processing params
	FRAME_INTERVAL_SECONDS: float = float(os.getenv("FRAME_INTERVAL_SECONDS", 1.0))
	MIN_SCENE_LENGTH_SECONDS: float = float(os.getenv("MIN_SCENE_LENGTH_SECONDS", 2.0))
	HISTOGRAM_DIFF_THRESHOLD: float = float(os.getenv("HISTOGRAM_DIFF_THRESHOLD", 0.4))

	# Transcription params
	WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
	TRANSCRIPT_CHUNK_SECONDS: float = float(os.getenv("TRANSCRIPT_CHUNK_SECONDS", 20.0))

	# Embedding models
	SBERT_MODEL_NAME: str = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
	CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
	BLIP_MODEL_NAME: str = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
	VIT_CLS_MODEL_NAME: str = os.getenv("VIT_CLS_MODEL_NAME", "google/vit-base-patch16-224")
	GENERATION_MODEL_NAME: str = os.getenv("GENERATION_MODEL_NAME", "google/flan-t5-base")

	# ChromaDB
	CHROMA_PERSIST_DIR: Path = DB_DIR
	COLLECTION_TRANSCRIPTS: str = "transcripts"
	COLLECTION_FRAMES: str = "frames"


def ensure_directories() -> None:
	AppConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
	AppConfig.DATA_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
	AppConfig.DATA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
	AppConfig.DATA_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
	AppConfig.LOGS_DIR.mkdir(parents=True, exist_ok=True)
	AppConfig.DB_DIR.mkdir(parents=True, exist_ok=True)
