from __future__ import annotations
from typing import List
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel
import torch
import imageio_ffmpeg

from .config import AppConfig
from .schemas import TranscriptSegment, TranscriptChunk


def extract_audio_to_wav(video_path: Path) -> Path:
	out_dir = Path(AppConfig.DATA_AUDIO_DIR) / video_path.stem
	out_dir.mkdir(parents=True, exist_ok=True)
	wav_path = out_dir / f"{video_path.stem}.wav"
	if wav_path.exists():
		return wav_path
	ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
	(
		ffmpeg
			.input(str(video_path))
			.output(str(wav_path), ac=1, ar=16000, format="wav")
			.overwrite_output()
			.run(cmd=ffmpeg_bin, quiet=True)
	)
	return wav_path


def transcribe_audio(wav_path: Path) -> List[TranscriptSegment]:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	compute_type = "float16" if device == "cuda" else "int8"
	model = WhisperModel(AppConfig.WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
	segments, _ = model.transcribe(str(wav_path), vad_filter=True)
	out: List[TranscriptSegment] = []
	for seg in segments:
		out.append(TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip()))
	return out


def chunk_transcript(segments: List[TranscriptSegment], chunk_seconds: float) -> List[TranscriptChunk]:
	chunks: List[TranscriptChunk] = []
	current_text: list[str] = []
	current_start: float | None = None
	current_end: float | None = None
	count: int = 0

	for seg in segments:
		if current_start is None:
			current_start = seg.start
		current_end = seg.end
		current_text.append(seg.text)
		count += 1

		if current_end - current_start >= chunk_seconds:
			chunks.append(
				TranscriptChunk(
					start=float(current_start),
					end=float(current_end),
					text=" ".join(current_text).strip(),
					segment_count=int(count),
				)
			)
			current_text = []
			current_start = None
			current_end = None
			count = 0

	if current_text and current_start is not None and current_end is not None:
		chunks.append(TranscriptChunk(start=float(current_start), end=float(current_end), text=" ".join(current_text).strip(), segment_count=int(count)))

	return chunks
