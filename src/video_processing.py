from __future__ import annotations
from dataclasses import dataclass
from typing import List
from pathlib import Path

import cv2
import numpy as np
import ffmpeg
import shutil
import os

from .config import AppConfig
from .schemas import Keyframe


@dataclass
class _FrameWithTime:
	frame: np.ndarray
	timestamp: float


def get_video_duration(video_path: Path) -> float:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		return 0.0
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
	cap.release()
	if frame_count <= 0:
		return 0.0
	return float(frame_count / fps)


def _histogram_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
	hist_size = 32
	channels = [0, 1, 2]
	ranges = [0, 256, 0, 256, 0, 256]
	hist_a = cv2.calcHist([frame_a], channels, None, [hist_size, hist_size, hist_size], ranges)
	hist_b = cv2.calcHist([frame_b], channels, None, [hist_size, hist_size, hist_size], ranges)
	cv2.normalize(hist_a, hist_a)
	cv2.normalize(hist_b, hist_b)
	# 1 - correlation in [0,2], we cap to [0,1]
	d = 1.0 - cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
	return float(max(0.0, min(1.0, d)))


def _save_frame_image(video_path: Path, frame: np.ndarray, timestamp: float) -> Path:
	video_name = video_path.stem
	out_dir = Path(AppConfig.DATA_FRAMES_DIR) / video_name
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{int(timestamp*1000)}.jpg"
	cv2.imwrite(str(out_path), frame)
	return out_path


def _extract_with_ffmpeg_periodic(video_path: Path, frame_interval_seconds: float) -> List[Keyframe]:
	"""Fallback periodic frame extraction using ffmpeg when OpenCV fails."""
	video_name = video_path.stem
	out_dir = Path(AppConfig.DATA_FRAMES_DIR) / video_name
	# Clean old frames for this video folder
	if out_dir.exists():
		for f in out_dir.glob("*.jpg"):
			try:
				f.unlink()
			except Exception:
				pass
	else:
		out_dir.mkdir(parents=True, exist_ok=True)

	tmp_dir = out_dir / "_tmp"
	tmp_dir.mkdir(parents=True, exist_ok=True)
	for f in tmp_dir.glob("*.jpg"):
		f.unlink()

	fps_expr = max(0.0001, 1.0 / max(0.0001, frame_interval_seconds))
	(
		ffmpeg
			.input(str(video_path))
			.output(str(tmp_dir / "frame_%06d.jpg"), vf=f"fps={fps_expr}", vsync=0)
			.overwrite_output()
			.run(quiet=True)
	)

	keyframes: List[Keyframe] = []
	images = sorted(tmp_dir.glob("frame_*.jpg"))
	for idx, img in enumerate(images):
		ts = float(idx) * float(frame_interval_seconds)
		target = out_dir / f"{int(ts*1000)}.jpg"
		try:
			shutil.move(str(img), str(target))
		except Exception:
			# Fallback to copy if moving fails across devices
			shutil.copy2(str(img), str(target))
			try:
				img.unlink()
			except Exception:
				pass
		keyframes.append(Keyframe(timestamp=ts, image_path=target))

	# Cleanup tmp dir
	try:
		shutil.rmtree(tmp_dir)
	except Exception:
		pass

	return keyframes


def extract_keyframes(
	video_path: Path,
	min_scene_length_seconds: float = AppConfig.MIN_SCENE_LENGTH_SECONDS,
	frame_interval_seconds: float = AppConfig.FRAME_INTERVAL_SECONDS,
	histogram_threshold: float = AppConfig.HISTOGRAM_DIFF_THRESHOLD,
) -> List[Keyframe]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		# Try ffmpeg fallback directly
		return _extract_with_ffmpeg_periodic(video_path, frame_interval_seconds)

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	step = max(1, int(frame_interval_seconds * fps))

	keyframes: List[Keyframe] = []
	last_scene_time = -9999.0
	prev_sample: _FrameWithTime | None = None

	frame_index = 0
	last_saved_index = -999999
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if frame_index % step != 0:
			frame_index += 1
			continue

		timestamp = float(frame_index / fps)

		# Always save at a coarser interval as a fallback (e.g., every 5*step)
		should_save_periodic = (frame_index - last_saved_index) >= (step * 5)

		if prev_sample is None:
			prev_sample = _FrameWithTime(frame=frame, timestamp=timestamp)
			last_scene_time = timestamp
			image_path = _save_frame_image(video_path, frame, timestamp)
			keyframes.append(Keyframe(timestamp=timestamp, image_path=image_path))
			last_saved_index = frame_index
		else:
			diff = _histogram_distance(prev_sample.frame, frame)
			if diff >= histogram_threshold and (timestamp - last_scene_time) >= min_scene_length_seconds:
				image_path = _save_frame_image(video_path, frame, timestamp)
				keyframes.append(Keyframe(timestamp=timestamp, image_path=image_path))
				prev_sample = _FrameWithTime(frame=frame, timestamp=timestamp)
				last_scene_time = timestamp
				last_saved_index = frame_index
			elif should_save_periodic:
				image_path = _save_frame_image(video_path, frame, timestamp)
				keyframes.append(Keyframe(timestamp=timestamp, image_path=image_path))
				last_saved_index = frame_index

		frame_index += 1

	cap.release()

	# If no frames were saved, fallback to ffmpeg
	if not keyframes:
		return _extract_with_ffmpeg_periodic(video_path, frame_interval_seconds)

	return keyframes
