from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Keyframe:
	timestamp: float
	image_path: Path


@dataclass
class TranscriptSegment:
	start: float
	end: float
	text: str


@dataclass
class TranscriptChunk:
	start: float
	end: float
	text: str
	segment_count: int
