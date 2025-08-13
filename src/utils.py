import re
from pathlib import Path


def seconds_to_hhmmss(seconds: float) -> str:
	seconds = int(seconds)
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	secs = seconds % 60
	if hours > 0:
		return f"{hours:d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:d}:{secs:02d}"


def slugify(value: str) -> str:
	value = value.lower()
	value = re.sub(r"[^a-z0-9]+", "-", value)
	value = re.sub(r"-+", "-", value).strip("-")
	return value or "item"
