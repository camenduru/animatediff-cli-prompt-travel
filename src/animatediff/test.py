import ffmpeg
import json
from pathlib import Path

movie_path=Path("data\video\dance00001.mp4")
probe = ffmpeg.probe(movie_path.resolve())
