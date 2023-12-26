import ffmpeg
import json
from pathlib import Path
import re


movie_path=Path(r"data\video\dance00001.mp4").resolve()
print(f"movie_path.resolve(): {movie_path}")
probe = ffmpeg.probe(movie_path)
