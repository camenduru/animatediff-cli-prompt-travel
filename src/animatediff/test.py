import ffmpeg
import json
from pathlib import Path
import re

movie_path=Path("data\video\dance00001.mp4")
movie_path = movie_path.resolve()
escaped = re.escape(movie_path)
print(f"movie_path.resolve(): {escaped}")
probe = ffmpeg.probe(escaped)
