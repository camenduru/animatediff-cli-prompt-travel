import ffmpeg
import json
from pathlib import Path

movie_path=Path("data\\video\\dance00001.mp4")
movie_path = movie_path.resolve()
print(f"movie_path.resolve(): {movie_path}")
probe = ffmpeg.probe(movie_path)
