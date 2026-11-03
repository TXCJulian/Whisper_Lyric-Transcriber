"""Lyric Transcriber service.

Load a .env file from the working directory before any app module runs:
several modules (config, job_manager, separation) read their settings from
the environment at import time. Values already in the environment win.
"""
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
