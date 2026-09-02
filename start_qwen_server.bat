@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
uv run --group nlp --group server python -m app.server --host 0.0.0.0 --port 8770 --judge qwen-omni
pause
