#!/usr/bin/env python3
"""Run the modular ODMAP-aligned historical ENM pipeline."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ["_pipeline_part1.py", "_pipeline_part2.py", "_pipeline_part3.py", "_pipeline_part4.py"]:
    source = (HERE / name).read_text(encoding="utf-8")
    exec(compile(source, str(HERE / name), "exec"), globals(), globals())
