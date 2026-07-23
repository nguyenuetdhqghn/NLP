from datetime import datetime
from pathlib import Path

RESULT_FILE = Path(__file__).resolve().parent / "result.txt"


def append_result(task_name, message):
    content = f"\n{task_name}\n{message.strip()}\n"
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(content)
