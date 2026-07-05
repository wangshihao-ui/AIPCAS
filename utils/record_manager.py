import json
import os
from config import MAX_RECORDS, BASE_DIR

RECORD_FILE = os.path.join(BASE_DIR, "detection_records.json")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

_records_cache = None


def _ensure_dirs():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def _flush(records):
    _ensure_dirs()
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _delete_screenshot(path):
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def load_records():
    global _records_cache
    if _records_cache is not None:
        return _records_cache
    if not os.path.exists(RECORD_FILE):
        _records_cache = []
        return _records_cache
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _records_cache = data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        _records_cache = []
    return _records_cache


def add_record(pest_type, image_path):
    global _records_cache
    records = load_records()
    record = {
        "time": os.path.basename(image_path).split("_", 1)[-1].replace(".jpg", "")
                if "_" in os.path.basename(image_path) else pest_type,
        "type": pest_type,
        "path": image_path,
    }
    records.insert(0, record)

    while len(records) > MAX_RECORDS:
        oldest = records.pop()
        _delete_screenshot(oldest.get("path", ""))

    _flush(records)
    _records_cache = records
    return records


def clear_records():
    global _records_cache
    records = load_records()
    for r in records:
        _delete_screenshot(r.get("path", ""))
    _flush([])
    _records_cache = []
    if os.path.isdir(SCREENSHOT_DIR):
        for name in os.listdir(SCREENSHOT_DIR):
            fp = os.path.join(SCREENSHOT_DIR, name)
            if os.path.isfile(fp):
                _delete_screenshot(fp)