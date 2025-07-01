import json
import os

SETTINGS_FILE = os.path.join(os.getcwd(), "mic_settings.json")

def save_mic_index(device_index):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"device_index": device_index}, f)

def load_mic_index():
    if not os.path.exists(SETTINGS_FILE):
        return None
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("device_index")
    except Exception:
        return None
