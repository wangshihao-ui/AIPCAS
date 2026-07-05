import os
import re
import asyncio
import threading
import tempfile
import subprocess
from config import TTS_VOICE, TTS_RATE

VOICE = TTS_VOICE or "zh-CN-XiaoxiaoNeural"
RATE = TTS_RATE or "+0%"


def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5\d\s，。、；：！？（）《》\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _kill_proc(proc):
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


class VoiceService:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_error = None
        self._playing = False
        self._proc = None

    @property
    def is_playing(self):
        return self._playing

    def stop(self):
        with self._lock:
            self._playing = False
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            _kill_proc(proc)

    def speak(self, text):
        if not text:
            return
        cleaned = clean_text(text)
        if not cleaned:
            return
        self.stop()
        thread = threading.Thread(target=self._speak_impl, args=(cleaned,), daemon=True)
        thread.start()

    def _speak_impl(self, text):
        with self._lock:
            self._playing = True
        try:
            asyncio.run(self._tts(text))
        except Exception as e:
            self.last_error = f"语音播报失败 {e}"
            print(f"[Voice] {self.last_error}")
        finally:
            with self._lock:
                self._playing = False
                self._proc = None

    async def _tts(self, text):
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
            await communicate.save(tmp_path)

            if os.path.getsize(tmp_path) == 0:
                return

            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            with self._lock:
                self._proc = proc
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill_proc(proc)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


voice_service = VoiceService()
