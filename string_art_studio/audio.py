# -*- coding: utf-8 -*-
"""
Голосовой ассистент ручного плетения на базе Windows Speech API (SAPI).
"""

import logging
import queue
import threading
from typing import Optional

try:
    import pythoncom
    import win32com.client

    SAPI_AVAILABLE = True
except Exception:
    pythoncom = None
    win32com = None
    SAPI_AVAILABLE = False


class AudioAssistant:
    """Потокобезопасный голосовой диктор номеров гвоздей для работы без экрана (RU / EN)."""

    def __init__(self, lang: str = "ru"):
        self.enabled: bool = False
        self.lang: str = "ru" if str(lang).lower().startswith("ru") else "en"
        self.available: bool = SAPI_AVAILABLE
        self.speech_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        if SAPI_AVAILABLE:
            self._start_worker()

    def set_language(self, lang: str) -> None:
        self.lang = "ru" if str(lang).lower().startswith("ru") else "en"

    def _start_worker(self) -> None:
        self.worker_thread = threading.Thread(target=self._worker_loop, name="VoiceWorker", daemon=True)
        self.worker_thread.start()

    def _worker_loop(self) -> None:
        if pythoncom is None or win32com is None:
            return
        pythoncom.CoInitialize()
        speaker = None
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 1
            while not self._stop_event.is_set():
                try:
                    text = self.speech_queue.get(timeout=0.35)
                except queue.Empty:
                    continue
                if text is None:
                    break
                # Оставляем только самый свежий шаг при быстром перелистывании
                while not self.speech_queue.empty():
                    try:
                        nxt = self.speech_queue.get_nowait()
                        if nxt is None:
                            return
                        text = nxt
                    except queue.Empty:
                        break
                if self.enabled and speaker:
                    try:
                        speaker.Speak(text)
                    except Exception:
                        pass
        except Exception as exc:
            logging.warning(f"Не удалось инициализировать Windows SAPI SpVoice: {exc}")
            self.available = False
        finally:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def announce_chord(self, from_pin: int, to_pin: int) -> None:
        if self.enabled and self.available:
            phrase = f"на {to_pin}" if self.lang == "ru" else f"to {to_pin}"
            self.speech_queue.put(phrase)

    def toggle(self, state: bool) -> None:
        self.enabled = bool(state)

    def shutdown(self) -> None:
        self._stop_event.set()
        self.speech_queue.put(None)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
