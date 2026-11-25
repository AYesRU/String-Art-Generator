# -*- coding: utf-8 -*-
"""
Голосовой ассистент ручного плетения на базе Windows Speech API (SAPI).
"""

import logging
import queue
import threading
from typing import Any, Optional

try:
    import pythoncom
    import win32com.client

    SAPI_AVAILABLE = True
except Exception:
    pythoncom = None
    win32com = None
    SAPI_AVAILABLE = False


def _detect_voice_lang(voice_token: Any) -> Optional[str]:
    """Определяет язык ('ru' или 'en') для SAPI voice token по LCID Language, Id или Description."""
    try:
        raw_lang = str(voice_token.GetAttribute("Language") or "")
        for tok in raw_lang.split(";"):
            clean = tok.strip()
            if not clean:
                continue
            try:
                primary = int(clean, 16) & 0x3FF
                if primary == 0x19:
                    return "ru"
                if primary == 0x09:
                    return "en"
            except ValueError:
                pass
    except Exception:
        pass

    text_parts = []
    try:
        fn = getattr(voice_token, "GetDescription", None)
        if callable(fn):
            text_parts.append(str(fn() or ""))
    except Exception:
        pass
    try:
        text_parts.append(str(getattr(voice_token, "Id", "") or ""))
    except Exception:
        pass
    joined = " ".join(text_parts).lower()
    if any(k in joined for k in ("russian", "ru-ru", "ru_ru")):
        return "ru"
    if any(k in joined for k in ("english", "en-us", "en-gb", "en_")):
        return "en"
    return None


def _pick_voice_for_lang(speaker: Any, lang: str) -> Optional[Any]:
    """Подбирает установленный голос Windows SAPI под указанный язык ('ru' или 'en')."""
    target = "ru" if str(lang).lower().startswith("ru") else "en"
    fallback = None
    try:
        voices = speaker.GetVoices()
        for idx in range(voices.Count):
            v = voices.Item(idx)
            if _detect_voice_lang(v) != target:
                continue
            if target == "en":
                try:
                    raw_l = str(v.GetAttribute("Language") or "").lower()
                    desc_l = str(v.GetDescription() or "").lower()
                    if any(t.strip() == "409" for t in raw_l.split(";")) or "en-us" in desc_l or "united states" in desc_l:
                        return v
                except Exception:
                    pass
            if fallback is None:
                fallback = v
            if target == "ru":
                return v
    except Exception:
        pass
    return fallback


class AudioAssistant:
    """Потокобезопасный голосовой диктор номеров гвоздей для работы без экрана (RU / EN)."""

    def __init__(self, lang: str = "ru"):
        self.enabled: bool = False
        self.lang: str = "ru" if str(lang).lower().startswith("ru") else "en"
        self.available: bool = SAPI_AVAILABLE
        self.active_voice_desc: str = ""
        self.speech_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        if SAPI_AVAILABLE:
            self._start_worker()

    def set_language(self, lang: str) -> None:
        self.lang = "ru" if str(lang).lower().startswith("ru") else "en"
        if SAPI_AVAILABLE and not self._stop_event.is_set():
            self.speech_queue.put(("__switch__", self.lang))

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
            voices_by_lang = {
                "ru": _pick_voice_for_lang(speaker, "ru"),
                "en": _pick_voice_for_lang(speaker, "en"),
            }
            applied_lang: Optional[str] = None

            def _apply_voice(target_lang: str) -> None:
                nonlocal applied_lang
                if applied_lang == target_lang:
                    return
                v = voices_by_lang.get(target_lang)
                if v is not None:
                    try:
                        speaker.Voice = v
                        self.active_voice_desc = str(v.GetDescription() or "")
                        logging.info(f"SAPI голос ({target_lang}): {self.active_voice_desc}")
                    except Exception as exc:
                        logging.warning(f"Не удалось переключить SAPI голос на '{target_lang}': {exc}")
                else:
                    logging.warning(f"Не найден установленный Windows SAPI голос для языка '{target_lang}'.")
                applied_lang = target_lang

            _apply_voice(self.lang)
            while not self._stop_event.is_set():
                try:
                    item = self.speech_queue.get(timeout=0.35)
                except queue.Empty:
                    continue
                if item is None:
                    break
                # Оставляем только самый свежий шаг при быстром перелистывании
                while not self.speech_queue.empty():
                    try:
                        nxt = self.speech_queue.get_nowait()
                        if nxt is None:
                            return
                        if nxt[0] == "__switch__":
                            _apply_voice(nxt[1])
                            if item[0] == "__switch__":
                                item = nxt
                        else:
                            item = nxt
                    except queue.Empty:
                        break
                cmd, payload = item
                if cmd == "__switch__":
                    _apply_voice(payload)
                    continue
                _apply_voice(cmd)
                if self.enabled and speaker and payload:
                    try:
                        speaker.Speak(payload)
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
            cur_lang = self.lang
            phrase = f"на {to_pin}" if cur_lang == "ru" else f"to {to_pin}"
            self.speech_queue.put((cur_lang, phrase))

    def toggle(self, state: bool) -> None:
        self.enabled = bool(state)

    def shutdown(self) -> None:
        self._stop_event.set()
        self.speech_queue.put(None)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
