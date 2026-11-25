# -*- coding: utf-8 -*-
import os
import re
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from string_art_studio import TOOLTIPS
from string_art_studio.audio import AudioAssistant
from string_art_studio.color_catalog import COLOR_CATALOG, COLOR_NAME_EN, COLOR_NAME_RU
from string_art_studio.geometry import CanvasManager
from string_art_studio.i18n import (
    COLOR_MODE_EN,
    LAYER_NAME_EN,
    METRIC_EN,
    PRESET_EN,
    SHAPE_EN,
    STRINGS,
    TOOLTIPS_EN,
    detect_system_language,
    get_display_color_modes,
    get_display_metrics,
    get_display_presets,
    get_display_shapes,
    get_tooltip,
    layer_short_code,
    normalize_lang,
    to_canonical_color_mode,
    to_canonical_metric,
    to_canonical_preset,
    to_canonical_shape,
    to_display_color_mode,
    to_display_metric,
    to_display_preset,
    to_display_shape,
    tr,
    translate_layer_name,
)
from string_art_studio.preprocessing import auto_recommend_parameters


class TestI18n(unittest.TestCase):
    def test_ru_en_dictionaries_keys_and_placeholders_match(self):
        ru_keys = set(STRINGS["ru"].keys())
        en_keys = set(STRINGS["en"].keys())
        self.assertEqual(ru_keys, en_keys)

        placeholder_re = re.compile(r"\{([a-zA-Z0-9_]+)(?::[^}]*)?\}")
        for key in ru_keys:
            ru_ph = set(placeholder_re.findall(STRINGS["ru"][key]))
            en_ph = set(placeholder_re.findall(STRINGS["en"][key]))
            self.assertEqual(ru_ph, en_ph, f"Mismatching format placeholders for key '{key}'")

        for key in TOOLTIPS:
            self.assertIn(key, TOOLTIPS_EN, f"Missing English tooltip for '{key}'")
            self.assertTrue(get_tooltip(key, "en", TOOLTIPS))
            self.assertTrue(get_tooltip(key, "ru", TOOLTIPS))

    def test_env_override_and_normalize_lang(self):
        self.assertEqual(normalize_lang("ru_RU"), "ru")
        self.assertEqual(normalize_lang("RU"), "ru")
        self.assertEqual(normalize_lang("en_US"), "en")
        self.assertEqual(normalize_lang("de_DE"), "en")
        self.assertEqual(normalize_lang(None), "en")

        with mock.patch.dict(os.environ, {"STRINGART_LANG": "en"}, clear=False):
            self.assertEqual(detect_system_language(respect_saved=True), "en")
        with mock.patch.dict(os.environ, {"STRINGART_LANG": "ru"}, clear=False):
            self.assertEqual(detect_system_language(respect_saved=True), "ru")
        with mock.patch.dict(os.environ, {"STRINGART_LANG": "fr_FR"}, clear=False):
            self.assertEqual(detect_system_language(respect_saved=True), "en")

    def test_os_language_detection_win32_and_posix(self):
        env_clean = {k: v for k, v in os.environ.items() if k not in ("STRINGART_LANG", "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE")}

        if sys.platform == "win32":
            import ctypes

            with mock.patch.dict(os.environ, env_clean, clear=True):
                with mock.patch.object(ctypes.windll.kernel32, "GetUserDefaultUILanguage", return_value=0x0419):
                    self.assertEqual(detect_system_language(respect_saved=False), "ru")
                with mock.patch.object(ctypes.windll.kernel32, "GetUserDefaultUILanguage", return_value=0x0409):
                    self.assertEqual(detect_system_language(respect_saved=False), "en")
                with mock.patch.object(ctypes.windll.kernel32, "GetUserDefaultUILanguage", return_value=0x0407):
                    self.assertEqual(detect_system_language(respect_saved=False), "en")

        # POSIX fallback check
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch("locale.getlocale", return_value=("ru_RU", "UTF-8")):
                with mock.patch.dict(os.environ, env_clean, clear=True):
                    self.assertEqual(detect_system_language(respect_saved=False), "ru")
            with mock.patch("locale.getlocale", return_value=("en_US", "UTF-8")):
                with mock.patch.dict(os.environ, env_clean, clear=True):
                    self.assertEqual(detect_system_language(respect_saved=False), "en")

    def test_canonical_display_roundtrips(self):
        for ru, en in SHAPE_EN.items():
            self.assertEqual(to_display_shape(ru, "en"), en)
            self.assertEqual(to_display_shape(en, "ru"), ru)
            self.assertEqual(to_canonical_shape(en), ru)
            self.assertEqual(to_canonical_shape(ru), ru)
        self.assertEqual(len(get_display_shapes("en")), len(SHAPE_EN))

        for ru, en in COLOR_MODE_EN.items():
            self.assertEqual(to_display_color_mode(ru, "en"), en)
            self.assertEqual(to_canonical_color_mode(en), ru)
        self.assertEqual(len(get_display_color_modes("en")), len(COLOR_MODE_EN))

        for ru, en in METRIC_EN.items():
            self.assertEqual(to_display_metric(ru, "en"), en)
            self.assertEqual(to_canonical_metric(en), ru)
        self.assertEqual(get_display_metrics("en", True), ["Distance Transform", "Contours"])

        for ru, en in PRESET_EN.items():
            self.assertEqual(to_display_preset(ru, "en"), en)
            self.assertEqual(to_canonical_preset(en), ru)
        self.assertEqual(len(get_display_presets("en")), len(PRESET_EN))

        for ru, en in LAYER_NAME_EN.items():
            self.assertEqual(translate_layer_name(ru, "en"), en)
            self.assertEqual(translate_layer_name(en, "ru"), ru)

        self.assertEqual(layer_short_code("Циан (Cyan)", "en"), "C")
        self.assertEqual(layer_short_code("Циан (Cyan)", "ru"), "Ц")
        self.assertEqual(layer_short_code("Черный (Black)", "en"), "K")
        self.assertEqual(tr("player_step", "en", step=3, total=10, p1=5, p2=42), "Step 3/10: pin 5 → 42")

    def test_english_shape_names_in_canvas_and_auto_recommend(self):
        cm_ru = CanvasManager(size=220, num_pins=60, shape="Прямоугольник (16:9)")
        cm_en = CanvasManager(size=220, num_pins=60, shape="Rectangle (16:9)")
        self.assertEqual(cm_en.shape, "Прямоугольник (16:9)")
        np.testing.assert_array_equal(cm_ru.pin_coords, cm_en.pin_coords)

        img = Image.new("RGB", (200, 200), (220, 220, 220))
        rec_en = auto_recommend_parameters(img, shape="Circle", lang="en")
        self.assertIn("Details:", rec_en["summary"])
        self.assertIn("pins", rec_en["summary"])
        self.assertIn("threads", rec_en["summary"])

        audio = AudioAssistant(lang="en")
        self.assertEqual(audio.lang, "en")
        audio.set_language("ru")
        self.assertEqual(audio.lang, "ru")
        audio.shutdown()

    def test_color_catalog_english_translations_and_short_codes(self):
        self.assertEqual(len(COLOR_CATALOG), 1017)
        cyr_re = re.compile(r"[\u0400-\u04ff]")
        en_names = set()
        for entry in COLOR_CATALOG:
            ru_name = entry["name"]
            en_name = entry.get("name_en", "")
            self.assertTrue(en_name, f"Empty name_en for {ru_name}")
            self.assertIsNone(cyr_re.search(en_name), f"Cyrillic left in name_en: {ru_name} -> {en_name}")
            en_names.add(en_name)
            self.assertEqual(COLOR_NAME_EN[ru_name], en_name)
            self.assertEqual(COLOR_NAME_RU[en_name], ru_name)
            self.assertEqual(translate_layer_name(ru_name, "en"), en_name)
            if en_name not in LAYER_NAME_EN.values():
                self.assertEqual(translate_layer_name(en_name, "ru"), ru_name)

        self.assertEqual(len(en_names), len(COLOR_CATALOG), "Duplicate name_en entries in COLOR_CATALOG")
        self.assertNotEqual(layer_short_code("Темный желтовато-коричневый", "ru"), "Ж")
        code_en = layer_short_code("Темный желтовато-коричневый", "en")
        self.assertIsNone(cyr_re.search(code_en))


if __name__ == "__main__":
    unittest.main()
