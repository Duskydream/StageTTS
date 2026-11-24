import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app
from backend import config
from backend.services import training_settings as service


class TrainingSettingsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target, name, value in [(service, 'SETTINGS_PATH', self.root/'training.json'),
                                     (config, 'GPT_SOVITS_ROOT', self.root),
                                     (config, 'GPT_SOVITS_CONFIG', str(self.root/'tts.yaml'))]:
            context = patch.object(target, name, value)
            context.start(); self.addCleanup(context.stop)
        (self.root/'tts.yaml').write_text('custom:\n  version: v2\n', encoding='utf-8')
        self.client = TestClient(app)

    def test_defaults_report_recipe_separately_from_inference(self):
        data = self.client.get('/api/settings/training').json()
        self.assertEqual(data['settings'], dict(base_model='v2Pro', gpt_epochs=15, sovits_epochs=10))
        self.assertEqual(data['inference_version'], 'v2')
        self.assertFalse(data['training_started'])
        self.assertTrue(all(not option['available'] for option in data['models']))
        self.assertFalse(service.SETTINGS_PATH.exists())

    def test_round_trip_persists_and_does_not_change_inference(self):
        before = (self.root/'tts.yaml').read_bytes()
        recipe = dict(base_model='v2ProPlus', gpt_epochs=20, sovits_epochs=12)
        response = self.client.put('/api/settings/training', json=recipe)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get('/api/settings/training').json()['settings'], recipe)
        self.assertEqual((self.root/'tts.yaml').read_bytes(), before)
        self.assertFalse(response.json()['training_started'])

    def test_invalid_values_do_not_replace_saved_recipe(self):
        valid = dict(base_model='v2Pro', gpt_epochs=15, sovits_epochs=10)
        self.client.put('/api/settings/training', json=valid)
        before = service.SETTINGS_PATH.read_bytes()
        for patch in [dict(gpt_epochs=0), dict(sovits_epochs=101), dict(gpt_epochs=1.5),
                      dict(gpt_epochs=True), dict(base_model='arbitrary'), dict(start_training=True)]:
            with self.subTest(patch=patch):
                self.assertEqual(self.client.put('/api/settings/training', json={**valid, **patch}).status_code, 422)
                self.assertEqual(service.SETTINGS_PATH.read_bytes(), before)

    def test_model_availability_requires_matching_files(self):
        files = [*service.BASE_MODELS['v2Pro'], 'sv/pretrained_eres2netv2w24s4ep4.ckpt']
        for file in files:
            path = self.root/'GPT_SoVITS'/'pretrained_models'/file
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        options = self.client.get('/api/settings/training').json()['models']
        selected = next(option for option in options if option['id']=='v2Pro')
        self.assertTrue(selected['available']); self.assertEqual(selected['missing_files'], [])

    def test_failed_write_keeps_previous_recipe(self):
        service.save_training_settings(service.TrainingSettings())
        before = service.SETTINGS_PATH.read_bytes()
        with patch.object(service.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                service.save_training_settings(service.TrainingSettings(gpt_epochs=25))
        self.assertEqual(service.SETTINGS_PATH.read_bytes(), before)
        self.assertFalse(list(self.root.glob('.training-settings-*.tmp')))
