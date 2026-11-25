"""UI LLM preferences, secret handling and immutable request configuration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from backend import config
from backend.main import app
from backend.services import llm_settings as settings
from backend.services import translation


class LLMSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'llm_settings.local.json'
        for target, value in [('SETTINGS_PATH', self.path)]:
            context = patch.object(settings, target, value)
            context.start()
            self.addCleanup(context.stop)
        for key, value in [
            ('LLM_API_KEY', 'old-secret-key'),
            ('LLM_BASE_URL', 'https://old.example/v1'),
            ('LLM_MODEL', 'old-model'),
            ('DIRECTOR_LLM_API_KEY', ''),
            ('DIRECTOR_LLM_BASE_URL', ''),
            ('DIRECTOR_LLM_MODEL', ''),
        ]:
            context = patch.object(config, key, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict('os.environ', {'STAGETTS_LLM_SNAPSHOT': '0'})
        context.start()
        self.addCleanup(context.stop)
        self.client = TestClient(app)

    def put(self, **changes):
        return self.client.put('/api/settings/llm', json={'base_url': 'https://new.example/v1', 'model': 'new-model', **changes})

    def test_get_uses_environment_defaults_but_never_returns_key(self):
        response = self.client.get('/api/settings/llm')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['model'], 'old-model')
        self.assertTrue(response.json()['api_key_configured'])
        self.assertNotIn('old-secret-key', response.text)
        self.assertFalse(self.path.exists())

    def test_save_persists_across_reads_and_blank_key_preserves_secret(self):
        response = self.put(api_key='new-secret')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn('new-secret', response.text)
        self.assertEqual(json.loads(self.path.read_text())['api_key'], 'new-secret')
        response = self.put(model='third-model', api_key='')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(settings.get_llm_snapshot().api_key, 'new-secret')
        self.assertEqual(self.client.get('/api/settings/llm').json()['model'], 'third-model')
        self.assertNotIn('new-secret', repr(settings.get_llm_snapshot()))

    def test_clear_key_does_not_fall_back_to_environment_secret(self):
        response = self.put(clear_api_key=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['configured'])
        self.assertFalse(config.is_llm_configured())
        self.assertEqual(settings.get_llm_snapshot().api_key, '')

    def test_invalid_settings_never_replace_existing_configuration(self):
        self.put()
        before = self.path.read_bytes()
        for changes in [{'base_url': 'file:///private'}, {'base_url': 'https://user:pass@example.com/v1'},
                        {'base_url': 'https://example.com/v1?key=secret'}, {'model': ' '},
                        {'clear_api_key': True, 'api_key': 'new-key'}]:
            with self.subTest(changes=changes):
                self.assertIn(self.put(**changes).status_code, [400, 422])
                self.assertEqual(self.path.read_bytes(), before)

    def test_atomic_write_failure_preserves_previous_settings(self):
        self.put()
        before = self.path.read_bytes()
        with patch.object(settings.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                settings.save_llm_settings(settings.LLMSettingsUpdate(base_url='https://other.example/v1', model='other'))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('.llm-settings-*.tmp')), [])

    def test_client_keeps_model_url_and_key_snapshot_after_settings_change(self):
        self.put(api_key='first-key')
        with patch.object(translation, 'OpenAI') as factory:
            first = MagicMock()
            second = MagicMock()
            factory.side_effect = [first, second]
            self.assertIs(translation.create_openai_client(), first)
            self.assertEqual(factory.call_args.kwargs['api_key'], 'first-key')
            self.assertEqual(factory.call_args.kwargs['base_url'], 'https://new.example/v1')
            self.put(model='another-model', api_key='second-key')
            self.assertIs(translation.create_openai_client(), second)
            first.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[]'))])
            translation.complete_dialogue(first, 'test', {})
            self.assertEqual(first.chat.completions.create.call_args.kwargs['model'], 'new-model')
            self.assertEqual(translation.client_model(second), 'another-model')

    def test_real_sdk_client_accepts_bound_model_without_any_network_request(self):
        self.put(api_key='sdk-test-key')
        client = translation.create_openai_client()
        try:
            self.assertEqual(translation.client_model(client), 'new-model')
            self.assertEqual(str(client.base_url), 'https://new.example/v1/')
        finally:
            client.close()

    def test_parse_snapshot_does_not_read_later_ui_changes(self):
        self.put(model='ui-model', api_key='ui-key')
        with patch.dict('os.environ', {'STAGETTS_LLM_SNAPSHOT': '1'}):
            snapshot = settings.get_llm_snapshot()
        self.assertEqual(snapshot.model, 'old-model')
        self.assertEqual(snapshot.api_key, 'old-secret-key')

    def test_corrupt_settings_reports_safe_error_without_secret_contents(self):
        self.path.write_text('{broken: super-secret-key', encoding='utf-8')
        response = self.client.get('/api/settings/llm')
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('super-secret-key', response.text)

    def test_legacy_three_key_file_loads_cleanly_without_director_fields(self):
        self.path.write_text(
            json.dumps({'base_url': 'https://legacy.example/v1', 'model': 'legacy-model', 'api_key': 'legacy-key'}),
            encoding='utf-8',
        )
        response = self.client.get('/api/settings/llm')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['model'], 'legacy-model')
        self.assertFalse(body['director_configured'])
        self.assertEqual(body['director_model'], '')
        fallback = MagicMock()
        self.assertIs(translation.create_director_client(fallback), fallback)

    def test_director_model_inherits_or_overrides_and_never_leaks_secret(self):
        # 1) Custom director_model alone inherits primary base_url & api_key
        resp = self.put(api_key='primary-key', director_model='director-pro')
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body['director_model'], 'director-pro')
        self.assertEqual(body['director_base_url'], '')
        self.assertFalse(body['director_api_key_configured'])
        self.assertTrue(body['director_configured'])

        with patch.object(translation, 'OpenAI') as factory:
            d_client = MagicMock()
            factory.return_value = d_client
            self.assertIs(translation.create_director_client(MagicMock()), d_client)
            self.assertEqual(factory.call_args.kwargs['api_key'], 'primary-key')
            self.assertEqual(factory.call_args.kwargs['base_url'], 'https://new.example/v1')
            self.assertEqual(translation.client_model(d_client), 'director-pro')

        # 2) Independent director URL + API key
        resp2 = self.put(
            director_model='claude-director',
            director_base_url='https://director.example/v1/',
            director_api_key='director-secret',
        )
        self.assertEqual(resp2.status_code, 200, resp2.text)
        self.assertNotIn('director-secret', resp2.text)
        self.assertNotIn('director-secret', repr(settings.get_llm_snapshot()))
        self.assertEqual(resp2.json()['director_base_url'], 'https://director.example/v1')
        self.assertTrue(resp2.json()['director_api_key_configured'])

        with patch.object(translation, 'OpenAI') as factory:
            d_client2 = MagicMock()
            factory.return_value = d_client2
            translation.create_director_client()
            self.assertEqual(factory.call_args.kwargs['api_key'], 'director-secret')
            self.assertEqual(factory.call_args.kwargs['base_url'], 'https://director.example/v1')
            self.assertEqual(translation.client_model(d_client2), 'claude-director')

        # 3) Legacy PUT without director fields preserves director config; invalid director update rejected
        self.assertEqual(self.put(model='updated-primary').status_code, 200)
        self.assertEqual(settings.get_llm_snapshot().director_model, 'claude-director')
        self.assertEqual(settings.get_llm_snapshot().director_api_key, 'director-secret')
        self.assertIn(self.put(clear_director_api_key=True, director_api_key='x').status_code, [400, 422])
        self.assertIn(self.put(director_base_url='ftp://bad.example').status_code, [400, 422])

        # 4) Clearing director key falls back to primary key; clearing director_model reverts to primary client
        resp3 = self.put(clear_director_api_key=True)
        self.assertEqual(resp3.status_code, 200)
        self.assertFalse(resp3.json()['director_api_key_configured'])
        self.assertTrue(resp3.json()['director_configured'])
        resp4 = self.put(director_model='', director_base_url='')
        self.assertEqual(resp4.status_code, 200)
        self.assertFalse(resp4.json()['director_configured'])
        fallback = MagicMock()
        self.assertIs(translation.create_director_client(fallback), fallback)

    def test_parse_snapshot_reads_director_config_defaults(self):
        self.put(model='ui-model', api_key='ui-key', director_model='ui-dir', director_api_key='ui-dir-key')
        with (
            patch.object(config, 'DIRECTOR_LLM_MODEL', 'env-dir-model'),
            patch.object(config, 'DIRECTOR_LLM_BASE_URL', 'https://env-dir.example/v1'),
            patch.object(config, 'DIRECTOR_LLM_API_KEY', 'env-dir-key'),
            patch.dict('os.environ', {'STAGETTS_LLM_SNAPSHOT': '1'}),
        ):
            snapshot = settings.get_llm_snapshot()
        self.assertEqual(snapshot.director_model, 'env-dir-model')
        self.assertEqual(snapshot.director_base_url, 'https://env-dir.example/v1')
        self.assertEqual(snapshot.director_api_key, 'env-dir-key')


if __name__ == '__main__':
    unittest.main()
