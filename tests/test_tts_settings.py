import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.schemas import CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment
from backend.services import task_runner, tts_client, tts_settings
from workers import cosyvoice3_server


def _write_pcm_wav(path: Path, seconds: float = 0.6, sr: int = 24000) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"\x00\x01" * int(seconds * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)
    return path.read_bytes()


class TtsSettingsAndRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

        gsv_root = self.root / "gsv"
        gsv_root.mkdir()
        (gsv_root / "api_v2.py").write_text("# mock", encoding="utf-8")
        gsv_py = gsv_root / "python.exe"
        gsv_py.write_text("exe", encoding="utf-8")

        cv_root = self.root / "cosyvoice"
        (cv_root / "cosyvoice" / "cli").mkdir(parents=True)
        (cv_root / "cosyvoice" / "cli" / "cosyvoice.py").write_text("# mock", encoding="utf-8")
        cv_model = self.root / "cv_model"
        cv_model.mkdir()
        (cv_model / "cosyvoice3.yaml").write_text("sample_rate: 24000\n", encoding="utf-8")
        cv_py = self.root / "cv_py.exe"
        cv_py.write_text("exe", encoding="utf-8")

        self.gsv_root = gsv_root
        self.cv_root = cv_root
        self.cv_model = cv_model
        self.cv_py = cv_py

        patches = [
            patch.object(tts_settings, "SETTINGS_PATH", self.root / "tts_settings.local.json"),
            patch.object(config, "WORKSPACE_DIR", self.root / "workspace"),
            patch.object(config, "ASSETS_DIR", self.root / "assets"),
            patch.object(config, "TTS_AUTO_MANAGE", True),
            patch.object(config, "GPT_SOVITS_ROOT", gsv_root),
            patch.object(config, "GPT_SOVITS_PYTHON", str(gsv_py)),
            patch.object(config, "TTS_API_URL", "http://127.0.0.1:9880"),
            patch.object(config, "COSYVOICE3_ROOT", cv_root),
            patch.object(config, "COSYVOICE3_MODEL_DIR", cv_model),
            patch.object(config, "COSYVOICE3_PYTHON", str(cv_py)),
            patch.object(config, "COSYVOICE3_API_URL", "http://127.0.0.1:9881"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(app)

    def test_default_engine_and_switch_stops_previous_tts(self) -> None:
        res = self.client.get("/api/settings/tts")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["engine"], "gpt_sovits")
        by_id = {row["id"]: row for row in body["engines"]}
        self.assertTrue(by_id["gpt_sovits"]["available"])
        self.assertTrue(by_id["cosyvoice3"]["available"])

        with patch.object(tts_settings, "ensure_tts_stopped") as mock_stop:
            put_res = self.client.put("/api/settings/tts", json={"engine": "cosyvoice3"})
        self.assertEqual(put_res.status_code, 200, put_res.text)
        self.assertEqual(put_res.json()["engine"], "cosyvoice3")
        mock_stop.assert_called_once()
        self.assertEqual(self.client.get("/api/settings/tts").json()["engine"], "cosyvoice3")

    def test_unavailable_cosyvoice3_rejected(self) -> None:
        with patch.object(config, "COSYVOICE3_MODEL_DIR", None):
            res = self.client.get("/api/settings/tts")
            by_id = {row["id"]: row for row in res.json()["engines"]}
            self.assertFalse(by_id["cosyvoice3"]["available"])
            self.assertIn("COSYVOICE3_MODEL_DIR/cosyvoice3.yaml", by_id["cosyvoice3"]["missing_files"])

            put_res = self.client.put("/api/settings/tts", json={"engine": "cosyvoice3"})
            self.assertEqual(put_res.status_code, 400)
            self.assertEqual(tts_settings.read_tts_engine(), "gpt_sovits")

    def test_cosyvoice3_prompt_formatting_and_launch_spec(self) -> None:
        self.assertEqual(
            cosyvoice3_server.format_cosyvoice3_prompt("希望你以后能够做的比我还好呦。"),
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        )
        already = "You are a helpful assistant.<|endofprompt|>原句"
        self.assertEqual(cosyvoice3_server.format_cosyvoice3_prompt(already), already)

        spec = task_runner._build_tts_launch_spec("cosyvoice3", "127.0.0.1", 9881)
        self.assertIsNotNone(spec)
        assert spec is not None
        cmd, cwd_path, cfg_dir = spec
        self.assertIsNone(cfg_dir)
        self.assertEqual(cwd_path, config.ROOT_DIR)
        self.assertEqual(Path(cmd[0]), self.cv_py)
        self.assertEqual(Path(cmd[2]).name, "cosyvoice3_server.py")
        self.assertIn("9881", cmd)

    def test_cosyvoice3_routing_skips_gsv_weight_switch(self) -> None:
        ref_wav = config.WORKSPACE_DIR / "cv_proj" / "references" / "ref.wav"
        wav_bytes = _write_pcm_wav(ref_wav, seconds=0.8, sr=24000)

        script = ProjectScript(
            project_id="cv_proj",
            source_lang="ja",
            media=MediaInfo(source_video="/workspace/cv_proj/in.mp4", duration=10.0),
            characters={
                "S01": CharacterConfig(
                    display_name="角色A",
                    gpt_weights="GPT_weights_v2Pro/s1.ckpt",
                    sovits_weights="SoVITS_weights_v2Pro/s2.pth",
                    emotion_refs={
                        "neutral": EmotionRef(
                            wav="/workspace/cv_proj/references/ref.wav",
                            text="おはようございます",
                            lang="ja",
                        )
                    },
                )
            },
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=3.0,
                    speaker="S01",
                    orig_text="おはようございます",
                    target_text="早上好。",
                    target_lang="zh",
                )
            ],
        )

        mock_resp = MagicMock(status_code=200, content=wav_bytes)
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = mock_resp

        with patch.object(tts_client, "_active_tts_engine", return_value="cosyvoice3"), \
             patch.object(tts_client, "ensure_tts_running") as mock_ensure, \
             patch.object(tts_client, "_switch_weights") as mock_switch, \
             patch.object(tts_client.httpx, "Client", return_value=fake_client):
            synth_ids, errors = tts_client.synthesize_segments_grouped(script)

        self.assertEqual(errors, [])
        self.assertEqual(synth_ids, [0])
        mock_ensure.assert_called_once_with(engine="cosyvoice3")
        mock_switch.assert_not_called()
        posted_url = fake_client.post.call_args.args[0]
        self.assertEqual(posted_url, "http://127.0.0.1:9881/tts")
        self.assertEqual(script.segments[0].status, "ready")
        out_file = config.WORKSPACE_DIR / "cv_proj" / "tts" / "seg_0000.wav"
        self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()
