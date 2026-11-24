import json
import math
import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent.tools import load_project_script, save_project_script
from backend.main import app
from backend.schemas import CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment
from backend.services import emotion_routing, gsv_finetune, training_settings, tts_client, voice_presets
from backend.services.translation import DialogueContext


def _write_tone_wav(path: Path, segment_amps: list[tuple[float, float, float]], sample_rate: int = 32000) -> None:
    """Create a 32kHz mono PCM16 WAV with controlled per-window sine amplitudes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_end = max(end for _, end, _ in segment_amps)
    total_frames = round(total_end * sample_rate)
    samples = [0] * total_frames
    for start, end, amp in segment_amps:
        first = max(0, round(start * sample_rate))
        last = min(total_frames, round(end * sample_rate))
        for idx in range(first, last):
            val = int(max(-0.98, min(0.98, amp * math.sin(2 * math.pi * 240.0 * idx / sample_rate))) * 32767)
            samples[idx] = val
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class EmotionAndFinetuneTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.assets = self.root / "assets"
        self.gsv = self.root / "gsv"
        self.tts_cfg = self.root / "tts_v2.yaml"
        self.tts_cfg.write_text(
            "custom:\n  version: v2\n  t2s_weights_path: old.ckpt\n  vits_weights_path: old.pth\n",
            encoding="utf-8",
        )
        self.python_exe = self.root / "python.exe"
        self.python_exe.write_text("", encoding="utf-8")

        patches = [
            patch.object(config, "WORKSPACE_DIR", self.ws),
            patch.object(config, "ASSETS_DIR", self.assets),
            patch.object(config, "GPT_SOVITS_ROOT", self.gsv),
            patch.object(config, "GPT_SOVITS_CONFIG", str(self.tts_cfg)),
            patch.object(config, "GPT_SOVITS_PYTHON", str(self.python_exe)),
            patch.object(config, "TTS_AUTO_MANAGE", True),
            patch.object(config, "ENV_ANALYSIS_PYTHON", ""),
            patch.object(training_settings, "SETTINGS_PATH", self.root / "training_settings.local.json"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        gsv_finetune._jobs_by_project.clear()
        self.client = TestClient(app)

    def _populate_gsv_root(self, *, include_v2pro: bool = True, include_v2proplus: bool = False) -> None:
        pretrained = self.gsv / "GPT_SoVITS" / "pretrained_models"
        (pretrained / "chinese-roberta-wwm-ext-large").mkdir(parents=True, exist_ok=True)
        (pretrained / "chinese-hubert-base").mkdir(parents=True, exist_ok=True)
        configs_dir = self.gsv / "GPT_SoVITS" / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        (configs_dir / "s2v2Pro.json").write_text(
            json.dumps({"train": {}, "model": {}, "data": {}}), encoding="utf-8"
        )
        (configs_dir / "s1longer-v2.yaml").write_text(
            "train:\n  batch_size: 8\n  epochs: 20\ndata:\n  num_workers: 4\n",
            encoding="utf-8",
        )
        models_to_touch = []
        if include_v2pro:
            models_to_touch.extend(training_settings.BASE_MODELS["v2Pro"])
            models_to_touch.append("sv/pretrained_eres2netv2w24s4ep4.ckpt")
        if include_v2proplus:
            models_to_touch.extend(training_settings.BASE_MODELS["v2ProPlus"])
            models_to_touch.append("sv/pretrained_eres2netv2w24s4ep4.ckpt")
        for rel in models_to_touch:
            p = pretrained / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    def _create_project(self, project_id: str = "demo_proj", *, confirmed: bool = True, count: int = 7) -> ProjectScript:
        amplitudes = [0.16, 0.16, 0.70, 0.55, 0.03, 0.14, 0.65, 0.04][:count]
        while len(amplitudes) < count:
            amplitudes.append(0.15)
        spans = [(i * 4.5, i * 4.5 + 4.0, amp) for i, amp in enumerate(amplitudes)]
        wav_path = self.ws / project_id / "vocals.wav"
        _write_tone_wav(wav_path, spans, sample_rate=32000)

        segments = [
            Segment(
                id=i,
                start=i * 4.5,
                end=i * 4.5 + 4.0,
                speaker="S01",
                orig_text=f"テスト台詞番号 {i}",
                target_text=f"测试台词编号 {i}",
                target_lang="zh",
                alignment_granularity="segment",
                timing_source="forced_alignment" if confirmed else "moss_joint_generated",
                segment_timing_needs_review=not confirmed,
                speaker_needs_review=not confirmed,
                emotion="neutral",
            )
            for i in range(count)
        ]
        script = ProjectScript(
            project_id=project_id,
            source_lang="ja",
            media=MediaInfo(duration=count * 4.5 + 1.0, vocals_wav=f"/workspace/{project_id}/vocals.wav"),
            characters={"S01": CharacterConfig(display_name="角色A", speaking_style="干脆利落")},
            segments=segments,
        )
        save_project_script(script)
        return script

    def test_inference_base_model_sync_and_missing_rejection(self):
        self._populate_gsv_root(include_v2pro=True, include_v2proplus=False)
        before_yaml = self.tts_cfg.read_text(encoding="utf-8")

        bad = self.client.put("/api/settings/training/inference", json={"base_model": "v2ProPlus"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.tts_cfg.read_text(encoding="utf-8"), before_yaml)

        with patch.object(training_settings, "ensure_tts_stopped") as stop_tts, \
             patch.object(training_settings, "reset_active_weights_cache") as reset_cache:
            ok = self.client.put("/api/settings/training/inference", json={"base_model": "v2Pro"})
        self.assertEqual(ok.status_code, 200, ok.text)
        payload = ok.json()
        self.assertEqual(payload["inference_version"], "v2Pro")
        stop_tts.assert_called_once()
        reset_cache.assert_called_once()
        written = self.tts_cfg.read_text(encoding="utf-8")
        self.assertIn("version: v2Pro", written)
        self.assertIn("v2Pro/s2Gv2Pro.pth", written)

    def test_voice_preset_routes_round_trip(self):
        ref_wav = self.ws / "p1" / "references" / "neutral.wav"
        _write_tone_wav(ref_wav, [(0.0, 3.5, 0.2)])

        res_create = self.client.post(
            "/api/voice-presets",
            json={
                "name": "主角Preset",
                "speaking_style": "元气",
                "gpt_weights": "GPT_weights_v2Pro/hero.ckpt",
                "sovits_weights": "SoVITS_weights_v2Pro/hero.pth",
                "emotion_refs": {
                    "neutral": {"wav": "/workspace/p1/references/neutral.wav", "text": "こんにちは", "lang": "ja"}
                },
            },
        )
        self.assertEqual(res_create.status_code, 200, res_create.text)
        preset = res_create.json()
        self.assertEqual(preset["name"], "主角Preset")
        self.assertTrue((self.assets / "refs" / "presets" / preset["preset_id"] / "neutral.wav").is_file())

        res_list = self.client.get("/api/voice-presets")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual([item["preset_id"] for item in res_list.json()["presets"]], [preset["preset_id"]])

        res_del = self.client.delete(f"/api/voice-presets/{preset['preset_id']}")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.json()["deleted_id"], preset["preset_id"])
        self.assertEqual(self.client.get("/api/voice-presets").json()["presets"], [])

    def test_emotion_routing_acoustics_context_and_resolution(self):
        script = self._create_project("emo_proj", confirmed=True, count=7)
        script.segments[2].emotion = "angry"
        script.segments[3].emotion = "happy"
        script.segments[4].emotion = "whisper"
        save_project_script(script)

        fake_ser = {
            0: {"top_label": "neutral", "top_score": 0.85, "scores": {"neutral": 0.85}},
            2: {"top_label": "angry", "top_score": 0.78, "scores": {"angry": 0.78, "neutral": 0.12}},
            3: {"top_label": "happy", "top_score": 0.72, "scores": {"happy": 0.72, "neutral": 0.2}},
            4: {"top_label": "neutral", "top_score": 0.55, "scores": {"neutral": 0.55}},
            5: {"top_label": "sad", "top_score": 0.68, "scores": {"sad": 0.68}},
        }
        with patch.object(emotion_routing, "_ser_ready", return_value=True), \
             patch.object(emotion_routing, "_run_ser_batch", return_value=fake_ser):
            hints = emotion_routing.extract_segment_acoustics(script, use_ser=True)

        self.assertEqual(hints[2]["suggested_emotion"], "angry")
        self.assertEqual(hints[3]["suggested_emotion"], "happy")
        self.assertEqual(hints[4]["suggested_emotion"], "whisper")
        self.assertGreater(hints[2]["rms_z"], 0.8)
        self.assertLess(hints[4]["rms_z"], -1.0)

        ctx = DialogueContext(script, acoustic_hints=hints)
        entry_angry = ctx.entry(script.segments[2])
        self.assertEqual(entry_angry["acoustic_hint"]["ser_top"], "angry")
        self.assertEqual(entry_angry["acoustic_hint"]["suggested_emotion"], "angry")

        # Auto-resolution before manual config picks emotion-matched clip on angry segment
        auto_angry = tts_client._resolve_ref_for_segment(script, script.characters["S01"], script.segments[2])
        auto_whisper = tts_client._resolve_ref_for_segment(script, script.characters["S01"], script.segments[4])
        self.assertIsNotNone(auto_angry)
        self.assertIsNotNone(auto_whisper)
        self.assertNotEqual(auto_angry[1], auto_whisper[1])

        res = self.client.post("/api/projects/emo_proj/characters/S01/emotion-refs", json={"overwrite": False})
        self.assertEqual(res.status_code, 200, res.text)
        populated = res.json()["populated_emotions"]
        self.assertIn("neutral", populated)
        self.assertIn("angry", populated)
        self.assertIn("happy", populated)
        self.assertIn("whisper", populated)

        updated_char = res.json()["script"]["characters"]["S01"]
        self.assertNotEqual(
            updated_char["emotion_refs"]["angry"]["wav"],
            updated_char["emotion_refs"]["whisper"]["wav"],
        )

    def test_finetune_validation_and_epoch_divisors(self):
        self.assertEqual(gsv_finetune._save_every_epoch(15), 5)
        self.assertEqual(gsv_finetune._save_every_epoch(10), 5)
        self.assertEqual(gsv_finetune._save_every_epoch(8), 4)
        self.assertEqual(gsv_finetune._save_every_epoch(7), 1)

        self._populate_gsv_root(include_v2pro=True)
        # Too few clips (< 6)
        self._create_project("few_proj", confirmed=True, count=4)
        res_few = self.client.post("/api/projects/few_proj/characters/S01/finetune", json={})
        self.assertEqual(res_few.status_code, 400)
        self.assertIn("6 条", res_few.json()["detail"])

        # Unreviewed clips rejected when allow_unreviewed=False
        self._create_project("unrev_proj", confirmed=False, count=7)
        res_unrev = self.client.post(
            "/api/projects/unrev_proj/characters/S01/finetune",
            json={"allow_unreviewed": False},
        )
        self.assertEqual(res_unrev.status_code, 400)
        self.assertIn("允许使用未校对片段", res_unrev.json()["detail"])

    def test_finetune_full_mocked_pipeline_and_preset_registration(self):
        self._populate_gsv_root(include_v2pro=True)
        self._create_project("ft_proj", confirmed=True, count=7)
        executed_cmds: list[list[str]] = []

        def fake_slice(vocals_path: Path, start: float, duration: float, out_wav: Path) -> None:
            _write_tone_wav(out_wav, [(0.0, duration, 0.2)])

        def fake_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], on_line) -> None:
            executed_cmds.append(cmd)
            is_wrapped = Path(cmd[2]).name == "gsv_train_launcher.py"
            script_arg = cmd[3] if is_wrapped else cmd[2]
            cfg_path = cmd[5] if is_wrapped else (cmd[4] if len(cmd) > 4 else "")
            opt_dir = Path(env.get("opt_dir", "")) if env.get("opt_dir") else None
            if script_arg.endswith("1-get-text.py"):
                (opt_dir / "2-name2text-0.txt").write_text("seg_0000.wav\ta i u\t1 1 1\tあいう\n", encoding="utf-8")
            elif script_arg.endswith("2-get-sv.py"):
                sv_dir = opt_dir / "7-sv_cn"
                sv_dir.mkdir(parents=True, exist_ok=True)
                (sv_dir / "seg_0000.wav.pt").write_text("sv", encoding="utf-8")
            elif script_arg.endswith("3-get-semantic.py"):
                (opt_dir / "6-name2semantic-0.tsv").write_text("seg_0000.wav\t1 2 3\n", encoding="utf-8")
            elif script_arg.endswith("s2_train.py"):
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                out_dir = cwd / cfg["save_weight_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{cfg['name']}_e{cfg['train']['epochs']}_s20.pth").write_text("s2", encoding="utf-8")
            elif script_arg.endswith("s1_train.py"):
                import yaml

                cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
                out_dir = cwd / cfg["train"]["half_weights_save_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{cfg['train']['exp_name']}-e{cfg['train']['epochs']}.ckpt").write_text(
                    "s1", encoding="utf-8"
                )
            on_line(f"mocked {Path(script_arg).name}")

        with patch.object(gsv_finetune, "_slice_training_wav", side_effect=fake_slice), \
             patch.object(gsv_finetune, "_run_gsv_command", side_effect=fake_cmd), \
             patch.object(gsv_finetune, "_ensure_local_tts_released"):
            res = self.client.post(
                "/api/projects/ft_proj/characters/S01/finetune",
                json={"preset_name": "测试主角音色", "allow_unreviewed": False},
            )
            self.assertEqual(res.status_code, 200, res.text)

            final_status = None
            for _ in range(50):
                poll = self.client.get("/api/projects/ft_proj/finetune/status", params={"speaker": "S01"})
                self.assertEqual(poll.status_code, 200)
                final_status = poll.json()
                if not final_status["active"] and final_status["stage"] in {"completed", "error"}:
                    break
                time.sleep(0.05)

        self.assertIsNotNone(final_status)
        self.assertEqual(final_status["stage"], "completed", final_status)
        self.assertEqual(final_status["progress_pct"], 100)
        self.assertTrue(final_status["gpt_weights"].startswith("GPT_weights_v2Pro/"))
        self.assertTrue(final_status["sovits_weights"].startswith("SoVITS_weights_v2Pro/"))
        self.assertIsNotNone(final_status["preset_id"])

        steps = [
            Path(cmd[3] if Path(cmd[2]).name == "gsv_train_launcher.py" else cmd[2]).name
            for cmd in executed_cmds
        ]
        self.assertEqual(
            steps,
            [
                "1-get-text.py",
                "2-get-hubert-wav32k.py",
                "2-get-sv.py",
                "3-get-semantic.py",
                "s2_train.py",
                "s1_train.py",
            ],
        )

        saved_script = load_project_script("ft_proj")
        self.assertEqual(saved_script.characters["S01"].gpt_weights, final_status["gpt_weights"])
        self.assertEqual(saved_script.characters["S01"].sovits_weights, final_status["sovits_weights"])
        self.assertTrue(saved_script.characters["S01"].emotion_refs)

        presets = voice_presets.list_voice_presets()
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0].preset_id, final_status["preset_id"])
        self.assertEqual(presets[0].name, "测试主角音色")
        self.assertEqual(presets[0].gpt_weights, final_status["gpt_weights"])

    def test_windows_dataloader_launcher_forces_single_process(self) -> None:
        from workers import gsv_train_launcher

        class _FakeDataLoader:
            def __init__(
                self,
                dataset,
                batch_size=1,
                shuffle=None,
                sampler=None,
                batch_sampler=None,
                num_workers=0,
                collate_fn=None,
                *,
                prefetch_factor=None,
                persistent_workers=False,
            ) -> None:
                self.captured = (num_workers, persistent_workers, prefetch_factor)

        gsv_train_launcher.patch_windows_dataloader(_FakeDataLoader)
        gsv_train_launcher.patch_windows_dataloader(_FakeDataLoader)

        kw_loader = _FakeDataLoader("ds", num_workers=4, persistent_workers=True, prefetch_factor=2)
        self.assertEqual(kw_loader.captured, (0, False, None))

        pos_loader = _FakeDataLoader("ds", 4, False, None, None, 6, persistent_workers=True, prefetch_factor=16)
        self.assertEqual(pos_loader.captured, (0, False, None))

        with patch.object(gsv_finetune.os, "name", "nt"):
            nt_cmd = gsv_finetune._gsv_train_cmd("python.exe", "GPT_SoVITS/s2_train.py", ["--config", "cfg.json"])
        self.assertEqual(Path(nt_cmd[2]).name, "gsv_train_launcher.py")
        self.assertEqual(nt_cmd[3:], ["GPT_SoVITS/s2_train.py", "--config", "cfg.json"])

        with patch.object(gsv_finetune.os, "name", "posix"):
            posix_cmd = gsv_finetune._gsv_train_cmd("python", "GPT_SoVITS/s2_train.py", ["--config", "cfg.json"])
        self.assertEqual(posix_cmd, ["python", "-s", "GPT_SoVITS/s2_train.py", "--config", "cfg.json"])

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "dummy_train.py"
            script.write_text("VALUE = 1\n", encoding="utf-8")
            fake_sys_path = ["C:/embedded/python312.zip", "C:/embedded"]
            with patch.object(gsv_train_launcher.sys, "argv", ["launcher.py", str(script), "--flag", "1"]), \
                 patch.object(gsv_train_launcher.sys, "path", fake_sys_path), \
                 patch.object(gsv_train_launcher, "patch_windows_dataloader"), \
                 patch.object(gsv_train_launcher.runpy, "run_path") as mock_run_path:
                gsv_train_launcher.main()
                self.assertEqual(gsv_train_launcher.sys.argv, [str(script), "--flag", "1"])
            self.assertIn("C:/embedded/python312.zip", fake_sys_path)
            self.assertIn(str(script.parent.resolve()), fake_sys_path)
            mock_run_path.assert_called_once_with(str(script.resolve()), run_name="__main__")

    def test_patch_lightning_atomic_save_streams_and_cleans_temp_file(self) -> None:
        from types import SimpleNamespace
        from workers import gsv_train_launcher

        called_orig: list[tuple[dict, str]] = []

        def orig_save(ckpt, filepath) -> None:
            called_orig.append((ckpt, str(filepath)))

        cloud_io = SimpleNamespace(_atomic_save=orig_save)
        torch_io = SimpleNamespace(_atomic_save=orig_save)
        save_targets: list[str] = []

        def fake_save(ckpt, filepath) -> None:
            save_targets.append(filepath)
            Path(filepath).write_bytes(b"ckpt-bytes")

        gsv_train_launcher.patch_lightning_atomic_save(cloud_io, torch_io, save_fn=fake_save)
        gsv_train_launcher.patch_lightning_atomic_save(cloud_io, torch_io, save_fn=fake_save)
        self.assertIs(cloud_io._atomic_save, torch_io._atomic_save)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "nested" / "epoch=4.ckpt"
            cloud_io._atomic_save({"weights": 1}, dest)
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), b"ckpt-bytes")
            self.assertEqual(len(save_targets), 1)
            self.assertTrue(save_targets[0].endswith(".tmp"))
            self.assertFalse(Path(save_targets[0]).exists())

            cloud_io._atomic_save({"weights": 2}, "s3://bucket/model.ckpt")
            self.assertEqual(called_orig, [({"weights": 2}, "s3://bucket/model.ckpt")])

            def failing_save(ckpt, filepath) -> None:
                Path(filepath).write_bytes(b"partial")
                raise RuntimeError("disk full")

            err_cloud = SimpleNamespace(_atomic_save=orig_save)
            err_torch = SimpleNamespace(_atomic_save=orig_save)
            gsv_train_launcher.patch_lightning_atomic_save(err_cloud, err_torch, save_fn=failing_save)
            fail_dest = Path(tmpdir) / "fail.ckpt"
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                err_cloud._atomic_save({"weights": 3}, fail_dest)
            self.assertFalse(fail_dest.exists())
            self.assertEqual(list(Path(tmpdir).glob("fail.ckpt*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
