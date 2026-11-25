"""Regression tests for few-shot GPT-SoVITS training and the 4 natural dubbing upgrades."""
from __future__ import annotations

import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from backend import config
from backend.agent.tools import save_project_script
from backend.schemas import CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment
from backend.services import (
    emotion_routing,
    gsv_finetune,
    media_mixer,
    pause_alignment,
    training_settings,
    tts_client,
)
from backend.services.translation import DialogueContext


def _write_tone_blocks(
    path: Path,
    windows: list[tuple[float, float, float]],
    total_duration: float,
    *,
    sample_rate: int = 32000,
    noise_amp: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_total = round(total_duration * sample_rate)
    samples = [0.0] * n_total
    if noise_amp > 0:
        for i in range(n_total):
            samples[i] = noise_amp * math.sin(2.0 * math.pi * 1300.0 * i / sample_rate)
    for start, end, amp in windows:
        first = max(0, round(start * sample_rate))
        last = min(n_total, round(end * sample_rate))
        for i in range(first, last):
            samples[i] += amp * math.sin(2.0 * math.pi * 240.0 * i / sample_rate)
    pcm = [int(max(-0.98, min(0.98, v)) * 32767) for v in samples]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


class NaturalDubbingUpgradesTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.assets = self.root / "assets"
        self.gsv = self.root / "gsv"
        patches = [
            patch.object(config, "WORKSPACE_DIR", self.ws),
            patch.object(config, "ASSETS_DIR", self.assets),
            patch.object(config, "GPT_SOVITS_ROOT", self.gsv),
            patch.object(config, "GPT_SOVITS_PYTHON", str(self.root / "python.exe")),
            patch.object(config, "ENV_ANALYSIS_PYTHON", ""),
            patch.object(training_settings, "SETTINGS_PATH", self.root / "training_settings.local.json"),
        ]
        (self.root / "python.exe").write_text("", encoding="utf-8")
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _populate_gsv(self) -> None:
        pretrained = self.gsv / "GPT_SoVITS" / "pretrained_models"
        (pretrained / "chinese-roberta-wwm-ext-large").mkdir(parents=True, exist_ok=True)
        (pretrained / "chinese-hubert-base").mkdir(parents=True, exist_ok=True)
        cfg_dir = self.gsv / "GPT_SoVITS" / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "s2v2Pro.json").write_text('{"train":{},"model":{},"data":{}}', encoding="utf-8")
        (cfg_dir / "s1longer-v2.yaml").write_text("train:\n  batch_size: 4\ndata:\n  num_workers: 4\n", encoding="utf-8")
        for rel in [*training_settings.BASE_MODELS["v2Pro"], "sv/pretrained_eres2netv2w24s4ep4.ckpt"]:
            p = pretrained / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    def test_few_shot_long_monologue_split_and_skip_gpt(self) -> None:
        self._populate_gsv()
        wav_path = self.ws / "mono" / "vocals.wav"
        # 24s segment with 3 clauses separated by 0.5s pauses around 7.5s and 15.5s
        _write_tone_blocks(
            wav_path,
            [(0.2, 7.2, 0.25), (7.8, 15.2, 0.25), (15.8, 23.5, 0.25)],
            total_duration=25.0,
        )
        seg = Segment(
            id=0,
            start=0.0,
            end=24.0,
            speaker="S01",
            orig_text="今日は天気がとても良いですね。みんなで図書室に行きましょう！午後はお茶会ですよ。",
            target_text="今天天气真好啊，大家一起去图书室吧！下午还有茶会呢。",
            speaker_needs_review=False,
        )
        script = ProjectScript(
            project_id="mono",
            source_lang="ja",
            media=MediaInfo(duration=25.0, vocals_wav="/workspace/mono/vocals.wav"),
            characters={"S01": CharacterConfig(display_name="白羽")},
            segments=[seg],
        )
        slices = gsv_finetune._build_training_slices(seg, wav_path)
        self.assertGreaterEqual(len(slices), 2)
        self.assertTrue(all(1.0 <= (sl.end - sl.start) <= 20.0 for sl in slices))

        # Force few_shot mode and ensure s1_train is skipped + gpt_weights stays None
        save_project_script(script)
        executed: list[str] = []

        def fake_cmd(cmd, *, cwd, env, on_line):
            arg = cmd[3] if Path(cmd[2]).name == "gsv_train_launcher.py" else cmd[2]
            name = Path(arg).name
            executed.append(name)
            opt_dir = Path(env["opt_dir"])
            if name == "1-get-text.py":
                (opt_dir / "2-name2text-0.txt").write_text("a\tb\tc\td\n", encoding="utf-8")
            elif name == "2-get-sv.py":
                (opt_dir / "7-sv_cn").mkdir(parents=True, exist_ok=True)
                (opt_dir / "7-sv_cn" / "a.pt").write_text("sv", encoding="utf-8")
            elif name == "3-get-semantic.py":
                (opt_dir / "6-name2semantic-0.tsv").write_text("a\t1 2\n", encoding="utf-8")
            elif name == "s2_train.py":
                import json as _json
                cfg_arg = cmd[5] if Path(cmd[2]).name == "gsv_train_launcher.py" else cmd[4]
                cfg = _json.loads(Path(cfg_arg).read_text(encoding="utf-8"))
                out_dir = cwd / cfg["save_weight_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{cfg['name']}_e{cfg['train']['epochs']}_s10.pth").write_text("s2", encoding="utf-8")

        plan = gsv_finetune.validate_finetune_request(
            script, "S01", gsv_finetune.FinetuneStartRequest(mode="few_shot")
        )
        self.assertEqual(plan.training_mode, "few_shot")
        self.assertEqual(plan.effective_gpt_epochs, 0)
        gsv_finetune._jobs_by_project["mono"] = gsv_finetune.FinetuneJobStatus(
            job_id="ft_1", active=True, project_id="mono", speaker="S01",
            training_mode="few_shot", base_model="v2Pro",
            gpt_epochs=0, sovits_epochs=plan.effective_sovits_epochs,
        )
        with patch.object(gsv_finetune, "_slice_training_wav", side_effect=lambda *a: _write_tone_blocks(a[3], [(0.0, 3.0, 0.2)], 3.0)), \
             patch.object(gsv_finetune, "_run_gsv_command", side_effect=fake_cmd), \
             patch.object(gsv_finetune, "_ensure_local_tts_released"):
            gsv_finetune._execute_pipeline(plan, gsv_finetune.FinetuneStartRequest(mode="few_shot"))
        self.assertNotIn("s1_train.py", executed)
        self.assertIn("s2_train.py", executed)
        self.assertIsNone(gsv_finetune.get_finetune_status("mono").gpt_weights)
        self.assertTrue(gsv_finetune.get_finetune_status("mono").sovits_weights.endswith(".pth"))

    def test_clarity_scoring_and_aux_ref_collection(self) -> None:
        wav_path = self.ws / "clarity" / "vocals.wav"
        # seg 0: noisy background + abrupt cut tail; seg 1 & 2: clean quiet pauses + clear speech + soft tail
        _write_tone_blocks(
            wav_path,
            [
                (0.0, 4.0, 0.25),          # constant noise floor added below
                (5.2, 8.6, 0.30),          # clean voiced inside [5.0, 9.0] with trailing fade room
                (10.2, 13.7, 0.28),
            ],
            total_duration=15.0,
        )
        # Inject noise onto [0.0..4.0] only
        with wave.open(str(wav_path), "rb") as wf:
            pcm = list(struct.unpack(f"<{wf.getnframes()}h", wf.readframes(wf.getnframes())))
        for i in range(0, 4 * 32000):
            pcm[i] = max(-32000, min(32000, pcm[i] + int(0.18 * 32767 * math.cos(2 * math.pi * 1900 * i / 32000))))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(32000)
            wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))

        script = ProjectScript(
            project_id="clarity",
            source_lang="ja",
            media=MediaInfo(duration=15.0, vocals_wav="/workspace/clarity/vocals.wav"),
            characters={
                "S01": CharacterConfig(
                    display_name="角色1",
                    emotion_refs={"neutral": EmotionRef(wav="/workspace/clarity/references/main.wav", text="a", lang="ja")},
                )
            },
            segments=[
                Segment(id=0, start=0.0, end=4.0, speaker="S01", orig_text="noisy", speaker_needs_review=False, segment_timing_needs_review=False),
                Segment(id=1, start=5.0, end=9.0, speaker="S01", orig_text="clean one", speaker_needs_review=False, segment_timing_needs_review=False),
                Segment(id=2, start=10.0, end=14.0, speaker="S01", orig_text="clean two", speaker_needs_review=False, segment_timing_needs_review=False),
            ],
        )
        main_ref = self.ws / "clarity" / "references" / "main.wav"
        _write_tone_blocks(main_ref, [(0.0, 3.5, 0.2)], 3.5)

        hints = emotion_routing.extract_segment_acoustics(script, use_ser=False)
        self.assertGreater(hints[1]["clarity_score"], hints[0]["clarity_score"])
        aux = tts_client._collect_aux_ref_paths(script, script.characters["S01"], "S01", main_ref, acoustics=hints)
        self.assertTrue(aux)
        self.assertTrue(all(p.resolve() != main_ref.resolve() for p in aux))

    def test_speaker_pace_budget_and_pause_alignment(self) -> None:
        wav_path = self.ws / "pace" / "vocals.wav"
        # Segment 0: slow pauses (two 1.2s islands separated by 1.6s silence inside a 5s slot)
        # Segment 1: dense fast speaker
        _write_tone_blocks(
            wav_path,
            [
                (0.2, 1.4, 0.25),
                (3.0, 4.4, 0.25),
                (5.6, 9.6, 0.25),
                (10.2, 14.0, 0.25),
            ],
            total_duration=15.0,
        )
        script = ProjectScript(
            project_id="pace",
            source_lang="ja",
            media=MediaInfo(duration=15.0, vocals_wav="/workspace/pace/vocals.wav"),
            characters={
                "S01": CharacterConfig(display_name="慢速角色"),
                "S02": CharacterConfig(display_name="快速角色"),
            },
            segments=[
                Segment(id=0, start=0.0, end=5.0, speaker="S01", orig_text="あの……本当ですか？", target_text="那个……这是真的吗？"),
                Segment(id=1, start=5.5, end=9.7, speaker="S02", orig_text="えっとですね実は昨日図書館でとても不思議な本を見つけたんです！"),
                Segment(id=2, start=10.1, end=14.1, speaker="S02", orig_text="しかもその本には古い地図が挟まっていたんですよ！"),
            ],
        )
        ctx = DialogueContext(script)
        e0 = ctx.entry(script.segments[0])
        self.assertIn("acoustic_hint", e0)
        self.assertIn("effective_speech_sec", e0["acoustic_hint"])
        self.assertLess(e0["acoustic_hint"]["effective_speech_sec"], 5.0)
        self.assertGreaterEqual(e0["acoustic_hint"].get("internal_pause_sec", 0.0), 0.35)

        islands = pause_alignment.detect_speech_islands(wav_path, 0.0, 5.0)
        self.assertEqual(len(islands), 2)
        aligned = pause_alignment.align_clauses_to_islands("那个，我觉得这是真的吧！", "zh", islands)
        self.assertIsNotNone(aligned)
        self.assertEqual(len(aligned), 2)

        with tempfile.TemporaryDirectory() as tmp:
            p0 = Path(tmp) / "p0.wav"
            p1 = Path(tmp) / "p1.wav"
            _write_tone_blocks(p0, [(0.05, 1.05, 0.3)], 1.1)
            _write_tone_blocks(p1, [(0.05, 1.25, 0.3)], 1.3)
            assembled = Path(tmp) / "out.wav"
            pause_alignment.assemble_island_wavs([p0, p1], aligned, 5.0, assembled, Path(tmp))
            with wave.open(str(assembled), "rb") as wf:
                frames = struct.unpack(f"<{wf.getnframes()}h", wf.readframes(wf.getnframes()))
                rate = wf.getframerate()
                mid_rms = math.sqrt(sum(v * v for v in frames[int(1.8 * rate):int(2.6 * rate)]) / int(0.8 * rate))
                self.assertLess(mid_rms, 50.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_vocal_space_filter_and_source_loudness_profile(self) -> None:
        wav_path = self.ws / "space" / "vocals.wav"
        _write_tone_blocks(wav_path, [(0.1, 2.8, 0.22)], total_duration=3.0)
        script = ProjectScript(
            project_id="space",
            media=MediaInfo(duration=3.0, vocals_wav="/workspace/space/vocals.wav"),
            segments=[Segment(id=0, start=0.0, end=2.9, speaker="S01")],
        )
        dbfs, wet = media_mixer._measure_source_dialogue_profile(script)
        self.assertGreaterEqual(dbfs, -21.0)
        self.assertLessEqual(dbfs, -15.0)
        self.assertGreaterEqual(wet, 0.06)
        self.assertLessEqual(wet, 0.16)
        flt = media_mixer._build_vocal_space_filter(wet)
        self.assertIn("highpass=", flt)
        self.assertIn("aecho=", flt)

    def test_runaway_silence_seed_retry_and_trim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ref_wav = tmp_dir / "ref.wav"
            bad_wav = tmp_dir / "bad.wav"
            good_wav = tmp_dir / "good.wav"
            out_wav = tmp_dir / "out.wav"
            _write_tone_blocks(ref_wav, [(0.0, 3.5, 0.25)], 3.5)
            _write_tone_blocks(bad_wav, [(12.0, 14.5, 0.25)], 15.0)
            _write_tone_blocks(good_wav, [(0.8, 3.0, 0.25)], 3.2)

            class _Resp:
                def __init__(self, content: bytes) -> None:
                    self.status_code = 200
                    self.content = content

            class _FakeClient:
                def __init__(self) -> None:
                    self.seeds: list[int] = []

                def post(self, _url, *, json, timeout):
                    self.seeds.append(json["seed"])
                    return _Resp(bad_wav.read_bytes() if len(self.seeds) == 1 else good_wav.read_bytes())

            fake = _FakeClient()
            seg = Segment(id=27, start=0.0, end=2.8, speaker="S03", orig_text="x", target_text="备好了吗？")
            ref = EmotionRef(wav="/workspace/x/ref.wav", text="a", lang="ja")
            tts_client._synthesize_single_line(fake, seg, ref, ref_wav, out_wav)
            self.assertEqual(len(fake.seeds), 2)
            self.assertEqual((fake.seeds[0] + 1) % (2**31 - 1), fake.seeds[1])
            with wave.open(str(out_wav), "rb") as wf:
                trimmed_dur = wf.getnframes() / float(wf.getframerate())
            self.assertLess(trimmed_dur, 2.6)


if __name__ == "__main__":
    unittest.main()
