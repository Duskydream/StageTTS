"""Tests for dubbing length calibration, MOSS acoustic segmentation, and autonomous Agent QA."""

from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent import orchestrator, tools
from backend.main import app
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from backend.services import pause_alignment, text_normalization, translation, tts_client
from workers.moss_segmentation import refine_moss_segments


def _write_tone_blocks(path: Path, blocks: list[tuple[float, float, float]], total_sec: float, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(round(total_sec * rate))
    buf = [0] * nframes
    for start, end, amp in blocks:
        i0 = max(0, int(round(start * rate)))
        i1 = min(nframes, int(round(end * rate)))
        for idx in range(i0, i1):
            buf[idx] = int(amp * 32767 * math.sin(2 * math.pi * 440.0 * idx / rate))
    with wave.open(str(path), "wb") as wf:
        wf.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wf.writeframes(struct.pack(f"<{nframes}h", *buf))


def _completion(content=None, tool_calls=()):
    payload = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload, tool_calls=tool_calls))]
    )


class TextNormalizationAndPreQaTests(unittest.TestCase):
    def test_clean_dubbing_text_and_short_ellipsis_clamping(self):
        self.assertEqual(text_normalization.clean_dubbing_text("银发小姐，耳环／耳钉如何？"), "银发小姐，耳环如何？")
        self.assertEqual(text_normalization.clean_dubbing_text("小三门（暂译）前辈"), "小三门前辈")
        self.assertEqual(text_normalization.clean_dubbing_text("A/B 测试   保留 "), "A/B 测试 保留")
        self.assertEqual(text_normalization.prepare_tts_text("可是……", "zh"), "可是。")
        self.assertEqual(text_normalization.prepare_tts_text("就算再怎么说也不该这样……", "zh"), "就算再怎么说也不该这样……")

    def test_director_pre_qa_tightens_dense_or_slashed_lines_and_persists_show_bible(self):
        script = ProjectScript(
            project_id="pre_qa",
            source_lang="ja",
            media=MediaInfo(duration=6.0),
            characters={"S01": CharacterConfig(display_name="让叶")},
            segments=[
                Segment(id=0, start=0.0, end=2.5, speaker="S01", orig_text="とても素敵な空間ですね。"),
            ],
        )
        director = MagicMock()
        translator = MagicMock()
        director.chat.completions.create.side_effect = [
            _completion({
                "genre": "日常",
                "tone": "轻松",
                "character_styles": {"S01": "干练温和"},
                "relationships": [],
                "glossary": {},
                "uncertain_terms": [],
            }),
            _completion([
                {
                    "id": 0,
                    "target_text": "真是太棒的空间呢，耳环／耳钉很配，整体气氛简直好得没话说！",
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
            _completion([
                {
                    "id": 0,
                    "target_text": "这地方真棒，气氛真好。",
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
        ]
        translator.chat.completions.create.side_effect = [
            _completion([
                {
                    "id": 0,
                    "target_text": "这里真是不错的空间啊，让人心情格外舒畅呢。",
                    "emotion": "happy",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
        ]
        ids = translation.translate_script(
            script,
            translator,
            enable_analysis=True,
            enable_polish=True,
            director_client=director,
        )
        self.assertEqual(ids, [0])
        self.assertEqual(director.chat.completions.create.call_count, 3)
        self.assertEqual(script.segments[0].target_text, "这地方真棒，气氛真好。")
        self.assertIsNotNone(script.show_bible)
        self.assertEqual(script.show_bible["genre"], "日常")


class MossSegmentationTests(unittest.TestCase):
    def test_snaps_clipped_tail_splits_long_monologue_and_merges_tiny_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "dialogue.wav"
            # Seg 0: 0.2..13.8 with genuine internal pause 6.6..7.2 (split candidate!)
            # Seg 1: 14.5..15.1 ("奶——") + Seg 2: 15.6..19.0 (same speaker -> merge!)
            # Seg 3: 20.0..22.0 speech extends slightly to 22.18 (tail snap!)
            _write_tone_blocks(
                wav,
                [
                    (0.25, 6.6, 0.25),
                    (7.2, 13.7, 0.25),
                    (14.55, 15.05, 0.25),
                    (15.65, 18.95, 0.25),
                    (20.05, 22.18, 0.25),
                ],
                total_sec=23.0,
            )
            raw = [
                {
                    "start": 0.2,
                    "end": 13.8,
                    "speaker": "S03",
                    "orig_text": "一階は普通の客向けで、二階から三階が貴賓客用です。そして四階が私たちの住居になっています。",
                    "words": [],
                    "alignment_granularity": "segment",
                    "timing_source": "moss_joint_generated",
                    "alignment_needs_review": True,
                    "segment_timing_needs_review": True,
                    "speaker_needs_review": True,
                },
                {
                    "start": 14.5,
                    "end": 15.1,
                    "speaker": "S03",
                    "orig_text": "おば。",
                    "words": [],
                    "alignment_granularity": "segment",
                    "timing_source": "moss_joint_generated",
                    "alignment_needs_review": True,
                    "segment_timing_needs_review": True,
                    "speaker_needs_review": True,
                },
                {
                    "start": 15.6,
                    "end": 19.0,
                    "speaker": "S03",
                    "orig_text": "咲さん、今日はマリッチリングの話だったよね。",
                    "words": [],
                    "alignment_granularity": "segment",
                    "timing_source": "moss_joint_generated",
                    "alignment_needs_review": True,
                    "segment_timing_needs_review": True,
                    "speaker_needs_review": True,
                },
                {
                    "start": 20.0,
                    "end": 22.0,
                    "speaker": "S01",
                    "orig_text": "とても素敵な空間ですね。",
                    "words": [],
                    "alignment_granularity": "segment",
                    "timing_source": "moss_joint_generated",
                    "alignment_needs_review": True,
                    "segment_timing_needs_review": True,
                    "speaker_needs_review": True,
                },
            ]
            refined = refine_moss_segments(raw, wav)
            self.assertEqual(len(refined), 4)
            # Seg 0 split into two balanced children (~6.7s each)
            self.assertLess(refined[0]["end"] - refined[0]["start"], 9.0)
            self.assertLess(refined[1]["end"] - refined[1]["start"], 9.0)
            self.assertEqual(refined[0]["orig_text"] + refined[1]["orig_text"], raw[0]["orig_text"])
            # Tiny fragment merged into following S03 line
            self.assertEqual(refined[2]["orig_text"], "おば。咲さん、今日はマリッチリングの話だったよね。")
            # Clipped tail extended cleanly past 22.15s
            self.assertGreater(refined[3]["end"], 22.15)
            # Preserves MOSS segment metadata invariants
            self.assertTrue(all(row["alignment_granularity"] == "segment" and row["words"] == [] for row in refined))

    def test_pause_alignment_ignores_brief_mid_sentence_breath(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "brief.wav"
            # 10s speech with only one brief 0.42s breath (ratio < 5%) -> should NOT split into stuttery chunks
            _write_tone_blocks(wav, [(0.1, 4.9, 0.25), (5.32, 9.8, 0.25)], total_sec=10.0)
            seg = Segment(id=0, start=0.0, end=10.0, speaker="S03", target_text="虽然听不清她们说了什么，但一定是祝福的话吧。")
            script = ProjectScript(project_id="p", media=MediaInfo(duration=10.0), segments=[seg])
            self.assertIsNone(pause_alignment.plan_segment_pause_synthesis(script, seg, vocals_path=wav))


class AgentQaToolsAndLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Path(self.temp.name)
        patcher = patch.object(config, "WORKSPACE_DIR", self.ws)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_inspect_split_merge_and_auto_qa_loop(self):
        proj = self.ws / "qa_proj"
        vocals = proj / "dialogue.wav"
        _write_tone_blocks(
            vocals,
            [(0.1, 6.0, 0.25), (6.7, 13.0, 0.25), (14.0, 14.7, 0.25), (15.1, 17.5, 0.25)],
            total_sec=18.5,
        )
        tts0 = proj / "tts" / "seg_0000.wav"
        _write_tone_blocks(tts0, [(0.1, 2.0, 0.25)], total_sec=4.2)  # low active ratio + overflow vs 2s? Wait seg 0 is 13.5s, make seg 3 overflow

        script = ProjectScript(
            project_id="qa_proj",
            source_lang="ja",
            show_bible={"genre": "日常", "tone": "轻快"},
            media=MediaInfo(duration=18.5, vocals_wav="/workspace/qa_proj/dialogue.wav"),
            characters={"S01": CharacterConfig(display_name="凛")},
            segments=[
                Segment(
                    id=0, start=0.0, end=13.5, speaker="S01",
                    orig_text="今日はいい天気ですね。一緒に散歩しましょう。",
                    target_text="今天天气真好啊。我们一起去散步吧。",
                ),
                Segment(
                    id=1, start=14.0, end=14.8, speaker="S01",
                    orig_text="あ。", target_text="啊……",
                    tts_wav="/workspace/qa_proj/tts/seg_0001.wav",
                    actual_duration=1.5, status="ready",
                ),
                Segment(
                    id=2, start=15.0, end=17.6, speaker="S01",
                    orig_text="これって耳飾りですか？",
                    target_text="请问这个是耳环／耳钉吗？",
                    tts_wav="/workspace/qa_proj/tts/seg_0002.wav",
                    actual_duration=2.4, status="ready",
                ),
            ],
        )
        _write_tone_blocks(proj / "tts" / "seg_0001.wav", [(0.05, 0.4, 0.25)], total_sec=1.5)  # tail silence anomaly + overflow
        _write_tone_blocks(proj / "tts" / "seg_0002.wav", [(0.05, 2.35, 0.25)], total_sec=2.4)
        tools.save_project_script(script)

        report = tools.inspect_project_quality("qa_proj")
        self.assertEqual(report["issue_count"], 3)
        codes_by_id = {item["segment_id"]: item["codes"] for item in report["issues"]}
        self.assertIn("OVERLONG_SEGMENT", codes_by_id[0])
        self.assertIn("TINY_FRAGMENT", codes_by_id[1])
        self.assertIn("FORBIDDEN_TEXT_PATTERN", codes_by_id[1])
        self.assertIn("AUDIO_ANOMALY", codes_by_id[1])
        self.assertIn("FORBIDDEN_TEXT_PATTERN", codes_by_id[2])

        # Test direct split_segment & merge_segments tools
        split_res = tools.split_segment("qa_proj", 0)
        self.assertEqual(split_res["left_id"], 0)
        self.assertEqual(split_res["right_id"], 3)
        self.assertTrue(5.0 <= split_res["split_time_sec"] <= 8.0)

        merge_res = tools.merge_segments("qa_proj", [1, 2])
        self.assertEqual(merge_res["kept_id"], 1)
        self.assertEqual(merge_res["removed_ids"], [2])

        # Test auto-QA REST endpoint and tool loop
        tool_update = SimpleNamespace(
            id="call_fix",
            function=SimpleNamespace(
                name="batch_update_segments",
                arguments=json.dumps({"updates": [{"id": 1, "target_text": "啊，请问这是耳环吗？"}]}),
            ),
        )
        with (
            TestClient(app) as http,
            patch.object(orchestrator, "create_openai_client") as mock_llm,
            patch.object(tts_client, "ensure_tts_running"),
            patch.object(tts_client, "_resolve_ref_for_segment", return_value=(None, None)),
            patch.object(
                tts_client,
                "_synthesize_single_line",
                side_effect=lambda _s, _seg, _r, _p, out, _g=None, **_kw: _write_tone_blocks(out, [(0.05, 1.8, 0.25)], 1.9),
            ),
        ):
            mock_llm.return_value.chat.completions.create.side_effect = [
                _completion(tool_calls=[tool_update]),
                _completion("已将碎片台词精简清洗并重合成。"),
            ]
            resp = http.post("/api/projects/qa_proj/agent/auto-qa", json={"max_rounds": 2, "auto_synthesize": True})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("精简", body["reply"])
        self.assertIn(1, body["synthesized_ids"])
        saved = tools.load_project_script("qa_proj")
        seg1 = next(s for s in saved.segments if s.id == 1)
        self.assertEqual(seg1.target_text, "啊，请问这是耳环吗？")

    def test_pronunciation_guide_and_tail_slack(self) -> None:
        from backend.agent import prompts
        from backend.services.synthesis import _check_duration, compute_tail_slack_map

        for prompt_text in (
            prompts.DIRECTOR_TRANSLATE_PROMPT,
            prompts.DIALOGUE_POLISH_PROMPT,
            prompts.DIALOGUE_QA_TIGHTEN_PROMPT,
            prompts.TRANSLATION_REPAIR_PROMPT,
        ):
            self.assertIn("同音字", prompt_text)
            self.assertIn("严禁修改 orig_text", prompt_text)

        proj = self.ws / "slack_proj"
        (proj / "tts").mkdir(parents=True, exist_ok=True)
        seg_open_gap = Segment(
            id=0, start=0.0, end=2.0, speaker="S01",
            orig_text="Wait a second.", target_text="稍等一下，我马上就好。",
            tts_wav="/workspace/slack_proj/tts/seg_0000.wav",
            actual_duration=2.40, status="ready",
        )
        seg_tight_gap = Segment(
            id=1, start=3.2, end=5.2, speaker="S01",
            orig_text="Are you sure?", target_text="你真的确定要这么做吗？",
            tts_wav="/workspace/slack_proj/tts/seg_0001.wav",
            actual_duration=2.40, status="ready",
        )
        seg_tail = Segment(
            id=2, start=5.3, end=7.0, speaker="S01",
            orig_text="Yes.", target_text="当然确定。",
            tts_wav="/workspace/slack_proj/tts/seg_0002.wav",
            actual_duration=1.60, status="ready",
        )
        _write_tone_blocks(proj / "tts" / "seg_0000.wav", [(0.05, 2.35, 0.25)], total_sec=2.40)
        _write_tone_blocks(proj / "tts" / "seg_0001.wav", [(0.05, 2.35, 0.25)], total_sec=2.40)
        _write_tone_blocks(proj / "tts" / "seg_0002.wav", [(0.05, 1.55, 0.25)], total_sec=1.60)
        script = ProjectScript(
            project_id="slack_proj",
            media=MediaInfo(duration=8.0, vocals_wav="/workspace/slack_proj/dialogue.wav"),
            characters={"S01": CharacterConfig(display_name="凛")},
            segments=[seg_open_gap, seg_tight_gap, seg_tail],
        )
        tools.save_project_script(script)

        slacks = compute_tail_slack_map(script.segments, script.media.duration)
        self.assertAlmostEqual(slacks[0], 0.44, places=2)
        self.assertEqual(slacks[1], 0.0)
        self.assertIsNone(_check_duration(seg_open_gap, tail_slack=slacks[0]))
        tight_check = _check_duration(seg_tight_gap, tail_slack=slacks[1])
        self.assertIsNotNone(tight_check)
        self.assertEqual(tight_check.duration_state, "too_long")

        report = tools.inspect_project_quality("slack_proj")
        codes_by_id = {item["segment_id"]: item["codes"] for item in report["issues"]}
        self.assertNotIn(0, codes_by_id)
        self.assertIn("DURATION_OVERFLOW", codes_by_id[1])


if __name__ == "__main__":
    unittest.main()
