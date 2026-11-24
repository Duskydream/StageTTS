import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from backend.services import translation, web_search


def _completion(payload_obj):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload_obj, ensure_ascii=False)))]
    )


def _make_script(*, user_glossary=None, style_a="", style_b="沉稳冷静，说话偏简练"):
    return ProjectScript(
        project_id="natural_trans",
        media=MediaInfo(duration=20),
        source_lang="ja",
        glossary=dict(user_glossary if user_glossary is not None else {"ProjectX": "X计划"}),
        characters={
            "S01": CharacterConfig(display_name="凛", speaking_style=style_a),
            "S02": CharacterConfig(display_name="澈", speaking_style=style_b),
        },
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=4.0,
                speaker="S01",
                orig_text="ProjectXは3日後に始まるんだよ。",
                target_lang="zh",
            ),
            Segment(
                id=1,
                start=4.5,
                end=8.0,
                speaker="S02",
                orig_text="Aegisのことを忘れるな。",
                target_lang="zh",
            ),
        ],
    )


class WebSearchUnitTests(unittest.TestCase):
    def test_parse_ddg_html_strips_markup_and_dedupes(self):
        markup = """
        <div class="result results_links">
          <a class="result__a" href="#">Aegis &amp; StageTTS &lt;Official&gt;</a>
          <a class="result__snippet">Official Chinese name: <b>埃癸斯</b> &mdash; character guide.</a>
        </div>
        <div class="result results_links">
          <a class="result__a" href="#">Aegis &amp; StageTTS &lt;Official&gt;</a>
          <a class="result__snippet">Official Chinese name: <b>埃癸斯</b> &mdash; character guide.</a>
        </div>
        <div class="result results_links">
          <a class="result__a" href="#">Second Result</a>
          <div class="result__snippet">Extra note</div>
        </div>
        """
        hits = web_search._parse_ddg_html(markup, limit=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["title"], "Aegis & StageTTS <Official>")
        self.assertEqual(hits[0]["snippet"], "Official Chinese name: 埃癸斯 — character guide.")
        self.assertEqual(hits[1]["title"], "Second Result")

    def test_search_terms_merges_wiki_and_ddg_and_isolates_failures(self):
        def fake_http(url, *, data=None, timeout_sec=8.0):
            if "list=search" in url:
                return json.dumps({
                    "query": {"search": [{"pageid": 101, "title": "Aegis", "snippet": "mythic shield"}]}
                })
            if "pageids=101" in url:
                return json.dumps({
                    "query": {
                        "pages": {
                            "101": {
                                "title": "Aegis",
                                "extract": "Shield associated with Athena.",
                                "langlinks": [{"lang": "zh", "*": "埃癸斯"}],
                            }
                        }
                    }
                })
            if "duckduckgo" in url:
                raise TimeoutError("network timeout")
            return "{}"

        with patch.object(web_search, "_http_request", side_effect=fake_http):
            found = web_search.search_terms(
                ["  Aegis ", "aegis", ""],
                source_lang="en",
                target_lang="zh",
                max_terms=8,
                max_results_per_term=3,
            )

        self.assertEqual(list(found.keys()), ["Aegis"])
        self.assertEqual(len(found["Aegis"]), 1)
        self.assertEqual(found["Aegis"][0]["source"], "en.wikipedia")
        self.assertIn("[zh维基对应词条: 埃癸斯]", found["Aegis"][0]["snippet"])


class NaturalTranslationPipelineTests(unittest.TestCase):
    def test_full_three_pass_flow_updates_styles_glossary_and_polishes_spoken_dialogue(self):
        script = _make_script()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _completion({
                "genre": "校园悬疑",
                "tone": "紧张克制又带点少年感",
                "character_styles": {
                    "S01": "语速偏快、直爽少年音，常带感叹语气",
                    "S02": "想要强行覆盖已有风格（应被忽略）",
                },
                "relationships": ["S01 对 S02 语气温和尊重"],
                "glossary": {
                    "ProjectX": "错误改写（不得覆盖用户条目）",
                    "Aegis": "埃癸斯",
                },
                "uncertain_terms": ["Aegis"],
            }),
            _completion([
                {
                    "id": 0,
                    "target_text": "X计划将在3天后开始。",
                    "emotion": "happy",
                    "action": "point",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    "id": 1,
                    "target_text": "不要忘记埃癸斯的事。",
                    "emotion": "sad",
                    "action": "nod",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
            _completion([
                {
                    "id": 0,
                    "target_text": "X计划，3天后就要启动了！",
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    "id": 1,
                    "target_text": "别忘了埃癸斯那件事。",
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
        ]

        logs: list[str] = []
        translated = translation.translate_script(
            script,
            client,
            log=logs.append,
            enable_analysis=True,
            enable_polish=True,
            enable_web_search=False,
        )

        self.assertEqual(translated, [0, 1])
        self.assertEqual(client.chat.completions.create.call_count, 3)

        # User-specified style and glossary must never be overwritten; empty style/missing terms filled.
        self.assertEqual(script.characters["S01"].speaking_style, "语速偏快、直爽少年音，常带感叹语气")
        self.assertEqual(script.characters["S02"].speaking_style, "沉稳冷静，说话偏简练")
        self.assertEqual(script.glossary, {"ProjectX": "X计划", "Aegis": "埃癸斯"})

        # show_bible must reach both Pass 1 and Pass 2 payloads.
        calls = client.chat.completions.create.call_args_list
        pass1_payload = json.loads(calls[1].kwargs["messages"][-1]["content"])
        pass2_payload = json.loads(calls[2].kwargs["messages"][-1]["content"])
        self.assertEqual(pass1_payload["show_bible"]["genre"], "校园悬疑")
        self.assertEqual(pass2_payload["show_bible"]["tone"], "紧张克制又带点少年感")
        self.assertEqual(pass2_payload["lines"][0]["draft_text"], "X计划将在3天后开始。")

        # Polished text applied while Pass 1 emotion & action metadata remain preserved.
        self.assertEqual(script.segments[0].target_text, "X计划，3天后就要启动了！")
        self.assertEqual(script.segments[0].emotion, "happy")
        self.assertEqual(script.segments[0].action, "point")
        self.assertEqual(script.segments[1].target_text, "别忘了埃癸斯那件事。")
        self.assertEqual(script.segments[1].emotion, "sad")
        self.assertEqual(script.segments[1].action, "nod")

    def test_polish_fidelity_guard_preserves_pass1_draft_when_glossary_or_number_dropped(self):
        script = _make_script()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _completion([
                {
                    "id": 0,
                    "target_text": "X计划3天后就要开始了。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    "id": 1,
                    "target_text": "别忘了这件事。",
                    "emotion": "angry",
                    "action": "shake_head",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
            _completion([
                {
                    # Dropped both glossary term "X计划" and number "3"
                    "id": 0,
                    "target_text": "那个行动很快要开始了。",
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    # Failed self-check
                    "id": 1,
                    "target_text": "记住这件事。",
                    "self_check": {"semantic_complete": False, "omitted_info": "省略否定"},
                },
            ]),
        ]

        translated = translation.translate_script(
            script,
            client,
            enable_analysis=False,
            enable_polish=True,
            enable_web_search=False,
        )

        self.assertEqual(translated, [0, 1])
        self.assertEqual(script.segments[0].target_text, "X计划3天后就要开始了。")
        self.assertEqual(script.segments[1].target_text, "别忘了这件事。")
        self.assertEqual(script.segments[1].emotion, "angry")

    def test_web_search_verifies_terms_and_degrades_gracefully_on_errors(self):
        script = _make_script(user_glossary={"ProjectX": "X计划"})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _completion({
                "genre": "机战",
                "tone": "严肃",
                "character_styles": {},
                "relationships": [],
                "glossary": {"Aegis": "临时猜测盾牌"},
                "uncertain_terms": ["Aegis"],
            }),
            _completion({
                "glossary": {"ProjectX": "试图覆盖用户词（禁止）", "Aegis": "埃癸斯神盾"},
                "background_notes": ["Aegis 为本作核心防御系统官方译名"],
            }),
            _completion([
                {
                    "id": 0,
                    "target_text": "X计划3天后启动。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    "id": 1,
                    "target_text": "别忘了埃癸斯神盾。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
        ]

        with patch(
            "backend.services.web_search.search_terms",
            return_value={"Aegis": [{"source": "ja.wikipedia", "title": "イージス", "snippet": "官方译名 埃癸斯神盾"}]},
        ) as mock_search:
            translated = translation.translate_script(
                script,
                client,
                enable_analysis=True,
                enable_polish=False,
                enable_web_search=True,
            )

        self.assertEqual(translated, [0, 1])
        mock_search.assert_called_once()
        self.assertEqual(script.glossary["ProjectX"], "X计划")
        self.assertEqual(script.glossary["Aegis"], "埃癸斯神盾")
        pass1_payload = json.loads(client.chat.completions.create.call_args_list[2].kwargs["messages"][-1]["content"])
        self.assertEqual(pass1_payload["show_bible"]["background_notes"], ["Aegis 为本作核心防御系统官方译名"])

        # Now verify search error does not break translation.
        script2 = _make_script()
        client2 = MagicMock()
        client2.chat.completions.create.side_effect = [
            _completion({
                "genre": "日常",
                "tone": "轻松",
                "character_styles": {},
                "relationships": [],
                "glossary": {},
                "uncertain_terms": ["Aegis"],
            }),
            _completion([
                {
                    "id": 0,
                    "target_text": "X计划3天后开始。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
                {
                    "id": 1,
                    "target_text": "别忘记埃癸斯。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                },
            ]),
        ]
        with patch("backend.services.web_search.search_terms", side_effect=RuntimeError("offline")):
            self.assertEqual(
                translation.translate_script(
                    script2,
                    client2,
                    enable_analysis=True,
                    enable_polish=False,
                    enable_web_search=True,
                ),
                [0, 1],
            )

    def test_translate_endpoint_persists_discovered_styles_and_glossary(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            script = _make_script(user_glossary={"ProjectX": "X计划"}, style_a="")
            proj_dir = Path(temp) / "natural_trans"
            proj_dir.mkdir(parents=True, exist_ok=True)
            script_path = proj_dir / "script.json"
            script_path.write_text(script.model_dump_json(), encoding="utf-8")

            with TestClient(app) as http_client, patch("backend.main.create_openai_client") as factory, patch(
                "backend.services.web_search.search_terms",
                return_value={"Aegis": [{"source": "web", "title": "Aegis", "snippet": "埃癸斯"}]},
            ):
                factory.return_value.chat.completions.create.side_effect = [
                    _completion({
                        "genre": "悬疑",
                        "tone": "紧张",
                        "character_styles": {"S01": "急性子、口语短句"},
                        "relationships": [],
                        "glossary": {"ProjectX": "不得覆盖", "Aegis": "旧词"},
                        "uncertain_terms": ["Aegis"],
                    }),
                    _completion({
                        "glossary": {"Aegis": "埃癸斯"},
                        "background_notes": ["重要防御装备"],
                    }),
                    _completion([
                        {
                            "id": 0,
                            "target_text": "X计划将在3天后启动。",
                            "emotion": "happy",
                            "action": "point",
                            "keep_original": False,
                            "self_check": {"semantic_complete": True, "omitted_info": ""},
                        },
                        {
                            "id": 1,
                            "target_text": "别忘了埃癸斯。",
                            "emotion": "neutral",
                            "action": "idle",
                            "keep_original": False,
                            "self_check": {"semantic_complete": True, "omitted_info": ""},
                        },
                    ]),
                    _completion([
                        {
                            "id": 0,
                            "target_text": "X计划，3天后准时开始！",
                            "self_check": {"semantic_complete": True, "omitted_info": ""},
                        },
                        {
                            "id": 1,
                            "target_text": "可别忘了埃癸斯。",
                            "self_check": {"semantic_complete": True, "omitted_info": ""},
                        },
                    ]),
                ]
                resp = http_client.post(
                    "/api/projects/natural_trans/translate",
                    json={"target_lang": "zh", "natural_polish": True, "web_search": True},
                )

            self.assertEqual(resp.status_code, 200, resp.text)
            saved = ProjectScript.model_validate_json(script_path.read_text(encoding="utf-8"))
            self.assertEqual(saved.characters["S01"].speaking_style, "急性子、口语短句")
            self.assertEqual(saved.characters["S02"].speaking_style, "沉稳冷静，说话偏简练")
            self.assertEqual(saved.glossary, {"ProjectX": "X计划", "Aegis": "埃癸斯"})
            self.assertEqual(saved.segments[0].target_text, "X计划，3天后准时开始！")
            self.assertEqual(saved.segments[0].emotion, "happy")

    def test_mixed_cjk_term_matching_and_parse_endpoint_forwards_flags(self):
        self.assertTrue(translation._term_in_text("X计划3天后就要开始了。", "X计划"))
        self.assertFalse(translation._term_in_text("WAIT", "AI"))
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            proj_dir = Path(temp) / "parse_flags"
            proj_dir.mkdir(parents=True, exist_ok=True)
            media = proj_dir / "input.mp4"
            media.write_bytes(b"fake")
            script = _make_script()
            script.project_id = "parse_flags"
            (proj_dir / "script.json").write_text(script.model_dump_json(), encoding="utf-8")

            with TestClient(app) as http_client, patch("backend.main.run_parse_worker") as mock_parse:
                resp = http_client.post(
                    "/api/projects/parse_flags/parse",
                    json={"target_lang": "zh", "natural_polish": False, "web_search": True},
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            mock_parse.assert_called_once()
            self.assertFalse(mock_parse.call_args.kwargs["natural_polish"])
            self.assertTrue(mock_parse.call_args.kwargs["web_search"])


if __name__ == "__main__":
    unittest.main()
