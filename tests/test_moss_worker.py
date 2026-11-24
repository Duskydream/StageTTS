"""MOSS short-sample protocol, source truth and offline preflight regressions."""
import hashlib
import importlib.metadata
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
from argparse import Namespace
from types import SimpleNamespace

from backend.schemas import ProjectScript
from workers import moss_worker, parse_worker


class MossWorkerTest(unittest.TestCase):
    def fixture(self, root):
        model = root / "model"
        model.mkdir()
        names = [*moss_worker.CUSTOM_FILES, "config.json", "processor_config.json", "preprocessor_config.json",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json",
                 "model-00000-of-00001.safetensors"]
        files = []
        for name in names:
            path = model / name
            path.write_bytes(b"mock reviewed local file")
            files.append({"path": name, "sha256": moss_worker.sha256(path)})
        manifest = {"model_id": moss_worker.MODEL_ID, "revision": moss_worker.REVISION, "files": files}
        (model / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (model / "custom_code_review.json").write_text(json.dumps({"revision": moss_worker.REVISION, "approved": True,
                  "files": {row["path"]: row["sha256"] for row in files if row["path"] in moss_worker.CUSTOM_FILES}}), encoding="utf-8")
        audio = root / "source.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 16000 * 10)
        return {"version": "1.0", "model": {"id": moss_worker.MODEL_ID, "revision": moss_worker.REVISION,
                  "local_path": str(model.resolve()), "custom_code_approved": True,
                  "transformers_version": "5.6.0", "torch_version": "2.8.0+cu128"},
                "source": {"id": "fixture-sha", "kind": "dialogue", "audio_path": str(audio.resolve())},
                "device": "cuda:0", "max_audio_seconds": 30, "max_new_tokens": 512}

    def versions(self, name):
        return {"transformers": "5.6.0", "torch": "2.8.0+cu128"}[name]

    def test_parse_segment_timestamps_never_fabricates_word_alignment(self):
        rows, reasons = moss_worker.parse_generation("[0.1][S01]Hello.[1.2][1.3][S02]Hi.[2.0]", 10)
        self.assertEqual(reasons, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["words"], [])
        self.assertIsNone(rows[0]["score"])
        self.assertTrue(rows[0]["needs_review"])
        self.assertIn("anonymous_speaker_within_request_only", rows[0]["review_reasons"])

    def test_invalid_and_unknown_markup_has_review_not_silent_drop(self):
        rows, reasons = moss_worker.parse_generation("garbage[0.1][S01]Hello.[1.0][2][Sxx]bad[3][9][S02]late[11]", 10)
        self.assertEqual(len(rows), 1)
        self.assertIn("unparsed_generation_content", reasons)
        self.assertIn("invalid_generated_segment", reasons)
        rows, reasons = moss_worker.parse_generation("[2][S01]reverse[1]", 10)
        self.assertEqual(rows, [])
        self.assertIn("no_valid_generated_segments", reasons)

    def test_explicit_fifty_millisecond_timestamp_correction_retains_generated_bounds(self):
        rows, reasons = moss_worker.parse_generation("[23.98][S02]Full final sentence.[30.02]", 30)
        self.assertEqual(reasons, [])
        self.assertEqual(rows[0]["text"], "Full final sentence.")
        self.assertEqual(rows[0]["end"], 30)
        self.assertEqual(rows[0]["generated_end"], 30.02)
        self.assertIn("timestamp_clamped_to_source_end", rows[0]["review_reasons"])
        rows, reasons = moss_worker.parse_generation("[23.98][S02]Too far.[30.5]", 30)
        self.assertEqual(rows, [])
        self.assertIn("invalid_generated_segment", reasons)
        rows, reasons = moss_worker.parse_generation("[-0.02][S01]Start.[1]", 30)
        self.assertEqual(rows[0]["start"], 0)
        self.assertEqual(rows[0]["generated_start"], -0.02)
        self.assertIn("timestamp_clamped_to_source_start", rows[0]["review_reasons"])
        rows, reasons = moss_worker.parse_generation("[-0.5][S01]Bad.[1]", 30)
        self.assertEqual(rows, [])
        self.assertIn("invalid_generated_segment", reasons)

    def test_unknown_speaker_and_malformed_request_are_explicit(self):
        rows, reasons = moss_worker.parse_generation("[0][S00]Unknown.[1]", 10)
        self.assertEqual(rows[0]["speaker"], "UNKNOWN")
        self.assertIn("unknown_generated_speaker", reasons)
        self.assertEqual(moss_worker.run_request([])["status"], "not_run")
        self.assertEqual(moss_worker.run_request({"version": "1.0", "model": []})["status"], "not_run")

    def test_unknown_tags_preserved_reviewed_and_duplicate_ids_scoped(self):
        rows, reasons = moss_worker.parse_generation("[0][S01]Hello [UNKNOWN] bye[laugh][1]", 10,
                                                     source_id="source-A", window_id="window-1")
        self.assertEqual(rows[0]["text"], "Hello [UNKNOWN] bye[laugh]")
        self.assertIn("unclassified_bracketed_tag_in_text", reasons)
        self.assertIn("unclassified_bracketed_tag_in_text", rows[0]["review_reasons"])
        text = "[0][S01]Hello.[1][0][S01]Hello.[1]"
        rows, reasons = moss_worker.parse_generation(text, 10, source_id="source-A", window_id="window-1")
        self.assertEqual(len({row["id"] for row in rows}), 2)
        self.assertIn("duplicate_generated_segment", reasons)
        source_b, _ = moss_worker.parse_generation(text, 10, source_id="source-B", window_id="window-1")
        window_b, _ = moss_worker.parse_generation(text, 10, source_id="source-A", window_id="window-2")
        self.assertNotEqual(rows[0]["id"], source_b[0]["id"])
        self.assertNotEqual(rows[0]["id"], window_b[0]["id"])
        self.assertNotEqual(rows[0]["speaker_scope"], window_b[0]["speaker_scope"])

    def test_generated_overlapping_speakers_are_retained_and_reviewed(self):
        rows, reasons = moss_worker.parse_generation("[0][S01]First.[2][1][S02]Together.[3]", 10)
        self.assertEqual(len(rows), 2)
        self.assertIn("overlapping_generated_segments_require_review", rows[1]["review_reasons"])

    def test_manifest_hash_and_review_required_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.fixture(Path(directory))
            request["model"]["custom_code_approved"] = False
            with patch.object(moss_worker, "_infer") as infer:
                result = moss_worker.run_request(request)
            self.assertEqual(result["status"], "not_run")
            infer.assert_not_called()
            request["model"]["custom_code_approved"] = True
            (Path(request["model"]["local_path"]) / moss_worker.CUSTOM_FILES[0]).write_bytes(b"modified")
            with patch.object(moss_worker, "_infer") as infer:
                result = moss_worker.run_request(request)
            self.assertEqual(result["status"], "not_run")
            self.assertIn("hash", result["error"])
            infer.assert_not_called()

    def test_manifest_all_files_validated_and_traversal_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root)
            model = Path(request["model"]["local_path"])
            manifest_path = model / "download_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            extra = model / "generation_config.json"
            extra.write_bytes(b"original")
            manifest["files"].append({"path": extra.name, "sha256": moss_worker.sha256(extra)})
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            extra.write_bytes(b"tampered")
            with patch.object(moss_worker, "_infer") as infer:
                result = moss_worker.run_request(request)
            self.assertEqual(result["status"], "not_run")
            self.assertIn("generation_config", result["error"])
            infer.assert_not_called()
            manifest["files"][-1] = {"path": "../outside.json", "sha256": "bad"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("inside model root", moss_worker.run_request(request)["error"])

    def test_exact_versions_and_short_limit_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.fixture(Path(directory))
            with patch.object(importlib.metadata, "version", return_value="wrong"):
                self.assertEqual(moss_worker.run_request(request)["status"], "not_run")
            request["max_audio_seconds"] = 5
            with patch.object(importlib.metadata, "version", side_effect=self.versions):
                self.assertIn("duration", moss_worker.run_request(request)["error"])

    def test_main_moss_route_retains_raw_stream_scope_and_segment_granularity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "dialogue.wav"
            with wave.open(str(vocals), "wb") as stream:
                stream.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0\0\0" * 48000 * 4)
            def conversion(command, **kwargs):
                Path(command[-1]).write_bytes(b"mock converted WAV")
                return SimpleNamespace(returncode=0, stderr="")
            output = {"status": "ok", "raw_text": "[0][S01]Hello.[4]", "review_reasons": [],
                      "segments": [{"start": 0, "end": 4, "speaker": "S01", "text": "Hello."}]}
            with patch.object(parse_worker.config, "MOSS_MODEL_PATH", root), \
                 patch.object(parse_worker.config, "MOSS_REVISION", moss_worker.REVISION), \
                 patch.object(parse_worker.config, "MOSS_CUSTOM_CODE_APPROVED", True), \
                 patch.object(parse_worker.subprocess, "run", side_effect=conversion), \
                 patch.object(moss_worker, "run_request", return_value=output) as run:
                rows, context, language = parse_worker._stage_moss_transcribe(vocals, "cuda")
            self.assertEqual(rows[0]["words"], [])
            self.assertEqual(rows[0]["alignment_granularity"], "segment")
            self.assertEqual(rows[0]["timing_source"], "moss_joint_generated")
            self.assertTrue(rows[0]["alignment_needs_review"])
            self.assertTrue(rows[0]["segment_timing_needs_review"])
            self.assertTrue(rows[0]["speaker_needs_review"])
            self.assertTrue(rows[0]["speaker"].startswith("MOSS_"))
            self.assertEqual(context["raw_generation"], output["raw_text"])
            self.assertEqual(context["raw_segments"][0]["words"], [])
            self.assertEqual(run.call_args.args[0]["device"], "cuda:0")
            args = Namespace(project_id="moss-main", input=vocals, output_dir=root, mode="video_dubbing",
                             target_lang="zh", source_lang="en", skip_llm=True)
            saved = parse_worker._build_and_save_script(args, 4, vocals, None, rows, language, evidence_context=context)
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            self.assertEqual(script.segments[0].alignment_granularity, "segment")
            self.assertIsNone(script.evidence.source_tokens[0].start)
            self.assertIsNone(script.evidence.source_tokens[0].end)
            self.assertEqual(script.evidence.source_utterances[0].text, "Hello.")
            self.assertEqual(script.evidence.provenance[0].model, moss_worker.MODEL_ID)
            self.assertEqual(script.evidence.raw_generation, output["raw_text"])
            self.assertEqual(script.evidence.speaker_scope, context["speaker_scope"])
            self.assertTrue(script.evidence.diarization_spans[0].review.needs_review)
            self.assertIn("moss_joint_generated_activity_unvalidated", script.evidence.diarization_spans[0].review.reasons)
            self.assertTrue(all(turn.review.needs_review for turn in script.evidence.source_turns))
            self.assertEqual(script.source_lang, "en")
            self.assertEqual(script.evidence.provenance[0].parameters["source_language_kind"], "user_declared")
            self.assertEqual(script.evidence.provenance[0].parameters["source_audio_id"], context["source_audio_id"])

    def test_main_route_preserves_clamped_timestamp_evidence_in_saved_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "dialogue.wav"
            vocals.write_bytes(b"source fixture")
            rows, _ = moss_worker.parse_generation("[23.98][S02]Complete last sentence.[30.02]", 30,
                                                   source_id="fixture", window_id="whole")
            output = {"status": "partial", "raw_text": "[23.98][S02]Complete last sentence.[30.02]",
                      "review_reasons": ["source_wave_tail_within_explicit_0.05s_budget_tolerance"], "segments": rows,
                      "duration_seconds": 60.000375, "audio_budget_seconds": 60,
                      "media_tail_tolerance_seconds": 0.05}
            def conversion(command, **kwargs):
                Path(command[-1]).write_bytes(b"converted fixture")
                return SimpleNamespace(returncode=0, stderr="")
            with patch.object(parse_worker.config, "MOSS_MODEL_PATH", root), \
                 patch.object(parse_worker.config, "MOSS_REVISION", moss_worker.REVISION), \
                 patch.object(parse_worker.config, "MOSS_CUSTOM_CODE_APPROVED", True), \
                 patch.object(parse_worker.subprocess, "run", side_effect=conversion), \
                 patch.object(parse_worker, "_release_cuda_memory"), \
                 patch.object(moss_worker, "run_request", return_value=output):
                segments, context, language = parse_worker._stage_moss_transcribe(vocals, "cuda")
            args = Namespace(project_id="clamp-save", input=vocals, output_dir=root, mode="video_dubbing",
                             target_lang="zh", source_lang="en", skip_llm=True)
            saved = parse_worker._build_and_save_script(args, 30, vocals, None, segments, language, evidence_context=context)
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            utterance = script.evidence.source_utterances[0]
            self.assertEqual(utterance.end, 30)
            self.assertEqual(utterance.generated_end, 30.02)
            self.assertIn("timestamp_clamped_to_source_end", utterance.review.reasons)
            self.assertIn("source_wave_tail_within_explicit_0.05s_budget_tolerance", utterance.review.reasons)
            self.assertEqual(script.evidence.provenance[0].parameters["source_wave_duration"], 60.000375)
            self.assertEqual(script.evidence.provenance[0].parameters["input_audio_budget_seconds"], 60)
            self.assertEqual(script.evidence.provenance[0].parameters["input_media_tail_tolerance_seconds"], 0.05)
            self.assertEqual(utterance.text, "Complete last sentence.")

    def test_duration_limit_fails_before_banquet_and_preserves_existing_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            source.touch()
            output = root / "output"
            output.mkdir()
            script = output / "script.json"
            script.write_text("previous script sentinel", encoding="utf-8")
            args = Namespace(project_id="duration", input=source, output_dir=output, device="cpu",
                             parse_backend="moss")
            def extract(input_path, target):
                with wave.open(str(target), "wb") as stream:
                    stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    stream.writeframes(b"\0\0" * 16000 * 61)
                return 60.042  # Container metadata cannot override actual 61s waveform.
            with patch.object(parse_worker.config, "MOSS_MAX_AUDIO_SECONDS", 60), \
                 patch.object(parse_worker, "_preflight_moss"), \
                 patch.object(parse_worker, "_extract_wav_and_probe", side_effect=extract), \
                 patch.object(parse_worker, "_stage_banquet") as banquet, \
                 patch.object(parse_worker, "_stage_whisperx_asr_align") as legacy:
                with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
                    parse_worker.run_pipeline(args)
            banquet.assert_not_called()
            legacy.assert_not_called()
            self.assertEqual(script.read_text(encoding="utf-8"), "previous script sentinel")
            self.assertEqual(list(output.glob("cinematic-*")), [])

    def test_main_moss_missing_configuration_never_calls_legacy(self):
        with patch.object(parse_worker.config, "MOSS_MODEL_PATH", None), \
             patch.object(parse_worker, "_stage_whisperx_asr_align") as legacy:
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                parse_worker._stage_moss_transcribe(Path("unused.wav"), "cuda")
            legacy.assert_not_called()

    def test_ninety_minute_budget_accepts_long_audio_and_larger_output(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.fixture(Path(directory))
            request.update(max_audio_seconds=5400, max_new_tokens=65536)
            audio = Path(request["source"]["audio_path"])
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * (16000 * 575))
            with patch.object(importlib.metadata, "version", side_effect=self.versions):
                self.assertEqual(moss_worker.validate_request(request)[2], 575)
                request["max_audio_seconds"] = 5401
                with self.assertRaisesRegex(ValueError, "5400"):
                    moss_worker.validate_request(request)
                request.update(max_audio_seconds=5400, max_new_tokens=65537)
                with self.assertRaisesRegex(ValueError, "65536"):
                    moss_worker.validate_request(request)

    def test_sixty_second_wave_budget_uses_frames_and_bounded_media_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.fixture(root)
            request.update(max_audio_seconds=60, max_new_tokens=4096)
            audio = Path(request["source"]["audio_path"])
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * 960006)  # Exactly 60.000375 seconds.
            with patch.object(importlib.metadata, "version", side_effect=self.versions):
                _, _, duration, _ = moss_worker.validate_request(request)
            self.assertEqual(duration, 60.000375)
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * 960816)  # 60.051 seconds exceeds explicit .05.
            with patch.object(importlib.metadata, "version", side_effect=self.versions):
                with self.assertRaisesRegex(ValueError, "budget"):
                    moss_worker.validate_request(request)

    def test_success_preserves_raw_source_and_marks_token_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.fixture(Path(directory))
            text = "[0.1][S01]Hello.[1.0]"
            generation = {"raw_text": text, "generation_limit_reached": True,
                          "peak_allocated_bytes": 123, "peak_reserved_bytes": 456}
            with patch.object(importlib.metadata, "version", side_effect=self.versions), \
                 patch.object(moss_worker, "_infer", return_value=generation):
                result = moss_worker.run_request(request)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["raw_text"], text)
        self.assertEqual(result["source"], request["source"])
        self.assertFalse(result["word_alignment_available"])
        self.assertFalse(result["cross_window_speaker_identity_available"])
        self.assertIn("generation_token_limit_reached_possible_truncation", result["review_reasons"])


if __name__ == "__main__":
    unittest.main()
