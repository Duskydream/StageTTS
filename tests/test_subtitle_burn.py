import math
import shutil
import subprocess
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent import tools
from backend.main import app
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from backend.services import media_mixer, subtitles


def _write_tone_wav(path: Path, seconds: float, *, rate: int = 44100) -> None:
    total_frames = int(seconds * rate)
    samples = array("h")
    for idx in range(total_frames):
        val = int(6000 * math.sin(2 * math.pi * 440.0 * idx / rate))
        samples.append(val)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


class SubtitleBurnTests(unittest.TestCase):
    def test_generate_ass_subtitles_layouts(self):
        script = ProjectScript(
            project_id="sub_layout",
            mode="video_dubbing",
            media=MediaInfo(duration=4.0),
            characters={"S01": CharacterConfig(display_name="角色A")},
            segments=[
                Segment(
                    id=0,
                    start=0.2,
                    end=1.8,
                    speaker="S01",
                    orig_text="今日はいい天気ですね。",
                    target_text="今天天气真不错呀。",
                )
            ],
        )

        source_ass = subtitles.generate_ass_subtitles(script, layout="source")
        self.assertIn("今日はいい天気ですね。", source_ass)
        self.assertNotIn("今天天气真不错呀。", source_ass)

        target_ass = subtitles.generate_ass_subtitles(script, layout="target")
        self.assertIn("今天天气真不错呀。", target_ass)
        self.assertNotIn("今日はいい天気ですね。", target_ass)

        bilingual_ass = subtitles.generate_ass_subtitles(script, layout="bilingual")
        self.assertIn(r"今天天气真不错呀。\N{\rSecondary}今日はいい天気ですね。", bilingual_ass)

    def test_build_ass_subtitle_filter_escapes_windows_paths(self):
        with patch.object(Path, "resolve", return_value=Path(r"C:\Temp\sub:1\it's.ass")):
            filt = media_mixer._build_ass_subtitle_filter(Path(r"C:\Temp\sub:1\it's.ass"))
        self.assertIn(r"C\:/Temp/sub\:1/it\'s.ass", filt)
        self.assertTrue(filt.startswith("ass='"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_mix_and_render_burns_source_subtitles_and_render_endpoint(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "burn_src"
            tts_dir = project_dir / "tts"
            tts_dir.mkdir(parents=True)

            input_mp4 = project_dir / "input.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x180:d=1.2",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=44100:cl=stereo:d=1.2",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(input_mp4),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            _write_tone_wav(tts_dir / "seg_0000.wav", 0.8)

            script = ProjectScript(
                project_id="burn_src",
                mode="video_dubbing",
                media=MediaInfo(source_video="/workspace/burn_src/input.mp4", duration=1.2),
                characters={"S01": CharacterConfig(display_name="角色A")},
                segments=[
                    Segment(
                        id=0,
                        start=0.1,
                        end=1.0,
                        speaker="S01",
                        orig_text="約束の時間ギリギリになってしまったね。",
                        target_text="都拖到约定时间的最后关头了呢。",
                        tts_wav="/workspace/burn_src/tts/seg_0000.wav",
                        status="ready",
                    )
                ],
            )
            tools.save_project_script(script)

            out_url, dur = media_mixer.mix_and_render(script, burn_subtitles="source")
            self.assertEqual(out_url, "/workspace/burn_src/output.mp4")
            self.assertGreater(dur, 0.5)
            self.assertTrue((project_dir / "output.mp4").is_file())
            self.assertGreater((project_dir / "output.mp4").stat().st_size, 1000)

            ass_file = project_dir / "subtitles.ass"
            self.assertTrue(ass_file.is_file())
            ass_content = ass_file.read_text(encoding="utf-8-sig")
            self.assertIn("約束の時間ギリギリになってしまったね。", ass_content)
            self.assertNotIn("都拖到约定时间的最后关头了呢。", ass_content)

            with TestClient(app) as client:
                resp = client.post(
                    "/api/projects/burn_src/render",
                    json={"burn_watermark": False, "burn_subtitles": "source"},
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            self.assertEqual(payload["burn_subtitles"], "source")
            self.assertEqual(payload["output_video"], "/workspace/burn_src/output.mp4")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_auto_fit_overflow_resolves_mild_spillover_on_render(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "fit_spill"
            tts_dir = project_dir / "tts"
            tts_dir.mkdir(parents=True)

            input_mp4 = project_dir / "input.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x180:d=2.0",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=44100:cl=stereo:d=2.0",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(input_mp4),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            _write_tone_wav(tts_dir / "seg_0000.wav", 1.10)

            script = ProjectScript(
                project_id="fit_spill",
                mode="video_dubbing",
                media=MediaInfo(source_video="/workspace/fit_spill/input.mp4", duration=2.0),
                characters={"S01": CharacterConfig(display_name="角色A")},
                segments=[
                    Segment(
                        id=0,
                        start=0.0,
                        end=1.0,
                        speaker="S01",
                        orig_text="First line",
                        target_text="第一句配音稍微长一点",
                        tts_wav="/workspace/fit_spill/tts/seg_0000.wav",
                        applied_speed_factor=1.0,
                        status="ready",
                    ),
                    Segment(
                        id=1,
                        start=1.05,
                        end=1.8,
                        speaker="S01",
                        orig_text="Second line",
                        target_text="第二句",
                        keep_original=True,
                        status="ready",
                    ),
                ],
            )
            tools.save_project_script(script)

            with self.assertRaisesRegex(RuntimeError, "导出待审核"):
                media_mixer.mix_and_render(script)

            out_url, _dur = media_mixer.mix_and_render(script, auto_fit_overflow=True)
            self.assertEqual(out_url, "/workspace/fit_spill/output.mp4")

            with TestClient(app) as client:
                resp = client.post(
                    "/api/projects/fit_spill/render",
                    json={"burn_watermark": False, "burn_subtitles": "none"},
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(resp.json()["output_video"], "/workspace/fit_spill/output.mp4")


if __name__ == "__main__":
    unittest.main()
