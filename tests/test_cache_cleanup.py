"""Tests for automatic cache cleanup and post-finetune pruning."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent.tools import save_project_script
from backend.main import app
from backend.schemas import CharacterConfig, CinematicStems, EmotionRef, MediaInfo, ProjectScript
from backend.services import cache_cleanup


class CacheCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.assets = self.root / "assets"
        self.gsv = self.root / "gsv"
        self.tmp = self.root / "tmp"
        self.ws.mkdir()
        self.assets.mkdir()
        self.gsv.mkdir()
        self.tmp.mkdir()
        for target, attr, val in (
            (config, "WORKSPACE_DIR", self.ws),
            (config, "ASSETS_DIR", self.assets),
            (config, "GPT_SOVITS_ROOT", self.gsv),
            (cache_cleanup.tempfile, "gettempdir", lambda: str(self.tmp)),
        ):
            p = patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)

    def test_cleanup_finetune_artifacts_keeps_only_selected_weights(self) -> None:
        exp = "st_demo_abc_12345"
        opt_dir = self.gsv / "logs" / exp
        ds_dir = self.ws / "demo" / "finetune" / exp
        opt_dir.mkdir(parents=True)
        ds_dir.mkdir(parents=True)
        (opt_dir / "G_233.pth").write_bytes(b"g" * 1024)
        (ds_dir / "seg_0000.wav").write_bytes(b"w" * 512)

        s2_dir = self.gsv / "SoVITS_weights_v2Pro"
        s1_dir = self.gsv / "GPT_weights_v2Pro"
        s2_dir.mkdir(parents=True)
        s1_dir.mkdir(parents=True)
        mid_s2 = s2_dir / f"{exp}_e5_s100.pth"
        final_s2 = s2_dir / f"{exp}_e10_s200.pth"
        mid_s1 = s1_dir / f"{exp}-e5.ckpt"
        final_s1 = s1_dir / f"{exp}-e15.ckpt"
        for f in (mid_s2, final_s2, mid_s1, final_s1):
            f.write_bytes(b"x" * 256)

        freed = cache_cleanup.cleanup_finetune_artifacts(
            ds_dir,
            opt_dir,
            exp_name=exp,
            keep_gpt=final_s1,
            keep_sovits=final_s2,
            base_model="v2Pro",
            root=self.gsv,
        )
        self.assertGreaterEqual(freed, 1024 + 512 + 512)
        self.assertFalse(opt_dir.exists())
        self.assertFalse(ds_dir.exists())
        self.assertFalse(mid_s2.exists())
        self.assertFalse(mid_s1.exists())
        self.assertTrue(final_s2.is_file())
        self.assertTrue(final_s1.is_file())

    def test_cleanup_stage_caches_preserves_referenced_assets_and_respects_active_flag(self) -> None:
        proj = self.ws / "p1"
        active_cin = proj / "cinematic-active"
        stale_cin = proj / "cinematic-old"
        active_cin.mkdir(parents=True)
        stale_cin.mkdir(parents=True)
        for name in ("dialogue", "singing", "music", "effects"):
            (active_cin / f"{name}.wav").write_bytes(b"keep")
            (stale_cin / f"{name}.wav").write_bytes(b"drop" * 100)
        old_ts = time.time() - 3600
        os.utime(stale_cin, (old_ts, old_ts))

        refs_dir = proj / "references"
        refs_dir.mkdir(parents=True)
        keep_ref = refs_dir / "auto_keep.wav"
        old_unused_ref = refs_dir / "auto_stale.wav"
        keep_ref.write_bytes(b"keep_ref")
        old_unused_ref.write_bytes(b"stale_ref" * 50)
        very_old = time.time() - 10 * 86400
        os.utime(keep_ref, (very_old, very_old))
        os.utime(old_unused_ref, (very_old, very_old))

        save_project_script(
            ProjectScript(
                project_id="p1",
                media=MediaInfo(
                    duration=10.0,
                    vocals_wav="/workspace/p1/cinematic-active/dialogue.wav",
                    cinematic_stems=CinematicStems(
                        dialogue="/workspace/p1/cinematic-active/dialogue.wav",
                        singing="/workspace/p1/cinematic-active/singing.wav",
                        music="/workspace/p1/cinematic-active/music.wav",
                        effects="/workspace/p1/cinematic-active/effects.wav",
                    ),
                ),
                characters={
                    "S01": CharacterConfig(
                        display_name="A",
                        emotion_refs={"neutral": EmotionRef(wav="/workspace/p1/references/auto_keep.wav", text="hi", lang="ja")},
                    )
                },
            )
        )

        gsv_st = self.gsv / "logs" / "st_old"
        gsv_st.mkdir(parents=True)
        (gsv_st / "big.ckpt").write_bytes(b"z" * 1024)

        # While finetune is active, gsv logs must NOT be touched
        res_busy = cache_cleanup.cleanup_stage_caches(active_finetune=True)
        self.assertTrue(gsv_st.exists())
        self.assertFalse(stale_cin.exists())
        self.assertTrue(active_cin.is_dir())
        self.assertTrue(keep_ref.is_file())
        self.assertFalse(old_unused_ref.exists())
        self.assertGreater(res_busy["freed_bytes"], 0)

        # Startup lifespan cleans gsv_st automatically, and explicit API cleans new stale dirs
        with TestClient(app) as client:
            self.assertFalse(gsv_st.exists())
            gsv_st2 = self.gsv / "logs" / "st_old_2"
            gsv_st2.mkdir(parents=True)
            (gsv_st2 / "big.ckpt").write_bytes(b"z" * 1024)
            resp = client.post("/api/system/cleanup-cache")
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertFalse(gsv_st2.exists())
            self.assertGreaterEqual(resp.json()["removed_dirs"], 1)


if __name__ == "__main__":
    unittest.main()
