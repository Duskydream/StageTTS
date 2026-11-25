import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app


class DeleteProjectTests(unittest.TestCase):
    def test_delete_project_removes_directory_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            proj_dir = Path(temp) / "proj_del_test"
            (proj_dir / "tts").mkdir(parents=True)
            (proj_dir / "script.json").write_text("{}", encoding="utf-8")
            (proj_dir / "tts" / "seg_0000.wav").write_bytes(b"RIFF")

            with TestClient(app) as client:
                resp = client.delete("/api/projects/proj_del_test")
                self.assertEqual(resp.status_code, 200, resp.text)
                self.assertEqual(resp.json(), {"status": "deleted", "project_id": "proj_del_test"})
                self.assertFalse(proj_dir.exists())

                missing = client.delete("/api/projects/proj_del_test")
                self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
