"""Regression for character-pack dialogue blocking collection (2026-10-28)."""
import json
from pathlib import Path
import unittest

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "assets/resource/base"
FIXTURES = ROOT / "tools/fixtures/collect_dialogue"


class CollectDialogueTest(unittest.TestCase):
    def test_both_failed_runs_recognize_skip_button(self):
        pipeline = json.loads((BASE / "pipeline/Collect_Launcher.json").read_text())
        node = pipeline["Collect_FirstPack_Talk"]
        x, y, w, h = node["roi"]
        for fixture in sorted(FIXTURES.glob("*.png")):
            with self.subTest(fixture=fixture.name):
                image = cv2.imread(str(fixture))
                # Fixtures preserve the original recognition ROI only.
                self.assertEqual(image.shape[:2], (h, w))
                scores = []
                for template in node["template"]:
                    result = cv2.matchTemplate(
                        image, cv2.imread(str(BASE / "image" / template)),
                        cv2.TM_CCOEFF_NORMED,
                    )
                    scores.append(cv2.minMaxLoc(result)[1])
                self.assertGreaterEqual(max(scores), node.get("threshold", 0.7))
        self.assertEqual(len(list(FIXTURES.glob("*.png"))), 2)

    def test_sandbox_recovery_handles_dialogue(self):
        pipeline = json.loads((BASE / "pipeline/Global.json").read_text())
        for name in ("Global_ToSandBox", "Global_ToSandBox_Reset"):
            with self.subTest(node=name):
                self.assertIn("[JumpBack]AutoBattle_Egress_TurnBack_SkipTalk", pipeline[name]["next"])


if __name__ == "__main__":
    unittest.main()
