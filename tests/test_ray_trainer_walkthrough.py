import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "projects" / "agent-r1-ray-trainer-walkthrough"


class RayTrainerWalkthroughTest(unittest.TestCase):
    def assert_snapshot_matches_provenance(self, source_name: str) -> None:
        source = PAGE / source_name
        provenance = json.loads((PAGE / f"{Path(source_name).stem}.provenance.json").read_text())
        contents = source.read_bytes()
        lines = contents.decode().removesuffix("\n").splitlines()

        self.assertEqual(len(lines), provenance["line_count"])
        self.assertEqual(
            provenance["end_line"] - provenance["start_line"] + 1,
            provenance["line_count"],
        )
        self.assertEqual(hashlib.sha256(contents).hexdigest(), provenance["sha256"])

    def test_fit_snapshot_is_exact(self) -> None:
        self.assert_snapshot_matches_provenance("fit-source.py")

    def test_dynamic_helper_snapshot_is_exact(self) -> None:
        self.assert_snapshot_matches_provenance("dynamic-helper-source.py")

    def test_dynamic_sampling_anchors_exist_in_snapshot(self) -> None:
        fit_lines = (PAGE / "fit-source.py").read_text().splitlines()
        helper_lines = (PAGE / "dynamic-helper-source.py").read_text().splitlines()

        self.assertIn("filter_groups_config =", fit_lines[1016 - 961])
        self.assertIn("filter_informative_prompt_groups(", fit_lines[1105 - 961])
        self.assertEqual(fit_lines[1133 - 961].strip(), "continue")
        self.assertIn("compute_advantage(", fit_lines[1233 - 961])
        self.assertIn("trajectory_totals[trajectory_uid] +=", helper_lines[131 - 85])
        self.assertIn("np.std(prompt_returns[prompt_uid])", helper_lines[140 - 85])


if __name__ == "__main__":
    unittest.main()
