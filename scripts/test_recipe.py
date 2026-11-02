import unittest
from pathlib import Path
from reproduce import STAGES, MAIN_STAGES, command
class RecipeTests(unittest.TestCase):
    def test_every_parent_is_declared_before_child(self):
        seen = set()
        for stage, item in STAGES.items():
            if item:
                self.assertIn(item[0], seen)
            seen.add(stage)
    def test_matched_comparison_changes_only_reward(self):
        standard = STAGES["stage4-standard"]
        shaped = STAGES["stage4-balance"]
        self.assertEqual(standard[:2], shaped[:2])
        self.assertEqual(standard[2], "standard")
        self.assertEqual(shaped[2], "jump_balance")
    def test_baseline_has_no_checkpoint(self):
        self.assertNotIn("--checkpoint", command("baseline", "python", Path("runs")))
    def test_evaluation_dimensions_exist(self):
        import json
        cases = json.loads(Path(__file__).with_name("course_frontier.json").read_text())["cases"]
        ids = {case["id"] for case in cases}
        self.assertTrue({"gap75", "gap80", "gap85", "gap90"}.issubset(ids))

    def test_main_recipe_excludes_exploratory_arms(self):
        self.assertEqual(MAIN_STAGES, ("baseline", "first", "second", "stage3-exposure", "stage4-standard", "stage4-balance"))
        seen = set()
        for stage in MAIN_STAGES:
            if STAGES[stage]:
                self.assertIn(STAGES[stage][0], seen)
            seen.add(stage)
