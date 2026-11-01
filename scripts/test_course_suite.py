import json
import unittest
from pathlib import Path
from course_layout import build_course
from course_metrics import LandingGoal


class CourseTests(unittest.TestCase):
    def test_legacy_course_dimensions_unchanged(self):
        blocks, _, _ = build_course(json.loads(Path("course_progressive.json").read_text()))
        self.assertEqual(len(blocks), 11)
        self.assertAlmostEqual(blocks[-1][0] + blocks[-1][1], 20.8)
        self.assertAlmostEqual(blocks[-1][2], 1.05)

    def test_suite_is_contiguous_and_returns_to_ground(self):
        suite = json.loads(Path("course_suite.json").read_text())
        for case in suite["cases"]:
            with self.subTest(case=case["id"]):
                blocks, stages, width = build_course(case)
                self.assertEqual(width, 1.8)
                for left, right in zip(blocks, blocks[1:]):
                    self.assertAlmostEqual(left[0] + left[1], right[0])
                self.assertAlmostEqual(blocks[-1][2], 0.)
                self.assertGreaterEqual(blocks[-1][1], 2.)
                self.assertAlmostEqual(stages[0]["top_height_m"], case["stages"][0]["rise_m"] * case["stages"][0]["count"])
                # Nominal walking time plus 1s start and 1s confirmation fits the deadline.
                self.assertLess((blocks[-1][0] + 0.8) / 0.5 + 2, case["timeout_s"])

    def test_goal_needs_all_feet_contact_and_time(self):
        goal = LandingGoal((2., 2., 0.), 1.8)
        feet = [[2.8, 0.2, 0.03]] * 4
        zero = [[0., 0., 0.]] * 4
        contact = [[0., 0., 20.]] * 4
        for _ in range(60):
            self.assertFalse(goal.update(feet, zero, -1.))
        self.assertFalse(goal.update([[1., 0., .03]] * 4, contact, -1.))
        for _ in range(49):
            self.assertFalse(goal.update(feet, contact, -1.))
        self.assertTrue(goal.update(feet, contact, -1.))

    def test_below_ground_is_rejected(self):
        case = {"width_m": 1.8, "landing_m": 2., "first_edge_m": 1.5, "stages": [{"rise_m": -.1, "count": 1}]}
        with self.assertRaises(ValueError):
            build_course(case)

    def test_frontier_step_bound(self):
        case={"width_m":1.8,"landing_m":2.,"first_edge_m":1.5,"stages":[{"rise_m":.36,"count":1}]}
        self.assertAlmostEqual(build_course(case)[0][-1][2],.36)
        case['stages'][0]['rise_m']=.41
        with self.assertRaises(ValueError): build_course(case)

    def test_gap_goal_is_on_far_bank(self):
        case = {"kind":"gap", "gap_m":.4, "depth_m":.4, "width_m":1.8, "landing_m":2., "first_edge_m":1.5}
        blocks, stages, width = build_course(case)
        self.assertEqual(blocks, [(1.9,2.,0.)])
        self.assertEqual(stages[0]["id"], "cross")
        case["gap_m"] = float("nan")
        with self.assertRaises(ValueError): build_course(case)

    def test_hard_gap_far_bank_and_bound(self):
        case={"kind":"gap","gap_m":.9,"depth_m":.4,"width_m":1.8,"landing_m":2.,"first_edge_m":1.5}
        self.assertEqual(build_course(case)[0],[(2.4,2.,0.)])
        case["gap_m"]=.91
        with self.assertRaises(ValueError):build_course(case)

    def test_side_bypass_is_permanent_failure(self):
        goal = LandingGoal((2., 2., 0.), 1.8)
        forces = [[0., 0., 20.]] * 4
        goal.update([[1., 1.4, .03]] * 4, forces, -1.)
        for _ in range(100):
            self.assertFalse(goal.update([[2.8, .2, .03]] * 4, forces, -1.))
        self.assertTrue(goal.off_course)

    def test_airborne_window_is_not_stable_support(self):
        goal = LandingGoal((2., 2., 0.), 1.8)
        feet = [[2.8, .2, .03]] * 4
        for _ in range(49):
            self.assertFalse(goal.update(feet, [[0., 0., 0.]] * 4, -1.))
        self.assertFalse(goal.update(feet, [[0., 0., 20.]] * 4, -1.))


if __name__ == "__main__":
    unittest.main()
