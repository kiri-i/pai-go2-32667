"""Evaluation-only goal monitor. Never changes policy observations or actions."""
import math
from collections import deque


class LandingGoal:
    def __init__(self, block, width, dt=0.02):
        self.edge, self.length, self.height = block
        self.width = width
        self.required_steps = math.ceil(1.0 / dt)
        self.steps = 0
        self.contact_seen = [False] * 4
        self.success = False
        self.off_course = False
        self.support_window = deque(maxlen=self.required_steps)

    def update(self, feet, forces, gravity_z):
        if any(abs(foot[1]) >= self.width / 2 for foot in feet):
            self.off_course = True
        if self.off_course:
            return False
        inside = gravity_z < -0.7 and len(feet) == len(forces) == 4 and all(
            self.edge + 0.05 < x < self.edge + self.length - 0.05
            and abs(y) < self.width / 2 - 0.05
            and self.height - 0.02 < z < self.height + 0.15
            for x, y, z in feet
        )
        if not inside:
            self.steps = 0
            self.contact_seen = [False] * 4
            self.support_window.clear()
            return self.success
        # Every foot must have supported weight on the final landing during the interval.
        self.contact_seen = [seen or force[2] > 1.0 for seen, force in zip(self.contact_seen, forces)]
        self.steps += 1
        self.support_window.append([force[2] > 1.0 for force in forces])
        each_foot_supported = all(any(frame[i] for frame in self.support_window) for i in range(4))
        supported_frames = sum(any(frame) for frame in self.support_window)
        if len(self.support_window) == self.required_steps and each_foot_supported and supported_frames >= math.ceil(.8 * self.required_steps):
            self.success = True
        return self.success
