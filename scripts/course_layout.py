"""Geometry only; no terrain information is passed to the policy."""
import math


def build_course(config):
    width = float(config["width_m"])
    landing = float(config["landing_m"])
    edge = float(config["first_edge_m"])
    if not all(math.isfinite(v) for v in (width, landing, edge)) or width <= 0 or landing < 1.2 or edge < 0.8:
        raise ValueError("Invalid width, landing, or approach")
    if config.get("kind") == "gap":
        gap = float(config["gap_m"])
        depth = float(config["depth_m"])
        if not math.isfinite(gap) or not 0.05 <= gap <= .9 or not math.isfinite(depth) or not .2 <= depth <= 1.:
            raise ValueError("Invalid gap geometry")
        return [(edge + gap, landing, 0.)], [dict(id="cross", count=1, entry_x_m=edge+gap, base_height_m=0., end_x_m=edge+gap+landing, top_height_m=0.)], width
    blocks, stages = [], []
    top = 0.0
    for stage in config["stages"]:
        rise, count = float(stage["rise_m"]), stage["count"]
        if not math.isfinite(rise) or not 0 < abs(rise) <= 0.4 or not isinstance(count, int) or not 1 <= count <= 6:
            raise ValueError("Invalid stage")
        tread = float(stage.get("tread_m", landing))
        final_landing = float(stage.get("landing_m", landing))
        if not math.isfinite(tread) or tread < 0.25 or not math.isfinite(final_landing) or final_landing < 1.2:
            raise ValueError("Invalid tread or landing")
        item = dict(stage, entry_x_m=edge, base_height_m=top)
        for i in range(count):
            top += rise
            if top < -1e-8:
                raise ValueError("Course cannot descend below the ground plane")
            top = max(0.0, top)
            length = final_landing if i == count - 1 else tread
            blocks.append((edge, length, top))
            edge += length
        stages.append(dict(item, end_x_m=edge, top_height_m=top))
    if not stages:
        raise ValueError("Empty course")
    return blocks, stages, width
