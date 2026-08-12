"""Config file resolution and the planner-config schema defaults.

`CuRoboPlanner.from_config()` accepts either a filesystem path or the bare
name of a packaged default (rammp_curobo/configs/*.yaml). Robot/world files
referenced from a planner config resolve relative to that config's own
directory first, so a user can copy gen3.yaml next to their code, edit the
world line, and everything keeps working.
"""

import os
from pathlib import Path

import yaml

PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Defaults mirror RAMMP-Kinova's field-tuned planner parameters; every one
# of them can be overridden from the planner YAML. See configs/gen3.yaml for
# what each knob does.
PLANNER_DEFAULTS = {
    "joint_names": [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ],
    "home_pose_rad": [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571],
    "planner": {
        "interpolation_dt": 0.02,
        "max_attempts": 8,
        "finetune_attempts": 5,
        "enable_finetune": True,
        "enable_graph": False,
        "collision_cache_obb": 60,
        "collision_cache_mesh": 10,
        "collision_activation_distance": 0.03,
        "world_padding": 0.02,
        "no_pad_names": ["pedestal"],
        "joint_space_method": "auto",
        "limit_clamp_rad": 0.05,
        "warmup": True,
    },
    "tool": {
        "spin_deg": 0.0,
        "tip_offset_m": 0.0,
    },
    "execution": {
        "speed_scale": 0.25,
        "max_speed_scale": 1.0,
    },
}


def resolve_config(name_or_path, relative_to=None):
    """Resolve a config reference to an existing file path.

    Search order: absolute path -> relative to `relative_to` -> relative to
    the current directory -> packaged defaults. Raises FileNotFoundError
    naming every location tried.
    """
    p = Path(os.path.expanduser(str(name_or_path)))
    tried = []
    if p.is_absolute():
        if p.is_file():
            return p
        tried.append(p)
    else:
        candidates = []
        if relative_to is not None:
            candidates.append(Path(relative_to) / p)
        candidates.append(Path.cwd() / p)
        candidates.append(PACKAGED_CONFIG_DIR / p)
        for c in candidates:
            if c.is_file():
                return c
        tried.extend(candidates)
    raise FileNotFoundError(
        "config %r not found; tried: %s"
        % (str(name_or_path), ", ".join(str(t) for t in tried))
    )


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_planner_config(name_or_path):
    """Load a planner YAML merged over the defaults.

    Returns (config dict, config directory) — the directory anchors
    resolution of the robot/world files the config references.
    """
    path = resolve_config(name_or_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = _merge(PLANNER_DEFAULTS, data)
    for key in ("robot", "world"):
        if key not in cfg:
            raise KeyError("planner config %s is missing the %r entry" % (path, key))
    return cfg, path.parent
