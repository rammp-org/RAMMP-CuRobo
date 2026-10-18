"""GPU: the planner on the arm's own URDF, checked against NVIDIA's model.

The baked robot config moved the audited spheres from NVIDIA's links onto the
upstream ones (scripts/bake_robot_config.py). These tests prove the move
through cuRobo's own kinematics, the thing that actually plans: at a set of
configurations every sphere sits where it did on NVIDIA's model (within 1 mm)
and arm_tool_frame is NVIDIA's tool_frame (within 1 mm and 0.1 deg). The rest
pin what the arm's model changes for callers: the seven arm_joint_N names, the
URDF's limits, the arm_mount stamp, and planning on a tilted mount.

Skips itself cleanly on machines without CUDA/cuRobo, like test_smoke.py.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from rammp_curobo import CuRoboPlanner  # noqa: E402
from rammp_curobo.robot_config import load_robot_config  # noqa: E402
from rammp_curobo.urdf_model import ARM_JOINTS, GRIPPER_JOINTS  # noqa: E402

DATA = Path(__file__).parent / "data"
BAKE = Path(__file__).resolve().parents[2] / "scripts" / "bake_robot_config.py"
_spec = importlib.util.spec_from_file_location("bake_robot_config", BAKE)
_bake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bake)
LINK_MAP = _bake.LINK_MAP

TILT = math.radians(30.0)
# A world nothing can reach: the kitchen is laid out for a level arm, and a
# tilted one must not fail on it for reasons that have nothing to do with frames.
FAR_BOX = [{"name": "far", "position": [5.0, 5.0, 5.0], "dims": [0.1, 0.1, 0.1]}]


class Kin:
    """One robot config through cuRobo's CudaRobotModel, q in ARM_JOINTS order."""

    def __init__(self, cfg):
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.types.robot import RobotConfig

        self.cfg = cfg["kinematics"]
        self.model = CudaRobotModel(RobotConfig.from_dict(cfg).kinematics)
        # NVIDIA's model names the joints joint_N; ours arm_joint_N.
        names = [n.replace("arm_", "") for n in self.model.joint_names]
        self.order = [
            [a.replace("arm_", "") for a in ARM_JOINTS].index(n) for n in names
        ]

    def state(self, q):
        q = torch.tensor(
            [[float(q[i]) for i in self.order]], device="cuda", dtype=torch.float32
        )
        return self.model.get_state(q)

    def spheres(self, q, link):
        idx = self.model.kinematics_config.get_sphere_index_from_link_name(link)
        return self.state(q).link_spheres_tensor[0, idx].cpu().numpy()

    def ee_pose(self, q):
        s = self.state(q)
        return s.ee_position[0].cpu().numpy(), s.ee_quaternion[0].cpu().numpy()

    def limits(self):
        lim = self.model.get_joint_limits().position.cpu().numpy()
        back = np.argsort(self.order)  # cuRobo order -> ARM_JOINTS order
        return lim[0][back], lim[1][back]


@pytest.fixture(scope="module")
def new_kin(urdf_level):
    return Kin(load_robot_config("robot_gen3_2f85.yaml", urdf_path=urdf_level))


@pytest.fixture(scope="module")
def old_kin():
    return Kin(load_robot_config(DATA / "robot_gen3_2f85_nvidia.yaml"))


@pytest.fixture(scope="module")
def configs(new_kin, old_kin):
    """Retract, zeros, and 20 seeded-random configurations inside BOTH models'
    limits (the URDF's j2/j4/j6 are the tighter ones)."""
    lo_n, hi_n = new_kin.limits()
    lo_o, hi_o = old_kin.limits()
    lo, hi = np.maximum(lo_n, lo_o), np.minimum(hi_n, hi_o)
    rng = np.random.default_rng(0)
    retract = new_kin.cfg["cspace"]["retract_config"]
    return [np.asarray(retract), np.zeros(7), *rng.uniform(lo, hi, size=(20, 7))]


@pytest.fixture(scope="module")
def planner(urdf_level):
    return CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_level)


@pytest.fixture
def start(planner):
    return list(planner.retract_pose)


def _qmul(a, b):
    """Hamilton product, xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _angle(a, b):
    """Angle between two unit quaternions (either packing, the same in both)."""
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def test_spheres_match_the_old_model_within_1mm(new_kin, old_kin, configs):
    # Every sphere is paired: the mapped links are exactly our collision links.
    old_links = old_kin.cfg["collision_link_names"]
    assert sorted(LINK_MAP[n] for n in old_links) == sorted(
        new_kin.cfg["collision_link_names"]
    )
    n_spheres = 0
    for q in configs:  # world positions: arm_mount (new) vs base_link (old), level
        for old_link in old_links:
            new = new_kin.spheres(q, LINK_MAP[old_link])
            old = old_kin.spheres(q, old_link)
            assert new.shape == old.shape and len(new), old_link
            dist = np.linalg.norm(new[:, :3] - old[:, :3], axis=1).max()
            assert dist < 1e-3, "%s: %.3f mm at q=%s" % (old_link, dist * 1e3, q)
            assert np.allclose(new[:, 3], old[:, 3], atol=1e-6), old_link
            n_spheres += len(new)
    assert n_spheres == len(configs) * old_kin.model.kinematics_config.total_spheres


def test_tool_frame_matches_nvidia_within_1mm_and_0_1deg(new_kin, old_kin, configs):
    for q in configs:
        p_new, quat_new = new_kin.ee_pose(q)
        p_old, quat_old = old_kin.ee_pose(q)
        dp = float(np.linalg.norm(p_new - p_old))
        dang = _angle(quat_new, quat_old)
        assert dp < 1e-3 and dang < math.radians(0.1), "%.3f mm, %.4f deg at q=%s" % (
            dp * 1e3,
            math.degrees(dang),
            q,
        )


def test_gripper_is_locked_open_through_its_mimics(urdf_level):
    """lock_joints names only the actuated knuckle; cuRobo's mimic handling
    must hold the other five. Every gripper joint's child link sits where it
    does with all six joints at 0 (yourdfpy as the reference)."""
    import xml.etree.ElementTree as ET

    from yourdfpy import URDF

    root = ET.parse(urdf_level).getroot()
    links = [
        root.find("joint[@name='%s']/child" % j).get("link") for j in GRIPPER_JOINTS
    ]
    cfg = load_robot_config("robot_gen3_2f85.yaml", urdf_path=urdf_level)
    cfg["kinematics"]["link_names"] = links
    kin = Kin(cfg)
    retract = kin.cfg["cspace"]["retract_config"]
    q = torch.tensor([retract], device="cuda", dtype=torch.float32)
    got = kin.model.get_link_poses(q, links).position[0].cpu().numpy()

    ref = URDF.load(str(urdf_level), load_meshes=False)
    ref.update_cfg({**dict(zip(ARM_JOINTS, retract)), GRIPPER_JOINTS[0]: 0.0})
    for link, p in zip(links, got):
        want = ref.get_transform(link, "arm_mount")[:3, 3]
        assert np.linalg.norm(p - want) < 1e-6, "%s: %.3g m" % (
            link,
            np.linalg.norm(p - want),
        )


def test_kinematics_joint_names_are_the_seven_arm_joints(planner):
    assert planner._motion_gen.kinematics.joint_names == list(ARM_JOINTS)


def test_joint_limits_are_the_urdfs(planner):
    lo, hi = planner.joint_limits()["position"]
    assert hi[1] == pytest.approx(2.24) and lo[1] == pytest.approx(-2.24)
    assert hi[3] == pytest.approx(2.57) and lo[3] == pytest.approx(-2.57)
    assert hi[5] == pytest.approx(2.09) and lo[5] == pytest.approx(-2.09)
    for j in (0, 2, 4, 6):  # continuous joints: cuRobo's parser gives +-6.28
        assert hi[j] == pytest.approx(6.28, abs=0.01) and lo[j] == -hi[j]


def test_every_plan_is_stamped_arm_mount(planner, start):
    assert planner.base_frame == "arm_mount"
    goal = list(start)
    goal[1] -= 0.25
    res = planner.plan_to_joints(goal, start)
    assert res.success, res.error
    assert res.joint_traj.frame_id == "arm_mount"

    q_target = list(start)
    q_target[0] += 0.4
    res = planner.plan_to_pose(*planner.fk(q_target), start)
    assert res.success, res.error
    assert res.joint_traj.frame_id == "arm_mount"
    # a retimed trajectory keeps its frame
    assert res.joint_traj.scaled(0.5).frame_id == "arm_mount"


def test_tilted_model_plans_in_arm_mount(planner, urdf_tilt30):
    """The same arm-relative goal, rotated by Ry(30 deg) into arm_mount, is
    what the tilted model plans to. Joint solutions are NOT compared:
    redundant IK may pick another family for the identical pose."""
    tilt = CuRoboPlanner.from_config("gen3.yaml", urdf_path=urdf_tilt30)
    tilt.update_world(FAR_BOX)
    start = list(planner.retract_pose)
    q_target = list(start)
    q_target[0] += 0.4
    q_target[5] -= 0.3
    p, quat = planner.fk(q_target)  # level: arm_mount IS the arm's base

    R = np.array(
        [
            [math.cos(TILT), 0.0, math.sin(TILT)],
            [0.0, 1.0, 0.0],
            [-math.sin(TILT), 0.0, math.cos(TILT)],
        ]
    )
    goal_p = R @ np.asarray(p)
    goal_q = _qmul([0.0, math.sin(TILT / 2), 0.0, math.cos(TILT / 2)], quat)

    # The tilted model really is tilted: the same joints land on the rotated pose.
    tp, tq = tilt.fk(q_target)
    assert np.linalg.norm(np.asarray(tp) - goal_p) < 1e-3
    assert _angle(tq, goal_q) < math.radians(0.5)

    res = tilt.plan_to_pose(goal_p.tolist(), goal_q, start)
    assert res.success, res.error
    assert res.joint_traj.frame_id == "arm_mount"
    fp, fq = tilt.fk(res.final_joints)
    dp = float(np.linalg.norm(np.asarray(fp) - goal_p))
    dang = _angle(fq, goal_q)
    assert dp < 1e-3 and dang < math.radians(0.5), (
        "reached %.3f mm, %.3f deg from the goal" % (dp * 1e3, math.degrees(dang))
    )


def test_park_pose_beyond_clamp_is_refused_by_name(planner):
    park = list(planner.retract_pose)
    park[3] = -2.674  # the real Gen3's park; the URDF limit is +-2.57
    res = planner.plan_to_joints(planner.retract_pose, park)
    assert not res.success
    assert res.status == "START_OUTSIDE_LIMITS"
    assert "arm_joint_4" in res.error


def test_retract_is_a_valid_state(planner):
    ok, detail = planner.check_state_valid(planner.retract_pose)
    assert ok, detail
