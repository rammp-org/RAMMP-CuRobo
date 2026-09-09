# Contributing

This repo follows the RAMMP module workflow. It is a **planning service**: start
config plus end position in, collision-free joint trajectory out. It never
touches the arm (issue #6), and that constraint shapes both the branching and
what CI can honestly claim.

## Pre-commit hooks

Style is enforced automatically before each commit — Python via Ruff, Markdown
via mdformat, the Dockerfile via hadolint, the workflows via actionlint,
`docker/entrypoint.sh` via shellcheck, plus general file hygiene.

Run this once after cloning:

```bash
uv tool install pre-commit
pre-commit install
```

Without `uv`: `pip install pre-commit && pre-commit install`.

Hook revisions are pinned. Updating them is a deliberate PR
(`pre-commit autoupdate`), never silent drift.

Bulk reformats are recorded in `.git-blame-ignore-revs`. It is currently empty —
the repo-wide mdformat run was squash-merged with the change that added it, so
there is no formatting-only commit to ignore. Land future ones as their own
merge, and enable the file so `git blame` reaches the author rather than the
reformat:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Branches

- `main` — demo-ready, stable, deployable. Updated from `dev` by PR. Tags are
  cut here.
- `dev` — staging ground for tested new code. Feature PRs land here.
- `feature/<issue-number>-<brief-description>` — forked from the latest `dev`.
  Use `bug/<issue-number>-<brief-description>` for fixes.

## What CI actually gates

Three workflows, deliberately split by cost, because the container build is an
hour and the rest is minutes.

|                                                        | PR → `dev` | push `dev` | PR → `main` | push `main` | tag `v*` |
| ------------------------------------------------------ | ---------- | ---------- | ----------- | ----------- | -------- |
| `lint.yml` — pre-commit, all files                     | ✅         | ✅         | ✅          | ✅          | —        |
| `build.yml` — core tests + colcon, amd64 **and** arm64 | ✅         | ✅         | ✅          | ✅          | ✅       |
| `image.yml` — the JetPack container                    | —          | ✅         | ✅          | ✅          | ✅       |
| publish to ghcr                                        | —          | —          | —           | —           | ✅       |

A feature PR into `dev` gets the fast gates and does not wait an hour. The merge
to `dev` builds the container, so a break is blamed on one merge. The PR into
`main` is the promotion gate — the last full build before a tag.

### The container job is a build gate, not a functional one

Hosted runners have no GPU. Inside `image.yml`, `torch.cuda.is_available()` is
`False`, `core/tests/test_smoke.py` skips, the first-plan CUDA kernel JIT never
runs and `--runtime nvidia` never happens. What it does prove is everything the
fast gates cannot reach:

- the pinned dependency chain still resolves — torch 2.10.0 / torchvision 0.25.0
  from `pypi.jetson-ai-lab.io/jp6/cu126`, `nvidia-cudss-cu12` 0.8.0.10
- cuRobo v0.7.8 still builds — the commit-hash assertion catches a moved tag,
  and nvcc compiles the `sm_87` kernels against torch 2.10 / warp 1.5.1 /
  numpy 1.26.4
- the two hand-found runtime deps (`libopenblas0`, the cudss `ldconfig` entry):
  the Dockerfile's build-time `import torch` gate only fires there
- arm64 jammy ROS apt availability, including `rmw-cyclonedds-cpp`

**Green CI does not mean the planner plans.** That is the Jetson's job, below.

When `image.yml` fails on a merge or a tag — not on a PR, where the red X is
already visible — it files or updates a single `ci-failure` issue, and closes it
when a build goes green again.

## Building and testing

Locally, `make lint` / `make test` / `make colcon` are the fast ones.

The container is arm64-only (`nvcr.io/nvidia/l4t-jetpack` has no x86 variant),
so it builds on the Jetson:

```bash
uv run ~/.claude/skills/hardware-loop/scripts/hil.py sync
uv run ~/.claude/skills/hardware-loop/scripts/hil.py exec -- bash -lc \
  'cd RAMMP-CuRobo && make build'
```

Then the gate that CI cannot give you — the GPU half of the suite, in the image
that was actually built:

```bash
uv run ~/.claude/skills/hardware-loop/scripts/hil.py exec -- bash -lc \
  'cd RAMMP-CuRobo && make verify'
```

`ros2 run rammp_curobo_ros tour_demo` is the live end-to-end plan check.

An x86-64 CUDA image — which *could* be functionally tested in CI, unlike the
Jetson one — is tracked in issue #8.

## Versioning

Semantic versioning, from **1.0.0** on. **Downstream repos pin an exact tag,
never a branch.** `kinova_arm_ros2` is the consumer that matters.

- **MAJOR** — a breaking change to the public surface.
- **MINOR** — additive: new optional config, new core functions, a new service.
- **PATCH** — fixes that change no declared message, signature, or contract.

### The public surface

- `rammp_curobo_interfaces/` — the `.action` and `.srv` definitions. This is the
  contract `kinova_arm_ros2` compiles against, and the one that matters most.
- The `rammp_curobo` core API: everything in `core/rammp_curobo/__init__.py`'s
  `__all__`.
- The container's ROS surface: the action and service *names*, and the
  behavioural guarantees this file states — chiefly that `start_joints` is
  required and that nothing here can move the arm.

Not covered: `scripts/`, tests, docs, the tuning inside `configs/`, and the
internals of any module not exported above.

### The rule that is easy to get wrong

**In ROS 2, adding a field to an existing message is a MAJOR change**, even
though adding things is usually minor. rosidl folds the definition into a type
hash used for discovery and endpoint matching, so a node built against the old
`.action` and one built against the new one do not talk — they fail to match, or
fail on deserialization, rather than degrading gracefully.

This binds hardest on `PlanToPose.action` and `PlanToJoints.action`, which
`kinova_arm_ros2` builds against. Adding a *new* action or service alongside the
existing ones is genuinely minor; growing one of these three is not.

The milder case is the core Python API, where adding a keyword argument with a
default is additive as usual.

### Cutting a release

Open a `release: vX.Y.Z` PR into `dev` containing **only** the bump and the
changelog, so the diff is the claim and is reviewable on its own:

1. Bump the version in all five places it lives — `core/pyproject.toml`,
   `core/rammp_curobo/__init__.py`, `rammp_curobo_interfaces/package.xml`,
   `rammp_curobo_ros/package.xml`, and `rammp_curobo_ros/setup.py`. Unlike the
   driver, nothing here derives the number from a single file; a mismatch is a
   silently wrong `ros2 pkg` listing, so grep before you push.
1. In `CHANGELOG.md`, rename `Unreleased` to the new version, date it, and add
   the comparison links at the bottom.
1. Merge that PR, then open the `dev` → `main` promotion PR. That PR runs the
   full JetPack build — the last gate before a tag.
1. **Wait for CI to go green on `main`**, then tag it `vX.Y.Z` and push the tag,
   so the tag points at a commit that has passed the gates rather than one you
   hope will. The tag build refuses to publish a tag that is not on `main`.
1. Confirm the image is pullable, then move the consuming repo's pin to the new
   tag. This ordering is forced: a consumer can only pin a tag that exists.

Tags publish to `ghcr.io/rammp-org/rammp-curobo`. Semver tags only —
`{{version}}`, `{{major}}.{{minor}}`, `{{major}}` — and `latest` follows the
newest non-prerelease tag, which is what a bare `docker pull` gets.

Bump at release time, not on every merge — you cannot know whether the next
release is minor or major until you see what landed. And the sharpest signal for
MAJOR is not the diff: it is whether `kinova_arm_ros2` needed an adoption commit.

## On hardware

**Attended only.** Follow `docs/HARDWARE_BRINGUP.md`, with a human on the
physical e-stop. Never drive the real arm autonomously from an agent session.

Nothing in this repo can move the arm, and that is the safety model — one
execution authority, not two with different rules. `kinova_arm_ros2` owns
execution and its own gates. Do not add an edge back: no executor, no
`/joint_states` subscription, no controller client, no arm-package dependency in
`package.xml`.
