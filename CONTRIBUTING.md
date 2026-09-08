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

One bulk reformat is recorded in `.git-blame-ignore-revs` (mdformat, applied
repo-wide when CI started running the hooks over every file). Enable it so
`git blame` reaches the author rather than the reformat:

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

## Releases

Tags are cut on `main` and publish to
`ghcr.io/rammp-org/rammp-curobo`. `image.yml` refuses to publish a `v*` tag
that does not point at a commit on `main`, before spending the hour rather than
after. Semver tags only: `{{version}}`, `{{major}}.{{minor}}`, `{{major}}`, and
`latest` follows the newest non-prerelease tag.

## On hardware

**Attended only.** Follow `docs/HARDWARE_BRINGUP.md`, with a human on the
physical e-stop. Never drive the real arm autonomously from an agent session.

Nothing in this repo can move the arm, and that is the safety model — one
execution authority, not two with different rules. `kinova_arm_ros2` owns
execution and its own gates. Do not add an edge back: no executor, no
`/joint_states` subscription, no controller client, no arm-package dependency in
`package.xml`.
