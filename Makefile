# Local gates for RAMMP-CuRobo. The heavy ones run on the Jetson — .hil.yml is
# the remote path (`hil.py sync` then `hil.py exec`), and these targets are what
# it ends up invoking. Nothing here touches the arm; this repo has no executor.
#
#   make lint      pre-commit run --all-files
#   make test      the offline + GPU-smoke core suite
#   make colcon    build the two ament packages
#   make build     the Jetson container (~1 h, arm64 only)
#   make verify    run the core suite INSIDE the built image
#   make run       run the planner node from the image
#   make layers    regenerate the MEBot obstacle layers (expect no diff)
#   make sim-build build the MEBot sim stack into .sim_ws/ (Jetson)
#   make sim-test  the sim package's own tests (Jetson; no ROS graph)
#   make sim       build the sim stack and run its suite (Jetson; exits 0
#                  only when every gate passes)

IMAGE ?= rammp-curobo:jp6
# The ROS workspace that provides rammp_arm_interfaces, which the sim's
# driver stand-in serves; consumed as built.
SIM_OVERLAY ?= ~/rammp_deps_ws/install/setup.bash
# Arguments for the sim's launch, e.g. SIM_ARGS=port:=8766 when the live
# view's default port is taken.
SIM_ARGS ?=

.PHONY: lint test colcon build verify run shell bake layers sim-build sim-test sim

lint:                      ## pre-commit run --all-files
	pre-commit run --all-files

# test_smoke.py importorskips torch/cuRobo and needs a live CUDA device, so off
# the Jetson this runs the offline half and skips the rest. That is expected;
# `make verify` is the one that exercises the GPU path. On the Jetson,
# RAMMP_CUROBO_GPU=1 make test fails where the GPU half would skip.
test:                      ## Core tests (offline everywhere, GPU on the Jetson)
	python3 -m pytest core/tests -q
	python3 -m pytest rammp_curobo_ros/test -q -p no:anyio

colcon:                    ## Build the interfaces and the node
	. /opt/ros/humble/setup.sh && \
	  colcon build --symlink-install \
	    --packages-select rammp_curobo_interfaces rammp_curobo_ros

# arm64 only: the base image is nvcr.io/nvidia/l4t-jetpack, which has no x86
# variant. An x86-64 CUDA image is tracked in issue #8.
build:                     ## Build the Jetson image (~1 h, ~15 GB)
	docker build -f docker/Dockerfile -t $(IMAGE) .

# RAMMP_CUROBO_GPU=1: the GPU half is the point here, so where it would skip
# (no device passed through, a torch or cuRobo that does not import) the run
# fails instead of passing on the offline half.
verify:                    ## Run the core suite inside the image (needs a GPU)
	docker run --rm --runtime nvidia -e RAMMP_CUROBO_GPU=1 $(IMAGE) \
	  python3 -m pytest /opt/rammp_curobo_src/core/tests -q

run:                       ## Run the planner node from the image
	docker run --rm -it --runtime nvidia --network host --ipc host \
	  -e ROS_LOCALHOST_ONLY=1 -v rammp-curobo-cache:/root/.cache $(IMAGE)

shell:                     ## Interactive shell in the image
	docker run --rm -it --runtime nvidia --network host --ipc host $(IMAGE) bash

# Regenerating the baked robot config must be a no-op — RETRACT_POSE in that
# script is the only place the retract pose is defined. A diff here means
# someone hand-edited configs/robot_gen3_2f85.yaml. It runs in a released
# image for cuRobo's content tree (NVIDIA's model, the sphere source) and
# yourdfpy; this checkout's core comes first on the path, not the image's.
BAKE_IMAGE ?= ghcr.io/rammp-org/rammp-curobo:1.1.0

bake:                      ## Regenerate the robot config (expect no diff)
	docker run --rm --network none -u $$(id -u):$$(id -g) -v $(CURDIR):/w -w /w \
	  -e PYTHONPATH=/w/core $(BAKE_IMAGE) python3 scripts/bake_robot_config.py
	git diff --exit-code core/rammp_curobo/configs/robot_gen3_2f85.yaml \
	  core/tests/data/robot_gen3_2f85_moves.yaml \
	  core/rammp_curobo/configs/arm_kinematics.json

# Regenerating the MEBot layers must be a no-op too: scripts/mebot_layers.yaml
# and the URDF it pins (from a Demo-Software clone, the cache or GitHub raw)
# are their only inputs. A diff here means someone hand-edited a
# core/tests/data/mebot/layer_mebot_*.yaml, or changed the inputs without
# committing the layers they make.
#
# These are this checkout's copy: fixtures for the tests and the sim, NOT
# package data. The layers are one robot's geometry and ship with that robot,
# in rammp-deployments under config/robots/<robot>/curobo (issue #19). Write
# the robot's copy from the same run with OUTDIR, then commit it there:
#
#   make layers OUTDIR=~/rammp-deployments/december_2026/config/robots/mebot/curobo
#
# Two runs of a deterministic generator, not one rendering: the bytes match
# because the inputs are pinned and the output is snapped to a grid, which is
# also why the in-repo copy can be diff-asserted. The robot's copy has no
# gate here, nor yet in the deployment repo's CI.
layers:                    ## Regenerate the MEBot layers (expect no diff; OUTDIR= also writes a robot's copy)
	python3 scripts/generate_layers.py
	git diff --exit-code core/tests/data/mebot/layer_mebot_platform.yaml \
	  core/tests/data/mebot/layer_mebot_person.yaml \
	  core/tests/data/mebot/layer_mebot_floor.yaml
ifneq ($(OUTDIR),)
	python3 scripts/generate_layers.py --outdir "$(OUTDIR)"
endif

# The sim stack (sim/mebot_sim) and the two packages it runs, built into the
# git-ignored .sim_ws/ as links to these sources. sim/ carries a COLCON_IGNORE
# so that CI's colcon build needs no rammp_arm_interfaces; here the package is
# named as a base path instead. The sim and its planners then run from
# .sim_ws/install/setup.bash, never on the bench's ROS domain: 0, unset or
# anything else but a domain from 1 to 232 becomes 88.
sim-build: SHELL := /bin/bash
sim-build:                 ## Build the sim stack into .sim_ws/ (Jetson)
	. /opt/ros/humble/setup.bash && . $(SIM_OVERLAY) && \
	  colcon --log-base .sim_ws/log build --symlink-install \
	    --base-paths rammp_curobo_interfaces rammp_curobo_ros sim/mebot_sim \
	    --build-base .sim_ws/build --install-base .sim_ws/install

# The sim package's own tests: the domain rule, the scene, flights, contacts,
# the driver's ROS side on a stub node, the page and the launch description,
# with MuJoCo on the CPU and EGL and no ROS graph. pytest.ini's testpaths
# leaves them out of `make test`: they need ROS and the overlay's messages,
# and this checkout's core/ and sim/mebot_sim first on the path. A test
# skips without an input from outside the checkout (the menagerie, MuJoCo,
# EGL, ...); MEBOT_SIM_STRICT=1 makes that a failure here, so that the gate
# never passes on the tests that could still run.
# Non-interactive shells skip ~/.zshrc: without this, nodes split across DDS
# (CLAUDE.md, Environment). It is exported to every recipe these two run,
# unless a CYCLONEDDS_URI picks the interfaces: Cyclone then refuses `lo`
# selected twice and no node starts (abra's bench config pins `lo`).
sim-test sim: export ROS_LOCALHOST_ONLY := $(if $(CYCLONEDDS_URI),$(ROS_LOCALHOST_ONLY),1)
sim-test: SHELL := /bin/bash
sim-test:                  ## The sim package's tests (Jetson; no ROS graph)
	. /opt/ros/humble/setup.bash && . $(SIM_OVERLAY) && \
	  PYTHONPATH=$(CURDIR)/core:$(CURDIR)/sim/mebot_sim:$$PYTHONPATH \
	  MEBOT_SIM_STRICT=1 python3 -m pytest sim/mebot_sim/test -q

# The sim suite (sim/mebot_sim/README.md): the stack, built first, with
# run_suite beside it; the launch ends when run_suite does, and exits with its
# verdict. The planners run this checkout's core/, first on PYTHONPATH.
sim: SHELL := /bin/bash
sim: sim-build             ## Build the sim stack and run its suite (Jetson)
	. /opt/ros/humble/setup.bash && . $(SIM_OVERLAY) && \
	  . $(CURDIR)/.sim_ws/install/setup.bash && \
	  PYTHONPATH=$(CURDIR)/core:$$PYTHONPATH \
	  ros2 launch mebot_sim suite.launch.py $(SIM_ARGS)
