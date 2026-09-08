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

IMAGE ?= rammp-curobo:jp6

.PHONY: lint test colcon build verify run shell bake

lint:                      ## pre-commit run --all-files
	pre-commit run --all-files

# test_smoke.py importorskips torch/cuRobo and needs a live CUDA device, so off
# the Jetson this runs the offline half and skips the rest. That is expected;
# `make verify` is the one that exercises the GPU path.
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

verify:                    ## Run the core suite inside the image (needs a GPU)
	docker run --rm --runtime nvidia $(IMAGE) \
	  python3 -m pytest /opt/rammp_curobo_src/core/tests -q

run:                       ## Run the planner node from the image
	docker run --rm -it --runtime nvidia --network host --ipc host \
	  -e ROS_LOCALHOST_ONLY=1 -v rammp-curobo-cache:/root/.cache $(IMAGE)

shell:                     ## Interactive shell in the image
	docker run --rm -it --runtime nvidia --network host --ipc host $(IMAGE) bash

# Regenerating the baked robot config must be a no-op — RETRACT_POSE in that
# script is the only place the retract pose is defined. A diff here means
# someone hand-edited configs/robot_gen3_2f85.yaml.
bake:                      ## Regenerate the robot config (expect no diff)
	python3 scripts/bake_robot_config.py
	git diff --exit-code core/rammp_curobo/configs/robot_gen3_2f85.yaml
