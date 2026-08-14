# RAMMP-CuRobo planning service — Jetson (JetPack 6 / L4T r36) container.
#
# In: an end position (tool pose or joint goal) on the ROS 2 actions
#   /rammp_curobo/plan_to_pose  /rammp_curobo/plan_to_joints
# Out: the collision-free, time-parameterized JointTrajectory.
# The container never owns the arm: it ships planning-only (execute:=false)
# and holds no arm driver — the caller executes the trajectory.
#
# Build ON the Jetson (nvcc for sm_87 comes from the base image; the cuRobo
# kernel compile makes this a ~1 h, ~15 GB build):
#     docker build -f docker/Dockerfile -t rammp-curobo:jp6 .
# Run (see docker/README.md for the client side):
#     docker run --rm -it --runtime nvidia --network host --ipc host \
#         -e ROS_LOCALHOST_ONLY=1 -v rammp-curobo-cache:/root/.cache \
#         rammp-curobo:jp6
#
# Every version here is the bench-verified pin from the repo README.
# Do NOT bump any of them casually — cuRobo v0.8 is an API rewrite,
# newer warp breaks v0.7.8, and torch must come from the jp6/cu126 index.

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

# --- ROS 2 Humble (base image is Ubuntu 22.04 = Humble's target) ---------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release git git-lfs ca-certificates \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-control-msgs \
        ros-humble-controller-manager-msgs \
        ros-humble-example-interfaces \
        python3-colcon-common-extensions \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch: Jetson cu126 wheels (pinned known-good) --------------------
RUN python3 -m pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# --- cuRobo: PINNED v0.7.8, source build for Orin (sm_87) ----------------
# The tag is additionally pinned to its commit hash so a moved tag fails
# the build instead of silently changing the planner.
ENV TORCH_CUDA_ARCH_LIST="8.7+PTX" MAX_JOBS=4
RUN git lfs install \
    && git clone https://github.com/NVlabs/curobo.git /opt/curobo \
    && cd /opt/curobo \
    && git checkout tags/v0.7.8 \
    && test "$(git rev-parse HEAD)" = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" \
    && python3 -m pip install --no-cache-dir -U "packaging>=24.1" "setuptools>=70,<80" \
    && python3 -m pip install --no-cache-dir --no-build-isolation /opt/curobo \
    && python3 -m pip install --no-cache-dir "warp-lang==1.5.1"

# --- this repo: pip core + the two ament packages ------------------------
COPY . /opt/rammp_curobo_src
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        /opt/rammp_curobo_src/core \
    && . /opt/ros/humble/setup.sh \
    && cd /opt/rammp_curobo_src \
    && colcon build --install-base /opt/rammp_curobo \
        --packages-select rammp_curobo_interfaces rammp_curobo_ros \
    && rm -rf /opt/rammp_curobo_src/build /opt/rammp_curobo_src/log

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# First plan after a cold start compiles CUDA kernels (minutes); mount
# /root/.cache as a named volume to keep them across container restarts.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "rammp_curobo_ros", "planner.launch.py", "config:=gen3_real.yaml"]
