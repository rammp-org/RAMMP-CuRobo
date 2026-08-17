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
        curl gnupg lsb-release git ca-certificates \
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

# --- torch runtime deps the base image lacks ------------------------------
# The Jetson jp6/cu126 torch wheel links two libraries l4t-jetpack does NOT ship
# (a JetPack *host* has them system-wide, so this only bites in a container).
# cuRobo's setup.py imports torch during metadata generation, so without these
# the next layer dies with an ImportError before compiling anything:
#   libopenblas.so.0 -> apt libopenblas0
#   libcudss.so.0    -> nvidia-cudss-cu12 wheel (no apt package for tegra)
# These are the ONLY two missing: ldd across torch/lib reports just these plus
# libcuda.so.1, which --runtime nvidia injects at run time by design.
#
# cudss is installed --no-deps deliberately: its declared deps pull a whole CUDA
# 12.9 toolkit in alongside the base image's 12.6. With --no-deps only the .so
# files land, and torch still reports cuda_built 12.6. The wheel unpacks under
# site-packages/nvidia/cu12/lib, which this torch build has no RPATH for, hence
# the explicit ldconfig entry rather than relying on LD_LIBRARY_PATH.
#
# The trailing import is a build-time gate: if either dep regresses, this layer
# fails loudly here instead of surfacing as a confusing pip metadata error.
#
# Kept as its own layer after the torch install purely so the multi-GB torch
# download stays cached; fold into the apt block above when upstreaming.
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --no-deps nvidia-cudss-cu12==0.8.0.10 \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    && echo /usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib \
        > /etc/ld.so.conf.d/nvidia-cudss.conf \
    && ldconfig \
    && python3 -c "import torch; print('torch import OK:', torch.__version__)"

# --- cuRobo: PINNED v0.7.8, source build for Orin (sm_87) ----------------
# The tag is additionally pinned to its commit hash so a moved tag fails
# the build instead of silently changing the planner. warp and numpy are
# pinned BEFORE the cuRobo install so pip's resolver keeps the
# bench-verified versions instead of pulling latest for cuRobo's bare
# 'warp-lang>=0.9.0' / 'numpy' requirements. ninja makes MAX_JOBS real
# (torch cpp_extension only parallelizes through it). v0.7.8 has no
# git-LFS content — plain clone is complete.
ENV TORCH_CUDA_ARCH_LIST="8.7+PTX" MAX_JOBS=4
RUN git clone https://github.com/NVlabs/curobo.git /opt/curobo \
    && cd /opt/curobo \
    && git checkout tags/v0.7.8 \
    && test "$(git rev-parse HEAD)" = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" \
    && python3 -m pip install --no-cache-dir -U \
        "packaging>=24.1" "setuptools>=70,<80" \
        "numpy==1.26.4" "warp-lang==1.5.1" ninja \
    && python3 -m pip install --no-cache-dir --no-build-isolation /opt/curobo

# --- this repo: pip core + the two ament packages ------------------------
COPY . /opt/rammp_curobo_src
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        /opt/rammp_curobo_src/core \
    && . /opt/ros/humble/setup.sh \
    && cd /opt/rammp_curobo_src \
    && colcon build --install-base /opt/rammp_curobo \
        --packages-select rammp_curobo_interfaces rammp_curobo_ros \
    && rm -rf /opt/rammp_curobo_src/build /opt/rammp_curobo_src/log

# The repo's own configs contain non-ASCII characters (degree signs in
# world_sim_kitchen.yaml comments, lines 183/202/204). Loading that world decodes
# the file with the ascii codec and dies:
#   UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2
# LANG=C.UTF-8 alone does not prevent it; UTF-8 mode does. Without this,
# core/tests/test_smoke.py::test_update_world_guards_and_round_trip fails (22/23).
ENV PYTHONUTF8=1

# Cyclone DDS RMW. The rest of the fleet is Cyclone (RAMMP-Software's base image
# installs this and its compose sets RMW_IMPLEMENTATION fleet-wide), and the
# gen3 driver container forces it too. Without this package the image can only
# run Fast DDS, and the driver -> planner link is cross-vendor: discovery
# succeeds, then samples fail with
#   [RTPS_READER_HISTORY Error] Change payload size of N bytes is larger than
#   the history payload size of M bytes
# Deliberately a late layer so the ~12 min cuRobo kernel compile stays cached;
# fold into the apt block above when upstreaming.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# First plan after a cold start compiles CUDA kernels (minutes); mount
# /root/.cache as a named volume to keep them across container restarts.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "rammp_curobo_ros", "planner.launch.py", "config:=gen3_real.yaml"]
