#!/usr/bin/env bash
set -euo pipefail

# Install the exact source revisions used by the optional --backend deepseek
# path. DeepEP deliberately follows its current V2 ElasticBuffer API.
# The defaults are the exact official revisions used while writing and checking
# the companion script. Override them only when deliberately updating the APIs.
DEEPEP_REF="${DEEPEP_REF:-01dc3aaac82068020353dce2c302e38153c0bfaa}"
DEEPGEMM_REF="${DEEPGEMM_REF:-559d79fb6994a58b8a15b4b93bf13ccc16edf247}"
INSTALL_ROOT="${MOE_HOPPER_ROOT:-${PWD}/.moe_hopper_deps}"
DEEP_EP_DIR="${INSTALL_ROOT}/DeepEP"
DEEP_GEMM_DIR="${INSTALL_ROOT}/DeepGEMM"

die() { echo "[setup_moe_hopper] ERROR: $*" >&2; exit 1; }
note() { echo "[setup_moe_hopper] $*"; }

command -v python3 >/dev/null || die "python3 is required"
command -v git >/dev/null || die "git is required"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "triton") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing Python packages: " + ", ".join(missing) + "; install a CUDA PyTorch/Triton environment first")
import torch
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.10'):
    raise SystemExit(f"DeepEP V2 requires PyTorch >= 2.10; found {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible. This setup targets H20/H100-class GPUs, not CPU/Gloo.")
major, minor = torch.cuda.get_device_capability()
if major not in (9, 10):
    raise SystemExit(f"need SM90/SM100 for DeepGEMM/DeepEP, found compute capability {major}.{minor}")
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; device {torch.cuda.get_device_name(0)}; SM {major}.{minor}")
PY

MOE_NVCC_BIN="${DG_JIT_NVCC_COMPILER:-nvcc}"
if command -v "${MOE_NVCC_BIN}" >/dev/null; then
  NVCC_VERSION="$("${MOE_NVCC_BIN}" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
  note "nvcc ${NVCC_VERSION:-unknown}"
else
  die "nvcc is required to JIT/build the official kernels; install CUDA Toolkit >= 12.3"
fi

command -v g++ >/dev/null || die "g++ with C++20 support is required by DeepGEMM"
CPP20_TMP="$(mktemp -d)"
trap 'rm -rf "${CPP20_TMP}"' EXIT
printf '#include <version>\nint main() { static_assert(__cplusplus >= 202002L); }\n' > "${CPP20_TMP}/check.cpp"
g++ -std=c++20 "${CPP20_TMP}/check.cpp" -o "${CPP20_TMP}/check" \
  || die "g++ cannot compile C++20; install a C++20-capable compiler"
rm -rf "${CPP20_TMP}"
trap - EXIT
note "C++20 compiler check passed"

"${PYTHON_BIN}" - <<'PY'
import os
import re
import subprocess
import torch

def version_tuple(value):
    out = []
    for part in value.split('.')[:3]:
        digits = ''.join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out + [0] * (3 - len(out)))

cuda = version_tuple(torch.version.cuda or '0')
if cuda < (12, 3, 0):
    raise SystemExit(f"CUDA >= 12.3 is required; PyTorch reports {torch.version.cuda}")
# The JIT compiler can differ from the CUDA runtime bundled with PyTorch.
# Validate the actual toolkit too, with the architecture-specific minimum.
major, _ = torch.cuda.get_device_capability()
required = (12, 9) if major == 10 else (12, 3)
compiler = os.environ.get('DG_JIT_NVCC_COMPILER', 'nvcc')
try:
    nvcc_text = subprocess.check_output([compiler, '--version'], text=True, stderr=subprocess.STDOUT)
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"cannot run CUDA compiler {compiler}: {exc}")
match = re.search(r'release\s+(\d+)\.(\d+)', nvcc_text)
if match is None:
    raise SystemExit("cannot parse CUDA Toolkit version from nvcc --version")
toolkit = tuple(map(int, match.groups()))
if toolkit < required:
    raise SystemExit(
        f"SM{major}0 requires CUDA Toolkit >= {required[0]}.{required[1]}; "
        f"{compiler} reports {toolkit[0]}.{toolkit[1]}"
    )
print(f"CUDA Toolkit {toolkit[0]}.{toolkit[1]} meets the SM{major}0 requirement")
try:
    import torch.distributed as dist
    nccl = dist.is_nccl_available()
except Exception:
    nccl = False
if not nccl:
    raise SystemExit("PyTorch was not built with NCCL; install a CUDA/NCCL PyTorch build")
nccl_version = torch.cuda.nccl.version()
if isinstance(nccl_version, tuple):
    nccl_tuple = tuple(nccl_version)
else:
    nccl_number = int(nccl_version or 0)
    nccl_tuple = (nccl_number // 10000, (nccl_number // 100) % 100, nccl_number % 100)
if nccl_tuple < (2, 30, 4):
    raise SystemExit(f"NCCL >= 2.30.4 is required by DeepEP V2; found {nccl_tuple}")
print(f"NCCL backend/version check passed: {nccl_tuple}")
print("A single-node run needs NVLink/P2P connectivity between all participating GPUs.")
PY

# DeepEP's build system must also be able to locate the NCCL and NVSHMEM
# headers/libraries. A working NCCL backend inside PyTorch is not sufficient.
"${PYTHON_BIN}" - <<'PY'
import os
from importlib.metadata import distributions

names = {(dist.metadata.get('Name') or '').lower() for dist in distributions()}

def available(package, *env_names):
    return any(os.environ.get(name) for name in env_names) or any(
        f'nvidia-{package}' in name or f'nvidia_{package}' in name
        for name in names
    )

if not available('nccl', 'EP_NCCL_ROOT_DIR', 'NCCL_DIR'):
    raise SystemExit(
        'DeepEP build cannot locate NCCL headers/libraries. Install the matching '
        'nvidia-nccl package recommended by the pinned DeepEP README, or set '
        'EP_NCCL_ROOT_DIR/NCCL_DIR.'
    )
if not available('nvshmem', 'EP_NVSHMEM_ROOT_DIR', 'NVSHMEM_DIR'):
    raise SystemExit(
        'DeepEP currently also builds its legacy backend and requires NVSHMEM. '
        'Install it using the pinned DeepEP docs, or set '
        'EP_NVSHMEM_ROOT_DIR/NVSHMEM_DIR.'
    )
print('DeepEP build dependencies are discoverable: NCCL and NVSHMEM')
PY

mkdir -p "${INSTALL_ROOT}"
clone_or_update() {
  local url="$1" dir="$2" ref="$3"
  if [[ -d "${dir}/.git" ]]; then
    note "using existing ${dir}"
    git -C "${dir}" fetch --tags --prune origin
  else
    note "cloning ${url} into ${dir}"
    git clone --recursive "${url}" "${dir}"
  fi
  git -C "${dir}" checkout --detach "${ref}"
  git -C "${dir}" submodule update --init --recursive
  note "${dir} pinned at $(git -C "${dir}" rev-parse HEAD)"
}

clone_or_update https://github.com/deepseek-ai/DeepEP.git "${DEEP_EP_DIR}" "${DEEPEP_REF}"
clone_or_update https://github.com/deepseek-ai/DeepGEMM.git "${DEEP_GEMM_DIR}" "${DEEPGEMM_REF}"

if [[ "${MOE_HOPPER_SKIP_INSTALL:-0}" == "1" ]]; then
  note "MOE_HOPPER_SKIP_INSTALL=1: source revisions checked out, packages not installed"
else
  note "installing DeepEP (V2 ElasticBuffer)"
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e "${DEEP_EP_DIR}"
  note "installing DeepGEMM"
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e "${DEEP_GEMM_DIR}"
fi

cat <<EOF

Setup complete.
  DeepEP:   ${DEEP_EP_DIR} @ $(git -C "${DEEP_EP_DIR}" rev-parse HEAD)
  DeepGEMM: ${DEEP_GEMM_DIR} @ $(git -C "${DEEP_GEMM_DIR}" rev-parse HEAD)

Run the baseline:
  torchrun --standalone --nproc-per-node=\$(nvidia-smi -L | wc -l) \\
    blogs/code/moe_ep_hopper.py --backend torch-triton --phase train --check

Run the official backend on one NVLink-connected H20/H100 node:
  torchrun --standalone --nproc-per-node=\$(nvidia-smi -L | wc -l) \\
    blogs/code/moe_ep_hopper.py --backend deepseek --phase decode --check

Compare both forward pipelines on identical inputs (dynamic preparation included):
  torchrun --standalone --nproc-per-node=\$(nvidia-smi -L | wc -l) \\
    blogs/code/moe_ep_hopper.py --backend both --phase decode --check --bench-iters 30

DeepEP V2 requires PyTorch >=2.10, NCCL >=2.30.4, and NVLink for intranode
operation. The CUDA Toolkit must be >=12.3 on SM90, or >=12.9 on SM100.
The DeepGEMM source build additionally requires a C++20 compiler and CUTLASS
submodules.  Standard Colab T4/L4 runtimes are intentionally rejected.
EOF
