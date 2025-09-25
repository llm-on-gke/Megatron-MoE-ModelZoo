#!/usr/bin/env bash
##############################################################################
# vLLM bootstrap script
# - Installs vLLM
# - Idempotent: re-runs will update existing repos instead of recloning
##############################################################################

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd  )"
source ${SCRIPT_DIR}/common.sh

##############################  configuration  ###############################
# Locations (override via env if desired)
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/app/vllm}"

# Repositories
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-main}"

banner "Environment summary"
echo "Python version      : ${PYTHON_VERSION}"
echo "Virtualenv path     : ${VENV_PATH}"
echo "uv binary           : ${UV}"
echo "vLLM repo / branch  : ${VLLM_REPO_URL}  (${VLLM_BRANCH})"
echo "vLLM commit         : ${VLLM_COMMIT:-<latest>}"
echo "====================================================================="

################################  vLLM  ######################################
clone_or_update "${VLLM_REPO_URL}" "${VLLM_SOURCE_DIR}" "${VLLM_BRANCH}" "${VLLM_COMMIT:-}"

banner "Installing vLLM (editable)"
pushd "${VLLM_SOURCE_DIR}" >/dev/null

# Default to precompiled binaries when on main
if [[ -z "${VLLM_COMMIT:-}" && "${VLLM_BRANCH}" == "main" && "${VLLM_REPO_URL}" == "https://github.com/vllm-project/vllm.git" ]]; then
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
fi

# TODO(tms): Work around for compressed_tensors bug in vLLM.
# Remove when no longer needed
upip accelerate

upip -e .

popd >/dev/null

banner "vLLM is ready"
