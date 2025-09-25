#!/usr/bin/env bash
##############################################################################
# FlashInfer installation script
##############################################################################

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd  )"
source ${SCRIPT_DIR}/common.sh

FLASHINFER_SOURCE_DIR="${FLASHINFER_SOURCE_DIR:-/app/flashinfer}"
FLASHINFER_REPO_URL="${FLASHINFER_REPO_URL:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_BRANCH="${FLASHINFER_BRANCH:-}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-}"

banner "Environment summary"
echo "Python version      : ${PYTHON_VERSION}"
echo "Virtualenv path     : ${VENV_PATH}"
echo "uv binary           : ${UV}"
echo "FlashInfer repo     : ${FLASHINFER_REPO_URL}"
echo "====================================================================="

if [[ -n "${FLASHINFER_BRANCH:-}" ]]; then
    # Dependencies
    upip cuda-python ninja
    # Remove if already installed to prevent versioning issues
    "${UV}" pip uninstall --python "${PYTHON}" flashinfer-python
    clone_or_update "${FLASHINFER_REPO_URL}" "${FLASHINFER_SOURCE_DIR}" "${FLASHINFER_BRANCH}" "${FLASHINFER_COMMIT}"
    pushd "${FLASHINFER_SOURCE_DIR}" >/dev/null
    if [[ -n "${FLASHINFER_AOT:-}" ]]; then
        "${PYTHON}" -m flashinfer.aot
    fi
    upip -e . --no-build-isolation
    popd >/dev/null
else
    upip flashinfer-python
fi
