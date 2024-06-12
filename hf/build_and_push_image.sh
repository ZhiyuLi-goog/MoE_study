set -euox pipefail
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ID=cloud-tpu-multipod-dev
DATE=$(date +%Y%m%d)
IMAGE=gcr.io/${PROJECT_ID}/${USER}-pytorch-xla-moe-${DATE}

pushd ${SCRIPTS_DIR}

docker build --network host \
  --file Dockerfile \
  --tag ${IMAGE} \
  .
popd

docker push ${IMAGE}