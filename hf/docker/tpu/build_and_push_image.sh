set -euox pipefail
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
DATE=$(date +%Y%m%d)
: ${PROJECT_ID:=cloud-tpu-multipod-dev}
: ${IMAGE:=gcr.io/${PROJECT_ID}/${USER}-pytorch-xla-moe-${DATE}}

pushd ${SCRIPTS_DIR}

docker build --network host \
  --file Dockerfile \
  --tag ${IMAGE} \
  .
popd

docker push ${IMAGE}