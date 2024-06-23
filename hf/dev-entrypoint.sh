set -ex
echo "[INFO]: this is an dev image"

GIT_REF=${GIT_REF:-main}
echo "[INFO]: GIT_REF=${GIT_REF}"
git clone https://github.com/ZhiyuLi-goog/MoE_study.git &&
    cd MoE_study && \
    git checkout ${GIT_REF}

exec bash -c "$@"