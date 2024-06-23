FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm 

WORKDIR /app

RUN pip install datasets accelerate evaluate scikit-learn

# Install jax, jaxlib, libtpu nightly
RUN pip install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
RUN pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
RUN pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre

# custom transformers with static mixtral implementation
# TODO: import changes in the current repo
ARG TRANSFORMERS_REVISION=lizhiyu/dpo_static
RUN git clone https://github.com/pytorch-tpu/transformers && \
    cd transformers && \
    echo TRANSFORMERS_REVISION=${TRANSFORMERS_REVISION} && \
    git checkout ${TRANSFORMERS_REVISION} && \
    echo TRANSFORMERS_REVISION=$(git rev-parse HEAD) && \
    pip install -e .

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

RUN pip install hydra-core

WORKDIR /app

# clone current repo
ARG GIT_REF=main
RUN git clone https://github.com/ZhiyuLi-goog/MoE_study.git &&
    cd MoE_study && \
    echo GIT_REF=${GIT_REF} && \
    git checkout ${GIT_REF}