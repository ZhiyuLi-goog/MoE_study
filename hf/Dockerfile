FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm 

WORKDIR /app

RUN pip install datasets accelerate evaluate scikit-learn
RUN pip install trl wandb

# Install jax, jaxlib, libtpu nightly
RUN pip install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
RUN pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
RUN pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre

RUN git clone https://github.com/pytorch-tpu/transformers.git

WORKDIR /app/transformers

RUN git fetch origin alanwaketan/moe &&\
   git checkout alanwaketan/moe &&\
   pip install -e .

COPY fsdp_config.json /app/transformers
COPY mixtral81.json /app/transformers

WORKDIR /app/transformers