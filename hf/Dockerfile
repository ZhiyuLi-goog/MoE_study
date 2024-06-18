FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm 

WORKDIR /app

RUN pip install datasets accelerate evaluate scikit-learn

# Install jax, jaxlib, libtpu nightly
RUN pip install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
RUN pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
RUN pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre

RUN git clone https://github.com/pytorch-tpu/transformers.git

WORKDIR /app/transformers

RUN git fetch origin alanwaketan/moe &&\
   git checkout alanwaketan/moe &&\
   pip install -e .

RUN pip install trl wandb

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

WORKDIR /app/transformers