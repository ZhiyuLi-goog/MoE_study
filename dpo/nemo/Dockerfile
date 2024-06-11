# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:24.04-py3
WORKDIR /workspace/ft-llm

RUN git config --global user.name "a" && \
    git config --global user.email "a"

## NeMo
ARG NEMO_REVISION=r2.0.0rc0

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    echo NEMO_REVISION=${NEMO_REVISION} && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install --no-build-isolation -e ".[nlp]"

## Megatron-core
ARG MEGATRON_REVISION=72560a8bc0f0b0246bae477bd2a6a997850dc017

RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    echo MEGATRON_REVISION=${MEGATRON_REVISION} && \
    git checkout ${MEGATRON_REVISION} && \
    echo MEGATRON_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install . && \
    cd megatron/core/datasets && \
    make
ENV PYTHONPATH "${PYTHONPATH}:/workspace/ft-llm/megatron-lm"

ENV NVTE_FRAMEWORK=pytorch

## TransformerEngine
# the following two variables are needed when building TE
ARG TE_REVISION=090e72412e06f44fe43aa4c4564ae11469961c9a
ENV CUSTOM_TE_REVISION ${TE_REVISION}

# https://github.com/NVIDIA/TransformerEngine/pull/784
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/TransformerEngine.git && \
      cd TransformerEngine && \
      git submodule init && git submodule update && \
      echo TE_REVISION=${TE_REVISION} && \
      git checkout ${CUSTOM_TE_REVISION} && \
      echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
      NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps . \
    ; fi


ARG NEMO_ALIGNER_REVISION=v0.3.0.trtllm

RUN git clone https://github.com/NVIDIA/NeMo-Aligner.git && \
    cd NeMo-Aligner && \
    echo NEMO_ALIGNER_REVISION=${NEMO_ALIGNER_REVISION} && \
    git checkout ${NEMO_ALIGNER_REVISION} && \
    echo NEMO_ALIGNER_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install .

RUN pip install --force-reinstall transformers==4.40.1

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk

