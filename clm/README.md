# 1. MLPerf Training: MoE Benchmark
This benchmark focuses on training Mixtral8x22B with a 32,768 token sequence length with key features:
* spare mixture-of-experts architecture: we specifically use the [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) architecture and checkpoint. This allows for greater computational efficiency compared to dense models like GPT-3 or LLaMA2-3, as only a subset of experts are activated during training and inferencing.
* Extended sequence length: Handles sequences up to 32,768 tokens long, enabling larger contexts window

# 2. Dataset
This benchmark uses the
[C4](https://www.tensorflow.org/datasets/catalog/c4)/en/3.0.1
dataset from TensorFlow Dataset. A version is 
[available](https://huggingface.co/datasets/allenai/c4/tree/main/en) 
from Hugging Face.

The benchmark uses a different split than the original C4/en/3.0.1:

| Split | What's in it | What it is used for | number of samples |
| - | - | - | - |
| train1 | first 768 of 1024 files of C4/en/3.0.1:train | training the initial checkpoint | 274,651,678 |
| train2 | last 256 of 1024 files of C4/en/3.0.1:train | training dataset of the benchmark | 91,217,223 |
| validation\_24567exp | 1/20th of C4/en/3.0.1:validation | validation datset of the benchmark | 24,567 |

The resplit dataset uses 3.0.4 as its version to differenciate from the original
3.0.1 version, and it's available on
[GCS](https://console.cloud.google.com/storage/browser/mlperf-llm-public2/c4/en/3.0.4)

Note this benchmark uses the same dataset as gpt3-175b benchmark see [dataset in gpt3-175b benchmark for reference](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#2-dataset).

# 3. Model, Checkpoint, & Optimizer
we specifically use the [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) architecture and checkpoint. 

| Config | Value |
| - | - |
| num_hidden_layers | 56 |
| num_attention_heads | 48 |
| num_experts_per_tok | 2 |
| num_key_value_heads | 8 |
| num_local_experts | 8 |
| vocab_size | 32000 |
| hidden_size | 6144 |
| intermediate_size | 16384 |

As for optimizer, we use [adamw](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

# 4. Evaluation
## Evaluation loss metric
Negative log likelihood loss for next token prediction

## Target Evaluation Loss
1.8

## Evaluation frequency
24 * 1024 * 2048 tokens

# 5.Training with [torch_xla](https://github.com/pytorch/xla/tree/master) on TPU Device
## Environment Setup

Docker image is used in this repo for environment setup.

The following command is to create an enviroment with necessary libraries, mainly including:
* ML Framework: [torch_xla](https://github.com/pytorch/xla.git)
* Models: [transformers](https://github.com/huggingface/transformers.git)
* config tool: [hydra-core](https://hydra.cc/)
```bash
# This command will create, tag, and push an image default to gcr.io/${PROJECT_ID}/${USER}-pytorch-xla-moe-${DATE}
bash docker/tpu/build_and_push_image.sh
```

```bash
# Alternatively, create, tag, and push an image with different name
IMAGE=<my_image> bash docker/tpu/build_and_push_image.sh
```

The tpu_type is `v5p-512` or 256 chips to train Mixtral8x22B with a 32,768 token sequence length.
## [recommended] Run Experiments in GKE

### Install XPK and create GKE cluster.
```
pip install xpk
xpk cluster create --cluster <cluster_name> --tpu-type=<tpu_type> --num-slices=<num_slices>
```

### Run workload in GKE
```bash
# login token required since 
# the mixtral model is a restricted model 
# that requires users e-signed agreement in place before accessing it
export HF_TOKEN=<your_hf_token>

xpk workload create \
--cluster <cluster_name> \
--base-docker-image ${IMAGE} \
--workload ${USER}-run \
--tpu-type=<tpu_type> \
--num-slices=<num_slices> \
--command="
# Setup envs
export HF_HOME=/tmp
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1

# Debug info
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

cd MoE_study/hf
git pull
huggingface-cli login --token ${HF_TOKEN}
# workload script
# equivalent of 
#   python run_clm.py +experiment=gbs256_tpu
python run_clm.py model.config_path=mixtral822.json per_device_train_batch_size=1 optimizer=ADAMW_TORCH_XLA checkpoint_manager_path=gs://lizhiyu-multipods-eu-west/moe/checkpoints-20240803/mixtral822/ model.name_or_path=mistralai/Mixtral-8x22B-v0.1 dataset.dataset_name=c4_mlperf max_steps=250 max_grad_norm=1.0 seed=4321 model.dtype=bfloat16 output_dir=/app/output max_length=32768 dataset.streaming=True tensor_parallelism=1 exp_name=convergence_exp model.capacity_factor=1.25 lr=4e-5 sched=WarmupHoldPolicy
EOF
"
```

## Run Experiments in GCE

### set project and zone

```bash
# change to a valid PROJECT_ID and ZONE
export PROJECT_ID=cloud-tpu-multipod-dev
export ZONE=us-central2-b

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
```

### Create TPU VMs
```bash
# create tpu vm say v4-8 as an example
export RUNTIME_VERSION=tpu-ubuntu2204-base
export TPU_NAME=${USER}-mlperf
gcloud compute tpus tpu-vm create ${TPU_NAME} --zone=${ZONE} --accelerator-type='v4-8' --version=${RUNTIME_VERSION}
```


### ssh to TPU VMs and Run Workloads
Pull docker image, say a pre-built image `gcr.io/cloud-tpu-multipod-dev/lizhiyu-pytorch-xla-moe-20241031`
```bash
# change to a valid docker image
export IMAGE=gcr.io/cloud-tpu-multipod-dev/lizhiyu-pytorch-xla-moe-20241031

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--worker=all \
--command="
yes Y | sudo gcloud auth configure-docker
sudo docker pull ${IMAGE}
"
```

Run workloads
```bash
# login token required since 
# the mixtral model is a restricted model 
# that requires users e-signed agreement in place before accessing it
export HF_TOKEN=<your_hf_token>

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--worker=all \
--command="
sudo docker run --privileged --net host --shm-size=16G --interactive -v /tmp:/tmp ${IMAGE} bash -s <<EOF

# Setup envs
export HF_HOME=/tmp
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1

# Debug info
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

cd MoE_study/clm
git pull
huggingface-cli login --token ${HF_TOKEN}
# workload script
# equivalent of 
#   python run_clm.py +experiment=gbs256_tpu
python run_clm.py model.config_path=mixtral822.json per_device_train_batch_size=1 optimizer=ADAMW_TORCH_XLA checkpoint_manager_path=gs://lizhiyu-multipods-eu-west/moe/checkpoints-20240803/mixtral822/ model.name_or_path=mistralai/Mixtral-8x22B-v0.1 dataset.dataset_name=c4_mlperf max_steps=250 max_grad_norm=1.0 seed=4321 model.dtype=bfloat16 output_dir=/app/output max_length=32768 dataset.streaming=True tensor_parallelism=1 exp_name=convergence_exp model.capacity_factor=1.25 lr=2e-5 sched=WarmupHoldPolicy
EOF
"
```

## Dry-run Tests
To perform dry-run tests with different models/parameters on a smaller scale like `v4-8`, use the following python commands as workload:
### Test with a smaller mixtral model
```
python run_clm.py model.config_path=mixtral80.json eval_frequency=3 n_eval_examples=100 per_device_train_batch_size=4 max_steps=30
``` 
### Test with a gpt2 model
```
python run_clm.py model.name_or_path=gpt2 eval_frequency=3 n_eval_examples=100 per_device_train_batch_size=4 gradient_accumulation_steps=2 sched.warmup_ratio=0. max_steps=30
```

# 6. Training Mixtral 8x7B with NeMo on GPU Device

## Docker Image

Build and push docker image:

```shell
docker build -t <regristry_path_image_name>:<image_tag> -f nemo_example.Dockerfile .
docker push <registry_path_image_name>:<image_tag>
```

## Run workflow

In order for this workflow to function, in the ```helm-context``` directory, there must exist a **_select-configuration.yaml_** file.

Package and schedule job. An example job name could be "nemo-gpt3-175b-nemo-16gpus". Use whatever is convenient when searching for later.


```shell
helm install <username_workload_job_name> helm-context/
```

## Monitor workflow

Check pod status (use this to find the name of the pod you want logs from)


```shell
kubectl get pods | grep "<some_part_of_username_workload_job_name>"
```


Check job status


```shell
kubectl get jobs | grep "<some_part_of_username_workload_job_name>"
```


Get logs (Using pod name from earlier)


```shell
kubectl logs "<pod_name>"
```

# 7. Reference
* [MLPerf Training: MoE Benchmark Proposal from Nvidia](https://docs.google.com/document/d/1NOJ_vt-o2WHFXmisLRk6Mn7Ki2CeB5UNeTkFrYHoE1I/edit?usp=sharing)
* [Mixtral of Experts](https://arxiv.org/pdf/2401.04088)

# 8. Lint

```
black clm/
```