# Running HuggingFace Mixtral DPO Training on Cloud TPUs

This guide offers a brief overview of the key steps needed to train HuggingFace's Mixtral model on Cloud TPUs.
The training tasks is [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

This repo take reference to the following repos:
* [Author's DPO Repo from Eric Mitchell](https://github.com/eric-mitchell/direct-preference-optimization)
* [Huggingface's TRL](https://github.com/huggingface/trl/tree/main)

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
Pull docker image, say a pre-built image `gcr.io/cloud-tpu-multipod-dev/lizhiyu-pytorch-xla-moe-20240820`
```bash
# change to a valid docker image
export IMAGE=gcr.io/cloud-tpu-multipod-dev/lizhiyu-pytorch-xla-moe-20240908

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
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

cd MoE_study/hf
git pull
huggingface-cli login --token ${HF_TOKEN}
python run_dpo_no_trainer.py model.config_path=mixtral80.json max_length=4096 per_device_train_batch_size=1
EOF
"
```

## Run Experiments in GKE

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
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

cd MoE_study/hf
git pull
huggingface-cli login --token ${HF_TOKEN}
python run_dpo_no_trainer.py model.config_path=mixtral80.json max_length=4096 per_device_train_batch_size=1
EOF
"
```

## Experiments
To recreate the experiments, substitute the Python scripts with the following commands.
### Reproduce Pythia2.8b DPO experiments
Best to run in v4-128 or v5p-128, and it is possible to run in v4-8 as well.  
Remove `dry_run=True` for a true experiment.
```bash
python run_dpo_no_trainer.py per_device_train_batch_size=1 optimizer=RMSprop report_metrics_freq=1 use_synthetic_data=False model.name_or_path=EleutherAI/pythia-2.8b datasets=hh max_steps=2600 eval_frequency=312 do_first_eval=True model.policy_dtype=float32 model.reference_dtype=float32 max_grad_norm=10.0 shuffle=True seed=4321 full_precision=True dry_run=True
```

### Mixtral8x22b
```bash
python run_dpo_no_trainer.py model.config_path=mixtral822.json per_device_train_batch_size=1 optimizer=RMSprop checkpoint_manager_path=gs://lizhiyu-multipods-eu-west/moe/checkpoints-20240803/mixtral822/ report_metrics_freq=1 use_synthetic_data=False model.name_or_path=mistralai/Mixtral-8x22B-v0.1 datasets=hh max_steps=2600 eval_frequency=312 do_first_eval=True model.policy_dtype=float32 model.reference_dtype=bfloat16 max_grad_norm=10.0 seed=4321
```
