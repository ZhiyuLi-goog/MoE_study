# Docker Image Build
create docker image
```
bash build_and_push_image.sh 
```

# Run DPO Trainer

```
# inside a VM
export IMAGE=<pre-built image>
export BRANCH=lizhiyu/dpo_static  # some branch
export TOKEN=<hf token>

sudo docker run --privileged --net host --shm-size=16G --interactive -v /tmp:/tmp "${IMAGE}" bash -s <<EOF
# inside /app/transformers installed from https://github.com/pytorch-tpu/transformers
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
huggingface-cli login --token $TOKEN
pip install --no-deps --force-reinstall git+https://github.com/ZhiyuLi-goog/trl.git@"${BRANCH}"

WANDB_MODE=offline XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 XLA_USE_SPMD=1 PJRT_DEVICE=TPU PROFILE_EPOCH=0 python dpo.py --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style --model_name_or_path=mistralai/Mixtral-8x7B-v0.1 --per_device_train_batch_size 1 --learning_rate 1e-3 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --output_dir="dpo_anthropic_hh" --warmup_steps 150 --logging_first_step --no_remove_unused_columns --config_name mixtral81.json --torch_dtype=bfloat16 --fsdp "full_shard" --fsdp_config fsdp_config.json --max_steps 6
EOF
```

experimental branch:
* [transformers @lizhiyu/dpo_static](https://github.com/pytorch-tpu/transformers/tree/lizhiyu/dpo_static)
* [trl @lizhiyu/dpo_static](https://github.com/ZhiyuLi-goog/trl/tree/lizhiyu/dpo_static)
