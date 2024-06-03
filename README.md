# Install

```
pip install -r requirements.txt
```

# Benchmark DPO

## Run
```
# single gpu run
accelerate launch --num_processes 1 loss.py

# works for cpu as well
accelerate launch dpo/loss.py

```

## Results
```
The following values were not passed to `accelerate launch` and had defaults used instead:
                More than one GPU was found, enabling multi-GPU training.
                If this was unintended please pass in `--num_processes=1`.
        `--num_machines` was set to a value of `1`
        `--mixed_precision` was set to a value of `'no'`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
device=device(type='cuda', index=0)
STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          reference_dpo         0.82%     578.000us        58.39%      41.241ms      41.241ms       0.000us         0.00%      60.034ms      60.034ms             1
                                          reference_dpo         0.00%       0.000us         0.00%       0.000us       0.000us      55.989ms        53.31%      55.989ms      55.989ms             1
                                              aten::sub         2.67%       1.886ms        22.89%      16.168ms       2.695ms      24.292ms        23.13%      35.288ms       5.881ms             6
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      24.292ms        23.13%      24.292ms       2.024ms            12
                                              aten::mul         0.23%     161.000us        16.15%      11.406ms       1.901ms      16.497ms        15.71%      16.497ms       2.749ms             6
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      16.497ms        15.71%      16.497ms       1.375ms            12
                                       cudaLaunchKernel        47.31%      33.413ms        47.31%      33.413ms       1.114ms       5.502ms         5.24%       5.502ms     183.400us            30
                                             cudaMalloc         6.93%       4.896ms         6.93%       4.896ms     699.429us       5.494ms         5.23%       5.494ms     784.857us             7
                                      aten::log_sigmoid         0.03%      23.000us         5.41%       3.821ms       1.911ms       0.000us         0.00%       5.493ms       2.747ms             2
                              aten::log_sigmoid_forward         0.14%      97.000us         5.38%       3.798ms       1.899ms       5.493ms         5.23%       5.493ms       2.747ms             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.493ms         5.23%       5.493ms       1.373ms             4
                                              aten::neg         0.09%      65.000us        13.09%       9.246ms       9.246ms       2.756ms         2.62%       2.756ms       2.756ms             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.756ms         2.62%       2.756ms       1.378ms             2
                                  cudaStreamIsCapturing         0.02%      13.000us         0.02%      13.000us       1.857us       0.000us         0.00%       0.000us       0.000us             7
                                       aten::empty_like         0.05%      34.000us         2.31%       1.630ms     815.000us       0.000us         0.00%       0.000us       0.000us             2
                                    aten::empty_strided         0.06%      41.000us         2.26%       1.596ms     798.000us       0.000us         0.00%       0.000us       0.000us             2
                                            aten::empty         0.02%      12.000us         0.02%      12.000us       6.000us       0.000us         0.00%       0.000us       0.000us             2
                                           aten::detach         0.02%      13.000us         0.02%      13.000us       6.500us       0.000us         0.00%       0.000us       0.000us             2
                                                 detach         0.01%       9.000us         0.01%       9.000us       4.500us       0.000us         0.00%       0.000us       0.000us             2
                                  cudaDeviceSynchronize        41.61%      29.389ms        41.61%      29.389ms      29.389ms       0.000us         0.00%       0.000us       0.000us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 70.630ms
Self CUDA time total: 105.027ms

STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                 hf_dpo         0.00%       0.000us         0.00%       0.000us       0.000us      49.077ms        50.03%      49.077ms      49.077ms             1
                                                 hf_dpo         0.33%     168.000us         4.12%       2.089ms       2.089ms       0.000us         0.00%      49.020ms      49.020ms             1
                                              aten::sub         3.22%       1.634ms         3.36%       1.702ms     283.667us      24.287ms        24.76%      24.287ms       4.048ms             6
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      24.287ms        24.76%      24.287ms       2.024ms            12
                                              aten::mul         0.11%      57.000us         0.23%     115.000us      19.167us      16.486ms        16.81%      16.486ms       2.748ms             6
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      16.486ms        16.81%      16.486ms       1.374ms            12
                                      aten::log_sigmoid         0.01%       4.000us         0.15%      75.000us      37.500us       0.000us         0.00%       5.498ms       2.749ms             2
                              aten::log_sigmoid_forward         0.05%      23.000us         0.14%      71.000us      35.500us       5.498ms         5.60%       5.498ms       2.749ms             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       5.498ms         5.60%       5.498ms       1.375ms             4
                                              aten::neg         0.02%      12.000us         0.04%      21.000us      21.000us       2.749ms         2.80%       2.749ms       2.749ms             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.749ms         2.80%       2.749ms       1.375ms             2
                                       cudaLaunchKernel         0.30%     154.000us         0.30%     154.000us       5.133us       0.000us         0.00%       0.000us       0.000us            30
                                               aten::to         0.00%       1.000us         0.00%       1.000us       0.167us       0.000us         0.00%       0.000us       0.000us             6
                                       aten::empty_like         0.03%      13.000us         0.05%      24.000us      12.000us       0.000us         0.00%       0.000us       0.000us             2
                                    aten::empty_strided         0.02%      11.000us         0.02%      11.000us       5.500us       0.000us         0.00%       0.000us       0.000us             2
                                            aten::empty         0.01%       5.000us         0.01%       5.000us       2.500us       0.000us         0.00%       0.000us       0.000us             2
                                           aten::detach         0.01%       4.000us         0.01%       7.000us       3.500us       0.000us         0.00%       0.000us       0.000us             2
                                                 detach         0.01%       3.000us         0.01%       3.000us       1.500us       0.000us         0.00%       0.000us       0.000us             2
                                  cudaDeviceSynchronize        95.88%      48.607ms        95.88%      48.607ms      48.607ms       0.000us         0.00%       0.000us       0.000us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 50.696ms
Self CUDA time total: 98.097ms

STAGE:2024-06-02 03:57:04 6093:6093 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-06-02 03:57:05 6093:6093 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-06-02 03:57:05 6093:6093 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          nemo_dpo_loss         0.00%       0.000us         0.00%       0.000us       0.000us     124.910ms        73.82%     124.910ms     124.910ms             1
                                          nemo_dpo_loss         0.46%     599.000us       100.00%     130.239ms     130.239ms       0.000us         0.00%      44.308ms      44.308ms             1
                                              aten::cat         1.30%       1.696ms         4.96%       6.455ms       3.228ms      11.821ms         6.99%      11.821ms       5.910ms             2
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      11.821ms         6.99%      11.821ms       5.910ms             2
                                              aten::sub         0.07%      85.000us        29.52%      38.442ms      19.221ms       9.304ms         5.50%       9.304ms       4.652ms             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       9.304ms         5.50%       9.304ms       1.861ms             5
                                              aten::mul         0.06%      77.000us         3.81%       4.965ms       2.482ms       8.099ms         4.79%       8.099ms       4.050ms             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       8.097ms         4.78%       8.097ms       2.024ms             4
                                               aten::to        -0.00%      -1.000us        32.99%      42.964ms      14.321ms       0.000us         0.00%       6.042ms       2.014ms             3
                                            aten::copy_         0.05%      61.000us         7.97%      10.383ms       5.191ms       6.042ms         3.57%       6.042ms       3.021ms             2
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       6.042ms         3.57%       6.042ms       1.208ms             5
                                         aten::_to_copy         0.04%      52.000us        32.97%      42.948ms      21.474ms       0.000us         0.00%       6.039ms       3.019ms             2
                                               aten::gt         0.08%      99.000us         7.21%       9.389ms       4.694ms       3.842ms         2.27%       3.842ms       1.921ms             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.841ms         2.27%       3.841ms     960.250us             4
                                              aten::sum         0.08%      98.000us         7.29%       9.492ms       9.492ms       2.645ms         1.56%       2.645ms       2.645ms             1
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       2.645ms         1.56%       2.645ms     661.250us             4
                                       aten::zeros_like         0.01%      15.000us         8.16%      10.627ms      10.627ms       0.000us         0.00%       2.540ms       2.540ms             1
                                            aten::zero_         0.02%      25.000us         7.31%       9.519ms       9.519ms       0.000us         0.00%       2.540ms       2.540ms             1
                                            aten::fill_         0.03%      45.000us         7.29%       9.494ms       9.494ms       2.540ms         1.50%       2.540ms       2.540ms             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.540ms         1.50%       2.540ms     635.000us             4
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 130.245ms
Self CUDA time total: 169.218ms
```

The slight delay observed in the nemo_dpo_loss application, attributed to concat and split_output operations, could potentially enhance MFU during model function calls with larger batch sizes beyond the DPO loss context.

# Dataset Info

```
python preference_datasets_metrics.py
```
## [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)
```
shp: num_samples=198556
shp: num_tokens_chosen_input=73847785
shp: num_tokens_rejected_input=59121849
shp: avg_tokens_chosen_input=371.9242178528979
shp: avg_tokens_rejected_input=297.75906545256754
```

## [Anthropic Helpful-Harmless dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
```
hh: num_samples=160800
hh: num_tokens_chosen_input=39369144
hh: num_tokens_rejected_input=38779153
hh: avg_tokens_chosen_input=244.83298507462686
hh: avg_tokens_rejected_input=241.1638868159204
```

