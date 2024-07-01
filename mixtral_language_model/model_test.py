import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import copy

from transformers import AutoTokenizer, AutoConfig, MixtralForCausalLM, AutoModelForCausalLM

from model import (MixtralXLAConfig, MixtralXLAForCausalLM)

# AutoConfig.register("mixtral_xla", MixtralXLAConfig)
# AutoModelForCausalLM.register(MixtralXLAConfig, MixtralXLAForCausalLM)

model_id = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(
    model_id,
    vocab_size=len(tokenizer),
    torch_dtype=torch.bfloat16,
    num_hidden_layers=1,
    hidden_size=512,
    intermediate_size=2048,
    num_local_experts=4,
)
print(config)
config.flash_attention = False
config.static = False

device = 'xla'

torch.manual_seed(42)
dynamic_model = MixtralForCausalLM(config).to(device)

new_config = copy.deepcopy(config)
new_config.flash_attention_xla = True
new_config.static = True
torch.manual_seed(42)
static_model = MixtralXLAForCausalLM(new_config).to(device)
static_model.load_state_dict(dynamic_model.state_dict())
torch.manual_seed(42)

print(f"Model parameters: {dynamic_model.num_parameters()/2**20:.2f}M params")

# This is a custom config to enable the static mode of expert computation.
print(f"Model parameters: {static_model.num_parameters()/2**20:.2f}M params")

input_sizes = [8, 128, 256, 512, 1024]
for input_size in input_sizes:
    input = torch.randint(128, ((2, input_size // 2))).to(device)
    static_output = static_model(input)
    print(static_output.logits.shape)
    print(static_output.logits)
    dynamic_output = dynamic_model(input)
    print(dynamic_output.logits.shape)
    print(dynamic_output.logits)
    assert torch.allclose(static_output.logits, dynamic_output.logits, atol=1e-2, rtol=1e-2), "logits are not equal"

device = xm.xla_device()
model = static_model.to(device)
output = model(torch.randint(128, ((2, 128))).to(device))
loss = torch.sum(output.logits)
loss.backward()
xm.mark_step()
print(met.metrics_report())


# 1e-5 cpu, static
# 1e-2 xla, static + flash attention