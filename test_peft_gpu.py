import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

path = "/tmp/peft/2787"

torch.manual_seed(0)
device = 0
x = torch.arange(10).unsqueeze(0).to(device)
model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
with torch.inference_mode():
    base_out = model(x).logits
    print("base model output:\n", base_out[0])

config = LoraConfig(init_lora_weights=False)
model = get_peft_model(model, config)
model.eval()
with torch.inference_mode():
    peft_out = model(x).logits
    print(f"output after applying LoRA, device {device} (should be != base output):\n", peft_out[0])

model.save_pretrained(path)
del model

device = 1
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model = PeftModel.from_pretrained(model, path)
with torch.inference_mode():
    peft_out_loaded = model(x.to(device)).logits
    print(f"output after loading LorA, device {device} (should be == previous LoRA output):\n", peft_out_loaded[0])