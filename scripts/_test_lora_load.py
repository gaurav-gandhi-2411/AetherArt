"""Diagnose LoRA adapter name at load time and test full application."""

from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

LORA_PATH = Path("data/lora/ukiyo-e/ukiyo-e-lora.safetensors")
MODEL = "sd2-community/stable-diffusion-2-1"

print("Loading SD 2.1...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

print(
    "\nAdapters BEFORE load:",
    pipe.get_list_adapters() if hasattr(pipe, "get_list_adapters") else "N/A",
)

print(f"\nLoading LoRA from {LORA_PATH}...")
pipe.load_lora_weights(str(LORA_PATH.parent), weight_name=LORA_PATH.name)

print(
    "Adapters AFTER load:",
    pipe.get_list_adapters() if hasattr(pipe, "get_list_adapters") else "N/A",
)
if hasattr(pipe, "get_active_adapters"):
    print("Active adapters:", pipe.get_active_adapters())

# Check UNet adapter state
if hasattr(pipe.unet, "get_adapter_state_dict"):
    print("UNet adapter state available")
# Check what PEFT registered
if hasattr(pipe.unet, "peft_config"):
    print("PEFT config keys:", list(pipe.unet.peft_config.keys()))

print("\nDONE — adapter name identified above")
