"""Resume sample generation: depth + quantized tiers only (canny already done)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from scripts.generate_samples import (
    generate_controlnet_depth,
    generate_quantized_8bit,
    generate_quantized_4bit,
    SAMPLES_DIR,
)

generate_controlnet_depth()
generate_quantized_8bit()
generate_quantized_4bit()

total = sum(len(list((SAMPLES_DIR / t).glob("*.png")))
            for t in ["standard_fp16", "lcm", "turbo", "lora_ukiyo_e",
                       "controlnet_canny", "controlnet_depth",
                       "quantized_8bit", "quantized_4bit"]
            if (SAMPLES_DIR / t).is_dir())
print(f"\n[samples] Resume done. {total} total sample images in {SAMPLES_DIR}")
