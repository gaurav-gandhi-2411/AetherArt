from safetensors import safe_open

path = "data/lora/ukiyo-e/ukiyo-e-lora.safetensors"
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())

print("Total keys:", len(keys))
print("First 15 keys:")
for k in keys[:15]:
    print(" ", k)

prefixes = set()
for k in keys:
    parts = k.split(".")
    if len(parts) >= 2:
        prefixes.add(parts[0])
print("\nUnique top-level prefixes:", sorted(prefixes)[:20])
