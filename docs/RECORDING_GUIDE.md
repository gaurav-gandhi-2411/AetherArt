# AetherArt — Screencast Recording Guide

60–90 second demo on your local RTX 3070. Captures each speed tier visually so
viewers understand the trade-offs at a glance.

## Setup

- Use **OBS Studio** (or any screen recorder)
- Dark mode / dark desktop preferred for contrast with the Gradio UI
- Start the app: `conda activate aetherart && python app.py`
- Open `http://localhost:7860` in a browser, **full-screen the Gradio UI**
- Have `nvidia-smi` open in a terminal (optional VRAM indicator)

## Script — 5 scenes

Keep the same base prompt throughout for a clear visual comparison:
> **"a lone samurai standing in a misty forest, dramatic lighting, cinematic"**

### Scene 1 — Standard fp16 (10–15 s)

1. Speed Mode: **Standard**  
2. Resolution: 512×512, Steps: 30, Guidance: 7.5  
3. Click **Generate** — let the timer run visibly  
4. Result appears (~3.2 s on RTX 3070) — pause 2 s on it

### Scene 2 — LCM 4-step (10–15 s)

1. Switch Speed Mode to **Fast (LCM)** — same prompt  
2. Click **Generate** — result appears in ~0.6 s  
3. Visually compare sharpness vs Scene 1 — hold 2 s

### Scene 3 — SDXL Turbo (10–15 s)

1. Switch to **Turbo (SDXL)** — same prompt  
2. Click **Generate** — 1-step, ~3.3 s  
3. Note the different aesthetic (SDXL architecture) — hold 2 s

### Scene 4 — 8-bit INT8 quantization (10–15 s)

1. Memory / VRAM Mode: **8-bit INT8**  
2. Speed Mode: Standard  
3. Click **Generate** — show nvidia-smi if possible (~2.2 GB vs ~3.1 GB fp16)  
4. Highlight VRAM difference in caption

### Scene 5 — Outro (5–10 s)

1. Switch to the **Sample Outputs** tab — scroll through the gallery  
2. End on the repo README header (Alt+Tab to browser or GitHub)

## Export

- Save as **MP4** (H.264, ~15–20 Mbps)
- Compress with HandBrake: CRF 23, 720p — target **< 10 MB**
- Place at: `docs/aetherart_demo.mp4`
- Optional GIF (< 3 MB, 15 fps, 720×540) for README embedding:
  ```
  ffmpeg -i docs/aetherart_demo.mp4 \
    -vf "fps=12,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    docs/aetherart_demo.gif
  ```
- Place GIF at: `docs/aetherart_demo.gif`

## After recording

```bash
git add docs/aetherart_demo.mp4 docs/aetherart_demo.gif
git commit -m "docs: add screencast demo (RTX 3070 local, 5 speed/memory tiers)"
git push origin main
python scripts/deploy_to_hf.py
```

Then add to `README.md` (replace the placeholder line):
```markdown
![AetherArt demo](docs/aetherart_demo.gif)
```
