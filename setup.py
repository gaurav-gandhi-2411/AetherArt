from setuptools import setup, find_packages

setup(
    name="aetherart",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gradio>=3.24",
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.18.0",
        "huggingface_hub[hf_xet]>=0.15.0",
        "safetensors",
        "Pillow",
        "python-dotenv",
    ],
)
