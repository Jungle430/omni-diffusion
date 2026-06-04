# UV Debug Environment

This branch uses `uv` for the server-side debug environment.

The requirements file intentionally does not include `torch` or `torchaudio`.
Install the CUDA-specific PyTorch wheels first, then sync the project
dependencies.

```bash
cd /root/autodl-fs/Omni-Diffusion
git fetch origin
git switch debug/od-compare-logs
git pull
git submodule update --init --recursive

# Create or reuse the local venv.
uv venv --python 3.12
source .venv/bin/activate

# Install CUDA-specific PyTorch wheels separately.
# Example for the current debug server:
uv pip install torch==2.10.0+cu128 torchaudio==2.10.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Install Omni-Diffusion debug dependencies without replacing torch.
uv pip install -r requirements-uv-debug.txt \
  -i https://mirrors.aliyun.com/pypi/simple
uv pip install -e . --no-deps
```

Run the T2I parity script with unbuffered logging:

```bash
PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/autodl-fs/Omni-Diffusion:/root/autodl-fs/Omni-Diffusion/third_party/GLM-4-Voice:/root/autodl-fs/Omni-Diffusion/third_party/GLM-4-Voice/third_party/Matcha-TTS:$PYTHONPATH \
python -u tools/inference.py \
  --model_name_or_path /root/autodl-tmp/models/Omni-Diffusion \
  --output_dir /tmp/omni_official_test \
  --image_tokenizer_path /root/autodl-tmp/models/magvitv2 \
  --audio_tokenizer_path /root/autodl-tmp/models/THUDM/glm-4-voice-tokenizer \
  --flow_path /root/autodl-tmp/models/THUDM/glm-4-voice-decoder \
  2>&1 | tee /tmp/omni_official_inference.log
```

Extract parity logs:

```bash
grep "OD-COMPARE" /tmp/omni_official_inference.log
```
