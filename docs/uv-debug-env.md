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

## Generation Profiling

The profiling and golden-trace modes must be run separately. Golden tracing
copies compact tensor summaries to the CPU and therefore is not a valid
performance measurement.

Run the official S2I example with CUDA-event stage profiling:

```bash
PYTHONUNBUFFERED=1 \
PYTHONPATH=$PWD:$PWD/third_party/GLM-4-Voice:$PWD/third_party/GLM-4-Voice/third_party/Matcha-TTS:$PYTHONPATH \
python -u tools/inference.py \
  --model_name_or_path /root/autodl-tmp/models/Omni-Diffusion \
  --output_dir /tmp/omni_official_profile \
  --image_tokenizer_path /root/autodl-tmp/models/magvitv2 \
  --audio_tokenizer_path /root/autodl-tmp/models/THUDM/glm-4-voice-tokenizer \
  --flow_path /root/autodl-tmp/models/THUDM/glm-4-voice-decoder \
  --task s2i \
  --profile_json /tmp/omni_official_s2i_profile.json \
  2>&1 | tee /tmp/omni_official_s2i_profile.log
```

Print the aggregate timing table:

```bash
python - <<'PY'
import json

with open("/tmp/omni_official_s2i_profile.json") as source:
    profile = json.load(source)

for stage, values in profile["summary"].items():
    print(
        f"{stage:28s} count={values['count']:4d} "
        f"total={values['total_ms']:10.3f} ms "
        f"mean={values['mean_ms']:9.3f} ms "
        f"min={values['min_ms']:9.3f} ms "
        f"max={values['max_ms']:9.3f} ms"
    )
PY
```

## Golden Trace

Run the same request again with per-step state tracing:

```bash
PYTHONUNBUFFERED=1 \
PYTHONPATH=$PWD:$PWD/third_party/GLM-4-Voice:$PWD/third_party/GLM-4-Voice/third_party/Matcha-TTS:$PYTHONPATH \
python -u tools/inference.py \
  --model_name_or_path /root/autodl-tmp/models/Omni-Diffusion \
  --output_dir /tmp/omni_official_trace \
  --image_tokenizer_path /root/autodl-tmp/models/magvitv2 \
  --audio_tokenizer_path /root/autodl-tmp/models/THUDM/glm-4-voice-tokenizer \
  --flow_path /root/autodl-tmp/models/THUDM/glm-4-voice-decoder \
  --task s2i \
  --trace_jsonl /tmp/omni_official_s2i_trace.jsonl \
  2>&1 | tee /tmp/omni_official_s2i_trace.log
```

Inspect the trace without printing the full JSONL payload:

```bash
python - <<'PY'
import json

path = "/tmp/omni_official_s2i_trace.jsonl"
with open(path) as source:
    records = [json.loads(line) for line in source]

steps = [record for record in records if record["event"] == "denoise_step"]
print("records:", len(records))
print("denoise steps:", len(steps))
for record in steps[:3] + steps[-3:]:
    print(
        "global_step={global_step:3d} local_step={local_step:3d} "
        "mask={mask_count_before:3d}->{mask_count_after:3d} "
        "accepted={accepted_count:3d}".format(**record)
    )
PY
```

Run the CPU smoke tests for the diagnostics hooks:

```bash
python -m pytest tests/test_generation_diagnostics.py -q
```
