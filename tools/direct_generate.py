from __future__ import annotations

import argparse
import inspect
import random
import re
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

DEFAULT_PROMPT = (
    "Generate an image based on the provided text description.\n"
    "The image shows a landscape background with double exposure glasses of wine, "
    "displaying a hyperealistic and detailed view of the subject."
)

CHAT_TEMPLATE = (
    "\n"
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- messages[0]['content'] }}\n"
    "    {%- endif %}\n"
    '    {{- "\\n\\n# Tools\\n\\n'
    "You may call one or more functions to assist with the user query.\\n\\n"
    'You are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n'
    "    {%- for tool in tools %}\n"
    '        {{- "\\n" }}\n'
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    '    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name '
    'and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n'
    '{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n'
    '</tool_call><|im_end|>\\n" }}\n'
    "{%- else %}\n"
    "    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    '    {%- if (message.role == "user") or (message.role == "system" and not loop.first) '
    'or (message.role == "assistant" and not message.tool_calls) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
    '    {%- elif message.role == "assistant" %}\n'
    "        {{- '<|im_start|>' + message.role }}\n"
    "        {%- if message.content %}\n"
    "            {{- '\\n' + message.content }}\n"
    "        {%- endif %}\n"
    "        {%- for tool_call in message.tool_calls %}\n"
    "            {%- if tool_call.function is defined %}\n"
    "                {%- set tool_call = tool_call.function %}\n"
    "            {%- endif %}\n"
    "            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n"
    "            {{- tool_call.name }}\n"
    "            {{- '\", \"arguments\": ' }}\n"
    "            {{- tool_call.arguments | tojson }}\n"
    "            {{- '}\\n</tool_call>' }}\n"
    "        {%- endfor %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "tool" %}\n'
    '        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n'
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    "        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- message.content }}\n"
    "        {{- '\\n</tool_response>' }}\n"
    '        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n'
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}\n\n"
)


@contextmanager
def allow_legacy_generation_config_validate():
    original_update = GenerationConfig.update

    def legacy_update(self: GenerationConfig, **kwargs: Any) -> dict[str, Any]:
        try:
            return original_update(self, **kwargs)
        except TypeError as exc:
            if "validate() got an unexpected keyword argument 'user_set_attributes'" not in str(exc):
                raise
            self.validate()
            return {key: value for key, value in kwargs.items() if not hasattr(self, key)}

    GenerationConfig.update = legacy_update
    try:
        yield
    finally:
        GenerationConfig.update = original_update


def ensure_default_rope() -> None:
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default(config, device=None, seq_len=None, **_):
        del seq_len
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        partial = rope_parameters.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device,
                    dtype=torch.float,
                )
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def format_debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 8:
            return value.detach().cpu().tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value


def summarize_generation_config(config: Any) -> str:
    fields = (
        "max_length",
        "max_new_tokens",
        "eps",
        "steps",
        "alg",
        "alg_temp",
        "temperature",
        "top_p",
        "top_k",
        "num_return_sequences",
        "return_dict_in_generate",
        "output_history",
        "do_sample",
        "num_beams",
        "use_cache",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "mask_token_id",
        "_bos_token_tensor",
        "_eos_token_tensor",
        "_pad_token_tensor",
        "_mask_token_tensor",
    )
    parts = [f"type={type(config).__name__}"]
    for field in fields:
        parts.append(f"{field}={format_debug_value(getattr(config, field, None))!r}")
    return " ".join(parts)


def install_generation_debug_hooks(model: Any) -> None:
    original_prepare = model._prepare_generation_config
    original_sample = model._sample

    def prepare_wrapper(generation_config=None, **kwargs):
        print(
            "[OD-COMPARE][official-direct] prepare_generation_config_input: "
            f"generation_config={summarize_generation_config(generation_config) if generation_config is not None else None} "
            f"kwargs_keys={sorted(kwargs)} "
            f"kwargs_subset={{{', '.join(f'{key}={kwargs[key]!r}' for key in sorted(kwargs) if key in {'max_new_tokens', 'steps', 'temperature', 'top_p', 'alg_temp', 'output_history', 'return_dict_in_generate'})}}}",
            flush=True,
        )
        try:
            prepared = original_prepare(generation_config, **kwargs)
        except Exception as exc:
            print(
                "[OD-COMPARE][official-direct] prepare_generation_config_error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        print(
            "[OD-COMPARE][official-direct] prepare_generation_config_output: "
            f"{summarize_generation_config(prepared)}",
            flush=True,
        )
        return prepared

    def sample_wrapper(*args, **kwargs):
        generation_config = kwargs.get("generation_config")
        if generation_config is None and len(args) >= 3:
            generation_config = args[2]
        print(
            "[OD-COMPARE][official-direct] sample_input: "
            f"{summarize_generation_config(generation_config)} "
            f"kwargs_subset={{{', '.join(f'{key}={kwargs[key]!r}' for key in sorted(kwargs) if key in {'alg', 'block_size', 'cfg', 'add_boa_token', 'max_position_penalty', 'repeat_penalty'})}}}",
            flush=True,
        )
        return original_sample(*args, **kwargs)

    model._prepare_generation_config = prepare_wrapper
    model._sample = sample_wrapper


def tensor_stats(tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return f"shape={list(tensor.shape)} dtype={tensor.dtype} device={tensor.device} finite=0"
    stats_tensor = finite.float()
    return (
        f"shape={list(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"finite={finite.numel()} min={stats_tensor.min().item():.4f} "
        f"max={stats_tensor.max().item():.4f} mean={stats_tensor.mean().item():.4f} "
        f"std={stats_tensor.std(unbiased=False).item():.4f}"
    )


def make_generation_step_hooks(tokenizer: Any, mask_token_id: int):
    image_offset = tokenizer.convert_tokens_to_ids("<|image_0|>")
    log_steps = {0, 1}

    def summarize_x(step: int | None, x: torch.Tensor, label: str) -> None:
        x_cpu = x[0].detach().cpu()
        image_ids = [
            int(token_id - image_offset)
            for token_id in x_cpu.tolist()
            if image_offset <= token_id < image_offset + 8192
        ]
        print(
            f"[OD-COMPARE][official-direct] {label}: "
            f"step={step} shape={list(x.shape)} mask_count={int((x == mask_token_id).sum().item())} "
            f"nonzero_count={int((x != 0).sum().item())} "
            f"seq_head={x_cpu[:12].tolist()} seq_tail={x_cpu[-12:].tolist()} "
            f"image_count={len(image_ids)} image_head={image_ids[:8]} image_tail={image_ids[-8:]}",
            flush=True,
        )

    def tokens_hook(step, x, logits):
        del logits
        if step is None or step in log_steps:
            summarize_x(step, x, "tokens_hook")
        return x

    def logits_hook(step, x, logits):
        if step not in log_steps:
            return logits
        mask_index = x == mask_token_id
        mask_count = int(mask_index.sum().item())
        print(
            f"[OD-COMPARE][official-direct] logits_hook: step={step} "
            f"mask_count={mask_count} logits_{tensor_stats(logits)}",
            flush=True,
        )
        if mask_count > 0:
            first_mask = torch.where(mask_index[0])[0][0].item()
            first_logits = logits[0, first_mask]
            top_values, top_ids = torch.topk(first_logits.float(), k=12)
            top_ids_list = top_ids.detach().cpu().tolist()
            top_values_list = [round(float(value), 4) for value in top_values.detach().cpu().tolist()]
            top_tokens = tokenizer.convert_ids_to_tokens(top_ids_list)
            image_top = [
                int(token_id - image_offset)
                for token_id in top_ids_list
                if image_offset <= token_id < image_offset + 8192
            ]
            print(
                f"[OD-COMPARE][official-direct] logits_hook_top: step={step} "
                f"first_mask_pos={first_mask} top_ids={top_ids_list} "
                f"top_values={top_values_list} top_tokens={top_tokens} image_top={image_top}",
                flush=True,
            )
        return logits

    return tokens_hook, logits_hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Omni-Diffusion's remote AutoModel.generate directly inside the current Python environment."
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Omni-Diffusion",
        help="Local Omni-Diffusion model directory.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--alg", default="entropy-penalty")
    parser.add_argument("--repeat-penalty", type=float, default=1.2)
    parser.add_argument("--max-position-penalty", type=float, default=2.0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--task", default="T2I")
    parser.add_argument(
        "--pass-generation-config",
        action="store_true",
        help="Pass model.generation_config to generate(). The official inference script does not do this by default.",
    )
    parser.add_argument(
        "--no-debug-hooks",
        action="store_true",
        help="Disable OD-COMPARE hooks around Dream generation config preparation and sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_default_rope()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        chat_template=CHAT_TEMPLATE,
    )

    with allow_legacy_generation_config_validate():
        model = (
            AutoModel.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to("cuda")
            .eval()
        )

    if getattr(model, "generation_config", None) is None:
        with allow_legacy_generation_config_validate():
            model.generation_config = GenerationConfig.from_pretrained(
                args.model_path,
                trust_remote_code=True,
            )

    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
        if getattr(model.generation_config, token_name, None) is not None:
            continue
        token_id = getattr(model.config, token_name, None)
        if token_id is None:
            token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            setattr(model.generation_config, token_name, token_id)

    for name, value in {
        "eps": 1e-3,
        "steps": 512,
        "alg": "origin",
        "alg_temp": None,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "output_history": False,
    }.items():
        setattr(model.generation_config, name, value)

    model.generation_config.max_new_tokens = 8192
    model.generation_config.chat_format = "chatml"
    model.generation_config.max_window_size = 8192
    model.generation_config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = args.top_k
    model.generation_config.top_p = 1.0
    model.generation_config.num_beams = 1
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if not args.no_debug_hooks:
        install_generation_debug_hooks(model)

    prompt_token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_token_ids = normalize_token_ids(prompt_token_ids)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device="cuda")
    prompt_preview = tokenizer.decode(
        prompt_token_ids,
        skip_special_tokens=False,
    )[:500].replace("\n", "\\n")

    print(
        "[OD-COMPARE][official-direct] environment: "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"transformers={transformers.__version__}",
        flush=True,
    )
    print(
        "[OD-COMPARE][official-direct] model_source: "
        f"model_class={type(model).__module__}.{type(model).__name__} "
        f"model_source={inspect.getsourcefile(type(model))}",
        flush=True,
    )
    print(
        "[OD-COMPARE][official-direct] generation_config: "
        f"type={type(model.generation_config).__name__} "
        f"max_new_tokens={model.generation_config.max_new_tokens} "
        f"max_window_size={getattr(model.generation_config, 'max_window_size', None)} "
        f"use_cache={model.generation_config.use_cache} "
        f"do_sample={model.generation_config.do_sample} "
        f"temperature={model.generation_config.temperature} "
        f"top_k={model.generation_config.top_k} "
        f"top_p={model.generation_config.top_p} "
        f"num_beams={model.generation_config.num_beams} "
        f"pad_token_id={model.generation_config.pad_token_id} "
        f"mask_token_id={getattr(model.generation_config, 'mask_token_id', None)}",
        flush=True,
    )
    print(
        "[OD-COMPARE][official-direct] generate_start: "
        f"prompt_tokens={input_ids.shape[1]} task={args.task!r} steps={args.steps} "
        f"max_new_tokens={args.max_new_tokens} alg={args.alg} cfg={args.cfg} "
        f"temperature={args.temperature} top_p={args.top_p} add_boa_token=0 "
        f"max_position_penalty={args.max_position_penalty} repeat_penalty={args.repeat_penalty} "
        f"output_text_only=False seed={args.seed} pass_generation_config={args.pass_generation_config} "
        f"prompt_head={input_ids[0, :12].detach().cpu().tolist()} "
        f"prompt_tail={input_ids[0, -12:].detach().cpu().tolist()} "
        f"prompt_preview={prompt_preview!r}",
        flush=True,
    )

    tokens_hook, logits_hook = make_generation_step_hooks(
        tokenizer,
        mask_token_id=getattr(model.generation_config, "mask_token_id"),
    )

    generate_kwargs = {}
    if args.pass_generation_config:
        generate_kwargs["generation_config"] = model.generation_config

    with torch.no_grad():
        with allow_legacy_generation_config_validate():
            outputs, histories = model.generate(
                input_ids,
                **generate_kwargs,
                audios=None,
                audio_indices=None,
                temperature=args.temperature,
                top_p=args.top_p,
                steps=args.steps,
                max_new_tokens=args.max_new_tokens,
                alg=args.alg,
                cfg=args.cfg,
                tokenizer=tokenizer,
                add_boa_token=0,
                max_position_penalty=args.max_position_penalty,
                repeat_penalty=args.repeat_penalty,
                output_text_only=False,
                task=args.task,
                generation_tokens_hook_func=tokens_hook,
                generation_logits_hook_func=logits_hook,
            )
    del histories

    generated_token_ids = outputs[0][input_ids.shape[1] :]
    generated_token_ids_cpu = generated_token_ids.detach().cpu()
    decoded_output = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    image_code_matches = re.findall(r"<\|image_(\d+)\|>", decoded_output)
    image_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_image|>")
    image_end_id = tokenizer.convert_tokens_to_ids("<|end_of_image|>")

    image_offset = tokenizer.convert_tokens_to_ids("<|image_0|>")
    image_token_ids = [
        int(token_id - image_offset)
        for token_id in generated_token_ids_cpu.tolist()
        if image_offset <= token_id < image_offset + 8192
    ]
    image_min = min(image_token_ids) if image_token_ids else None
    image_max = max(image_token_ids) if image_token_ids else None
    image_unique = len(set(image_token_ids))

    print(
        "[OD-COMPARE][official-direct] generate_raw_output: "
        f"generated={generated_token_ids.numel()} "
        f"begin_count={int((generated_token_ids == image_start_id).sum().item())} "
        f"end_count={int((generated_token_ids == image_end_id).sum().item())} "
        f"regex_image_tokens={len(image_code_matches)} "
        f"output_head={generated_token_ids_cpu[:12].tolist()} "
        f"output_tail={generated_token_ids_cpu[-12:].tolist()} "
        f"output_preview={decoded_output[:500].replace(chr(10), chr(92) + 'n')!r}",
        flush=True,
    )
    print(
        "[OD-COMPARE][official-direct] split_tokens: "
        f"image={len(image_token_ids)} image_min={image_min!r} image_max={image_max!r} "
        f"image_unique={image_unique} image_head={image_token_ids[:8]} image_tail={image_token_ids[-8:]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
