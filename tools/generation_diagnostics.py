from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _identity_tokens_hook(step: int | None, x: torch.Tensor, logits: torch.Tensor | None):
    del step, logits
    return x


def _identity_logits_hook(step: int, x: torch.Tensor, logits: torch.Tensor):
    del step, x
    return logits


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return tensor.numpy().tobytes()


def _tensor_summary(tensor: torch.Tensor, max_values: int = 16) -> dict[str, Any]:
    detached = tensor.detach()
    flat = detached.flatten()
    head = flat[:max_values].cpu().tolist()
    tail = flat[-max_values:].cpu().tolist() if flat.numel() > max_values else head
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "sha256": hashlib.sha256(_tensor_bytes(detached)).hexdigest(),
        "head": head,
        "tail": tail,
    }
    if detached.numel() and (detached.is_floating_point() or detached.is_complex()):
        stats = detached.float()
        summary.update(
            {
                "min": float(stats.min().item()),
                "max": float(stats.max().item()),
                "mean": float(stats.mean().item()),
                "std": float(stats.std(unbiased=False).item()),
            }
        )
    return summary


class GenerationDiagnostics:
    """Collect Dream generation timings or a compact per-step golden trace.

    Profiling and tracing are intentionally separate modes. Golden tracing
    copies tensors to the CPU and therefore must not be used for performance
    measurements.
    """

    def __init__(
        self,
        *,
        profile_path: str | None = None,
        trace_path: str | None = None,
        mask_token_id: int | None = None,
        trace_topk: int = 8,
    ) -> None:
        if profile_path and trace_path:
            raise ValueError("Profiling and golden tracing must run separately.")
        if trace_topk <= 0:
            raise ValueError("trace_topk must be positive.")

        self.profile_path = Path(profile_path) if profile_path else None
        self.trace_path = Path(trace_path) if trace_path else None
        self.mask_token_id = mask_token_id
        self.trace_topk = trace_topk

        self._active_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._profile_events: list[dict[str, Any]] = []
        self._module_handles: list[Any] = []
        self._restore_callbacks: list[Callable[[], None]] = []
        self._inside_sample = False
        self._global_step = 0
        self._previous_x: torch.Tensor | None = None
        self._last_sample: dict[str, torch.Tensor] | None = None
        self._last_logits_snapshot: dict[str, Any] | None = None
        self._recorded_backbone_input = False
        self._closed = False
        self._request_index = -1
        self._request_metadata: dict[str, Any] = {}

        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.profile_path is not None or self.trace_path is not None

    @property
    def profiling(self) -> bool:
        return self.profile_path is not None

    @property
    def tracing(self) -> bool:
        return self.trace_path is not None

    def install(self, model: Any) -> None:
        if not self.enabled:
            return
        if self.mask_token_id is None:
            self.mask_token_id = getattr(model.generation_config, "mask_token_id", None)
        if self.mask_token_id is None:
            raise ValueError("Dream generation diagnostics require mask_token_id.")

        self._wrap_sample_tokens(model)
        self._wrap_sample_loop(model)
        self._wrap_forward_dream(model)
        self._register_module_stage(model.model.embed_tokens, "token_embedding")
        self._register_module_stage(model.lm_head, "lm_head")

        audio_model = getattr(model.model, "audio_model", None)
        if audio_model is not None:
            self._register_module_stage(audio_model, "audio_encoder")
        audio_projection = getattr(model.model, "audio_projection", None)
        if audio_projection is not None:
            self._register_module_stage(audio_projection, "audio_projection")

    def start_request(self, **metadata: Any) -> None:
        self._request_index += 1
        self._request_metadata = metadata
        self._recorded_backbone_input = False
        if self.tracing:
            self._write_trace(
                {
                    "event": "request_start",
                    "request_index": self._request_index,
                    **metadata,
                }
            )

    def _wrap_sample_loop(self, model: Any) -> None:
        original_sample = model._sample

        def wrapped_sample(*args: Any, **kwargs: Any):
            self._inside_sample = True
            self._global_step = 0
            self._previous_x = None
            try:
                return original_sample(*args, **kwargs)
            finally:
                self._inside_sample = False

        model._sample = wrapped_sample
        self._restore_callbacks.append(lambda: setattr(model, "_sample", original_sample))

    def _wrap_forward_dream(self, model: Any) -> None:
        original_forward = model.forward_dream

        def wrapped_forward(*args: Any, **kwargs: Any):
            if self.tracing and not self._recorded_backbone_input:
                inputs_embeds = kwargs.get("inputs_embeds")
                if isinstance(inputs_embeds, torch.Tensor):
                    self._write_trace(
                        {
                            "event": "first_backbone_input",
                            "inputs_embeds": _tensor_summary(inputs_embeds),
                        }
                    )
                    self._recorded_backbone_input = True
            with self.stage("backbone", global_step=self._global_step):
                return original_forward(*args, **kwargs)

        model.forward_dream = wrapped_forward
        self._restore_callbacks.append(lambda: setattr(model, "forward_dream", original_forward))

    def _wrap_sample_tokens(self, model: Any) -> None:
        sample_globals = getattr(model._sample, "__globals__", None)
        if sample_globals is None:
            sample_globals = getattr(getattr(model._sample, "__func__", None), "__globals__", None)
        if not sample_globals or "sample_tokens" not in sample_globals:
            raise RuntimeError("Could not locate Dream sample_tokens in the remote generation module.")

        original_sample_tokens = sample_globals["sample_tokens"]

        def wrapped_sample_tokens(*args: Any, **kwargs: Any):
            with self.stage("candidate_sampling", global_step=self._global_step):
                confidence, candidate_tokens = original_sample_tokens(*args, **kwargs)
            if self.tracing:
                self._last_sample = {
                    "confidence": confidence.detach().clone(),
                    "candidate_tokens": candidate_tokens.detach().clone(),
                }
            return confidence, candidate_tokens

        sample_globals["sample_tokens"] = wrapped_sample_tokens
        self._restore_callbacks.append(lambda: sample_globals.__setitem__("sample_tokens", original_sample_tokens))

    def _register_module_stage(self, module: torch.nn.Module, name: str) -> None:
        def before(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
            stage_name = name
            if name == "token_embedding" and not self._inside_sample:
                stage_name = "input_embedding"
            self._begin_stage(stage_name, global_step=self._global_step)

        def after(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            _output: Any,
        ) -> None:
            stage_name = name
            if name == "token_embedding" and not self._inside_sample:
                stage_name = "input_embedding"
            self._end_stage(stage_name)

        self._module_handles.append(module.register_forward_pre_hook(before))
        self._module_handles.append(module.register_forward_hook(after))

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        record_cuda: bool = True,
        **metadata: Any,
    ) -> Iterator[None]:
        if not self.profiling:
            yield
            return
        self._begin_stage(name, record_cuda=record_cuda, **metadata)
        try:
            yield
        finally:
            self._end_stage(name)

    def _begin_stage(
        self,
        name: str,
        *,
        record_cuda: bool = True,
        **metadata: Any,
    ) -> None:
        if not self.profiling:
            return
        event: dict[str, Any] = {
            "stage": name,
            "cpu_start_ns": time.perf_counter_ns(),
            "metadata": {
                "request_index": self._request_index,
                **self._request_metadata,
                **metadata,
            },
        }
        if record_cuda and torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            event["cuda_start"] = cuda_start
            event["cuda_end"] = cuda_end
        self._active_events[name].append(event)

    def _end_stage(self, name: str) -> None:
        if not self.profiling:
            return
        if not self._active_events[name]:
            raise RuntimeError(f"Generation profile stage ended without a matching start: {name}")
        event = self._active_events[name].pop()
        if "cuda_end" in event:
            event["cuda_end"].record()
        event["cpu_end_ns"] = time.perf_counter_ns()
        self._profile_events.append(event)

    def wrap_generation_hooks(
        self,
        tokens_hook: Callable[..., torch.Tensor] | None = None,
        logits_hook: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
        base_tokens_hook = tokens_hook or _identity_tokens_hook
        base_logits_hook = logits_hook or _identity_logits_hook

        def wrapped_logits_hook(step: int, x: torch.Tensor, logits: torch.Tensor):
            logits = base_logits_hook(step, x, logits)
            if self.tracing:
                self._last_logits_snapshot = self._summarize_active_logits(x, logits)
            self._begin_stage(
                "sampler_update",
                global_step=self._global_step,
                local_step=step,
            )
            return logits

        def wrapped_tokens_hook(
            step: int | None,
            x: torch.Tensor,
            logits: torch.Tensor | None,
        ):
            if step is not None:
                self._end_stage("sampler_update")
            x = base_tokens_hook(step, x, logits)

            if step is None:
                if self.tracing:
                    self._write_trace(
                        {
                            "event": "initial_canvas",
                            "global_step": None,
                            "local_step": None,
                            "mask_count": int((x == self.mask_token_id).sum().item()),
                            "token_ids": _tensor_summary(x),
                        }
                    )
                    self._previous_x = x.detach().clone()
                return x

            if self.tracing:
                self._trace_step(local_step=step, x=x)
                self._previous_x = x.detach().clone()
                self._last_sample = None
                self._last_logits_snapshot = None
            self._global_step += 1
            return x

        return wrapped_tokens_hook, wrapped_logits_hook

    def _summarize_active_logits(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict[str, Any]:
        active_logits = logits[x == self.mask_token_id]
        if active_logits.shape[0] == 0:
            return {"active_count": 0, "rows": []}

        row_indices = sorted({0, active_logits.shape[0] // 2, active_logits.shape[0] - 1})
        rows = active_logits[row_indices]
        k = min(self.trace_topk, rows.shape[-1])
        top_values, top_ids = torch.topk(rows, k=k, dim=-1)
        return {
            "active_count": active_logits.shape[0],
            "sampled_row_indices": row_indices,
            "top_ids": top_ids.detach().cpu().tolist(),
            "top_values": top_values.float().detach().cpu().tolist(),
        }

    def _trace_step(self, *, local_step: int, x: torch.Tensor) -> None:
        if self._previous_x is None:
            raise RuntimeError("Golden trace did not receive the initial canvas.")

        previous_mask = self._previous_x == self.mask_token_id
        accepted_mask = previous_mask & (x != self.mask_token_id)
        accepted_positions = torch.where(accepted_mask[0])[0]
        record: dict[str, Any] = {
            "event": "denoise_step",
            "request_index": self._request_index,
            **self._request_metadata,
            "global_step": self._global_step,
            "local_step": local_step,
            "mask_count_before": int(previous_mask.sum().item()),
            "mask_count_after": int((x == self.mask_token_id).sum().item()),
            "accepted_count": accepted_positions.numel(),
            "accepted_positions": accepted_positions.detach().cpu().tolist(),
            "accepted_token_ids": x[0, accepted_positions].detach().cpu().tolist(),
            "token_ids_after": _tensor_summary(x),
            "logits_snapshot": self._last_logits_snapshot,
        }
        if self._last_sample is not None:
            record["candidate_token_ids"] = self._last_sample["candidate_tokens"].detach().cpu().tolist()
            record["confidence"] = self._last_sample["confidence"].float().detach().cpu().tolist()
        self._write_trace(record)

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        record = {
            "request_index": self._request_index,
            **self._request_metadata,
            **record,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")

    def close(self, metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True

        if self.profiling:
            if any(self._active_events.values()):
                active = {name: len(events) for name, events in self._active_events.items() if events}
                raise RuntimeError(f"Unclosed generation profile stages: {active}")
            if torch.cuda.is_available():
                torch.accelerator.synchronize()
            self._write_profile(metadata or {})

        for handle in reversed(self._module_handles):
            handle.remove()
        for restore in reversed(self._restore_callbacks):
            restore()

    def _write_profile(self, metadata: dict[str, Any]) -> None:
        if self.profile_path is None:
            return

        serialized_events = []
        grouped: dict[str, list[float]] = defaultdict(list)
        for event in self._profile_events:
            cpu_ms = (event["cpu_end_ns"] - event["cpu_start_ns"]) / 1_000_000
            cuda_ms = None
            if "cuda_start" in event:
                cuda_ms = float(event["cuda_start"].elapsed_time(event["cuda_end"]))
                grouped[event["stage"]].append(cuda_ms)
            else:
                grouped[event["stage"]].append(cpu_ms)
            serialized_events.append(
                {
                    "stage": event["stage"],
                    "cpu_ms": cpu_ms,
                    "cuda_ms": cuda_ms,
                    **event["metadata"],
                }
            )

        summary = {}
        for stage_name, values in sorted(grouped.items()):
            summary[stage_name] = {
                "count": len(values),
                "total_ms": sum(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }

        payload = {
            "metadata": metadata,
            "summary": summary,
            "events": serialized_events,
        }
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
