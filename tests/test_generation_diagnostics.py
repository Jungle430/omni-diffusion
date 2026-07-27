import json
from types import SimpleNamespace

import torch
from tools.generation_diagnostics import GenerationDiagnostics


def sample_tokens(logits, **_):
    probabilities = logits.softmax(dim=-1)
    return probabilities.max(dim=-1)


class DummyDreamModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(mask_token_id=9)
        self.model = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(10, 4),
            audio_model=torch.nn.Identity(),
            audio_projection=torch.nn.Identity(),
        )
        self.lm_head = torch.nn.Linear(4, 10, bias=False)

    def forward_dream(
        self,
        _input_ids=None,
        _attention_mask=None,
        _tok_idx=None,
        *,
        inputs_embeds,
    ):
        return SimpleNamespace(logits=self.lm_head(inputs_embeds))

    def _sample(
        self,
        *,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ):
        token_ids = torch.tensor([[1, 2, 9, 9]])
        token_ids = generation_tokens_hook_func(None, token_ids, None)
        for step in range(2):
            embeddings = self.model.embed_tokens(token_ids)
            logits = self.forward_dream(inputs_embeds=embeddings).logits
            logits = generation_logits_hook_func(step, token_ids, logits)
            mask = token_ids == self.generation_config.mask_token_id
            _, candidates = sample_tokens(logits[mask][:, :9])
            first_mask = torch.where(mask[0])[0][0]
            token_ids[0, first_mask] = candidates[0]
            token_ids = generation_tokens_hook_func(step, token_ids, logits)
        return token_ids


def test_generation_profile_collects_nested_stages(tmp_path):
    profile_path = tmp_path / "profile.json"
    diagnostics = GenerationDiagnostics(profile_path=str(profile_path))
    model = DummyDreamModel()
    diagnostics.install(model)
    diagnostics.start_request(task="dummy")
    tokens_hook, logits_hook = diagnostics.wrap_generation_hooks()

    with diagnostics.stage("generate_total"):
        model._sample(
            generation_tokens_hook_func=tokens_hook,
            generation_logits_hook_func=logits_hook,
        )
    diagnostics.close({"kind": "unit"})

    profile = json.loads(profile_path.read_text())
    assert profile["summary"]["backbone"]["count"] == 2
    assert profile["summary"]["candidate_sampling"]["count"] == 2
    assert profile["summary"]["sampler_update"]["count"] == 2


def test_generation_trace_records_canvas_updates(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    diagnostics = GenerationDiagnostics(trace_path=str(trace_path))
    model = DummyDreamModel()
    diagnostics.install(model)
    diagnostics.start_request(task="dummy")
    tokens_hook, logits_hook = diagnostics.wrap_generation_hooks()

    model._sample(
        generation_tokens_hook_func=tokens_hook,
        generation_logits_hook_func=logits_hook,
    )
    diagnostics.close()

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    steps = [record for record in records if record["event"] == "denoise_step"]
    assert len(steps) == 2
    assert [record["accepted_count"] for record in steps] == [1, 1]
    assert steps[-1]["mask_count_after"] == 0
