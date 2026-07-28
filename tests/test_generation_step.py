import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "omni_diffusion"
    / "models"
    / "dream"
    / "generation_utils.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "dream_generation_utils_under_test",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
GENERATION_UTILS = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = GENERATION_UTILS
MODULE_SPEC.loader.exec_module(GENERATION_UTILS)
DreamGenerationMixin = GENERATION_UTILS.DreamGenerationMixin
DreamGenerationState = GENERATION_UTILS.DreamGenerationState


class DummyDreamGenerationModel(DreamGenerationMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(embed_tokens=torch.nn.Embedding(10, 4))

    def forward_dream(
        self,
        _input_ids,
        _attention_mask,
        _tok_idx,
        *,
        inputs_embeds,
    ):
        logits = torch.zeros((*inputs_embeds.shape[:2], 10))
        logits[:, 0, 3] = 10
        logits[:, 1, 4] = 5
        return SimpleNamespace(logits=logits)


def test_denoise_step_updates_selected_mask_and_preserves_hook_order():
    model = DummyDreamGenerationModel()
    token_ids = torch.tensor([[1, 9, 9]])
    mask_index = token_ids == 9
    block_mask = torch.tensor([False, True, True])
    state = DreamGenerationState(
        x=token_ids,
        mask_token_id=9,
        histories=[],
        all_logits=[],
    )
    state.start_block(
        block_index=0,
        steps=2,
        timesteps=torch.tensor([1.0, 0.5, 0.001]),
        block_mask=block_mask,
    )
    assert torch.equal(state.start_step(0), mask_index)
    hook_calls = []

    def logits_hook(step, x, logits):
        hook_calls.append(("logits", step, x.clone()))
        return logits

    def tokens_hook(step, x, logits):
        hook_calls.append(("tokens", step, x.clone()))
        return x.clone()

    logits = model._denoise_step(
        state=state,
        input_ids=torch.tensor([[1]]),
        attention_mask=None,
        inputs_embeds=None,
        tok_idx=None,
        un_x=None,
        cfg=0,
        alg="entropy-penalty",
        alg_temp=0,
        temperature=0,
        top_p=1,
        top_k=None,
        max_position_penalty=1,
        repeat_penalty=1,
        generation_tokens_hook_func=tokens_hook,
        generation_logits_hook_func=logits_hook,
    )
    state.record_step(logits)

    assert state.x.tolist() == [[1, 3, 9]]
    assert logits.shape == (1, 3, 10)
    assert state.block_index == 0
    assert state.step == 0
    assert state.global_step == 1
    assert state.histories[0].tolist() == [[1, 3, 9]]
    assert len(state.all_logits) == 1
    assert [call[:2] for call in hook_calls] == [
        ("logits", 0),
        ("tokens", 0),
    ]
    assert hook_calls[0][2].tolist() == [[1, 9, 9]]
    assert hook_calls[1][2].tolist() == [[1, 3, 9]]
