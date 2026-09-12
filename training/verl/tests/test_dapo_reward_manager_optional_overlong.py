from types import SimpleNamespace

import torch

from verl.workers.reward_manager.dapo import DAPORewardManager


class _Tokenizer:
    eos_token = "<eos>"

    def decode(self, _ids, skip_special_tokens=True):
        return "decoded"


class _Data:
    def __init__(self):
        self.batch = {"responses": torch.zeros((1, 2), dtype=torch.long)}
        self._item = SimpleNamespace(
            batch={
                "prompts": torch.tensor([1, 2]),
                "responses": torch.tensor([3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
            },
            non_tensor_batch={
                "reward_model": {"ground_truth": "answer"},
                "data_source": "if",
                "extra_info": {},
            },
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self._item


def test_dapo_reward_manager_allows_missing_overlong_config():
    manager = DAPORewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        compute_score=lambda **kwargs: 1.0,
        max_resp_len=2,
        overlong_buffer_cfg=None,
    )

    rewards = manager(_Data())

    assert rewards.shape == (1, 2)
    assert rewards[0, 0].item() == 1.0
