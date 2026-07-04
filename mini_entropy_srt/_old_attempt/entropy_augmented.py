"""
Entropy-augmented reward manager.

Implements:
    R_total = R_consistency + alpha * H_batch

H_batch is the normalized Shannon entropy of the extracted-answer distribution
across all model rollouts for the same prompt. When the model reward-hacks by
collapsing to one answer, H_batch -> 0 and the entropy bonus disappears.
When responses are diverse, H_batch -> 1.0 and the bonus is alpha.
"""
import torch
from collections import defaultdict
from verl import DataProto
from verl.utils.reward_score import _extract_verifiable_part_of_solution
from verl.utils.reward_score.entropy_reward import batch_shannon_entropy
from verl.workers.reward_manager.self_learning import SelfLearningRewardManager


class EntropyAugmentedRewardManager(SelfLearningRewardManager):
    """
    Adds a per-prompt Shannon entropy bonus on top of the SRT consistency reward.

        R_total = R_consistency + alpha * H_batch

    alpha = 0.0  ->  identical to the original SRT baseline
    alpha = 0.1  ->  recommended starting point
    alpha = 0.5  ->  strong diversity pressure (may prevent convergence)
    """

    def __init__(self, *args, alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        print(f"[EntropyAugmentedRewardManager] alpha = {self.alpha}")

    def _compute_prompt_entropy_map(self, data: DataProto) -> dict:
        """
        For every prompt in the batch, collect all extracted answers across
        rollouts and compute H_batch (normalized Shannon entropy).

        Uses the same get_prompt_and_response_and_ground_truth helper as the
        parent class so decoding logic stays in one place.
        """
        prompt_to_answers = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_str, response_str, _, _ = self.get_prompt_and_response_and_ground_truth(
                data_item=data_item,
            )

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            try:
                extracted = _extract_verifiable_part_of_solution(
                    data_source=data_source,
                    solution_str=response_str,
                )
            except NotImplementedError:
                extracted = None

            prompt_to_answers[prompt_str].append(extracted)

        return {
            prompt: batch_shannon_entropy(answers, normalize=True)
            for prompt, answers in prompt_to_answers.items()
        }

    def __call__(self, data: DataProto, return_dict=False, log_threshold_plot=False):
        # Run the original SRT majority-voting reward computation.
        result = super().__call__(data, return_dict=True, log_threshold_plot=log_threshold_plot)

        reward_tensor = result["reward_tensor"]

        if self.alpha > 0.0:
            prompt_entropy_map = self._compute_prompt_entropy_map(data)

            for i in range(len(data)):
                data_item = data[i]

                # Use parent helper — same pattern as the parent __call__ loop.
                prompt_str, _, _, _ = self.get_prompt_and_response_and_ground_truth(
                    data_item=data_item,
                )

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                last_token_idx = valid_response_length - 1

                h_batch = prompt_entropy_map.get(prompt_str, 0.0)
                reward_tensor[i, last_token_idx] += self.alpha * h_batch

        result["reward_tensor"] = reward_tensor

        if return_dict:
            return result
        return reward_tensor
