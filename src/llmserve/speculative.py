from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpeculativeResult:
    token_ids: list[int]
    proposed_tokens: int
    accepted_tokens: int
    target_passes: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.proposed_tokens, 1)


class SpeculativeDecoder:
    """Greedy draft/target verification with a single target pass per proposal block."""

    def __init__(
        self,
        draft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        *,
        speculation_tokens: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.draft = draft_model.eval()
        self.target = target_model.eval()
        self.speculation_tokens = speculation_tokens
        self.device = torch.device(device)

    @torch.inference_mode()
    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> SpeculativeResult:
        sequence = list(prompt_token_ids)
        generated: list[int] = []
        proposed = accepted = target_passes = 0
        while len(generated) < max_new_tokens:
            proposal: list[int] = []
            draft_sequence = list(sequence)
            budget = min(self.speculation_tokens, max_new_tokens - len(generated))
            for _ in range(budget):
                draft_logits = self.draft(torch.tensor(draft_sequence, device=self.device)[None])
                token = int(draft_logits[0, -1].argmax().item())
                proposal.append(token)
                draft_sequence.append(token)
            proposed += len(proposal)

            verification_input = torch.tensor(sequence + proposal, device=self.device)[None]
            target_logits = self.target(verification_input)[0]
            target_passes += 1
            base = len(sequence) - 1
            all_accepted = True
            for index, draft_token in enumerate(proposal):
                target_token = int(target_logits[base + index].argmax().item())
                if target_token == draft_token:
                    sequence.append(draft_token)
                    generated.append(draft_token)
                    accepted += 1
                else:
                    sequence.append(target_token)
                    generated.append(target_token)
                    all_accepted = False
                    break
                if len(generated) >= max_new_tokens:
                    break
            if all_accepted and len(generated) < max_new_tokens:
                bonus = int(target_logits[-1].argmax().item())
                sequence.append(bonus)
                generated.append(bonus)
        return SpeculativeResult(generated, proposed, accepted, target_passes)
