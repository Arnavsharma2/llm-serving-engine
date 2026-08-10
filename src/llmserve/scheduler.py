from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

TokenCallback = Callable[[int], Awaitable[None] | None]


@dataclass
class RequestState:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    callback: TokenCallback | None = None
    priority: int = 0
    arrival_order: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    generated_token_ids: list[int] = field(default_factory=list)
    processed_tokens: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    future: asyncio.Future[list[int]] | None = None

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def next_input_token(self) -> int:
        return self.all_tokens[self.processed_tokens]

    @property
    def remaining_output_tokens(self) -> int:
        return self.max_new_tokens - len(self.generated_token_ids)


class IterationScheduler:
    """Iteration-level admission for static or continuous batching."""

    def __init__(self, max_batch_size: int, policy: str = "fcfs", mode: str = "continuous") -> None:
        if policy not in {"fcfs", "sjf", "priority"}:
            raise ValueError("policy must be fcfs, sjf, or priority")
        if mode not in {"continuous", "static"}:
            raise ValueError("mode must be continuous or static")
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.mode = mode
        self.waiting: list[RequestState] = []
        self.active: list[RequestState] = []
        self._counter = itertools.count()

    def add(self, request: RequestState) -> None:
        request.arrival_order = next(self._counter)
        self.waiting.append(request)

    def _sort_waiting(self) -> None:
        if self.policy == "fcfs":

            def key(item: RequestState):
                return item.arrival_order
        elif self.policy == "sjf":

            def key(item: RequestState):
                return (item.remaining_output_tokens, item.arrival_order)
        else:

            def key(item: RequestState):
                return (-item.priority, item.arrival_order)

        self.waiting.sort(key=key)

    def admit(self) -> list[RequestState]:
        if self.mode == "static" and self.active:
            return []
        self._sort_waiting()
        count = min(self.max_batch_size - len(self.active), len(self.waiting))
        admitted = self.waiting[:count]
        del self.waiting[:count]
        self.active.extend(admitted)
        return admitted

    def finish(self, request: RequestState) -> None:
        self.active.remove(request)

    def preempt(self, request: RequestState) -> None:
        self.active.remove(request)
        request.recomputed_tokens += request.processed_tokens
        request.processed_tokens = 0
        request.preemptions += 1
        self.waiting.append(request)

    @property
    def empty(self) -> bool:
        return not self.waiting and not self.active
