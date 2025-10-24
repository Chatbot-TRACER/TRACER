"""Shared token bucket rate limiter for chatbot interactions."""

from __future__ import annotations

from random import SystemRandom
from threading import Lock
from time import monotonic, sleep

from tracer.constants import (
    CHATBOT_MIN_INTER_MESSAGE_GAP_SECONDS,
    CHATBOT_READING_DELAY_PER_CHARACTER_SECONDS,
    CHATBOT_TOKEN_BUCKET_CAPACITY,
    CHATBOT_TOKEN_BUCKET_REFILL_RATE_PER_SECOND,
    CHATBOT_TYPING_DELAY_JITTER_SECONDS,
    CHATBOT_TYPING_DELAY_PER_CHARACTER_SECONDS,
)
from tracer.utils.logging_utils import get_logger

logger = get_logger()


class TokenBucket:
    """Thread-safe token bucket to throttle chatbot traffic."""

    def __init__(self, capacity: float, refill_rate_per_second: float) -> None:
        """Initialize a token bucket with *capacity* and *refill_rate_per_second*."""
        self.capacity = capacity
        self.refill_rate_per_second = refill_rate_per_second
        self._tokens = 0.0  # start empty to avoid initial burst
        self._last_refill_ts = monotonic()
        self._lock = Lock()
        self._rng = SystemRandom()

    def _refill_locked(self) -> None:
        now = monotonic()
        elapsed = now - self._last_refill_ts
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate_per_second)
        self._last_refill_ts = now

    def consume(self, tokens: float = 1.0) -> float:
        """Consume *tokens*, blocking until they are available. Returns total wait duration."""
        if tokens <= 0:
            return 0.0

        total_sleep = 0.0
        while True:
            wait_time = 0.0
            with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    break
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.refill_rate_per_second

            jitter_ceiling = min(wait_time, 0.5)
            jitter = self._rng.random() * jitter_ceiling if jitter_ceiling > 0 else 0.0
            sleep_time = max(wait_time + jitter, 0.05)
            sleep(sleep_time)
            total_sleep += sleep_time

        return total_sleep


GLOBAL_CHATBOT_TOKEN_BUCKET: TokenBucket | None = None
if CHATBOT_TOKEN_BUCKET_CAPACITY > 0 and CHATBOT_TOKEN_BUCKET_REFILL_RATE_PER_SECOND > 0:
    GLOBAL_CHATBOT_TOKEN_BUCKET = TokenBucket(
        capacity=float(CHATBOT_TOKEN_BUCKET_CAPACITY),
        refill_rate_per_second=float(CHATBOT_TOKEN_BUCKET_REFILL_RATE_PER_SECOND),
    )


_human_delay_rng = SystemRandom()


def enforce_chatbot_rate_limit(tokens: float = 1.0) -> float:
    """Block until the chatbot rate limit allows another request. Returns time slept."""
    if GLOBAL_CHATBOT_TOKEN_BUCKET is None:
        return 0.0

    waited = GLOBAL_CHATBOT_TOKEN_BUCKET.consume(tokens)
    if waited > 0:
        logger.debug("Throttled chatbot request by %.2fs to respect rate limits.", waited)
    return waited


def apply_human_like_delay(
    message: str,
    *,
    include_thinking_delay: bool = True,
    previous_received_message: str | None = None,
) -> float:
    """Mimic human typing before sending *message* by enforcing a contextual delay."""
    text = message if isinstance(message, str) else str(message)
    base_gap = max(CHATBOT_MIN_INTER_MESSAGE_GAP_SECONDS, 0.0)
    typing_delay = len(text) * CHATBOT_TYPING_DELAY_PER_CHARACTER_SECONDS if include_thinking_delay else 0.0
    reading_delay = (
        len(previous_received_message) * CHATBOT_READING_DELAY_PER_CHARACTER_SECONDS
        if include_thinking_delay and previous_received_message
        else 0.0
    )
    jitter = _human_delay_rng.uniform(-CHATBOT_TYPING_DELAY_JITTER_SECONDS, CHATBOT_TYPING_DELAY_JITTER_SECONDS)
    wait_time = max(base_gap, base_gap + typing_delay + reading_delay + jitter)
    if wait_time > 0:
        sleep(wait_time)
        logger.debug("Simulated human-like delay of %.2fs before sending chatbot message.", wait_time)
    return wait_time
