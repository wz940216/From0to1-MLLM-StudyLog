"""Simple in-process queue and fixed-window rate limiter."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import HTTPException


class FixedWindowRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        if self.requests_per_minute <= 0:
            return

        now = time.monotonic()
        window_start = now - 60.0
        async with self._lock:
            events = self._events[key]
            while events and events[0] < window_start:
                events.popleft()
            if len(events) >= self.requests_per_minute:
                raise HTTPException(status_code=429, detail="rate limit exceeded")
            events.append(now)


class RequestQueue:
    def __init__(self, max_concurrency: int, queue_timeout_s: float):
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.queue_timeout_s = queue_timeout_s

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=503, detail="request queue timeout") from exc

        try:
            yield
        finally:
            self._semaphore.release()

