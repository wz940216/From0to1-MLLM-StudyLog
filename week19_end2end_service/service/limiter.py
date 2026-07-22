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

        # 固定窗口限流：只保留最近 60 秒的请求时间戳，超过配额直接返回 429。
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
        # Semaphore 控制同一时间最多有多少请求进入模型推理阶段。
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.queue_timeout_s = queue_timeout_s

    @asynccontextmanager
    async def slot(self):
        try:
            # 请求可以等待一段时间拿推理名额；等不到就返回 503，避免无限堆积。
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=503, detail="request queue timeout") from exc

        try:
            yield
        finally:
            # 无论推理成功还是异常，都必须释放名额，否则后续请求会被永久阻塞。
            self._semaphore.release()

