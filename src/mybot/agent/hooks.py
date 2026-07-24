"""Stage 8.1: Lightweight hook system.

A :class:`HookManager` is a tiny event-emitter that lets callers register
callbacks for named events (e.g. ``"pre_llm_call"``) and invoke them all
when an event fires. Sync and async callbacks are both supported; one
callback raising does not affect the others.

Aligns with the spirit of nananobot's ``nanobot/agent/hook_manager.py``
but in the simplest possible form: a dict of event → list of callbacks,
no priority, no chaining, no cancel semantics. Hooks are observers.

Events currently emitted by the runner (Stage 8.1):

- ``pre_llm_call`` — kwargs: ``messages``, ``iteration``, ``max_iterations``
- ``post_llm_call`` — kwargs: ``response``, ``iteration``, ``max_iterations``

Subsequent stages will add pre/post tool call (8.2) and pre/post run (8.3).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class HookManager:
    """A minimal event hook registry."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[..., Any]]] = {}

    def on(
        self,
        event: str,
        callback: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """Register ``callback`` for ``event``.

        Supports both styles::

            @hooks.on("event")
            def handler(...): ...

            hooks.on("event", handler)

        Returns ``callback`` so it can be used as a decorator directly.
        """
        if callback is None:
            # Used as @hooks.on("event") — return a decorator.
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                self._hooks.setdefault(event, []).append(fn)
                return fn

            return decorator
        self._hooks.setdefault(event, []).append(callback)
        return callback

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Invoke every callback registered for ``event``. Errors are logged
        and isolated so one bad hook cannot break the run."""
        for callback in list(self._hooks.get(event, [])):
            try:
                result = callback(**kwargs)
                if inspect.iscoroutine(result):
                    await result
            except Exception as e:  # noqa: BLE001 — isolation is intentional
                logger.warning(
                    "Hook %r for %r raised: %s", callback, event, e
                )