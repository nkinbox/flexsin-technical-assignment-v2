"""In-process chat session store.

Deliberately simple: a dict guarded by a lock. Restarting the server clears
history, which is documented as a known limitation -- production would use
Redis or Postgres so sessions survive restarts and scale past one process.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

MAX_SESSIONS = 500
MAX_TURNS_PER_SESSION = 40


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    def new_session_id(self) -> str:
        return str(uuid.uuid4())

    def history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append({"role": role, "content": content})

            # Keep sessions bounded so a long conversation cannot grow without
            # limit; the rewriter only reads the most recent turns anyway.
            if len(turns) > MAX_TURNS_PER_SESSION:
                del turns[: len(turns) - MAX_TURNS_PER_SESSION]

            self._sessions.move_to_end(session_id)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
