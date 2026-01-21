from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FrameItem:
    frame: int
    data: Any


class SensorQueue:
    """A simple frame-keyed queue for CARLA sensors."""

    def __init__(self, maxsize: int = 128):
        self.q: queue.Queue[FrameItem] = queue.Queue(maxsize=maxsize)
        self.latest: Optional[FrameItem] = None

    def push(self, frame: int, data: Any) -> None:
        try:
            self.q.put_nowait(FrameItem(frame, data))
        except queue.Full:
            # Drop oldest
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(FrameItem(frame, data))
            except queue.Full:
                pass

    def pop_for_frame(self, frame: int, timeout_s: float = 0.2) -> Optional[Any]:
        """Pop until we reach requested frame. Returns data for that frame if found."""
        end = None
        try:
            item = self.q.get(timeout=timeout_s)
        except queue.Empty:
            return self.latest.data if self.latest else None

        # consume items until >= frame
        while item.frame < frame:
            self.latest = item
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                return self.latest.data if self.latest else None

        if item.frame == frame:
            self.latest = item
            return item.data
        else:
            # item is from future frame; store as latest and keep it (can't put back easily)
            self.latest = item
            return item.data
