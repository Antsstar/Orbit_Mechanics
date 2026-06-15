from __future__ import annotations
from typing import List, Optional, Any
from .types import Seconds

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .body import BaseBody
from .propagators import KeplerianPropagator

class Simulation:
    def __init__(self, start_epoch: Optional[datetime] = None) -> None:
        self.start_epoch: datetime = start_epoch if start_epoch else datetime.now()

        self.t: Seconds = 0.0
        self.bodies: List[BaseBody] = []
        self._history_buffer: List[dict[str, Any]] = []

    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)
    
    def add_body(self, body: BaseBody) -> None:
        self.bodies.append(body)

        if body.parent and body.elements is None:
            body.sync_elements()

    def step(self, dt: Seconds) -> None:
        for body in self.bodies:
            KeplerianPropagator.propagate(body, dt)
        
        self.t += dt
        self._record_state()

    def run(self, duration: Seconds, dt: Seconds) -> None:
        if self.t == 0:
            self._record_state()

        steps = int(duration/dt)
        for _ in range(steps):
            self.step(dt)

    def _record_state(self) -> None:
        """Internal helper to snap the current state of all bodies."""
        current_dt = self.start_epoch + timedelta(seconds=self.t)
        for body in self.bodies:
            self._history_buffer.append({
                "timestamp": current_dt,
                "seconds": self.t,
                "body": body.name,
                "x": body.r[0], "y": body.r[1], "z": body.r[2],
                "vx": body.v[0], "vy": body.v[1], "vz": body.v[2],
                "e": body.elements.e if body.elements else None,
                "theta": body.elements.theta if body.elements else None
            })

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history_buffer)
    
    def clear_history(self) -> None:
        self._history_buffer = []