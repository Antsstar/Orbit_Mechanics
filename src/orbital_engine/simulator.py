import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from .body import BaseBody
from .propagators import KeplerianPropagator

class Simulation:
    def __init__(self, start_epoch : datetime = None):
        self.start_epoch = start_epoch if start_epoch else datetime.now()

        self.t = 0.0
        self.bodies : List[BaseBody] = []
        self._history_buffer : List[dict] = []

    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)
    
    def add_body(self, body : BaseBody):
        self.bodies.append(body)

        if body.parent and body.elements == None:
            body.sync_elements()

    def step(self, dt : float):
        for body in self.bodies:
            KeplerianPropagator.propagate(body, dt)
        
        self.t += dt
        self._record_state()

    def run(self, duration: float, dt: float):
        if self.t == 0:
            self._record_state()

        steps = int(duration/dt)
        for _ in range(steps):
            self.step(dt)

    def _record_state(self):
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
    def history(self):
        return pd.DataFrame(self._history_buffer)
    
    def clear_history(self):
        self._history_buffer = []