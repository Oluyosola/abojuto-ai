"""
Vigirix Edge — Loitering Tracker
Centroid-based dwell-time tracker. No external dependencies beyond numpy.
Flags a person as loitering when they remain within `distance_threshold`
pixels of their entry position for longer than `dwell_seconds`.
"""
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Track:
    track_id:    int
    centroid:    Tuple[float, float]
    origin:      Tuple[float, float]   # position when first seen
    first_seen:  float = field(default_factory=time.monotonic)
    last_seen:   float = field(default_factory=time.monotonic)
    loitering:   bool = False
    disappeared: int = 0


class LoiteringTracker:
    """
    Maintains one Track per visible person.
    Call update() every detection cycle; it returns the set of track IDs
    that are currently loitering.
    """

    def __init__(
        self,
        dwell_seconds: float = 30.0,
        distance_threshold: float = 80.0,   # pixels — same-person radius
        max_disappeared: int = 10,          # frames before track is dropped
    ):
        self.dwell_seconds      = dwell_seconds
        self.distance_threshold = distance_threshold
        self.max_disappeared    = max_disappeared
        self._tracks:  Dict[int, Track] = {}
        self._next_id: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, person_boxes: List[List[int]]) -> List[int]:
        """
        person_boxes: list of [x1, y1, x2, y2] for each detected person.
        Returns: list of track IDs currently flagged as loitering.
        """
        centroids = [self._centroid(b) for b in person_boxes]
        now = time.monotonic()

        if not self._tracks:
            for c in centroids:
                self._register(c, now)
        else:
            self._match_and_update(centroids, now)

        self._drop_stale()
        return [t.track_id for t in self._tracks.values() if t.loitering]

    def loitering_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.loitering)

    def active_count(self) -> int:
        return len(self._tracks)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _centroid(box: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _register(self, centroid: Tuple[float, float], now: float):
        self._tracks[self._next_id] = Track(
            track_id=self._next_id,
            centroid=centroid,
            origin=centroid,
            first_seen=now,
            last_seen=now,
        )
        self._next_id += 1

    def _match_and_update(self, centroids: List[Tuple[float, float]], now: float):
        track_ids   = list(self._tracks.keys())
        used_tracks = set()
        used_cents  = set()

        # Greedy nearest-neighbour matching
        pairs = []
        for ci, c in enumerate(centroids):
            best_d, best_t = float("inf"), None
            for tid in track_ids:
                d = self._dist(c, self._tracks[tid].centroid)
                if d < best_d:
                    best_d, best_t = d, tid
            if best_t is not None and best_d < self.distance_threshold:
                pairs.append((ci, best_t, best_d))

        # Apply matches
        for ci, tid, _ in pairs:
            if ci in used_cents or tid in used_tracks:
                continue
            used_cents.add(ci)
            used_tracks.add(tid)
            t = self._tracks[tid]
            t.centroid    = centroids[ci]
            t.last_seen   = now
            t.disappeared = 0
            dwell = now - t.first_seen
            if dwell >= self.dwell_seconds and not t.loitering:
                t.loitering = True

        # Unmatched tracks → increment disappeared
        for tid in track_ids:
            if tid not in used_tracks:
                self._tracks[tid].disappeared += 1

        # Unmatched centroids → new tracks
        for ci, c in enumerate(centroids):
            if ci not in used_cents:
                self._register(c, now)

    def _drop_stale(self):
        stale = [tid for tid, t in self._tracks.items()
                 if t.disappeared > self.max_disappeared]
        for tid in stale:
            del self._tracks[tid]
