"""Reconstruct where a satellite actually looked, from its own fire detections.

MODIS (2330 km swath) leaves diamond-shaped gaps between orbits near the
equator; VIIRS (3060 km) does not. To compare the sensors fairly we need to
know whether MODIS observed a location at all, but FIRMS only lists fire
pixels. Every sun-synchronous orbit of one day is fixed by just two numbers:

  * the orbit phase  - when the satellite crosses the equator (mod the period)
  * C = 15 x local solar time of the ascending node (degrees)

Both are estimated from the detections:
  * phase: each detection's latitude gives its position along the orbit, and
    its UTC time then gives the node-crossing time (circular mode over all
    detections).
  * C: each detection's `scan` pixel size gives its distance from the ground
    track (pixels grow with scan angle), putting the track at one of two
    longitudes; the most common candidate across all detections wins.

With the tracks known, any location can be tested for coverage, and the
time each satellite passed over it can be computed.
"""

import math
from collections import Counter

R_EARTH = 6371.0
EARTH_DEG_PER_MIN = 0.25          # sun-synchronous: ground track drifts with the sun
KM_PER_DEG = 111.32

ORBITS = {
    # altitude km, period min, inclination deg, half swath km, max scan angle deg
    "MODIS": dict(alt=705.0, period=98.82, incl=98.2, half_swath=1165.0, max_scan=55.0),
    "VIIRS": dict(alt=828.0, period=101.44, incl=98.7, half_swath=1530.0, max_scan=56.3),
}
MIN_DETECTIONS_TO_FIT = 100
FIT_MAX_LAT = 60.0                # high latitudes: day/night passes are ambiguous


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class SwathModel:
    def __init__(self, instrument: str):
        o = ORBITS[instrument]
        self.instrument = instrument
        self.h, self.P, self.half = o["alt"], o["period"], o["half_swath"]
        self.sin_i = math.sin(math.radians(o["incl"]))
        self.cos_i = math.cos(math.radians(o["incl"]))
        # pixel area (scan x track, km^2) -> cross-track distance lookup. MODIS pixels
        # are 1 x 1 km at nadir and grow with scan angle; VIIRS aggregates pixels, so
        # its area is not a usable proxy and the VIIRS fit uses _fit_c_band instead.
        self.table = []
        k = (R_EARTH + self.h) / R_EARTH
        for step in range(1, int(o["max_scan"] * 20) + 1):
            th = math.radians(step / 20)
            x = k * math.sin(th)
            if x >= 1:
                break
            gamma = math.asin(x) - th                     # Earth central angle
            scan = ((R_EARTH + self.h) * math.cos(th) / math.sqrt(1 - x * x) - R_EARTH) / self.h
            track = R_EARTH * math.sin(gamma) / math.sin(th) / self.h   # slant range / altitude
            self.table.append((scan * track, R_EARTH * gamma))
        self.table.insert(0, (1.0, 0.0))
        self.fits: dict[str, tuple[float, float, int]] = {}   # date -> (phase, C, n)

    # -- orbit geometry ---------------------------------------------------
    def _u(self, lat: float, ascending: bool) -> float | None:
        """Argument of latitude (deg) where the track reaches `lat`."""
        x = math.sin(math.radians(lat)) / self.sin_i
        if abs(x) > 1:
            return None
        u = math.degrees(math.asin(x))
        return u if ascending else 180.0 - u

    def _offset(self, u: float) -> float:
        """Track longitude relative to the node longitude at argument of latitude u."""
        ur = math.radians(u)
        return math.degrees(math.atan2(self.cos_i * math.sin(ur), math.cos(ur))) \
            - EARTH_DEG_PER_MIN * u / 360.0 * self.P

    def _east_factor(self, u: float, lat: float) -> float:
        """East component of the cross-track direction (cos of track heading)."""
        du = 0.01
        dlon = _wrap(self._offset(u + du) - self._offset(u - du)) * math.cos(math.radians(lat))
        dlat = math.degrees(math.asin(self.sin_i * math.sin(math.radians(u + du)))) - \
            math.degrees(math.asin(self.sin_i * math.sin(math.radians(u - du))))
        heading = math.atan2(dlon, dlat)
        return abs(math.cos(heading)) or 1e-6

    def area_to_distance(self, area: float) -> float:
        t = self.table
        if area <= t[0][0]:
            return 0.0
        for (a0, d0), (a1, d1) in zip(t, t[1:]):
            if a0 <= area <= a1:
                return d0 + (d1 - d0) * (area - a0) / (a1 - a0)
        return t[-1][1]

    # -- fitting ----------------------------------------------------------
    def fit(self, detections) -> None:
        """detections: iterable of (date, minutes_utc, lat, lon, daynight, pixel_area_km2)."""
        by_date = {}
        for d in detections:
            by_date.setdefault(d[0], []).append(d)
        for day, dets in by_date.items():
            pts = []
            for _, t, lat, lon, dn, scan in dets:
                if abs(lat) > FIT_MAX_LAT or dn not in ("D", "N"):
                    continue
                u = self._u(lat, dn == "D")
                if u is not None:
                    pts.append((t - u / 360.0 * self.P, u, lat, lon, scan))
            if len(pts) < MIN_DETECTIONS_TO_FIT:
                continue
            # phase: circular mode of node-crossing times (0.25 min bins, 3-bin smoothing)
            nb = int(self.P * 4)
            hist = Counter(int((tn % self.P) * 4) % nb for tn, *_ in pts)
            best = max(range(nb), key=lambda b: hist[b - 1] + hist[b] + hist[(b + 1) % nb])
            window = [b % nb for b in (best - 1, best, best + 1)]
            vals = [(tn % self.P) for tn, *_ in pts if int((tn % self.P) * 4) % nb in window]
            ang = [v / self.P * 2 * math.pi for v in vals]
            phase = (math.atan2(sum(map(math.sin, ang)), sum(map(math.cos, ang))) % (2 * math.pi)) / (2 * math.pi) * self.P
            C = self._fit_c_scan(pts, phase) if self.instrument == "MODIS" else self._fit_c_band(pts, phase)
            self.fits[day] = (phase, C, len(pts))

    def _fit_c_scan(self, pts, phase) -> float:
        """MODIS: pixel area -> distance from track -> two candidate track longitudes each."""
        bins = 360 * 5
        cands = [0] * bins
        for tn, u, lat, lon, area in pts:
            tn_k = phase + round((tn - phase) / self.P) * self.P
            dist = self.area_to_distance(area)
            dlon = dist * self._east_factor(u, lat) / (KM_PER_DEG * math.cos(math.radians(lat)))
            for side in (-1, 1):
                c = (lon + side * dlon + EARTH_DEG_PER_MIN * tn_k - self._offset(u)) % 360
                cands[int(c * 5) % bins] += 1
        best = max(range(bins), key=lambda b: cands[b - 1] + cands[b] + cands[(b + 1) % bins])
        return best / 5 + 0.1

    def _fit_c_band(self, pts, phase) -> float:
        """VIIRS: every detection lies within +-half swath of its track, so the track
        sits at the centre of the longitude band that contains the most detections."""
        bins = 360 * 5
        diff = [0] * (bins + 1)
        for tn, u, lat, lon, scan in pts:
            tn_k = phase + round((tn - phase) / self.P) * self.P
            c = (lon + EARTH_DEG_PER_MIN * tn_k - self._offset(u)) % 360
            w = self.half * self._east_factor(u, lat) / (KM_PER_DEG * math.cos(math.radians(lat)))
            w = min(w, 179.0)
            lo, hi = int((c - w) * 5), int((c + w) * 5)
            for a, b in ((lo, hi),) if lo >= 0 and hi < bins else (((lo % bins), bins - 1), (0, hi % bins)) if lo < 0 or hi >= bins else ():
                diff[a] += 1
                diff[b + 1] -= 1
        counts, run = [], 0
        for b in range(bins):
            run += diff[b]
            counts.append(run)
        top = max(counts)
        best = [b for b in range(bins) if counts[b] >= top * 0.995]
        ang = [b / 5 / 360 * 2 * math.pi for b in best]
        return math.degrees(math.atan2(sum(map(math.sin, ang)), sum(map(math.cos, ang)))) % 360

    # -- queries ----------------------------------------------------------
    def nearest_pass(self, day: str, lat: float, lon: float, daynight: str):
        """Return (cross_track_km, minutes_utc) of the closest pass on `day`, or None if unfitted."""
        fit = self.fits.get(day)
        if fit is None or daynight not in ("D", "N"):
            return None
        phase, C, _ = fit
        u = self._u(lat, daynight == "D")
        if u is None:
            return None
        ef = self._east_factor(u, lat)
        best = None
        k0 = math.floor((-90 - phase) / self.P)
        for k in range(k0, k0 + int(1620 / self.P) + 2):
            tn = phase + k * self.P
            t = tn + u / 360.0 * self.P
            if not -90 <= t <= 1530:          # this UTC day, with a margin either side
                continue
            track_lon = C - EARTH_DEG_PER_MIN * tn + self._offset(u)
            dist = abs(_wrap(lon - track_lon)) * KM_PER_DEG * math.cos(math.radians(lat)) * ef
            if best is None or dist < best[0]:
                best = (dist, t)
        return best

    def last_look_before(self, day: str, lat, lon, before_min, margin_km: float = 50.0):
        """Vectorised: for each point, the latest pass on `day` (minutes UTC from that day's
        00:00, either day or night branch) that had the point inside its swath (with
        `margin_km` to spare) and came before `before_min`. NaN where there is none or
        the day is unfitted. Assumes daytime passes are ascending (Aqua, VIIRS), not Terra."""
        import numpy as np
        lat, lon, before = np.asarray(lat, float), np.asarray(lon, float), np.asarray(before_min, float)
        best = np.full(lat.shape, np.nan)
        fit = self.fits.get(day)
        if fit is None:
            return best
        phase, C, _ = fit
        x = np.sin(np.radians(lat)) / self.sin_i
        ok = np.abs(x) <= 1
        u_asc = np.degrees(np.arcsin(np.clip(x, -1, 1)))

        def offset(u):
            ur = np.radians(u)
            return np.degrees(np.arctan2(self.cos_i * np.sin(ur), np.cos(ur))) - EARTH_DEG_PER_MIN * u / 360.0 * self.P

        for u in (u_asc, 180.0 - u_asc):                       # day (ascending) and night branches
            du = 0.01
            dlon = ((offset(u + du) - offset(u - du) + 180) % 360 - 180) * np.cos(np.radians(lat))
            dlat = np.degrees(np.arcsin(self.sin_i * np.sin(np.radians(u + du)))) - \
                np.degrees(np.arcsin(self.sin_i * np.sin(np.radians(u - du))))
            ef = np.abs(np.cos(np.arctan2(dlon, dlat)))
            ef[ef == 0] = 1e-6
            off = offset(u)
            k0 = math.floor((-90 - phase) / self.P)
            for k in range(k0, k0 + int(1620 / self.P) + 2):
                tn = phase + k * self.P
                t = tn + u / 360.0 * self.P
                track_lon = C - EARTH_DEG_PER_MIN * tn + off
                dist = np.abs((lon - track_lon + 180) % 360 - 180) * KM_PER_DEG * np.cos(np.radians(lat)) * ef
                hit = ok & (dist <= self.half - margin_km) & (t < before) & (t >= -90) & (t <= 1530)
                best = np.where(hit & ~(best >= t), t, best)
        return best

    def next_look_after(self, day: str, lat, lon, after_min, margin_km: float = 50.0):
        """Vectorised mirror of last_look_before: the earliest pass on `day` after `after_min`
        (minutes UTC from that day's 00:00) that had the point inside its swath. NaN if none."""
        import numpy as np
        lat, lon, after = np.asarray(lat, float), np.asarray(lon, float), np.asarray(after_min, float)
        best = np.full(lat.shape, np.nan)
        fit = self.fits.get(day)
        if fit is None:
            return best
        phase, C, _ = fit
        x = np.sin(np.radians(lat)) / self.sin_i
        ok = np.abs(x) <= 1
        u_asc = np.degrees(np.arcsin(np.clip(x, -1, 1)))

        def offset(u):
            ur = np.radians(u)
            return np.degrees(np.arctan2(self.cos_i * np.sin(ur), np.cos(ur))) - EARTH_DEG_PER_MIN * u / 360.0 * self.P

        for u in (u_asc, 180.0 - u_asc):
            du = 0.01
            dlon = ((offset(u + du) - offset(u - du) + 180) % 360 - 180) * np.cos(np.radians(lat))
            dlat = np.degrees(np.arcsin(self.sin_i * np.sin(np.radians(u + du)))) - \
                np.degrees(np.arcsin(self.sin_i * np.sin(np.radians(u - du))))
            ef = np.abs(np.cos(np.arctan2(dlon, dlat)))
            ef[ef == 0] = 1e-6
            off = offset(u)
            k0 = math.floor((-90 - phase) / self.P)
            for k in range(k0, k0 + int(1620 / self.P) + 2):
                tn = phase + k * self.P
                t = tn + u / 360.0 * self.P
                track_lon = C - EARTH_DEG_PER_MIN * tn + off
                dist = np.abs((lon - track_lon + 180) % 360 - 180) * KM_PER_DEG * np.cos(np.radians(lat)) * ef
                hit = ok & (dist <= self.half - margin_km) & (t > after) & (t >= -90) & (t <= 1530)
                best = np.where(hit & ~(best <= t), t, best)
        return best

    def covers(self, day: str, lat: float, lon: float, daynight: str):
        """True / False, or None when the model could not be fitted for that day."""
        p = self.nearest_pass(day, lat, lon, daynight)
        return None if p is None else p[0] <= self.half
