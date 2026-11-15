"""Fire-weather for a zone, from Open-Meteo (free, no API key: https://open-meteo.com).

The zone is sampled on a small grid of points (up to 7 x 7, listed row by row from the south-west corner, so the
app can blend them into a smooth heatmap). For each point and each requested day we get the
weather that matters most for fire: the day's highest temperature, lowest relative humidity, strongest wind and
its dominant direction, gusts and rain. The response also has a zone-wide summary per day and a simple
fire-weather rating, so the app can show both the map markers and a short summary in the panel.
"""

import json
import math
import urllib.parse
import urllib.request
from datetime import date

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
DAILY = ["temperature_2m_max", "relative_humidity_2m_min", "wind_speed_10m_max",
         "wind_direction_10m_dominant", "wind_gusts_10m_max", "precipitation_sum"]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def grid_shape(w: float, s: float, e: float, n: float, most: int = 7) -> tuple[int, int]:
    """Columns and rows of the sample grid: more along the zone's longer side, 2 to `most` each way."""
    span_x = (e - w) * max(0.2, math.cos(math.radians((s + n) / 2)))   # east-west extent, in degrees of latitude
    span_y = n - s
    longer = max(span_x, span_y, 1e-9)
    nx = max(2, min(most, round(most * span_x / longer)))
    ny = max(2, min(most, round(most * span_y / longer)))
    return nx, ny


def grid(w: float, s: float, e: float, n: float) -> list[tuple[float, float]]:
    """(lat, lon) at the centres of the grid cells, row by row from the south-west corner."""
    nx, ny = grid_shape(w, s, e, n)
    return [(s + (j + .5) * (n - s) / ny, w + (i + .5) * (e - w) / nx) for j in range(ny) for i in range(nx)]


def mean_direction(degrees: list[float]) -> float | None:
    """Average of compass directions (so 350° and 10° average to 0°, not 180°)."""
    if not degrees:
        return None
    x = sum(math.cos(math.radians(d)) for d in degrees)
    y = sum(math.sin(math.radians(d)) for d in degrees)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None
    return round(math.degrees(math.atan2(y, x)) % 360)


def rating(temp: float, rh: float, wind: float, rain: float) -> str:
    """A rough fire-weather rating: hot, dry and windy days let fire spread; rain damps it."""
    score = 0
    score += 2 if temp >= 32 else 1 if temp >= 25 else 0
    score += 2 if rh <= 20 else 1 if rh <= 35 else 0
    score += 2 if wind >= 30 else 1 if wind >= 18 else 0
    if rain >= 5:
        score -= 3
    elif rain >= 1:
        score -= 1
    return "high" if score >= 4 else "moderate" if score >= 2 else "low"


def fetch_weather(w: float, s: float, e: float, n: float, start: date, end: date) -> dict:
    points = grid(w, s, e, n)
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
        "daily": ",".join(DAILY),
        "timezone": "GMT",
        "wind_speed_unit": "kmh",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    req = urllib.request.Request(f"{OPEN_METEO}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "fire-pattern-predictor"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.load(resp)
    if isinstance(raw, dict):             # a single point comes back as one object, several as a list
        raw = [raw]
    if len(raw) != len(points):
        raise ValueError("Unexpected reply from the weather service.")

    dates = raw[0]["daily"]["time"]
    out_points = []
    for (lat, lon), r in zip(points, raw):
        d = r["daily"]
        days = []
        for k in range(len(dates)):
            days.append({
                "temp": d["temperature_2m_max"][k], "rh": d["relative_humidity_2m_min"][k],
                "wind": d["wind_speed_10m_max"][k], "dir": d["wind_direction_10m_dominant"][k],
                "gust": d["wind_gusts_10m_max"][k], "rain": d["precipitation_sum"][k],
            })
        out_points.append({"lat": round(lat, 4), "lon": round(lon, 4), "days": days})

    summary = []
    for k, day in enumerate(dates):
        vals = [p["days"][k] for p in out_points]
        ok = lambda key: [v[key] for v in vals if v[key] is not None]
        temps, rhs, winds, gusts, rains, dirs = ok("temp"), ok("rh"), ok("wind"), ok("gust"), ok("rain"), ok("dir")
        if not (temps and rhs and winds):
            summary.append({"date": day, "available": False})
            continue
        t, h, wmax = max(temps), min(rhs), max(winds)
        rain = round(sum(rains) / len(rains), 1) if rains else 0.0
        bearing = mean_direction(dirs)
        summary.append({
            "date": day, "available": True,
            "temp_max": round(t, 1), "rh_min": round(h), "wind_max": round(wmax), "gust_max": round(max(gusts)) if gusts else None,
            "wind_from": bearing, "wind_from_compass": None if bearing is None else COMPASS[round(bearing / 45) % 8],
            "rain_mean": rain, "rating": rating(t, h, wmax, rain),
        })
    nx, ny = grid_shape(w, s, e, n)
    return {"dates": dates, "nx": nx, "ny": ny, "bbox": [w, s, e, n], "points": out_points, "summary": summary,
            "source": "Open-Meteo (open-meteo.com), daily forecast"}
