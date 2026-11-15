"""Soundtrack for the launch video: 120 BPM, A minor (Am - F - C - G), 30 s, every part on one beat grid from t = 0.
Hook: pad and a riser only; the beat lands on the reveal cut (2.5 s) and runs from bar 2 (2.5 s); the outro (27 s) drops the
drums and lets the last chord ring. Effects are in key and sit under the music: whooshes into each cut, soft tonal
blips on the pointer clicks, rising notes as the pipeline stages tick."""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR, BPM, DUR = 48000, 120, 30.0
BEAT = 60 / BPM
N = int(SR * DUR)
rng = np.random.default_rng(7)
t_all = np.arange(N) / SR


def hz(m):  # MIDI note -> Hz
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def env(n, a, d, sustain=0.0, r=None):
    """attack a s, exponential decay d s towards `sustain` (0..1)."""
    tt = np.arange(n) / SR
    e = np.minimum(1, tt / max(a, 1e-4)) * (sustain + (1 - sustain) * np.exp(-np.maximum(0, tt - a) / d))
    if r:
        e *= np.clip((n / SR - tt) / r, 0, 1)
    return e


def place(bus, x, at):
    i = int(round(at * SR))
    if i >= N:
        return
    x = x[: N - i]
    bus[i : i + len(x)] += x


def reverb(x, secs=1.8, mix=0.3):
    n = int(SR * secs)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * secs / 6))
    ir = lp(ir, 6000)
    ir /= np.sqrt((ir ** 2).sum())
    wet = fftconvolve(x, ir)[: len(x)]
    return x * (1 - mix) + wet * mix * 1.4


# chords per bar (2 s): Am, F, C, G   (MIDI roots and triads)
CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]
BARS = int(np.ceil(DUR / (4 * BEAT)))

pad, bass, pluck, drums, fx = (np.zeros(N) for _ in range(5))

# --- pad: soft detuned saws, low-passed, all through (swells in over the hook)
for b in range(BARS):
    root, tri = CHORDS[b % 4]
    n = int(4 * BEAT * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n)
    for m in tri + [tri[0] + 12]:
        for det in (-0.07, 0.07):
            f = hz(m + det)
            x += 2 * ((tt * f) % 1) - 1
    x = lp(x, 1600) * env(n, 0.35, 9, 0.85) * 0.05
    place(pad, x, b * 4 * BEAT)
pad *= np.clip(t_all / 2.5, 0.35, 1)

# --- bass: 8th-note root pulses from the reveal (3.5 s) to the outro (18.5 s)
for i in range(int(DUR / (BEAT / 2))):
    at = i * BEAT / 2
    if not (2.5 <= at < 27.0):
        continue
    root = CHORDS[int(at // (4 * BEAT)) % 4][0] - 12
    n = int(0.24 * SR)
    tt = np.arange(n) / SR
    f = hz(root)
    x = (np.sin(2 * np.pi * f * tt) + 0.35 * (2 * ((tt * f) % 1) - 1)) * env(n, 0.004, 0.12, 0.25, 0.03)
    place(bass, lp(x, 700) * (0.30 if i % 2 == 0 else 0.22), at)

# --- plucks: arpeggio of the chord on 8ths from bar 2, sparse in the hook and outro
for i in range(int(DUR / (BEAT / 2))):
    at = i * BEAT / 2
    bar = int(at // (4 * BEAT))
    tri = CHORDS[bar % 4][1]
    if at < 2.5 and i % 4 != 0:
        continue
    if at >= 27.0 and i % 2 != 0:
        continue
    m = (tri + [tri[0] + 12])[i % 4] + 12
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    f = hz(m)
    x = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)) * env(n, 0.003, 0.16)
    place(pluck, x * 0.07, at)

# --- drums from bar 2 to the outro: kick on 1 and 3 (plus a push), clap on 2 and 4, hats on 8ths
def kick():
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    f = 48 + 90 * np.exp(-tt / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.11) * 0.55


def clap():
    n = int(0.22 * SR)
    x = hp(lp(rng.standard_normal(n), 3500), 900) * env(n, 0.002, 0.06)
    return x * 0.16


def hat(open_=False):
    n = int((0.12 if open_ else 0.05) * SR)
    return hp(rng.standard_normal(n), 7000) * env(n, 0.001, 0.05 if open_ else 0.015) * 0.06


for i in range(int(DUR / (BEAT / 2))):
    at = i * BEAT / 2
    if not (2.5 <= at < 27.0) or 8.5 <= at < 9.0:
        continue
    beat_in_bar = (i // 2) % 4
    if i % 2 == 0 and beat_in_bar in (0, 2):
        place(drums, kick(), at)
    if i % 2 == 0 and beat_in_bar in (1, 3):
        place(drums, clap(), at)
    if i % 8 == 7:                       # push kick on the last 8th of the bar
        place(drums, kick() * 0.6, at)
    place(drums, hat(i % 2 == 1) * (1.0 if i % 2 else 0.6), at)

# --- effects (in key, under the music)
def whoosh(length=0.5):
    n = int(length * SR)
    x = rng.standard_normal(n)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(8):                   # band sweeping up
        seg = slice(k * n // 8, (k + 1) * n // 8)
        out[seg] = lp(hp(x, 300 + 400 * k), 1200 + 900 * k)[seg]
    return out * (tt / length) ** 2 * 0.09


def impact():
    n = int(1.2 * SR)
    tt = np.arange(n) / SR
    x = np.sin(2 * np.pi * hz(33) * tt) * env(n, 0.003, 0.35) * 0.45
    x += lp(rng.standard_normal(n), 900) * env(n, 0.002, 0.25) * 0.08
    return x


def blip(m, gain=0.08, dec=0.09):
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * hz(m) * tt) + 0.25 * np.sin(2 * np.pi * hz(m + 12) * tt)) * env(n, 0.002, dec) * gain


CUTS = [2.5, 5.0, 8.5, 13.0, 15.5, 17.5, 19.5, 22.0, 24.0, 27.0]
for c in CUTS:
    place(fx, whoosh(0.5), c - 0.5)
place(fx, impact(), 2.5)                 # the drop on the reveal
place(fx, impact() * 0.8, 9.0)           # the architecture build
place(fx, impact() * 0.8, 24.0)          # the proof
place(fx, impact() * 0.6, 27.0)          # into the outro

# riser over the hook (pitched noise + a rising tone into the reveal)
n = int(2.5 * SR)
tt = np.arange(n) / SR
f = hz(57) * 2 ** (tt / 2.5)
riser = np.sin(2 * np.pi * np.cumsum(f) / SR) * (tt / 2.5) ** 2 * 0.035
riser += lp(hp(rng.standard_normal(n), 2000), 6000) * (tt / 2.5) ** 3 * 0.03
place(fx, riser, 0)

for at in (4.5, 6.5, 13.0, 15.5, 17.5, 19.5, 20.5, 22.0, 29.0):             # pointer clicks: a short E5 / A5 pair
    place(fx, blip(76, 0.07, 0.05), at)
    place(fx, blip(81, 0.05, 0.08), at + BEAT / 4)

scale = [69, 72, 74, 76, 79, 81, 84, 86]  # A minor pentatonic, rising with the pipeline's stages
for k in range(6):
    place(fx, blip(scale[k], 0.05, 0.07), 9.75 + k * 0.25)

# --- mix: reverb on pad / plucks / effects, a little on drums; gentle bus, soft limiter
mix = reverb(pad, 2.4, 0.45) + reverb(pluck, 1.6, 0.35) + bass + reverb(drums, 0.9, 0.12) + reverb(fx, 1.8, 0.4) * 0.9
mix = hp(mix, 30)
mix *= np.clip((DUR - t_all) / 0.8, 0, 1)            # tail fade over the last 0.8 s
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix *= 0.89 / np.max(np.abs(mix))                    # peak about -1 dBFS
stereo = np.stack([mix, np.roll(mix, int(0.0006 * SR))], 1)   # a touch of width
pcm = (stereo * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", sys.argv[1], f"{DUR}s")
