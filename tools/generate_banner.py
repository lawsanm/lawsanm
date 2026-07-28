"""
Animated GitHub profile banner generator  —  assets/dark.svg + assets/light.svg

This script (plus tools/data/*.npy) is the source of truth. The SVGs are build
output; regenerate rather than hand-editing them.

    python tools/generate_banner.py            # full build
    python tools/generate_banner.py --preview   # also dump PNG previews

Pipeline
    1. segment the subject out of the source photo   (grabcut -> polynomial
       background residual -> texture gate)
    2. tone + 1-bit Floyd-Steinberg dither, serpentine, on a 300x340 grid
    3. run-length encode dots into <path> stroke runs, crispEdges
    4. two independent animated layers: a dense portrait that dissolves, and a
       sparse traveller swarm that morphs between three glyphs
"""

import argparse
import functools
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tools", "data")
ASSETS = os.path.join(ROOT, "assets")
PHOTO = os.path.join(ROOT, "Law.png")

# ── portrait ──────────────────────────────────────────────────────────────────
GW, GH = 300, 340                 # dither grid
CROP = (121, 0, 1024, 1024)       # head + shoulders, aspect 0.882
ERODE = 3                         # mask erosion, cells
FEATHER = 1.6                     # mask edge falloff, cells
GAMMA = 1.15                      # tone shaping (mild — keep gradation)
PEAK = 0.75                       # max dot density; below 1.0 so highlights
                                  # never become a solid field
DOTS_DARK = 18000
DOTS_LIGHT = 17000

# ── animation ─────────────────────────────────────────────────────────────────
INTRO = 3.2
INTRO_FADE = 2.0
INTRO_GROUPS = 60
LOOP = 14.2                       # 3.0 portrait + 3x(1.3 transition + 2.0 hold) + 1.3
T_PORTRAIT, T_TRANS, T_HOLD = 3.0, 1.3, 2.0
BANDS = 94                        # portrait drift bands
DRIFT = 0.42                      # fraction of the way to logo 1's centroid
BAND_NOISE = 6.0                  # per-dot jitter (cells) before grouping —
                                  # without this, quantizing a linear drift
                                  # field mathematically recreates a grid
WARP_SCALE = 18.0                 # low-frequency warp: correlation length
WARP_AMP = 34.0                   # low-frequency warp: amplitude, cells
TRAVELLERS = 900
TRAV_SIZE = 2.0                   # in grid cells — deliberately thicker

# ── layout (1180x610) ─────────────────────────────────────────────────────────
W, H = 1180, 610
WX, WY, WW, WH = 16, 16, 1148, 578
BAR = 38
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 34, 72, 424, 486
PX0, PY0 = 46.0, 96.0
CELL = 400.0 / GW                 # 1.3333 px per grid cell
RX, RY, RW = 482, 72, 664         # right readout column
ROW0, ROWSTEP, GROUPGAP = 128, 23.0, 12.0
FS = 14                           # row font-size

# The typeface is subset and embedded as a data URI, so the readout looks the
# same everywhere instead of resolving to Consolas / Menlo / DejaVu per OS.
FACE = "LawMono"
FONT_SRC = r"C:/Windows/Fonts/CascadiaMono.ttf"     # SIL OFL 1.1
FONT = "'%s','Cascadia Mono','DejaVu Sans Mono',Menlo,Consolas,monospace" % FACE
MONO_ADV = 1200 / 2048.0          # Cascadia Mono advance / upem — exact, so
                                  # textLength never has to stretch glyphs

THEMES = {
    # Matrix: black + electric green. The ramp is a single hue at four
    # brightnesses, so `dim` has to stay green-grey rather than slate — a blue
    # grey next to #00FF41 reads as a colour mistake, not as muted text.
    "dark": dict(
        bg="#000000", panel="#020A02", border="#003B00", grid="#001A00",
        chrome="#00FF41", portrait="#39FF14", accent="#00FF41",
        text="#C8FFD4", dim="#4F8A5C", leader="#003B00", pill_fg="#001A00",
    ),
    "light": dict(
        bg="#FFFFFF", panel="#F4FBF5", border="#BFE6C6", grid="#E4F5E8",
        chrome="#046A20", portrait="#0A7D26", accent="#046A20",
        text="#02220A", dim="#4A6B51", leader="#CDEBD4", pill_fg="#FFFFFF",
    ),
}

ROWS = [
    ("Subject", "Lawsan"),
    ("Role", "Cybersecurity  ·  Machine Learning"),
    ("Origin", "Sri Lanka"),
    ("Education", "BSc Computer Science  ·  UCSC"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code  ·  Git  ·  Docker  ·  Kali  ·  Burp"),
    None,
    ("Core.Lang", "C  ·  C++  ·  Java  ·  Python  ·  Go  ·  TypeScript"),
    ("Core.Frontend", "React  ·  Next.js  ·  Tailwind"),
    ("Core.Backend", "Node  ·  Express  ·  FastAPI"),
    ("Core.Database", "MongoDB  ·  MySQL  ·  SQLAlchemy"),
    ("Core.Infra", "Docker  ·  Vercel  ·  Vite"),
    None,
    ("Grid.Mail", "lawsanm@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "/in/lawsan"),
    ("Grid.GitHub", "@lawsanm"),
    ("Grid.Instagram", "@m.lawsan"),
]
HANDLE = "@lawsanm"
TITLE = "profile.sh --live"


def log(*a):
    print(*a, flush=True)


# ══════════════════════════════════════════════════════════════════ 1. segment

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def segment(rgb):
    """Subject mask. Three stages, because each one alone fails on this photo:
    grabcut finds the figure but keeps the backdrop's shadow halo; a polynomial
    background fit removes most of it; a texture gate kills what's left, since
    the halo is smooth and the subject is not."""
    import cv2

    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask,
                (60, 10, w - 100, h - 10), bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    gc = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    gc = ndi.binary_fill_holes(ndi.binary_closing(gc, np.ones((9, 9)), border_value=0))
    gc = keep_largest(gc)

    a = rgb.astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx / w - .5, yy / h - .5
    A = np.stack([(X ** i * Y ** j).ravel()
                  for i in range(5) for j in range(5) if i + j <= 4], 1)
    conf_bg = ~ndi.binary_dilation(gc, np.ones((41, 41)))
    idx = np.nonzero(conf_bg.ravel())[0]
    fit = np.empty_like(a)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[idx], a[..., c].ravel()[idx], rcond=None)
        fit[..., c] = (A @ coef).reshape(h, w)
    resid = np.sqrt(((a - fit) ** 2).sum(-1)) > 24

    E = ndi.gaussian_gradient_magnitude(a @ [.299, .587, .114], 2.0)
    near_texture = ndi.binary_dilation(E > np.percentile(E, 85), disk(14))

    m = resid & near_texture
    m = ndi.binary_fill_holes(ndi.binary_closing(m, disk(12), border_value=0))
    return ndi.binary_fill_holes(keep_largest(m))


def keep_largest(m):
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    return lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)


# ══════════════════════════════════════════════════════════════════ 2. dither

def floyd_steinberg(v, keep):
    """1-bit Floyd-Steinberg, serpentine scan. Error is hard-cleared outside
    `keep` — otherwise the diffusion piles up against the mask and draws a
    solid bright outline around the whole silhouette."""
    v = v.astype(np.float64) * 255.0
    out = np.zeros(v.shape, np.uint8)
    hh, ww = v.shape
    for y in range(hh):
        d = 1 if y % 2 == 0 else -1
        for x in (range(ww) if d == 1 else range(ww - 1, -1, -1)):
            if not keep[y, x]:
                v[y, x] = 0.0
                continue
            old = v[y, x]
            new = 255.0 if old > 127.5 else 0.0
            out[y, x] = 1 if new else 0
            e = old - new
            if 0 <= x + d < ww:
                v[y, x + d] += e * 7 / 16
            if y + 1 < hh:
                if 0 <= x - d < ww:
                    v[y + 1, x - d] += e * 3 / 16
                v[y + 1, x] += e * 5 / 16
                if 0 <= x + d < ww:
                    v[y + 1, x + d] += e * 1 / 16
    return out


def build_portrait():
    rgb = np.asarray(Image.open(PHOTO).convert("RGB"))
    cache = os.path.join(DATA, "mask.npy")
    if os.path.exists(cache):
        full = np.load(cache)
    else:
        log("segmenting (~20s) …")
        full = segment(rgb)
        np.save(cache, full)
    log("  mask coverage %.1f%% of frame" % (100 * full.mean()))

    src = Image.open(PHOTO).convert("RGB").crop(CROP)
    m = np.asarray(Image.fromarray((full * 255).astype(np.uint8)).crop(CROP)
                   .resize((GW, GH), Image.LANCZOS)) > 140
    m = ndi.binary_erosion(m, np.ones((ERODE, ERODE)))
    soft = np.clip((ndi.gaussian_filter(m.astype(np.float64), FEATHER) - .25) / .55, 0, 1)

    g = src.convert("L").resize((GW, GH), Image.LANCZOS)
    g = ImageOps.autocontrast(g, cutoff=0)          # cutoff=1 clips 19% of the
    g = ImageEnhance.Contrast(g).enhance(1.3)       # subject to pure white on
    g = g.filter(ImageFilter.UnsharpMask(3, 140))   # this already-graded photo
    g = np.clip(np.asarray(g, np.float64) / 255.0, 0, 1)

    out = {}
    for theme, base, target in (("dark", g, DOTS_DARK), ("light", 1 - g, DOTS_LIGHT)):
        b = np.clip(base, 0, 1) ** GAMMA
        s = min(PEAK, (target / m.sum()) / b[m].mean())
        d = floyd_steinberg(b * s * soft, m)
        log("  %-5s scale %.3f  %d dots  peak density %.2f" % (theme, s, d.sum(), s))
        out[theme] = d
    return out, m


# ═══════════════════════════════════════════════════════ 3. glyph point clouds

def glyph_clouds(n=TRAVELLERS):
    """Three marks the travellers morph between, drawn in grid space.

    These are constructed geometry / font outlines, NOT traced brand logos —
    the prompt's tracing step needs reference images, which we don't have.
    Swap in traced clouds here if you want real product marks."""
    cx, cy = GW / 2, GH / 2
    clouds = []

    def sample(img):
        pts = np.argwhere(np.asarray(img) > 127)[:, ::-1].astype(np.float64)
        if len(pts) > 40000:
            pts = pts[np.random.choice(len(pts), 40000, replace=False)]
        c, _ = kmeans2(pts, n, minit="++", iter=25, seed=7)
        return c

    # 1 — </>  code glyph, from a real font outline
    im = Image.new("L", (GW, GH), 0)
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 150)
    dr.text((cx, cy), "</>", font=f, fill=255, anchor="mm")
    clouds.append(sample(im))

    # 2 — shield (security): straight top, quadratic sides meeting at a point
    im = Image.new("L", (GW, GH), 0)
    dr = ImageDraw.Draw(im)
    hw, y0, y1 = 74.0, cy - 92, cy + 96
    u = np.linspace(0, 1, 160)[:, None]
    p0, p1, p2 = np.array([cx + hw, y0]), np.array([cx + hw, y0 + 0.72 * (y1 - y0)]), np.array([cx, y1])
    right = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2
    left = right * [-1, 1] + [2 * cx, 0]
    outline = np.vstack([[[cx - hw, y0]], right[::-1][::-1], left[::-1], [[cx - hw, y0]]])
    dr.line([tuple(p) for p in outline], fill=255, width=11, joint="curve")
    dr.ellipse([cx - 17, cy - 34, cx + 17, cy], fill=255)          # keyhole
    dr.ellipse([cx - 7, cy - 24, cx + 7, cy - 10], fill=0)
    dr.polygon([(cx - 8, cy - 8), (cx + 8, cy - 8), (cx + 5, cy + 36), (cx - 5, cy + 36)], fill=255)
    clouds.append(sample(im))

    # 3 — neural graph (machine learning)
    im = Image.new("L", (GW, GH), 0)
    dr = ImageDraw.Draw(im)
    cols = [(cx - 78, 3), (cx, 4), (cx + 78, 2)]
    nodes = [[(x, cy + (i - (k - 1) / 2) * 52) for i in range(k)] for x, k in cols]
    for a_, b_ in zip(nodes, nodes[1:]):
        for p in a_:
            for q in b_:
                dr.line([p, q], fill=255, width=4)
    for layer in nodes:
        for x, y in layer:
            dr.ellipse([x - 17, y - 17, x + 17, y + 17], fill=255)
    clouds.append(sample(im))
    return clouds


def chain_match(clouds, start):
    """Optimal transport between consecutive clouds so every traveller takes the
    shortest path instead of crossing the frame."""
    seq = [start]
    for c in clouds:
        cost = ((seq[-1][:, None, :] - c[None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        seq.append(c[col])
    return seq


# ═════════════════════════════════════════════════════════════ 4. drift bands

def drift_bands(pts, target, jitter=True):
    """Group dots into ~BANDS bands that each translate toward `target`.

    Drift is a linear function of position, so binning it directly reproduces a
    square grid and the dissolve looks blocky. Two perturbations fix it: a
    low-frequency warp field bends the band boundaries into waves, and per-dot
    noise frays them. Pass jitter=False to reproduce the grid trap."""
    rng = np.random.default_rng(11)
    noisy = pts.copy()
    if jitter:
        field = np.stack([ndi.gaussian_filter(rng.normal(size=(GH, GW)), WARP_SCALE)
                          for _ in range(2)], -1)
        field *= WARP_AMP / field.std()
        noisy = noisy + field[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        noisy = noisy + rng.normal(0, BAND_NOISE, pts.shape)
    if jitter:
        # Voronoi cells over the jittered field: boundaries run at every angle
        # instead of along the two axes the drift function is separable in.
        _, band = kmeans2(noisy, BANDS, minit="++", iter=30, seed=5, missing="warn")
    else:
        d0 = DRIFT * (target - noisy)
        k = int(round(np.sqrt(BANDS)))
        band = np.zeros(len(pts), np.int64)
        for ax in (0, 1):
            q = np.quantile(d0[:, ax], np.linspace(0, 1, k + 1)[1:-1])
            band = band * k + np.searchsorted(q, d0[:, ax])
    _, band = np.unique(band, return_inverse=True)
    nb = band.max() + 1
    cnt = np.bincount(band, None, nb)
    centre = np.stack([np.bincount(band, pts[:, a], nb) / cnt for a in (0, 1)], 1)
    return band, DRIFT * (target - centre), nb


def grid_affinity(band, band_grid):
    """Adjusted Rand index between the real bands and the pure axis-aligned
    quantization of the same drift field. 1.0 = the bands ARE the grid and the
    dissolve will look blocky; ~0 = the two partitions are unrelated."""
    n = len(band)
    k = np.bincount(band.astype(np.int64) * (band_grid.max() + 1) + band_grid)
    comb = lambda v: (v * (v - 1) / 2).sum()
    idx = comb(k.astype(np.float64))
    ea = comb(np.bincount(band).astype(np.float64))
    eb = comb(np.bincount(band_grid).astype(np.float64))
    exp = ea * eb / (n * (n - 1) / 2)
    return float((idx - exp) / (0.5 * (ea + eb) - exp))


def evenness(pts, group, ngroups, cells=6):
    """Mean total-variation distance between each intro group's spatial
    distribution and the portrait's. Small = every group is scattered over the
    whole face; large = groups are spatial patches and the intro reads as a
    wipe. Compare against the patchy baseline, not against zero: with ~200 dots
    per group there is an irreducible multinomial sampling floor."""
    gx = np.clip((pts[:, 0] / GW * cells).astype(int), 0, cells - 1)
    gy = np.clip((pts[:, 1] / GH * cells).astype(int), 0, cells - 1)
    cell = gy * cells + gx
    glob = np.bincount(cell, None, cells * cells).astype(float)
    glob /= glob.sum()
    tv = []
    for g in range(ngroups):
        sel = group == g
        h = np.bincount(cell[sel], None, cells * cells).astype(float)
        if h.sum():
            tv.append(0.5 * np.abs(h / h.sum() - glob).sum())
    return float(np.mean(tv))


# ═══════════════════════════════════════════════════════════ 4b. embedded font

@functools.lru_cache(maxsize=1)
def font_face():
    """Subset Cascadia Mono to the glyphs this banner actually uses and inline
    it as a data URI. GitHub renders the README SVG through <img>, which blocks
    every external reference — a data URI is part of the document, so it is the
    only way to control the typography. Two static instances are cut from the
    variable font; @font-face weight ranges are less reliably honoured in
    SVG-as-image."""
    import base64
    import io

    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    chars = {" "}
    for row in ROWS:
        if row:
            chars |= set(row[0]) | set(row[1])
    for s in (TITLE, HANDLE, "VISUAL.MAP", "SYSTEM.INFO", "LIVE"):
        chars |= set(s)

    css = []
    for weight in (400, 700):
        f = instancer.instantiateVariableFont(TTFont(FONT_SRC), {"wght": weight})
        opt = subset.Options(layout_features=[], notdef_outline=True,
                             desubroutinize=True, drop_tables=["DSIG"])
        s = subset.Subsetter(options=opt)
        s.populate(text="".join(sorted(chars)))
        s.subset(f)
        buf = io.BytesIO()
        f.save(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        log("    font w%d: %d glyphs, %.1f KB embedded" % (weight, len(chars), len(b64) / 1024))
        css.append("@font-face{font-family:'%s';font-style:normal;font-weight:%d;"
                   "src:url(data:font/ttf;base64,%s) format('truetype');}"
                   % (FACE, weight, b64))
    return "".join(css)


# ═════════════════════════════════════════════════════════════════ 5. SVG emit

def runs_path(pts_by_group, ngroups):
    """Horizontal run-length encode each group into one stroked <path>.
    Stroke runs are ~40% smaller than per-dot rects and stay crisp."""
    out = []
    for g in range(ngroups):
        p = pts_by_group[g]
        if not len(p):
            out.append("")
            continue
        order = np.lexsort((p[:, 0], p[:, 1]))
        p = p[order]
        d, i = [], 0
        while i < len(p):
            j = i
            while (j + 1 < len(p) and p[j + 1, 1] == p[i, 1]
                   and p[j + 1, 0] == p[j, 0] + 1):
                j += 1
            x = PX0 + p[i, 0] * CELL
            y = PY0 + (p[i, 1] + 0.5) * CELL
            d.append("M%s %sh%s" % (r1(x), r1(y), r1((j - i + 1) * CELL)))
            i = j + 1
        out.append("".join(d))
    return out


def r1(v):
    s = "%.1f" % v
    return s[:-2] if s.endswith(".0") else s


def kt(*times):
    return ";".join("%.4f" % (t / LOOP) for t in times)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, fill, size=FS, weight=400, anchor="start", extra=""):
    """Every row is locked with textLength + spacingAndGlyphs so the readout
    stays aligned whatever font the browser actually resolves."""
    tl = len(s) * size * MONO_ADV
    return ('<text x="%s" y="%s" font-size="%d" font-weight="%d" fill="%s" '
            'text-anchor="%s" textLength="%s" lengthAdjust="spacingAndGlyphs"%s>%s</text>'
            % (r1(x), r1(y), size, weight, fill, anchor, r1(tl), extra, esc(s)))


def build_svg(theme, dots, mask, clouds):
    C = THEMES[theme]
    pts = np.argwhere(dots == 1)[:, ::-1].astype(np.float64)   # (x, y) in cells
    rng = np.random.default_rng(3)

    # ── portrait layers ──────────────────────────────────────────────────────
    intro_group = rng.permutation(len(pts)) % INTRO_GROUPS
    ev = evenness(pts, intro_group, INTRO_GROUPS)
    patchy = evenness(pts, np.argsort(np.argsort(pts[:, 1] * GW + pts[:, 0]))
                      * INTRO_GROUPS // len(pts), INTRO_GROUPS)

    band, band_d, nb = drift_bands(pts, clouds[0].mean(0))
    grid = drift_bands(pts, clouds[0].mean(0), jitter=False)[0]
    ga = grid_affinity(band, grid)
    log("  %-5s intro evenness   %.3f  (spatial-patch baseline %.3f)" % (theme, ev, patchy))
    log("  %-5s band/grid affinity %.3f  (1.000 = the grid trap)  %d bands"
        % (theme, ga, nb))
    assert ev < 0.35 * patchy, "intro groups are not interleaved"
    assert ga < 0.30, "drift bands collapsed into a grid"

    ip = np.rint(pts).astype(int)
    intro_paths = runs_path([ip[intro_group == g] for g in range(INTRO_GROUPS)], INTRO_GROUPS)
    band_paths = runs_path([ip[band == b] for b in range(nb)], nb)

    # ── travellers ───────────────────────────────────────────────────────────
    start = pts[rng.choice(len(pts), TRAVELLERS, replace=False)]
    seq = chain_match(clouds, start)                    # start, L1, L2, L3
    hop = float(np.mean([np.linalg.norm(seq[i + 1] - seq[i], axis=1).mean()
                         for i in range(3)]))
    log("  %-5s traveller mean hop %.1f cells" % (theme, hop))

    t = [0.0, T_PORTRAIT]
    for _ in range(3):
        t += [t[-1] + T_TRANS, t[-1] + T_TRANS + T_HOLD]
    t.append(LOOP)                                       # 9 keyframes
    KT = kt(*t)
    frames = [seq[0], seq[0], seq[1], seq[1], seq[2], seq[2], seq[3], seq[3], seq[0]]

    trav = []
    sz = r1(TRAV_SIZE * CELL)
    off = TRAV_SIZE * CELL / 2
    for i in range(TRAVELLERS):
        xs = ";".join(r1(PX0 + f[i, 0] * CELL - off) for f in frames)
        ys = ";".join(r1(PY0 + f[i, 1] * CELL - off) for f in frames)
        trav.append(
            '<rect width="%s" height="%s" x="%s" y="%s">'
            '<animate attributeName="x" values="%s" keyTimes="%s" dur="%ss" '
            'begin="%ss" repeatCount="indefinite"/>'
            '<animate attributeName="y" values="%s" keyTimes="%s" dur="%ss" '
            'begin="%ss" repeatCount="indefinite"/>'
            '</rect>'
            % (sz, sz, xs.split(";")[0], ys.split(";")[0],
               xs, KT, LOOP, INTRO, ys, KT, LOOP, INTRO))

    # ── chrome ───────────────────────────────────────────────────────────────
    o = []
    a = o.append
    # crispEdges belongs on the dot layers only. Inherited from the root it
    # also lands on every <text>, which kills antialiasing and makes the
    # readout look chunky and broken.
    a('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
      'viewBox="0 0 %d %d" font-family="%s">' % (W, H, W, H, FONT))
    a('<defs><style type="text/css"><![CDATA[%s]]></style></defs>' % font_face())
    a('<rect width="%d" height="%d" fill="%s"/>' % (W, H, C["bg"]))
    a('<rect x="%d" y="%d" width="%d" height="%d" rx="10" fill="%s" stroke="%s"/>'
      % (WX, WY, WW, WH, C["panel"], C["border"]))
    a('<path d="M%d %dh%dv%da10 10 0 0 1-10 10H%da10 10 0 0 1-10-10z" fill="%s"/>'
      % (WX + 10, WY, WW - 20, BAR - 10, WX + 10, C["bg"]))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s"/>'
      % (WX, WY + BAR, WX + WW, WY + BAR, C["border"]))
    for i, col in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        a('<circle cx="%d" cy="%d" r="5.5" fill="%s"/>' % (WX + 24 + i * 20, WY + BAR / 2, col))
    a(text(W / 2, WY + BAR / 2 + 4.5, TITLE, C["dim"], 13, 400, "middle"))

    # portrait frame
    a('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="none" stroke="%s"/>'
      % (PANEL_X, PANEL_Y, PANEL_W, PANEL_H, C["border"]))
    a(text(PANEL_X + 12, PANEL_Y + 17, "VISUAL.MAP", C["chrome"], 11, 700,
           extra=' letter-spacing="1.6"'))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s"/>'
      % (PANEL_X, PANEL_Y + 24, PANEL_X + PANEL_W, PANEL_Y + 24, C["border"]))

    # ── portrait: intro layer (fades in once, then hands over) ───────────────
    a('<g stroke="%s" stroke-width="%s" opacity="1" shape-rendering="crispEdges">'
      % (C["portrait"], r1(CELL)))
    a('<animate attributeName="opacity" to="0" begin="%ss" dur="0.01s" fill="freeze"/>' % INTRO)
    for g, d in enumerate(intro_paths):
        if not d:
            continue
        a('<path d="%s" opacity="0"><animate attributeName="opacity" from="0" to="1" '
          'begin="%.3fs" dur="0.5s" fill="freeze"/></path>'
          % (d, g * INTRO_FADE / INTRO_GROUPS))
    a('</g>')

    # ── portrait: loop layer (drift bands) ──────────────────────────────────
    kt5 = kt(0, T_PORTRAIT, T_PORTRAIT + T_TRANS, LOOP - T_TRANS, LOOP)
    a('<g stroke="%s" stroke-width="%s" opacity="0" shape-rendering="crispEdges">'
      % (C["portrait"], r1(CELL)))
    a('<animate attributeName="opacity" to="1" begin="%ss" dur="0.01s" fill="freeze"/>' % INTRO)
    for b, d in enumerate(band_paths):
        if not d:
            continue
        dx, dy = band_d[b] * CELL
        a('<g><animateTransform attributeName="transform" type="translate" '
          'values="0 0;0 0;%s %s;%s %s;0 0" keyTimes="%s" dur="%ss" begin="%ss" '
          'repeatCount="indefinite"/>'
          '<path d="%s" opacity="1"><animate attributeName="opacity" '
          'values="1;1;0;0;1" keyTimes="%s" dur="%ss" begin="%ss" repeatCount="indefinite"/>'
          '</path></g>' % (r1(dx), r1(dy), r1(dx), r1(dy), kt5, LOOP, INTRO,
                           d, kt5, LOOP, INTRO))
    a('</g>')

    # ── travellers ───────────────────────────────────────────────────────────
    a('<g fill="%s" opacity="0" shape-rendering="crispEdges">' % C["chrome"])
    a('<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" keyTimes="%s" '
      'dur="%ss" begin="%ss" repeatCount="indefinite"/>' % (KT, LOOP, INTRO))
    a("".join(trav))
    a('</g>')

    # ── readout ──────────────────────────────────────────────────────────────
    a(text(RX, PANEL_Y + 17, "SYSTEM.INFO", C["chrome"], 11, 700,
           extra=' letter-spacing="1.6"'))
    a('<circle cx="%d" cy="%d" r="4" fill="#EF4444">'
      '<animate attributeName="opacity" values="1;0.15;1" dur="1.6s" repeatCount="indefinite"/>'
      '</circle>' % (RX + RW - 46, PANEL_Y + 13))
    a('<circle cx="%d" cy="%d" r="4" fill="none" stroke="#EF4444">'
      '<animate attributeName="r" values="4;11" dur="1.6s" repeatCount="indefinite"/>'
      '<animate attributeName="opacity" values="0.7;0" dur="1.6s" repeatCount="indefinite"/>'
      '</circle>' % (RX + RW - 46, PANEL_Y + 13))
    a(text(RX + RW, PANEL_Y + 17, "LIVE", "#EF4444", 12, 700, "end",
           extra=' letter-spacing="1.4"'))
    a('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s"/>'
      % (RX, PANEL_Y + 24, RX + RW, PANEL_Y + 24, C["border"]))

    y = ROW0
    for row in ROWS:
        if row is None:
            y += GROUPGAP
            continue
        label, value = row
        lw = len(label) * FS * MONO_ADV
        vw = len(value) * FS * MONO_ADV
        a(text(RX, y, label, C["dim"]))
        a(text(RX + RW, y, value, C["text"], anchor="end"))
        x1, x2 = RX + lw + 8, RX + RW - vw - 8
        if x2 - x1 > 6:
            a('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4" '
              'stroke-dasharray="1.4 4.6" stroke-linecap="round"/>'
              % (r1(x1), r1(y - 4), r1(x2), r1(y - 4), C["leader"]))
        y += ROWSTEP
    if y > PANEL_Y + PANEL_H - 34:
        log("  !! readout overflows the panel (y=%.0f)" % y)

    pw = len(HANDLE) * 14 * MONO_ADV + 30
    py = PANEL_Y + PANEL_H - 30
    a('<rect x="%d" y="%s" width="%s" height="26" rx="13" fill="%s"/>'
      % (RX, r1(py), r1(pw), C["accent"]))
    a(text(RX + pw / 2, py + 18, HANDLE, C["pill_fg"], 14, 700, "middle"))
    a('</svg>')
    return "".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)

    dots, mask = build_portrait()
    np.save(os.path.join(DATA, "dots_dark.npy"), dots["dark"])
    np.save(os.path.join(DATA, "dots_light.npy"), dots["light"])
    np.save(os.path.join(DATA, "gridmask.npy"), mask)

    clouds = glyph_clouds()
    np.save(os.path.join(DATA, "glyphs.npy"), np.stack(clouds))

    for theme in ("dark", "light"):
        svg = build_svg(theme, dots[theme], mask, clouds)
        p = os.path.join(ASSETS, "%s.svg" % theme)
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg)
        log("  wrote %s  %.0f KB" % (p, len(svg.encode()) / 1024))

    if args.preview:
        preview(dots, clouds)


def preview(dots, clouds):
    out = os.path.join(DATA, "preview_glyphs.png")
    im = Image.new("RGB", (GW * 3, GH), (10, 16, 31))
    d = ImageDraw.Draw(im)
    for k, c in enumerate(clouds):
        for x, y in c:
            d.rectangle([k * GW + x - 1, y - 1, k * GW + x + 1, y + 1], fill=(34, 211, 238))
    im.save(out)
    log("  wrote " + out)


if __name__ == "__main__":
    sys.exit(main())
