"""Low-level rendering primitives for DEV document generation.

Determinism strategy (see docs/DEV_DOCUMENT_SPEC.md "Reproducibility"):
- ``SOURCE_DATE_EPOCH`` is pinned at import so reportlab emits fixed
  PDF CreationDate/ModDate (PDF /ID is already a content digest).
- DOCX/XLSX are repacked: sorted entries, fixed ZipInfo timestamps, and
  fixed ``core.xml`` created/modified stamps.
- PNG/JPG carry no timestamps (Pillow/matplotlib do not emit any).
- All stochastic degradation uses ``random.Random(SEED)`` streams.
"""

import io
import os
import re
import zipfile
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")  # 2024-01-01Z
FIXED_ZIP_DT = (2024, 1, 1, 0, 0, 0)
FIXED_ISO_TS = "2024-01-01T00:00:00Z"

ARIAL_TTF = r"C:\Windows\Fonts\arial.ttf"
ARIAL_BOLD_TTF = r"C:\Windows\Fonts\arialbd.ttf"

PDF_META = {"author": "MizanIQ", "title": None,
            "creator": "MizanIQ-devgen-v0.1", "subject": "dataset_v0.1 DEV"}

# ---------------------------------------------------------------------------
# Text / money formatting
# ---------------------------------------------------------------------------
def ar(text: str) -> str:
    """Reshape + bidi-reorder logical Arabic text for visual renderers
    (reportlab, Pillow, matplotlib). NOT for DOCX (Word shapes natively)."""
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def fmt_money(value: Decimal) -> str:
    """Document-facing money: thousands separators, exactly 2dp."""
    return format(value, ",.2f")


def plain(value: Decimal) -> str:
    """Ground-truth numeric form: plain 2dp, no separators."""
    return format(value, ".2f")


# ---------------------------------------------------------------------------
# Native PDF (reportlab platypus)
# ---------------------------------------------------------------------------
_registered_fonts = False


def ensure_pdf_fonts():
    global _registered_fonts
    if _registered_fonts:
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("Arial", ARIAL_TTF))
    pdfmetrics.registerFont(TTFont("Arial-Bold", ARIAL_BOLD_TTF))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                  italic="Arial", boldItalic="Arial-Bold")
    _registered_fonts = True


def build_pdf(path, flowables, title: str, pagesize=None):
    """Build a native selectable-text PDF with fixed metadata."""
    from reportlab.platypus import SimpleDocTemplate
    ensure_pdf_fonts()
    meta = dict(PDF_META, title=title)
    doc = SimpleDocTemplate(str(path), author=meta["author"],
                            title=meta["title"], creator=meta["creator"],
                            subject=meta["subject"],
                            pagesize=pagesize) if pagesize else \
        SimpleDocTemplate(str(path), author=meta["author"],
                          title=meta["title"], creator=meta["creator"],
                          subject=meta["subject"])
    doc.build(flowables)
    return path


def pdf_from_image(path, image, title: str, jpeg_quality: int | None = None):
    """Single-raster-page PDF (scanned-style): no selectable body text.

    ``jpeg_quality`` set → JPEG-encoded page image (scan-realistic, small);
    None → lossless PNG (larger; use only when encoding must not be a tested
    variable). Pillow JPEG output is deterministic (no EXIF).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    ensure_pdf_fonts()
    buf = io.BytesIO()
    if jpeg_quality is None:
        image.save(buf, format="PNG")
    else:
        image.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality,
                                  subsampling=1, optimize=False)
    buf.seek(0)
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setAuthor(PDF_META["author"])
    canvas.setTitle(title)
    canvas.setCreator(PDF_META["creator"])
    canvas.setSubject(PDF_META["subject"])
    canvas.drawImage(ImageReader(buf), 0, 0, width=A4[0], height=A4[1],
                     preserveAspectRatio=True, anchor="c")
    canvas.showPage()
    canvas.save()
    return path


# ---------------------------------------------------------------------------
# DOCX / XLSX with metadata normalization (byte-determinism)
# ---------------------------------------------------------------------------
_DT_RE = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?")


def normalize_zip(path) -> None:
    """Repack a ZIP-based office file deterministically: sorted entries,
    fixed entry timestamps, fixed core-property datetimes."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zin:
        items = {}
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.startswith("docProps/") and \
                    info.filename.endswith(".xml"):
                data = _DT_RE.sub(FIXED_ISO_TS.encode(), data)
            items[info.filename] = (data, info.compress_type)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(items):
            data, compress_type = items[name]
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_DT)
            info.compress_type = compress_type
            info.create_system = 0
            zout.writestr(info, data)


# ---------------------------------------------------------------------------
# Image helpers (Pillow)
# ---------------------------------------------------------------------------
def new_page(width: int, height: int, color: str = "white"):
    from PIL import Image
    return Image.new("RGB", (width, height), color)


def get_font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(
        ARIAL_BOLD_TTF if bold else ARIAL_TTF, size)


def degrade(image, *, rotation: float, blur: float, brightness: float,
            noise_sigma: float, seed: int, bands=()):
    """CONTROLLED scan degradation (all params recorded in ground truth)."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    # NB: noise uses RandomState(seed); rotation/blur/brightness/bands are
    # fixed-parameter transforms, so the whole pipeline is deterministic.
    # 1. brightness (deterministic factor, no RNG consumed)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    # 2. Gaussian sensor noise, seeded
    npx = np.asarray(image).astype("float32")
    gauss = np.random.RandomState(seed).normal(0.0, noise_sigma, npx.shape)
    npx = np.clip(npx + gauss, 0, 255).astype("uint8")
    image = Image.fromarray(npx, mode="RGB")
    # 3. faint horizontal band artifacts at FIXED positions (photocopier look):
    #    deterministic multiplicative darkening, no RNG consumed.
    if bands:
        import numpy as np
        px = np.asarray(image).astype("float32")
        for y0, y1, darken in bands:  # darken: 0..1 fraction removed
            px[y0:y1, :, :] *= (1.0 - darken)
        image = Image.fromarray(np.clip(px, 0, 255).astype("uint8"),
                                mode="RGB")
    # 4. slight blur
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    # 5. rotation on a padded white canvas (nothing gets cropped)
    if rotation:
        pad = int(max(image.size) * 0.06)
        canvas = Image.new("RGB", (image.size[0] + 2 * pad,
                                   image.size[1] + 2 * pad), "white")
        canvas.paste(image, (pad, pad))
        image = canvas.rotate(rotation, resample=Image.BICUBIC, expand=False,
                              fillcolor="white")
    return image


# ---------------------------------------------------------------------------
# Chart / dashboard drawing (Pillow-native: deterministic, no heavy deps)
# ---------------------------------------------------------------------------
INK = (31, 58, 95)
GRID = (210, 210, 210)
SERIES_COLORS = [(31, 58, 95), (58, 122, 58), (138, 109, 31)]


def draw_line_chart(title, xlabel, ylabel, xs, ys, value_labels,
                    size=(1600, 1000)):
    """Deterministic line chart with markers + value labels. Returns Image."""
    from PIL import ImageDraw
    W, H = size
    img = new_page(W, H)
    draw = ImageDraw.Draw(img)
    f_title = get_font(34, bold=True)
    f_axis = get_font(26)
    f_tick = get_font(24)
    L, Rm, T, B = 150, 70, 130, 110
    box = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (box[2] - box[0])) / 2, 36), title, font=f_title,
              fill=(0, 0, 0))
    lo, hi = 0.0, max(ys) * 1.12
    steps = 6
    for i in range(steps + 1):
        v = hi * i / steps
        y = T + (H - T - B) * (1 - i / steps)
        draw.line([(L, y), (W - Rm, y)], fill=GRID, width=1)
        lab = f"{v / 1e6:.0f}M"
        lb = draw.textbbox((0, 0), lab, font=f_tick)
        draw.text((L - 12 - (lb[2] - lb[0]), y - (lb[3] - lb[1]) / 2), lab,
                  font=f_tick, fill=(60, 60, 60))
    n = len(xs)
    px = [L + (W - L - Rm) * (i / (n - 1) if n > 1 else 0.5)
          for i in range(n)]
    py = [T + (H - T - B) * (1 - (v - lo) / (hi - lo)) for v in ys]
    for i in range(n - 1):
        draw.line([(px[i], py[i]), (px[i + 1], py[i + 1])],
                  fill=INK, width=5)
    for i, (x, vlab) in enumerate(zip(xs, value_labels)):
        draw.ellipse([(px[i] - 9, py[i] - 9), (px[i] + 9, py[i] + 9)],
                     fill=INK)
        vb = draw.textbbox((0, 0), vlab, font=f_tick)
        draw.text((px[i] - (vb[2] - vb[0]) / 2, py[i] - 34), vlab,
                  font=f_tick, fill=(0, 0, 0))
        xb = draw.textbbox((0, 0), str(x), font=f_tick)
        draw.text((px[i] - (xb[2] - xb[0]) / 2, H - B + 24), str(x),
                  font=f_tick, fill=(40, 40, 40))
    draw.rectangle([(L, T), (W - Rm, H - B)], outline=(0, 0, 0), width=2)
    yb = draw.textbbox((0, 0), ylabel, font=f_axis)
    draw.text((28, (H / 2) - (yb[2] - yb[0]) / 2), ylabel, font=f_axis,
              fill=(40, 40, 40))
    xb2 = draw.textbbox((0, 0), xlabel, font=f_axis)
    draw.text(((W - (xb2[2] - xb2[0])) / 2, H - 56), xlabel, font=f_axis,
              fill=(40, 40, 40))
    return img


def draw_dashboard(title, cards, bar_title, bar_labels, bar_values,
                   panel_title, panel_lines, size=(1600, 1000)):
    """Deterministic KPI dashboard: cards + horizontal bars + text panel."""
    from PIL import ImageDraw
    W, H = size
    img = new_page(W, H)
    draw = ImageDraw.Draw(img)
    f_title = get_font(30, bold=True)
    f_card_t = get_font(24, bold=True)
    f_card_v = get_font(30, bold=True)
    f_body = get_font(26)
    f_small = get_font(23)
    box = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (box[2] - box[0])) / 2, 24), title, font=f_title,
              fill=(0, 0, 0))
    # KPI cards
    cw, chh, gap, top = 360, 210, 24, 100
    x0 = (W - (4 * cw + 3 * gap)) / 2
    for i, (label, value) in enumerate(cards):
        x = x0 + i * (cw + gap)
        draw.rectangle([(x, top), (x + cw, top + chh)], outline=INK, width=3)
        lb = draw.textbbox((0, 0), label, font=f_card_t)
        draw.text((x + (cw - (lb[2] - lb[0])) / 2, top + 26), label,
                  font=f_card_t, fill=INK)
        vb = draw.textbbox((0, 0), value, font=f_card_v)
        draw.text((x + (cw - (vb[2] - vb[0])) / 2, top + 100), value,
                  font=f_card_v, fill=(0, 0, 0))
    # branch bars (left)
    px0, py0, pw, ph = 90, 400, 660, 500
    tb = draw.textbbox((0, 0), bar_title, font=f_body)
    draw.text((px0, py0 - 44), bar_title, font=f_body, fill=(0, 0, 0))
    draw.rectangle([(px0, py0), (px0 + pw, py0 + ph)], outline=(0, 0, 0),
                   width=2)
    rows = len(bar_labels)
    rh = ph / rows
    vmax = max(bar_values) * 1.22
    for i, (lab, val) in enumerate(zip(bar_labels, bar_values)):
        yc = py0 + rh * i + rh / 2
        bw = pw * 0.52 * val / vmax
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        draw.rectangle([(px0 + pw * 0.30, yc - 22),
                        (px0 + pw * 0.30 + bw, yc + 22)], fill=color)
        draw.text((px0 + 8, yc - 16), lab, font=f_small, fill=(0, 0, 0))
        draw.text((px0 + pw * 0.30 + bw + 10, yc - 16),
                  f"{val / 1e6:.2f}M", font=f_small, fill=(0, 0, 0))
    # budget panel (right)
    qx0, qy0, qw, qh = 810, 400, 700, 500
    pb = draw.textbbox((0, 0), panel_title, font=f_body)
    draw.text((qx0, qy0 - 44), panel_title, font=f_body, fill=(0, 0, 0))
    draw.rectangle([(qx0, qy0), (qx0 + qw, qy0 + qh)], outline=INK, width=3)
    for i, line in enumerate(panel_lines):
        draw.text((qx0 + 30, qy0 + 40 + i * 62), line, font=f_body,
                  fill=(0, 0, 0))
    return img
