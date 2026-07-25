"""Generate the fictional receipt used by the default inference demo."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "assets" / "demo"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 1100), color="#f7f5ef")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((55, 40, 845, 1060), radius=18, fill="white", outline="#333333", width=3)
    draw.text((450, 95), "DEMO MART", font=_font(58, bold=True), fill="#111111", anchor="ma")
    draw.text((450, 170), "12 SAMPLE ROAD", font=_font(35), fill="#222222", anchor="ma")
    draw.text((450, 220), "FICTIONAL CITY 10000", font=_font(30), fill="#333333", anchor="ma")
    draw.line((95, 285, 805, 285), fill="#555555", width=2)
    draw.text((105, 330), "DATE", font=_font(31, bold=True), fill="#111111")
    draw.text((795, 330), "12/07/2026", font=_font(31), fill="#111111", anchor="ra")
    draw.line((95, 390, 805, 390), fill="#bbbbbb", width=2)
    draw.text((105, 435), "ITEM", font=_font(30, bold=True), fill="#111111")
    draw.text((795, 435), "AMOUNT", font=_font(30, bold=True), fill="#111111", anchor="ra")
    rows = (
        ("SAMPLE NOTEBOOK", "18.00"),
        ("DEMO COFFEE", "9.50"),
        ("FICTIONAL GROCERIES", "15.00"),
    )
    y = 505
    for item, amount in rows:
        draw.text((105, y), item, font=_font(29), fill="#222222")
        draw.text((795, y), amount, font=_font(29), fill="#222222", anchor="ra")
        y += 70
    draw.line((95, 745, 805, 745), fill="#555555", width=2)
    draw.text((105, 790), "TOTAL", font=_font(42, bold=True), fill="#111111")
    draw.text((795, 790), "42.50", font=_font(42, bold=True), fill="#111111", anchor="ra")
    draw.line((95, 860, 805, 860), fill="#bbbbbb", width=2)
    draw.text((450, 920), "THANK YOU", font=_font(34, bold=True), fill="#222222", anchor="ma")
    draw.text(
        (450, 980),
        "Synthetic demonstration receipt — not a real transaction",
        font=_font(22),
        fill="#555555",
        anchor="ma",
    )
    image.save(OUTPUT_DIR / "synthetic_receipt.png", format="PNG", optimize=True)
    expected = {
        "company": "DEMO MART",
        "address": "12 SAMPLE ROAD",
        "date": "12/07/2026",
        "total": "42.50",
    }
    (OUTPUT_DIR / "expected_output.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
