"""Regenerate the bundled CJK subset from the i18n catalog.

Why a subset at all: the full Noto Sans SC is ~10 MB, the app ships its fonts in
`app/assets/fonts/`, and every label it draws comes from `app/i18n.py`. So the file
only needs the glyphs that catalog can produce -- plus ASCII, because a path or a
model name printed through the CJK fallback must not turn into boxes either.

Why this script exists: the subset was first built by hand, and every catalog string
added afterwards (要/器/整 arrived with the settings titles) was missing from it, which
paints as tofu. `tests/test_fonts.py` now fails when a catalog glyph is not covered, and
this is the command that fixes it.

Usage (fontTools is a build-time tool, not a runtime dependency):

    python tools/make_cjk_subset.py [path-to-NotoSansSC-VF.ttf]

The default source is Windows' own variable font. It is instanced at wght=400 first:
the variable font's default named instance is Thin, and a name table that still says so
makes Qt pick a weight nobody asked for.
"""
from __future__ import annotations

import string
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "app" / "assets" / "fonts" / "NotoSansSC.ttf"
DEFAULT_SOURCE = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")


def catalog_text() -> str:
    """Every character any locale of the catalog can render, plus printable ASCII."""
    sys.path.insert(0, str(REPO))
    import app.i18n as i18n

    chars = set(string.printable)
    for row in i18n.CATALOG.values():
        for text in row:
            chars.update(text)
    return "".join(sorted(chars))


def _rename(font: TTFont) -> None:
    """Family/subfamily Qt can match, at the weight we instanced."""
    names = {1: "Noto Sans SC", 2: "Regular", 4: "Noto Sans SC Regular",
             6: "NotoSansSC-Regular", 16: "Noto Sans SC", 17: "Regular"}
    for record in font["name"].names:
        if record.nameID in names:
            record.string = names[record.nameID]
    font["OS/2"].usWeightClass = 400


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else DEFAULT_SOURCE
    if not source.is_file():
        print(f"source font missing: {source}", file=sys.stderr)
        return 2

    font = instantiateVariableFont(TTFont(str(source)), {"wght": 400})
    _rename(font)

    # The subsetter recalculates the OS/2 coverage bits from what it kept, and the
    # result is too narrow for DirectWrite: measured, a font whose cmap held every
    # glyph still answered "unsupported" for 229 of 273 catalog characters, because
    # `ulCodePageRange1` said Latin-1 only and the CJK unicode-range bits were clear.
    # Keep the source's coverage claim -- this is still Noto Sans SC, subset or not.
    os2 = font["OS/2"]
    coverage = (os2.ulUnicodeRange1, os2.ulUnicodeRange2, os2.ulUnicodeRange3,
                os2.ulUnicodeRange4, os2.ulCodePageRange1, os2.ulCodePageRange2)

    options = subset.Options()
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.notdef_outline = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=catalog_text())
    subsetter.subset(font)

    os2 = font["OS/2"]
    (os2.ulUnicodeRange1, os2.ulUnicodeRange2, os2.ulUnicodeRange3,
     os2.ulUnicodeRange4, os2.ulCodePageRange1, os2.ulCodePageRange2) = coverage
    os2.fsSelection = (os2.fsSelection & ~0x21) | 0x40       # regular, not italic/bold
    os2.usFirstCharIndex = 0x20
    os2.usLastCharIndex = 0xFFFF

    OUT.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{len(font.getBestCmap())} codepoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
