"""The bundled CJK font must cover every character the catalog can show.

Measured failure this guards: the subset was built once by hand, and each later catalog
addition (要/器/整 came with the settings titles) painted as a tofu box because the
glyph was simply absent. `tools/make_cjk_subset.py` regenerates the file from the
catalog; this test is the reason somebody will notice it is needed.

The oracle is the font's own `cmap`, read with fontTools (a build-time tool the generator
already needs), **not** `QFontMetrics.inFont`: measured on this subset, Qt reports
"unsupported" for 229 of 273 characters that sit in the cmap and render correctly in the
window -- so `inFont` would fail a good file. The rendering itself was checked by
screenshot under the real `windows` platform; what a test can pin is that the file ships
the glyphs.
"""
from pathlib import Path

from fontTools.ttLib import TTFont
from PySide6.QtGui import QFontDatabase

from app import i18n

REPO = Path(__file__).resolve().parents[1]
FONT = REPO / "app" / "assets" / "fonts" / "NotoSansSC.ttf"


def test_the_family_is_the_one_the_widget_chain_asks_for(qapp):
    """`app/shell.py` rewrites Fluent's families to ["Inter", "Noto Sans SC"]; a font
    whose name table says something else would load and never be used."""
    font_id = QFontDatabase.addApplicationFont(str(FONT))
    assert font_id != -1, f"Qt could not load {FONT}"
    families = QFontDatabase.applicationFontFamilies(font_id)
    assert families and families[0] == "Noto Sans SC", families
    assert "Noto Sans SC" in (REPO / "app" / "shell.py").read_text(encoding="utf-8")


def test_every_catalog_character_is_in_the_bundled_font():
    cmap = TTFont(str(FONT)).getBestCmap()
    wanted = {ch for row in i18n.CATALOG.values() for text in row[1:]
              for ch in text if not ch.isspace()}
    missing = sorted(ch for ch in wanted if ord(ch) not in cmap)
    assert not missing, (
        f"{len(missing)} catalog glyphs are not in {FONT.name}: "
        f"{''.join(missing)} -- run: python tools/make_cjk_subset.py")
