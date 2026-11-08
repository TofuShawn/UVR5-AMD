"""Front-end internationalisation: one catalog, three locales, no new dependency.

The GUI's user-facing copy lives here as `{key: (en, zh-Hant, zh-Hans)}` tuples rather
than three parallel dicts, so a key cannot exist in one language and be missing in
another -- completeness is a property of the shape, not of discipline. `tr()` formats
the row for the current locale; English is the source and the fallback.

Only the *front end* reads this. The engine, the CLI and the golden harness never import
it, so scripted runs and byte-pinned baselines are untouched by a language choice. Qt's
own `QLocale` supplies the system language (no third-party i18n library, in a fork that
hand-audits every pin); the choice is persisted in the `ui` block and applied at launch.

Locale is process-global and set once by `app.main` before any widget is built, because
a label constructed in one language and repainted in another is a half-translated page.
Tests build pages directly with the default `en`, which is why the suite's pinned English
copy stays green regardless of the machine's system language.

Rendering note: the offscreen QPA platform has no CJK font fallback, so a zh render grabbed
under `QT_QPA_PLATFORM=offscreen` shows tofu boxes even though the strings are correct;
the same page under the real `windows` platform paints CJK properly (measured). Judge zh
renders on the windows platform, not offscreen.
"""
from __future__ import annotations

import json
from pathlib import Path

import settings as settings_module
from PySide6.QtCore import QLocale
from selections import _read_raw

#: Own top-level block in the shared settings.json, read-modify-written like `ui` and
#: `selections`. Not a `UiPrefs` field (that dataclass is deliberately one field) and not a
#: 51st `Settings` field (the engine never reads a language). The three existing readers
#: ignore this block, so it survives their saves untouched.
_KEY = "i18n"

#: code -> native name, in selector order. The index into each catalog tuple.
LANGUAGES = (("en", "English"), ("zh-Hant", "繁體中文"), ("zh-Hans", "简体中文"))
_INDEX = {code: i for i, (code, _name) in enumerate(LANGUAGES)}

CATALOG: dict[str, tuple[str, str, str]] = {
    # masthead / orientation
    "orientation": (
        "New here? Add audio in 1, pick a model in 2, then press START.",
        "新手?在 1 加入音訊、在 2 選擇模型,然後按 START。",
        "新手?在 1 加入音频、在 2 选择模型,然后按 START。"),
    "fork_badge": ("AMD/ROCm fork", "AMD/ROCm 分支", "AMD/ROCm 分支"),
    # segment headings
    "seg1": ("1 · What to process", "1 · 處理什麼", "1 · 处理什么"),
    "seg2": ("2 · How to separate", "2 · 如何分離", "2 · 如何分离"),
    # field labels
    "input_file": ("Input file", "輸入檔案", "输入文件"),
    "output_dir": ("Output directory", "輸出目錄", "输出目录"),
    "model": ("Model", "模型", "模型"),
    "architecture": ("Architecture", "架構", "架构"),
    "models_installed": ("{n} model(s) available for this architecture",
                         "{n} 個模型可用（此架構）", "{n} 个模型可用（此架构）"),
    "models_none": ("No models for this architecture - add weights under Models",
                    "此架構尚無模型 - 到「Models」頁加入權重",
                    "此架构尚无模型 - 到「Models」页加入权重"),
    "output_format": ("Output format", "輸出格式", "输出格式"),
    "console": ("Console", "主控台", "控制台"),
    "group_input": ("Input", "輸入", "输入"),
    "group_model": ("Model & output", "模型與輸出", "模型与输出"),
    "group_device": ("Device", "裝置", "设备"),
    "row_input_desc": ("One file, or add a batch below", "單一檔案，或在下方加入批次", "单个文件，或在下方加入批次"),
    "row_batch": ("Batch", "批次", "批次"),
    "row_batch_desc": ("Add more files or a folder to the queue", "加入更多檔案或資料夾到佇列", "加入更多文件或文件夹到队列"),
    "row_output_desc": ("Leave empty for the default folder", "留空則用預設資料夾", "留空则用默认文件夹"),
    "row_arch_desc": ("Pick the model family", "選擇模型架構", "选择模型架构"),
    "row_model_desc": ("Choose a model for this architecture", "為此架構選擇模型", "为此架构选择模型"),
    "row_format_desc": ("How the stems are written", "輸出檔案的格式", "输出文件的格式"),
    "row_device": ("Device", "裝置", "设备"),
    "row_device_desc": ("Auto / ROCm / DirectML / CPU", "自動 / ROCm / DirectML / CPU", "自动 / ROCm / DirectML / CPU"),
    "language": ("Language", "語言", "语言"),
    "compute_device": ("Compute device", "運算裝置", "计算设备"),
    "lang_system": ("System (follow OS)", "跟隨系統", "跟随系统"),
    # buttons
    "browse": ("Browse", "瀏覽", "浏览"),
    "add_files": ("Add files…", "加入檔案…", "添加文件…"),
    "add_folder": ("Add folder…", "加入資料夾…", "添加文件夹…"),
    "clear": ("Clear", "清除", "清除"),
    "start": ("START", "START", "START"),
    "stop": ("STOP", "STOP", "STOP"),
    "settings": ("Settings", "設定", "设置"),
    "params_edit": ("Edit these in Settings", "到設定編輯", "到设置编辑"),
    # checkboxes / expander
    "default_output_check": (
        "Use this as my default output folder",
        "設為我的預設輸出資料夾", "设为我的默认输出文件夹"),
    "gpu_check": ("Use GPU", "使用 GPU", "使用 GPU"),
    "params_header": ("Model parameters", "模型參數", "模型参数"),
    # placeholders
    "input_ph": (
        "path to an audio file, or add a batch below",
        "音訊檔案路徑,或在下方加入批次", "音频文件路径,或在下方添加批次"),
    "queue_ph": (
        "nothing queued yet -- add files or a folder above",
        "尚無佇列 -- 在上方加入檔案或資料夾", "暂无队列 -- 在上方添加文件或文件夹"),
    "output_ph": (
        "empty = the default folder below",
        "留空 = 下方的預設資料夾", "留空 = 下方的默认文件夹"),
    "console_ph": (
        "engine output appears here",
        "引擎輸出會顯示在這裡", "引擎输出会显示在这里"),
    # START blockers
    "blocker_model": (
        "pick a model from the Model list first",
        "請先從 Model 清單選擇模型", "请先从 Model 列表选择模型"),
    "blocker_input": (
        "choose an input audio file first",
        "請先選擇輸入音訊檔案", "请先选择输入音频文件"),
    "blocker_missing": (
        "input audio not found: {head}{more}",
        "找不到輸入音訊:{head}{more}", "找不到输入音频:{head}{more}"),
    "blocker_more": (" (+{n} more)", "(還有 {n} 個)", "(还有 {n} 个)"),
    # count / summary
    "count_queued": ("{n} file(s) queued", "已佇列 {n} 個檔案", "已排队 {n} 个文件"),
    "count_progress": ("{done} / {total} file(s)", "{done} / {total} 個檔案", "{done} / {total} 个文件"),
    "summary_files": ("{n} file(s)", "{n} 個檔案", "{n} 个文件"),
    "summary_no_input": ("no input", "無輸入", "无输入"),
    "summary_output": ("{fmt} output", "{fmt} 輸出", "{fmt} 输出"),
    # default output caption
    "default_set": ("default output: {path}", "預設輸出:{path}", "默认输出:{path}"),
    "default_unset": (
        "default output: UVR_out beside each input file",
        "預設輸出:每個輸入檔旁的 UVR_out", "默认输出:每个输入文件旁的 UVR_out"),
    "default_not_set": (
        "default output not set: the Output directory box is empty -- choose a "
        "folder first, then tick the box",
        "預設輸出未設定:輸出目錄欄是空的 -- 先選資料夾,再勾選",
        "默认输出未设置:输出目录栏是空的 -- 先选文件夹,再勾选"),
    # parameter summary
    "params_none": (
        "{arch} reads no per-arch engine parameters; the general group's "
        "format and device choices are what apply to it.",
        "{arch} 不讀取各架构引擎參數;套用的一般群組格式與裝置選擇即為其設定。",
        "{arch} 不读取各架构引擎参数;套用的通用组格式与设备选择即为其设置。"),
    "params_line": (
        "{arch}: {n} engine parameters (read-only here -- use the button below)",
        "{arch}:{n} 個引擎參數(此处唯讀 -- 使用下方按鈕)",
        "{arch}:{n} 个引擎参数(此处只读 -- 使用下方按钮)"),
    "lang_restart": (
        "language saved; restart the app to apply it",
        "語言已儲存;重新啟動應用程式後生效", "语言已保存;重启应用后生效"),
}

_current = "en"


def set_locale(code: str) -> None:
    """Select the process locale; an unknown code falls back to English, never crashes."""
    global _current
    _current = code if code in _INDEX else "en"


def current_locale() -> str:
    return _current


def tr(key: str, /, **kw) -> str:
    """The current locale's copy for `key`, formatted with `kw`. Unknown key is a bug."""
    row = CATALOG[key]
    text = row[_INDEX[_current]]
    return text.format(**kw) if kw else text


def resolve_system() -> str:
    """Map the OS UI language to one of our three locales; anything else is English."""
    tags = QLocale.system().uiLanguages() or [QLocale.system().name()]
    for tag in tags:
        t = tag.replace("_", "-")
        if t.startswith("zh"):
            if "Hant" in t or any(r in t for r in ("TW", "HK", "MO")):
                return "zh-Hant"
            return "zh-Hans"
        if t.startswith("en"):
            return "en"
    return "en"


def load_language(path: Path | None = None) -> str:
    """The persisted choice: a locale code or `"system"` (follow the OS). Default system."""
    target = Path(path) if path is not None else settings_module.config_file()
    block = _read_raw(target).get(_KEY) or {}
    code = block.get("language", "system")
    return code if isinstance(code, str) else "system"


def save_language(code: str, path: Path | None = None) -> Path:
    """Merge the choice into the shared file without rewriting the other blocks."""
    target = Path(path) if path is not None else settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(target)
    raw[_KEY] = {"language": code}
    target.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target