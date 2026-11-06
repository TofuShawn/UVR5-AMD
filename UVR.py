# --------------------------------------------------------------------------- #
# The refusal is the FIRST statement of this file, above every import.
#
# Task 11d retired ModelData -- the only thing this block's MainWindow could ever
# resolve a model through -- and Plan 2 replaces this GUI with a PySide6 app. The
# refusal used to sit at the bottom of the file, below ~7,290 lines of module-level
# imports, which meant `python UVR.py` never reached it: this file is not importable
# in the venv it ships in (`import pyglet` fails here, and `separate` no longer
# exports `directml_available`), so a user got a ModuleNotFoundError traceback
# instead of the sentence telling them to use separate_cli.py. Nothing above a raise
# runs, so the raise has to be above everything.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    raise SystemExit("legacy tkinter GUI removed; use separate_cli.py, or the Fluent-Qt app: python -m app.main")

    # Plan 2 Task 8 deleted the layer this block used to launch: the nine GUI classes
    # (the batch/ensemble/audio-tool layers, the widgets, the window itself), the 16
    # `pop_up_*` dialogs, the drag-and-drop callback and the module's launcher body --
    # 7,001 lines, unreachable from the moment the raise above became the first
    # statement. What survives here is the header, the OS/path set-up and inert data
    # constants, and it reaches no GUI import: `tests/test_model_data_retired.py`'s
    # `test_tkinter_is_gone_and_the_engine_still_runs` asserts that absence *and* the
    # presence of the engine behind it, because a guard that only checks absence is
    # satisfied by deleting the feature. The front ends are `python -m app.main` (GUI)
    # and `python separate_cli.py` (GUI-free).
    #
    # The cut ran to the end of the file rather than stopping at :7243, which is where
    # `legacy_main` ends in the deleted copy. The five helpers below it
    # (`read_bulliten_text_mac`, `open_link`, `auto_hyperlink`, `vip_downloads`,
    # `extract_stems`) are not tkinter themselves, so each was checked before cutting:
    # every call of every one of them sat inside the range above, and none is reached
    # from outside this file. They went on that evidence, and
    # `tests/test_model_data_retired.py` names all five of those helpers in
    # GONE_FUNCTIONS. (The five named further below are a different five: survivors.)

    # The keys those dialogs collected are not lost: `app/dialogs.py` PARAM_FIELDS is
    # the live copy the app builds its tier-3 form from, and
    # `tests/test_dialogs.py::POPUP_EVIDENCE` is the frozen, provenanced record of what
    # the popup bodies themselves wrote, read from this file at commit 720ca46.


# Upstream labelled this block "GUI modules", and that label is the one thing Task 8
# could not delete along with the GUI. What it imported and no longer does: `tk`, `ttk`,
# `Font`, `filedialog` and `messagebox` from the windowing toolkit, the themed-widget
# binding and the drag-and-drop binding -- seven lines, none of them reachable once the
# classes that used them went.
#
# The sentence that used to justify everything BELOW those seven was "the rest stays,
# because the set-up below this raise still reads it, and because `pyglet` -- whose
# absence is why the refusal above has to be the first statement in the file -- is still
# among them." It was false, and this file is the refutation. Measured against its own
# AST, 40 of the 61 names its explicit imports bind are never referenced anywhere below
# -- `torch`, `librosa`, `onnx`, `wget`, `urllib.request`, `webbrowser`, `pyperclip`,
# `playsound`, `matchering`, the four model classes in the `separate` import and the
# three `cryptography` names are the heavy ones -- and `pyglet_font` is one of the 40, so
# the reason named for keeping the block is itself unread. `import pyglet` still *runs*,
# which is exactly why the refusal has to be the file's first statement; execution is not
# use. Both sets -- the 40 dead and the 21 read -- are pinned as DEAD_IMPORT_BINDINGS and
# READ_IMPORT_BINDINGS in `tests/test_model_data_retired.py`, whose guard fails if either
# set moves in either direction, so this paragraph and its counts cannot go quietly
# stale. Seven of the 21 (`json`, `time`, `queue`, `psutil`, `yaml`, `ConfigDict`,
# `KThread`) are read only inside the caller-less functions named below, so the set-up
# that earns its keep here touches fourteen names, not twenty-one.
#
# The real reason this block stayed is a boundary, not a reader. Task 8 cut the tkinter
# layer at one line -- the first GUI class definition -- rather than re-auditing 61
# import bindings inside the residue. The residue is *unreachable when run*, which is
# not the same as *inert*: the raise above makes `python UVR.py` stop at its first
# statement, and nothing that ships imports this file -- but everything below is at
# module level, NOT inside the `if __name__ == "__main__"` block, so an import *would*
# execute it, and it is not side-effect-free: `os.remove(SPLASH_DOC)`, three `os.mkdir`
# sites (`DEFAULT_SAVE_PATH`, `ENSEMBLE_TEMP_PATH`, `SAMPLE_CLIP_PATH`), the
# `file_check`/`remove_temps` writes and `data = load_data()` against a CWD-relative
# `data.pkl` -- and the module-level `os.chdir(BASE_PATH)` is what makes that CWD the
# repository root. "Unreachable when run" is the honest word, not "inert either way";
# that module-level I/O is the reason the unimportability the module docstring relies on
# must be preserved, not just tolerated. No line numbers are quoted for them because
# every one of them sits below this comment, so editing the comment moves them -- the
# names are the stable handle, and `tests/test_model_data_retired.py` is what keeps the
# set from growing. Trimming the unread imports is a separate call about how much of a
# run-dead file to keep, and it has not been made here.
#
# The same honesty is owed to the functions, and Task 8's report understated it (it
# named one of five): seven survive below, and five have no caller in any tracked file,
# inside this file included -- `get_execution_time`, `close_process`,
# `load_model_hash_data`, `font_checker`, `get_yaml_data`. Only `save_data` and
# `load_data` are reached, `load_data` from the last line of this file and `save_data`
# from inside `load_data`. `CALLER_LESS_FUNCTIONS` in
# `tests/test_model_data_retired.py` pins both halves.
import time
#start_time = time.time()
import audioread
import hashlib
import json
import librosa
import math
import natsort
import os
import pickle
import psutil
from pyglet import font as pyglet_font
import pyperclip
import base64
import queue
import shutil
import subprocess
import soundfile as sf
import torch
import urllib.request
import webbrowser
import wget
import traceback
import matchering as match
from collections import Counter
from __version__ import VERSION, PATCH, PATCH_MAC, PATCH_LINUX
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
from gui_data.constants import *
from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
from lib_v5.vr_network.model_param_init import ModelParameters
from lib_v5 import spec_utils
from lib_v5 import apollo_inference
from kthread import KThread
from pathlib  import Path
from separate import (
    SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR,  # Model-related
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available
)
from playsound import playsound
from typing import List
import onnx
import re
import sys
import yaml
from ml_collections import ConfigDict
from collections import Counter
from os.path import expanduser
# import faulthandler
# faulthandler.enable()

# The legacy GUI's DirectML device path is gone: `separate.py` no longer exports
# `directml_available` and `torch_directml` is not a dependency of this fork (Task 12
# moved the ONNX provider decision into backends.resolve(), which picks
# DmlExecutionProvider by the PCI vendor scan). These flags only fed the removed
# device pickers, so they are constants now instead of probes of a module that does
# not exist. The settings key `is_use_directml` survives in settings.py as inert
# data; nothing reads it as a device choice.
is_choose_arch = False
is_directml_only = False
is_cuda_only = False
is_gpu_available = cuda_available or mps_available

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

if os.path.isfile(SPLASH_DOC):
    os.remove(SPLASH_DOC)

def get_execution_time(function, name):
    start = time.time()
    function()
    end = time.time()
    time_difference = end - start
    print(f'{name} Execution Time: ', time_difference)

# Task 12: PREVIOUS_PATCH_WIN (the previous upstream patch id, used only to tell a
# roll-back from an update in the deleted start-up check) is gone with the rest of the
# application's own network.

is_dnd_compatible = True
banner_placement = -2

if OPERATING_SYSTEM=="Darwin":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["open", input_string])
    dnd_path_check = MAC_DND_CHECK
    banner_placement = -8
    current_patch = PATCH_MAC
    is_windows = False
    is_macos = True
    right_click_button = '<Button-2>'
    application_extension = ".dmg"
elif OPERATING_SYSTEM=="Linux":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["xdg-open", input_string])
    dnd_path_check = LINUX_DND_CHECK
    current_patch = PATCH_LINUX
    is_windows = False
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".zip"
elif OPERATING_SYSTEM=="Windows":
    OPEN_FILE_func = lambda input_string:os.startfile(input_string)
    dnd_path_check = WINDOWS_DND_CHECK
    current_patch = PATCH
    is_windows = True
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".exe"


if not is_windows:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
else:
    from ctypes import windll, wintypes
    
def close_process(q:queue.Queue):
    def close_splash():
        name = "UVR_Launcher.exe"
        for process in psutil.process_iter(attrs=["name"]):
            process_name = process.info.get("name")
            
            if process_name == name:
                try:
                    process.terminate()
                    q.put(f"{name} terminated.")  # Push message to queue
                    break
                except psutil.NoSuchProcess as e:
                    q.put(f"Error terminating {name}: {e}")  # Push error to queue
                    
                    try:
                        with open(SPLASH_DOC, 'w') as f:
                            f.write('1')
                    except:
                        print('No splash screen.')

    thread = KThread(target=close_splash)
    thread.start()

def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Paramters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    with open('data.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)

def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    try:
        with open('data.pkl', 'rb') as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it

        save_data(data=DEFAULT_DATA)

        return load_data()

def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''
    with open(dictionary, 'r') as d:
        return json.load(d)

def font_checker(font_file):
    chosen_font_name = None
    chosen_font_file = None
    
    try:
        if os.path.isfile(font_file):
            with open(font_file, 'r') as d:
                chosen_font = json.load(d)
                
            chosen_font_name = chosen_font["font_name"]
            if chosen_font["font_file"]:
                chosen_font_file = os.path.join(OTHER_FONT_PATH, chosen_font["font_file"])
                chosen_font_file = chosen_font_file if os.path.isfile(chosen_font_file) else None
    except Exception as e:
        print(e)
        
    chosen_font = chosen_font_name, chosen_font_file
     
    return chosen_font

def get_yaml_data(yamml_path, is_get_dict=False):
    config = None
    try:
        if os.path.isfile(yamml_path):
            with open(yamml_path) as f:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                
            if is_get_dict:
                config = dict(config)
                
    except Exception as e:
        print(e)
        
    return config
        
debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')
APOLLO_MODELS_DIR = os.path.join(MODELS_DIR, 'Apollo_Models')
APOLLO_CONFIG_PATH = os.path.join(APOLLO_MODELS_DIR, 'model_configs')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
APOLLO_HASH_DIR = os.path.join(APOLLO_MODELS_DIR, 'model_data')
APOLLO_HASH_JSON = os.path.join(APOLLO_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')#

APP_DIR_DATA = os.path.join(expanduser("~"), "UVR Conversions")
DEFAULT_SAVE_PATH = os.path.join(APP_DIR_DATA)

if not os.path.isdir(DEFAULT_SAVE_PATH):
    os.mkdir(DEFAULT_SAVE_PATH)

#CR Text
CR_TEXT = os.path.join(BASE_PATH, 'gui_data', 'cr_text.txt')

#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.ico')
if not is_windows:
    MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.png')

OWN_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'own_font.json')

MAIN_FONT_NAME = 'Montserrat'
SEC_FONT_NAME = 'Century Gothic'
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'Montserrat', 'Montserrat.ttf')#
SEC_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.ttf')#
OTHER_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'other')#

FONT_MAPPER = {MAIN_FONT_NAME:FONT_PATH,
               SEC_FONT_NAME:SEC_FONT_PATH}

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

# Task 12: MODEL_DATA_URLS -- a list of the four killed model-table URLs, and the last
# module-level load of them -- is deleted. Those tables are refreshed on request only,
# by model_refresh.py writing through Catalog.write_table.
MODEL_DATA_FILES = [VR_HASH_JSON, MDX_HASH_JSON, MDX_MODEL_NAME_SELECT, DEMUCS_MODEL_NAME_SELECT]

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}
data = load_data()
