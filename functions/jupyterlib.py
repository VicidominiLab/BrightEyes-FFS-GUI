# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:28:44 2026

@author: eslenders
"""

import sys, os, shutil, subprocess
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import json
from pathlib import Path


def find_jupyter_old():
    """Find jupyter executable, checking PATH and common install locations."""
    # 1. Standard PATH lookup
    jupyter = shutil.which("jupyter") or shutil.which("jupyter-lab")
    
    if jupyter:
        return jupyter

    # 2. Common locations missed when PATH is restricted inside a frozen exe
    candidates = []
    if sys.platform == 'win32':
        for base in [os.environ.get('LOCALAPPDATA', ''),
                     os.environ.get('APPDATA', ''),
                     os.path.expanduser('~')]:
            candidates += [
                os.path.join(base, 'Programs', 'Python', 'Scripts', 'jupyter.exe'),
                os.path.join(base, r'Programs\Python\Python3\Scripts\jupyter.exe'),
            ]
        # Also check all Python installs on PATH via py launcher
        try:
            out = subprocess.check_output(['py', '-c',
                'import shutil; print(shutil.which("jupyter") or "")'],
                timeout=5, text=True).strip()
            if out:
                return out
        except Exception:
            pass
    else:
        candidates += [
            os.path.expanduser('~/.local/bin/jupyter'),
            '/usr/local/bin/jupyter',
            '/usr/bin/jupyter',
            os.path.expanduser('~/anaconda3/bin/jupyter'),
            os.path.expanduser('~/miniconda3/bin/jupyter'),
        ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def get_app_dir():
    """
    Return the directory containing the .exe when frozen with PyInstaller,
    or the script directory when running from Python.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent
    

APP_DIR = get_app_dir()
SETTINGS_FILE = APP_DIR / "files" / "settings.json"


def can_run_jupyterlab(python_exe):
    """
    Return True if this Python executable has JupyterLab installed.
    """
    try:
        result = subprocess.run(
            [str(python_exe), "-m", "jupyterlab", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def find_jupyter_python_automatically():
    """
    Try to find a Python executable that can run JupyterLab.
    """
    candidates = []

    for command in ("jupyter-lab", "jupyter"):
        jupyter_cmd = shutil.which(command)

        if jupyter_cmd:
            jupyter_cmd = Path(jupyter_cmd)

            if os.name == "nt":
                possible_python = jupyter_cmd.parent.parent / "python.exe"
            else:
                possible_python = jupyter_cmd.parent.parent / "bin" / "python"

            candidates.append(possible_python)

    python_from_path = shutil.which("python")
    if python_from_path:
        candidates.append(Path(python_from_path))

    for python_exe in candidates:
        if python_exe and Path(python_exe).exists():
            if can_run_jupyterlab(python_exe):
                return Path(python_exe)

    return None


def load_saved_jupyter_python():
    if not SETTINGS_FILE.exists():
        return None

    try:
        data = json.loads(SETTINGS_FILE.read_text())
        python_exe = data.get("jupyter_python")

        if python_exe and Path(python_exe).exists() and can_run_jupyterlab(python_exe):
            return Path(python_exe)

    except Exception:
        pass

    return None


def save_jupyter_python(python_exe):
    settings = json.loads(SETTINGS_FILE.read_text())
    settings["jupyter_python"] = str(python_exe)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def ask_user_for_python_executable(parent=None):
    """
    Open a PyQt file dialog so the user can select the Python executable
    that has JupyterLab installed.
    """

    if os.name == "nt":
        title = "Select python.exe with JupyterLab installed"
        initial_dir = str(Path.home())
        file_filter = "Python executable (python.exe);;Executable files (*.exe);;All files (*)"
    else:
        title = "Select Python executable with JupyterLab installed"
        initial_dir = "/usr/bin"
        file_filter = "Python executable (python*);;All files (*)"

    selected_file, _ = QFileDialog.getOpenFileName(
        parent,
        title,
        initial_dir,
        file_filter
    )

    if not selected_file:
        return None

    selected_python = Path(selected_file)

    if can_run_jupyterlab(selected_python):
        return selected_python

    QMessageBox.warning(
        parent,
        "JupyterLab not found",
        "The selected Python executable does not seem to have JupyterLab installed.\n\n"
        "Please select a Python environment where JupyterLab is installed, or install it with:\n\n"
        f"{selected_python} -m pip install jupyterlab"
    )

    return None


def get_jupyter_python(parent=None):
    """
    Try, in order:
    1. saved user-selected Python;
    2. automatic detection;
    3. file dialog.
    """

    python_exe = load_saved_jupyter_python()
    if python_exe is not None:
        return python_exe

    python_exe = find_jupyter_python_automatically()
    if python_exe is not None:
        save_jupyter_python(python_exe)
        return python_exe

    QMessageBox.information(
        parent,
        "JupyterLab not found automatically",
        "The notebook was created, but JupyterLab could not be found automatically.\n\n"
        "Please select the Python executable that has JupyterLab installed."
    )

    python_exe = ask_user_for_python_executable(parent)

    if python_exe is not None:
        save_jupyter_python(python_exe)

    return python_exe


def open_notebook_in_jupyterlab(notebook_file, parent=None):
    notebook_file = Path(notebook_file).resolve()

    if not notebook_file.exists():
        QMessageBox.warning(
            parent,
            "Notebook not found",
            f"The notebook file does not exist:\n\n{notebook_file}"
        )
        return

    python_exe = get_jupyter_python(parent)

    if python_exe is None:
        QMessageBox.information(
            parent,
            "Notebook created",
            "The notebook was created, but JupyterLab could not be opened automatically.\n\n"
            f"You can open it manually here:\n\n{notebook_file}"
        )
        return

    try:
        cmd = [
            str(python_exe),
            "-m",
            "jupyterlab",
            str(notebook_file)
        ]

        kwargs = {
            "cwd": str(notebook_file.parent)
        }

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs["start_new_session"] = True
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL

        subprocess.Popen(cmd, **kwargs)

    except Exception as e:
        QMessageBox.critical(
            parent,
            "Error opening notebook",
            "The notebook was created, but JupyterLab could not be opened automatically.\n\n"
            f"Error: {e}"
        )