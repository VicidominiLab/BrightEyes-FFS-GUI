# -*- mode: python ; coding: utf-8 -*-

#from PyInstaller.utils.hooks import collect_submodules

# Collect all submodules in 'functions'
#hiddenimports_folder = collect_submodules('functions')

block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('pyqt_gui_brighteyes_ffs.py', '.'),               # Copies file1 to the root
        ('mplwidget.py', '.'),       # Copies file2 to the 'resources' folder
        ('mplwidgetcorrplot.py', '.'),       # Copies config.json to the 'config' folder
        ('mplwidgetdifflawplot.py', '.'),       # Copies config.json to the 'config' folder
        ('mplwidgetlineplot.py', '.'),       # Copies config.json to the 'config' folder
		('mplwidgetfingerprint.py', '.'),       # Copies config.json to the 'config' folder
		('files/ffs_icon.ico', 'files/ffs_icon.ico'),
		('files/brighteyes_ffs_startup_splash.png', 'files/brighteyes_ffs_startup_splash.png'),
		('files/Cells_DEKegfp_75x75um.jpg', 'files/Cells_DEKegfp_75x75um.jpg'),
		('functions', 'functions')
    ],
    hiddenimports=['matplotlib.backends.backend_qt5agg', 'brighteyes_ffs.fcs_gui.correlation_functions_class', 'functions', 'xsdata_pydantic_basemodel.hooks', 'xsdata_pydantic_basemodel.hooks.class_type'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    bundle_files=1,
    icon='files/ffs_icon.ico',
	splash='files/brighteyes_ffs_startup_splash.png',
	onefile=True,
)
