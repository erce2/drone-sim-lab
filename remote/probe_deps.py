import importlib
import shutil

for m in ["matplotlib", "imageio", "tensorboard", "numpy", "torch", "gymnasium"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:14s} OK   {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:14s} MISSING ({type(e).__name__})")

print("ffmpeg:", shutil.which("ffmpeg"))