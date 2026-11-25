"""检查 imageio 的 ffmpeg 后端是否可用 (录视频需要)。"""

import shutil

try:
    import imageio_ffmpeg

    print("imageio_ffmpeg OK", imageio_ffmpeg.get_ffmpeg_exe())
except Exception as e:  # noqa: BLE001
    print("imageio_ffmpeg MISSING:", e)

print("ffmpeg in PATH:", shutil.which("ffmpeg"))