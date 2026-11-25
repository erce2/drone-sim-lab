"""校验生成的 PNG 不是空白图 (统计像素方差)。"""

import glob
import os

from PIL import Image, ImageStat

for p in sorted(glob.glob("C:/work/drone_lab/out/*.png")):
    im = Image.open(p).convert("L")
    st = ImageStat.Stat(im)
    print(f"{os.path.basename(p):32s} size={im.size} mean={st.mean[0]:6.1f} stddev={st.stddev[0]:6.2f}")