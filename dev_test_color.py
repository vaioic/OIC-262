import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io
from skimage.color import rgb2hsv
from pathlib import Path

rgb_img = data.coffee()

print(f"Max: {rgb_img.dtype}")
print(f"Max: {np.max(rgb_img)}")

image_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\2026-02-10 Jason Exported\\daf2\\daf2')

img = io.imread(image_directory / "daf2_s01.tif")
print(f"Max: {img.dtype}")
print(f"Max: {np.max(img)}")