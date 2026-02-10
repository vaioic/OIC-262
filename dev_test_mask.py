from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
import skimage
import numpy as np
import sklearn

image_file = '..\\data\\isp1\\P14-Image.tiff'
mask_file = '..\\data\\isp1\\qupath\\export\\P14-Image-labels.png'

mask = skimage.io.imread(mask_file)

print(f"dtype: {mask.dtype}")
print(f"max: {np.max(mask)}")
print(f"min: {np.min(mask)}")

cell_mask = mask == 1
cell_labels, nlabels = skimage.measure.label(cell_mask, return_num=True)

print(f"nLabels: {nlabels}")

