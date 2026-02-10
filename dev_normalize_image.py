from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
import skimage
import numpy as np
import sklearn
from pylibCZIrw import czi as pyczi
# import bioio

# with pyczi.open_czi('..//data//Jason//isp1.czi') as czidoc:

# #     with open('metadata.txt', 'w') as file:

# #         file.write(czidoc.raw_metadata)

# #     # for prop, val in (czidoc.metadata).items():
# #     #     print(f"{prop}: {val}")

#     pixel_type = czidoc.pixel_types
#     print(f"Pixel Type: {pixel_type}")

#     print(f"Bounding box: {czidoc.total_bounding_box}")
#     print(f"Bounding box: {czidoc.total_bounding_rectangle}")

#     image = czidoc.read(plane={'C': 0, 'Z': 0, 'T': 0, 'S': 0}, roi=(0, 0, 1024, 1024))

#     plt.imshow(image)
#     plt.show()

# exit()
balance = [0.28010405233057, 0.57116589093515, 0.24675315098362]

# balance = [0.24675315098362, 0.57116589093515, 0.28010405233057]


image_file = '..\\data\\isp1\\P14-Image.tiff'
mask_file = '..\\data\\isp1\\qupath\\export\\P14-Image-labels.png'

image = skimage.io.imread(image_file)
print(f"Dtype: {image.dtype}")
mask = skimage.io.imread(mask_file)

# Try normalizing RGB
image_norm = image.astype(np.float32)

for c in range(3):
    image_norm[..., c] *= balance[c]
    # image_norm[..., c] = (image_norm[..., c] - np.min(image_norm[..., c])) / (np.max(image_norm[..., c]) - np.min(image_norm[..., c]))
    
image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))

# image_norm = np.clip(image_norm, 0, 255).astype(np.uint8)

# image_norm = np.clip(image_norm, 0, 1)

plt.imshow(image_norm)
plt.show()

# Get the gray region mask
gray_mask = mask == 2


