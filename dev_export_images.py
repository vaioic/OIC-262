from bioio import BioImage
import bioio_czi
from matplotlib import pyplot as plt
import skimage
import numpy as np

img = BioImage('..//data//Jason//isp1.czi', use_aicspylibczi=True)

print(f"Image shape: {img.data.shape}")
print(f"Image scenes: {img.scenes}")
print(f"Dimension order: {img.dims.order}")
print(f"Physical pixel sizes: {img.physical_pixel_sizes}")

img.set_scene(0)
data = (img.data).squeeze()

# Re-order the third axis
new_data = np.zeros_like(data)
new_data[..., 0] = data[..., 2]
new_data[..., 1] = data[..., 1]
new_data[..., 2] = data[..., 0]

skimage.io.imsave("..\\data\\" + img.current_scene + "Image.tiff", new_data)

img.set_scene(1)
data = (img.data).squeeze()

# Re-order the third axis
new_data = np.zeros_like(data)
new_data[..., 0] = data[..., 2]
new_data[..., 1] = data[..., 1]
new_data[..., 2] = data[..., 0]
skimage.io.imsave("..\\data\\" + img.current_scene + "Image.tiff", new_data)


img.set_scene(2)
data = (img.data).squeeze()

# Re-order the third axis
new_data = np.zeros_like(data)
new_data[..., 0] = data[..., 2]
new_data[..., 1] = data[..., 1]
new_data[..., 2] = data[..., 0]
skimage.io.imsave("..\\data\\" + img.current_scene + "Image.tiff", new_data)


# plt.imshow(img.data[..., 1].squeeze())
# plt.show()

