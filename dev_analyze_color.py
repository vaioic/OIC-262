from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
import skimage
import numpy as np
import sklearn

# image_file = '..\\data\\WT-300\\P5-Image.tiff'
# mask_file = '..\\data\\WT-300\\qupath\\export\\P5-Image-labels.png'

image_file = '..\\data\\isp1\\P14-Image.tiff'
mask_file = '..\\data\\isp1\\qupath\\export\\P14-Image-labels.png'

image = skimage.io.imread(image_file)
mask = skimage.io.imread(mask_file)

# Should I be doing this or does this mess up the original image?
# image = (image - np.min(image)) / (np.max(image) - np.min(image))

# plt.imshow(image)
# plt.show()

# Note: Mask is in RGB
mask_gray = skimage.color.rgb2gray(mask)
mask_gray = skimage.util.img_as_ubyte(mask_gray)
# plt.imshow(mask_gray)
# plt.show()

#Get values: [ 0 91 171 193]
unique = np.unique(mask_gray)
unique = np.delete(unique, np.where(unique == 0))

image_hsv = skimage.color.rgb2hsv(image)

# Quantify the HSV in each mask
for obj_id in unique:

    hue_channel = (image_hsv[..., 0]).squeeze()
    sat_channel = (image_hsv[..., 1]).squeeze()
    value_channel = (image_hsv[..., 2]).squeeze()

    # Calculate average of each metric
    mean_hue = np.mean(hue_channel[np.where(mask_gray == obj_id)])

    print(mean_hue)

    # KMeans expects each row to be a pixel and the coluns to be H, S, V
    X = image_hsv.reshape((-1, 3))
    
    # Try k-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init="auto")
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    print(f"k-means centers: {centers}")

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, :], projection='3d')

    # Pick a random selection of points
    rng = np.random.default_rng()
    pts = rng.integers(low=0, high=X.shape[0], size=200)

    # Scatter plot of the data points, colored by their assigned cluster label
    ax1.scatter(X[pts, 0], X[pts, 1], X[pts, 2], c=labels[pts], cmap='viridis', s=50, alpha=0.6)

    # Plot the centroids as well
    ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=200, c='red', label='Centroids')

    ax1.set_title('K-Means Clustering')
    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Saturation')
    ax1.set_zlabel('Value/Brightness')
    ax1.legend()

    # Make an inset that shows rectangles of the centroid colors
    rgb_color1 = mcolors.hsv_to_rgb(centers[0, :])
    rgb_color2 = mcolors.hsv_to_rgb(centers[1, :])
    
    inset1 = ax1.inset_axes([0.65, 0.65, 0.3, 0.3]) 
    box1 = mpatches.Rectangle((0.1, 0.1), 0.35, 0.8, color=rgb_color1, transform=inset1.transAxes)
    box2 = mpatches.Rectangle((0.55, 0.1), 0.35, 0.8, color=rgb_color2, transform=inset1.transAxes)

    inset1.add_patch(box1)
    inset1.add_patch(box2)
    inset1.text(0.275, 0.05, f'HSV={centers[0,:]}', ha='center', va='top', transform=inset1.transAxes, fontsize=8)
    inset1.text(0.725, 0.05, f'HSV={centers[1,:]}', ha='center', va='top',  transform=inset1.transAxes, fontsize=8)


    inset1.set_xlim(0, 1)
    inset1.set_ylim(0, 1)
    inset1.axis('off') # Hide the axes ticks and spines

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(X[pts, 0], X[pts, 1], c=labels[pts], cmap='viridis')
    ax2.scatter(centers[:, 0], centers[:, 1], marker='X', c='red')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Saturation')


    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(X[pts, 0], X[pts, 2], c=labels[pts], cmap='viridis')
    ax3.scatter(centers[:, 0], centers[:, 2], marker='X', c='red')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Hue')
    ax3.set_ylabel('Value')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(X[pts, 1], X[pts, 2], c=labels[pts], cmap='viridis')
    ax4.scatter(centers[:, 1], centers[:, 2], marker='X', c='red')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Saturation')
    ax4.set_ylabel('Value')

    plt.show()

    exit()