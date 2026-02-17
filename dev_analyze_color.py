from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
import skimage
import numpy as np
import sklearn
from pathlib import Path
import csv

# mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\2026-02-10 Jason Exported\\daf2\\qupath\\export')

# image_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\2026-02-10 Jason Exported\\daf2\\daf2')

mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\daf2 con 1_20 02112026\\qupath\\export')

image_directory =  Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\daf2 con 1_20 02112026')

output_directory = Path('..\\processed\\2026-02-17\\daf2 con')

# ---Begin processing code---
output_directory.mkdir(exist_ok=True)

results = []

for mask_file in mask_directory.glob('*-labels.png'):

    # Get image file    
    image_file = image_directory / (mask_file.stem[:-7] + '.tif')

    # Read in images
    image = skimage.io.imread(image_file)
    # print(image.dtype)
    mask = skimage.io.imread(mask_file)

    # Get cells
    cell_labels, nLabels = skimage.measure.label(mask == 1, return_num=True)

    # Convert image to HSV color space for color analysis
    image_hsv = skimage.color.rgb2hsv(image)

    # Create a List to store output data

    # Quantify the HSV in each mask
    for obj_id in range(nLabels):

        # Get the HSV values for current cell. Data is N x 3 where N is the pixel
        hsv_values = image_hsv[cell_labels == (obj_id + 1)]
        # print(hsv_values.shape)

        # Calculate the average H, S, V
        mean_HSV = np.mean(hsv_values, axis=0)
        # print(mean_HSV)

        # Calculate k-means clustering. Input should be N x 3.
        kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init="auto")
        labels = kmeans.fit_predict(hsv_values)
        centers = kmeans.cluster_centers_

        #print(f"k-means centers: {centers}")

        # Generate a dictionary of the measured data
        results.append({
            'filename': image_file.stem,
            'cell': obj_id,
            'mean_hue': mean_HSV[0],
            'mean_saturation': mean_HSV[1],
            'mean_value': mean_HSV[2],
            'kmeans_centroid1': centers[0, :],
            'kmeans_centroid2': centers[1, :]
        })

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        obj_mask = cell_labels == (obj_id + 1)
        outline = skimage.segmentation.find_boundaries(obj_mask, mode='thick')
        outline = skimage.morphology.dilation(outline, skimage.morphology.disk(4))

        image_out = image.copy()

        image_out_r = image_out[:, :, 0]
        image_out_g = image_out[:, :, 1]
        image_out_b = image_out[:, :, 2]

        image_out_r[outline] = 0
        image_out_g[outline] = 255
        image_out_b[outline] = 0

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image_out)
        
        ax1 = fig.add_subplot(gs[0, 1:2], projection='3d')

        # Pick a random selection of points
        rng = np.random.default_rng()
        pts = rng.integers(low=0, high=hsv_values.shape[0], size=200)

        # Scatter plot of the data points, colored by their assigned cluster label
        ax1.scatter(hsv_values[pts, 0], hsv_values[pts, 1], hsv_values[pts, 2], c=labels[pts], cmap='viridis', s=50, alpha=0.6)

        # Plot the centroids as well
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=200, c='red', label='Centroids')

        ax1.set_title('K-Means Clustering')
        ax1.set_xlabel('Hue')
        ax1.set_ylabel('Saturation')
        ax1.set_zlabel('Value/Brightness')
        #ax1.legend()

        # Make an inset that shows rectangles of the centroid colors
        rgb_color1 = skimage.color.hsv2rgb(centers[0, :])
        rgb_color2 = skimage.color.hsv2rgb(centers[1, :])
        
        inset1 = ax1.inset_axes([1.5, 0.5, 0.3, 0.3]) 
        box1 = mpatches.Rectangle((0.1, 0.1), 0.35, 0.35, color=rgb_color1, transform=inset1.transAxes)
        
        inset1.add_patch(box1)   
        
        inset1.text(0.275, 0.0, f'HSV={np.round(centers[0, :], 2)}', ha='center', va='top', transform=inset1.transAxes, fontsize=8)
        
        box2 = mpatches.Rectangle((0.1, 0.70), 0.35, 0.35, color=rgb_color2, transform=inset1.transAxes)
        inset1.add_patch(box2)
        inset1.text(0.275, 0.655, f'HSV={np.round(centers[1, :], 2)}', ha='center', va='top',  transform=inset1.transAxes, fontsize=8)

        inset1.set_xlim(0, 1)
        inset1.set_ylim(0, 1)
        inset1.axis('off') # Hide the axes ticks and spines

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(hsv_values[pts, 0], hsv_values[pts, 1], c=labels[pts], cmap='viridis')
        ax2.scatter(centers[:, 0], centers[:, 1], marker='X', c='red')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Hue')
        ax2.set_ylabel('Saturation')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(hsv_values[pts, 0], hsv_values[pts, 2], c=labels[pts], cmap='viridis')
        ax3.scatter(centers[:, 0], centers[:, 2], marker='X', c='red')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Hue')
        ax3.set_ylabel('Value')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(hsv_values[pts, 1], hsv_values[pts, 2], c=labels[pts], cmap='viridis')
        ax4.scatter(centers[:, 1], centers[:, 2], marker='X', c='red')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('Saturation')
        ax4.set_ylabel('Value')

        fig.savefig(output_directory / (mask_file.stem[:-7] + "_cell" + f"{obj_id + 1}" + ".jpg"))
        #plt.show()

        fig.clf()

if results:
    keys = results[0].keys()
    with open(output_directory / 'results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)