from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patches as mpatches
from matplotlib import colors as mcolors
import skimage
import numpy as np
import sklearn
from pathlib import Path
import csv
from tqdm import tqdm

# mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\2026-02-10 Jason Exported\\daf2\\qupath\\export')

# image_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\2026-02-10 Jason Exported\\daf2\\daf2')

# mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\daf2 con 1_20 02112026\\qupath\\export')

# image_directory =  Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\daf2 con 1_20 02112026')

# output_directory = Path('..\\processed\\2026-02-18\\daf2_con')

# # ---

mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\wt con 02112026\\qupath\\export')

image_directory =  Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\wt con 02112026')

output_directory = Path('..\\processed\\2026-02-18\\wt con 02112026')

# # ---

# mask_directory = Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\wt starve 02112026\\qupath\\export')

# image_directory =  Path('\\\\pn.vai.org\\projects\\burton\\VARI CORE GENERATED DATA\\OIC\\Oocyte glycogen staining\\Data\\lugols 1_20 02112026\\wt starve 02112026')

# output_directory = Path('..\\processed\\2026-02-18\\wt starve 02112026')

# ---Begin processing code---
output_directory.mkdir(parents=True, exist_ok=True)

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

    # Convert image to HSV and LAB color space for color analysis
    image_hsv = skimage.color.rgb2hsv(image)
    image_lab = skimage.color.rgb2lab(image)

    # Create a List to store output data

    # Quantify the HSV in each mask
    for obj_id in tqdm(range(nLabels)):

        # Get the HSV values for current cell. Data is N x 3 where N is the pixel
        hsv_values = image_hsv[cell_labels == (obj_id + 1)]
        lab_values = image_lab[cell_labels == (obj_id + 1)]
        # print(hsv_values.shape)

        # Calculate the average H, S, V, L, A, B
        mean_HSV = np.mean(hsv_values, axis=0)
        mean_LAB = np.mean(lab_values, axis=0)
        # print(mean_HSV)

        # Calculate k-means clustering using the LAB color space. Input should be N x 3.
        kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init="auto")
        labels = kmeans.fit_predict(lab_values)
        centers = kmeans.cluster_centers_

        #print(f"k-means centers: {centers}")

        # Generate a dictionary of the measured data
        results.append({
            'filename': image_file.stem,
            'cell': obj_id,
            'mean_hue': mean_HSV[0],
            'mean_saturation': mean_HSV[1],
            'mean_value': mean_HSV[2],
            'mean_lightness': mean_LAB[0],
            'mean_A': mean_LAB[1],
            'mean_B': mean_LAB[2],
            'kmeans_LAB_centroid1': centers[0, :],
            'kmeans_LAB_centroid2': centers[1, :],
            'centroid_distance': np.linalg.norm(centers[0, :] - centers[1, :]),
            'silhouette_score': sklearn.metrics.silhouette_score(lab_values, labels, sample_size=3000),
            'Calinski-Harabasz_score': sklearn.metrics.calinski_harabasz_score(lab_values, labels)
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
        pts = rng.integers(low=0, high=lab_values.shape[0], size=400)

        # Scatter plot of the data points, colored by their assigned cluster label
        ax1.scatter(lab_values[pts, 0], lab_values[pts, 1], lab_values[pts, 2], c=labels[pts], cmap='viridis', s=25, alpha=0.5)

        # Plot the centroids as well
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=75, c='red', label='Centroids')

        ax1.set_title('K-Means Clustering')
        ax1.set_xlabel('Lightness')
        ax1.set_ylabel('a*')
        ax1.set_zlabel('b*')
        #ax1.legend()

        # Make an inset that shows rectangles of the centroid colors
        # rgb_color1 = skimage.color.hsv2rgb(centers[0, :])
        # rgb_color2 = skimage.color.hsv2rgb(centers[1, :])
        rgb_color1 = skimage.color.lab2rgb(centers[0, :])
        rgb_color2 = skimage.color.lab2rgb(centers[1, :])
        
        inset1 = ax1.inset_axes([1.5, 0.5, 0.3, 0.3]) 
        box1 = mpatches.Rectangle((0.1, 0.1), 0.35, 0.35, color=rgb_color1, transform=inset1.transAxes)
        
        inset1.add_patch(box1)   
        
        inset1.text(0.275, 0.0, f'LAB={np.round(centers[0, :], 2)}', ha='center', va='top', transform=inset1.transAxes, fontsize=8)
        
        box2 = mpatches.Rectangle((0.1, 0.70), 0.35, 0.35, color=rgb_color2, transform=inset1.transAxes)
        inset1.add_patch(box2)
        inset1.text(0.275, 0.655, f'LAB={np.round(centers[1, :], 2)}', ha='center', va='top',  transform=inset1.transAxes, fontsize=8)
        inset1.set_xlim(0, 1)
        inset1.set_ylim(0, 1)
        inset1.axis('off') # Hide the axes ticks and spines

        # Chroma
        chroma = np.sqrt((lab_values[pts, 1] ** 2) + (lab_values[pts, 2] ** 2))
        chroma_center_1 = np.sqrt(centers[0, 1] ** 2 + centers[0, 2] ** 2)
        chroma_center_2 = np.sqrt(centers[1, 1] ** 2 + centers[1, 2] ** 2)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(chroma, lab_values[pts, 0], c=labels[pts], cmap='viridis')
        ax2.scatter(chroma_center_1, centers[0, 0], marker='X', c='red')
        ax2.scatter(chroma_center_2, centers[1, 0], marker='X', c='red')
        # ax2.set_xlim(0, 1)
        # ax2.set_ylim(0, 1)
        ax2.set_xlabel('Chroma')
        ax2.set_ylabel('Lightness')

        # ax3 = fig.add_subplot(gs[1, 1])
        # ax3.scatter(lab_values[pts, 0], lab_values[pts, 2], c=labels[pts], cmap='viridis')
        # ax3.scatter(centers[:, 0], centers[:, 2], marker='X', c='red')
        # # ax3.set_xlim(0, 1)
        # # ax3.set_ylim(0, 1)
        # ax3.set_xlabel('Lightness')
        # ax3.set_ylabel('b*')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(lab_values[pts, 2], lab_values[pts, 1], c=labels[pts], cmap='viridis')
        ax4.scatter(centers[:, 2], centers[:, 1], marker='X', c='red')
        # ax4.set_xlim(0, 1)
        # ax4.set_ylim(0, 1)
        ax4.set_xlabel('b*')
        ax4.set_ylabel('a*')

        fig.savefig(output_directory / (mask_file.stem[:-7] + "_cell" + f"{obj_id + 1}" + ".jpg"))
        #plt.show()

        fig.clf()

if results:
    keys = results[0].keys()
    with open(output_directory / 'results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)