import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"
from pathlib import Path, PurePath
from skimage import io, color, measure
from sklearn import cluster, metrics
import numpy as np
import xarray as xr
import pandas as pd

def analyze_color(input_directory_and_labels, output_directory):

    print(input_directory_and_labels)

    results = []

    for input_directory, input_label in input_directory_and_labels:

        # Find all cell label files in the input directory. These files are 
        # expected to be saved as PNG files ending in "-labels".

        input_path = Path(input_directory)        
        
        for label_file in input_path.glob('*-labels.png'):

            # Get image file
            image_file = input_path / (label_file.stem[:-7] + '.tif')

            # Read in the image and object labels
            image = io.imread(image_file)
            object_labels = io.imread(label_file)

            # Check if start and cell labels exists
            if not ((1 in object_labels) or (3 in object_labels)):
                continue

            # Get the cell mask and re-label
            cell_labels, nCells = measure.label(object_labels == 1, return_num=True)

            # Convert image to HSV and LAB color space for color analysis
            image_hsv = color.rgb2hsv(image)
            image_lab = color.rgb2lab(image)

            # Get the "Start" label (3)
            start_label = measure.label(object_labels == 3)

            # Find dark regions
            mask_dark_regions = image_lab[..., 0] < 30
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask_dark_regions)
            # plt.show()

            # Measure the centroid position - this is a dict with keys 'centroid-0' and 'centroid-1'
            start_props = measure.regionprops_table(start_label, properties=['centroid'])
            print(label_file, start_props)
            start_centroid = np.array([[start_props['centroid-0'][0], start_props['centroid-1'][0]]])
            
            # {'centroid-0': array([1555.60535021, 2918.63291515]), 'centroid-1': array([1497.46124521, 1551.34955156])}
            cell_props = measure.regionprops_table(cell_labels, properties=['centroid'])
            cell_centroids = np.column_stack((cell_props['centroid-0'], cell_props['centroid-1']))

            cell_distances = np.linalg.norm(cell_centroids - start_centroid, axis=1)

            sorted_indices = np.argsort(cell_distances)

            # Quantify color for each cell
            pos = 0
            for obj_id in sorted_indices:

                cell_position = pos
                pos += 1

                # Get the HSV values for current cell. Data is N x 3 where N is the pixel
                hsv_values = image_hsv[cell_labels == (obj_id + 1)]
                lab_values = image_lab[cell_labels == (obj_id + 1)]
                # print(hsv_values.shape)

                # Calculate the average H, S, V, L, A, B
                mean_HSV = np.mean(hsv_values, axis=0)
                mean_LAB = np.mean(lab_values, axis=0)
                # print(mean_HSV)

                # Calculate k-means clustering using the LAB color space. Input should be N x 3, where each row is data from a single pixel.
                kmeans = cluster.KMeans(n_clusters=2, n_init="auto")
                labels = kmeans.fit_predict(lab_values)
                centers = kmeans.cluster_centers_

                # Calculate percentage of dark region vs cell area
                num_pixels_dark_region = np.count_nonzero(mask_dark_regions[cell_labels == (obj_id + 1)])
                

                # Generate a dictionary of the measured data and append to list
                results.append({
                    'image': image_file.stem,
                    'exp_label': input_label,
                    'cell_id': obj_id,
                    'cell_position': f"M-{cell_position + 1}",
                    'cell_area_pixels': np.count_nonzero(cell_labels == (obj_id + 1)),
                    'cell_ratio_area_dark': num_pixels_dark_region / np.count_nonzero(cell_labels == (obj_id + 1)),
                    'mean_hue': mean_HSV[0],
                    'mean_saturation': mean_HSV[1],
                    'mean_value': mean_HSV[2],
                    'mean_lightness': mean_LAB[0],
                    'mean_A': mean_LAB[1],
                    'mean_B': mean_LAB[2],
                    'kmeans_centroid1_L': centers[0, 0],
                    'kmeans_centroid1_A': centers[0, 1],
                    'kmeans_centroid1_B': centers[0, 2],
                    'kmeans_centroid2_L': centers[0, 0],
                    'kmeans_centroid2_A': centers[0, 1],
                    'kmeans_centroid2_B': centers[0, 2],
                    'centroid_distance': np.linalg.norm(centers[0, :] - centers[1, :]),
                    'silhouette_score': metrics.silhouette_score(lab_values, labels, sample_size=3000),
                    'Calinski-Harabasz_score': metrics.calinski_harabasz_score(lab_values, labels)
                })

    # Create a dataset after processing all the datasets
    df = pd.DataFrame(results)

    ds = df.set_index(["image", "exp_label", "cell_position"]).to_xarray()

    # For debugging if you get netCDF issues
    # for var in ds.data_vars:
    #     print(f"{var}: {ds[var].dtype}")

    # Save to file
    if not isinstance(output_directory, PurePath):
        output_directory = Path(output_directory)

    if not output_directory.is_dir():
        output_directory.mkdir()

    ds.to_netcdf(output_directory / "results.nc")
    df.to_csv(output_directory / "results.csv")

    # Try reading the data
    # print(ds.sel(exp_label="WT con", cell_position="M-1"))
    exit()


if __name__ == "__main__":
    main_folder = Path('D:\\Projects\\OIC-262\\data\\single_images')

    analyze_color([
        [main_folder / 'daf2 300 02112026_ mislabled as nduf7', 'daf2 300'],
        # [main_folder / 'daf2 con 1_20 02112026', 'daf2 con'],
        # [main_folder / 'gsy1 300mM 1_20 02112026', 'gsy1 300'],
        # [main_folder / 'gsy1 con 02112026', 'gsy1 con'],
        # [main_folder / 'nduf7 300mM 1_20 02112026', 'nduf7 300'],
        # [main_folder / 'nudf7 con 02112026', 'nduf7 con'],
        # [main_folder / 'wt 300mM 02112026', 'wt 300'],
        # [main_folder / 'wt con 02112026', 'wt con'],
        # [main_folder / 'wt starve 02112026', 'wt starve']
        ], '..\\2026-03-03_test')