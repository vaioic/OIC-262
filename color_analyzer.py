import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"
from pathlib import Path, PurePath
from skimage import io, color, measure
from sklearn import cluster, metrics
import numpy as np
import xarray as xr
import pandas as pd

def analyze_color(input_directory_and_labels, output_directory, image_directory=None):

    results = []

    # Process each dataset (folder)
    for input_directory, input_label in input_directory_and_labels:
        
        # Find all cell label files in the input directory. These files are 
        # expected to be saved as PNG files ending in "-labels".

        # Each call of this loop processes all images for a specific experimental condition.
        
        input_path = Path(input_directory)

        if not input_path.exists():
            raise FileNotFoundError(f"No such directory: {input_path}")
        
        # Process each image
        for label_file in input_path.glob('*-labels.png'):

            print(f"Processing file: {label_file}")

            # Get image file
            if image_directory is None:
                image_file = input_path / (label_file.stem[:-7] + '.tif')

            else:
                image_directory = Path(image_directory)

                image_file = image_directory / label_file.parent.parent.name / (label_file.stem[:-7] + '.tif')
            
            if not image_file.exists():
                print(f"Trying jpg name")
                # Try
                image_file = image_file.parent / (label_file.stem[:-7] + '.jpg')
                if not image_file.exists():
                    print(f"Could not find image file. Skipping")
                    continue

            # Read in the image and object labels
            image = io.imread(image_file)
            object_labels = io.imread(label_file)

            # Check if start and cell labels exists
            if not ((1 in object_labels) or (3 in object_labels)):
                print("Skipped")
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

            if len(start_props['centroid-0']) == 0:
                print("No objects found. Skipping.")
                continue


            
            start_centroid = np.array([[start_props['centroid-0'][0], start_props['centroid-1'][0]]])
            
            # {'centroid-0': array([1555.60535021, 2918.63291515]), 'centroid-1': array([1497.46124521, 1551.34955156])}
            cell_props = measure.regionprops_table(cell_labels, properties=['centroid'])
            cell_centroids = np.column_stack((cell_props['centroid-0'], cell_props['centroid-1']))

            cell_distances = np.linalg.norm(cell_centroids - start_centroid, axis=1)

            sorted_indices = np.argsort(cell_distances)

            # Initialize a dict of lists to store data from this image
            cell_data = {
                'cell_label': [],
                'cell_position': [],
                'cell_area_pixels': [],
                'cell_ratio_area_dark': [],
                'mean_hue': [],
                'mean_saturation': [],
                'mean_value': [],
                'mean_lightness': [],
                'mean_A': [],
                'mean_B': [],
                'kmeans_centroid1_L': [],
                'kmeans_centroid1_A': [],
                'kmeans_centroid1_B': [],
                'kmeans_centroid2_L': [],
                'kmeans_centroid2_A': [],
                'kmeans_centroid2_B': [],
                'centroid_distance': [],
                'silhouette_score': [],
                'calinski_harabasz_score': [],
            }
            
            # Process each cell
            pos = 1
            for pos, cell_label in enumerate(sorted_indices):

                cell_data['cell_label'].append(cell_label)
                cell_data['cell_position'].append(f"M-{pos + 1}")

                # Get the HSV values for current cell. Data is N x 3 where N is the pixel
                hsv_values = image_hsv[cell_labels == (cell_label + 1)]
                mean_HSV = np.mean(hsv_values, axis=0)

                cell_data['mean_hue'].append(mean_HSV[0])
                cell_data['mean_saturation'].append(mean_HSV[1])
                cell_data['mean_value'].append(mean_HSV[2])

                lab_values = image_lab[cell_labels == (cell_label + 1)]
                mean_LAB = np.mean(lab_values, axis=0)
                cell_data['mean_lightness'].append(mean_LAB[0])
                cell_data['mean_A'].append(mean_LAB[1])
                cell_data['mean_B'].append(mean_LAB[2])

                # Calculate k-means clustering using the LAB color space. Input should be N x 3, where each row is data from a single pixel.
                kmeans = cluster.KMeans(n_clusters=2, n_init="auto")
                labels = kmeans.fit_predict(lab_values)
                centers = kmeans.cluster_centers_

                cell_data['kmeans_centroid1_L'].append(centers[0, 0])
                cell_data['kmeans_centroid1_A'].append(centers[0, 1])
                cell_data['kmeans_centroid1_B'].append(centers[0, 2])

                cell_data['kmeans_centroid2_L'].append(centers[0, 0])
                cell_data['kmeans_centroid2_A'].append(centers[0, 1])
                cell_data['kmeans_centroid2_B'].append(centers[0, 2])

                cell_data['centroid_distance'].append(np.linalg.norm(centers[0, :] - centers[1, :]))

                cell_data['silhouette_score'].append(metrics.silhouette_score(lab_values, labels, sample_size=3000))
                cell_data['calinski_harabasz_score'].append(metrics.calinski_harabasz_score(lab_values, labels))
                   
                # Calculate percentage of "dark" region vs cell area
                num_pixels_dark_region = np.count_nonzero(mask_dark_regions[cell_labels == (cell_label + 1)])

                cell_data['cell_area_pixels'].append(np.count_nonzero(cell_labels == (cell_label + 1)))
                cell_data['cell_ratio_area_dark'].append(num_pixels_dark_region / np.count_nonzero(cell_labels == 
                (cell_label + 1)))

            # Generate an xarray dataset
            key_list = ["cell_position"]

            num_cells = len(cell_data["cell_position"])

            curr_ds = xr.Dataset(
                data_vars={k: (("id"), v) for k, v in cell_data.items() if k not in key_list},
                coords={
                    "dataset": ("id", [str(image_file.parent.name)] * num_cells),
                    "image": ("id", [image_file.stem] * num_cells),
                    "exp_label": ("id", [input_label] * num_cells),
                    "cell_position": ("id", cell_data["cell_position"])
                }
            )

            # for var in curr_ds.variables:
            #     print(f"Variable: {var:20} | Dtype: {curr_ds[var].dtype}")

            # exit()

            results.append(curr_ds)

       
    # Merge the datasets
    combined_ds = xr.concat(results, dim="id")

    # For debugging if you get netCDF issues
    # for var in ds.data_vars:
    #     print(f"{var}: {ds[var].dtype}")

    # Save to file
    if not isinstance(output_directory, PurePath):
        output_directory = Path(output_directory)

    if not output_directory.is_dir():
        output_directory.mkdir(parents=True)

    combined_ds.to_netcdf(output_directory / "results.nc")

    # Convert to DataFrame and save
    combined_df = combined_ds.to_dataframe()
    combined_df.to_csv(output_directory / "results.csv")

    # Try reading the data
    # print(ds.sel(exp_label="WT con", cell_position="M-1"))

if __name__ == "__main__":
    # main_folder = Path('D:\\Projects\\OIC-262\\data\\single_images')

    # analyze_color([
    #     [main_folder / 'daf2 300 02112026_ mislabled as nduf7', 'daf2 300'],
    #     # [main_folder / 'daf2 con 1_20 02112026', 'daf2 con'],
    #     # [main_folder / 'gsy1 300mM 1_20 02112026', 'gsy1 300'],
    #     # [main_folder / 'gsy1 con 02112026', 'gsy1 con'],
    #     # [main_folder / 'nduf7 300mM 1_20 02112026', 'nduf7 300'],
    #     # [main_folder / 'nudf7 con 02112026', 'nduf7 con'],
    #     # [main_folder / 'wt 300mM 02112026', 'wt 300'],
    #     # [main_folder / 'wt con 02112026', 'wt con'],
    #     # [main_folder / 'wt starve 02112026', 'wt starve']
    #     ], '..\\2026-03-03_test')

    # main_folder = Path('D:\\Projects\\OIC-262 Worm\\data\\Timecourse_Feb 2026\\lugols NaCl timecourse 02202026')

    # analyze_color([
    #     [main_folder / 'wt 0 hr 300mM 02202026', 'wt 0h'],
    #     [main_folder / 'wt 1hr 300mM 02202026', 'wt 1h'],
    #     [main_folder / 'wt 3 hr 300 02202026', 'wt 3h'],
    #     [main_folder / 'wt 6 hr 300 02202026', 'wt 6h'],
    #     [main_folder / 'wt 24 hr 300 02202026', 'wt 24h']
    #     ], '../processed/2026-03-11c')
    
    # main_folder = Path('D:\\Projects\\OIC-262 Worm\\data\\Timecourse_Feb 2026\\lugols NaCl timecourse 02192026')

    # analyze_color([
    #     [main_folder / 'wt 0 hr 300mM 20192026', 'wt 0h'],
    #     [main_folder / 'wt 1 hr 300 mM 02192026', 'wt 1h'],
    #     [main_folder / 'wt 3 hr 300mM 02192026', 'wt 3h'],
    #     [main_folder / 'wt 6hr 300mM 02192026', 'wt 6h'],
    #     [main_folder / 'wt 24hr 300m 02192026', 'wt 24h']
    #     ], '../processed/2026-03-11b lugols NaCl timecourse 02192026')
    
    # main_folder = Path('D:\\Projects\\OIC-262 Worm\\data\\Timecourse_Feb 2026\\lugols 02182026 NaCl timecourse')

    # analyze_color([
    #     [main_folder / 'wt 0 hrs', 'wt 0h'],
    #     [main_folder / 'wt 1 hr', 'wt 1h'],
    #     [main_folder / 'wt 6 hr', 'wt 6h'],
    #     [main_folder / 'wt 300 24hr 02182026', 'wt 24h']
    #     ], '../processed/lugols 02182026 NaCl timecourse')
    
        # [main_folder / 'gsy1 300mM 1_20 02112026', 'gsy1 300'],
        # [main_folder / 'gsy1 con 02112026', 'gsy1 con'],
        # [main_folder / 'nduf7 300mM 1_20 02112026', 'nduf7 300'],
        # [main_folder / 'nudf7 con 02112026', 'nduf7 con'],
        # [main_folder / 'wt 300mM 02112026', 'wt 300'],
        # [main_folder / 'wt con 02112026', 'wt con'],
        # [main_folder / 'wt starve 02112026', 'wt starve']

    # main_folder = Path(r'D:\Projects\OIC-262 Worm\data\Germline RNAi\20feb26')

    # analyze_color([
    #     [main_folder / 'ctrl RNAi', 'ctrl'],
    #     [main_folder / 'ctrl rnai 300', 'ctrl 300'],
    #     [main_folder / 'gsy-1 rnai', 'gsy-1'],
    #     [main_folder / 'gsy-1 rnai 300', 'gsy-1 300'],
    #     [main_folder / 'mpk-1 rnai', 'mpk-1'],
    #     [main_folder / 'mpk-1 rnai 300', 'mpk-1 300'],
    #     [main_folder / 'pygl-1 rnai', 'pygl-1'],
    #     [main_folder / 'pygl-1 rnai 300', 'pygl-1 300']
    #     ], '../processed/2026-03-12 Germline RNAi/20feb26')

    # main_folder = Path(r'D:\Projects\OIC-262 Worm\data\Germline RNAi\lugols 02142026 1_20')

    # analyze_color([
    #     [main_folder / 'EV RNAi', 'ctrl'],
    #     [main_folder / 'EV RNAi 300', 'ctrl 300'],
    #     [main_folder / 'gsy1 RNAi', 'gsy-1'],
    #     [main_folder / 'gsy1 RNAi 300', 'gsy-1 300'],
    #     [main_folder / 'pygl1 rnai', 'pygl-1'],
    #     [main_folder / 'pygl1 rnai 300', 'pygl-1 300']
    #     ], '../processed/2026-03-12 Germline RNAi/lugols 02142026 1_20')

    # main_folder = Path(r'D:\Projects\OIC-262 Worm\QuPath projects\wt vs flcn-1\lugols 17 feb 26 flcn fzo1')

    # analyze_color([
    #     [main_folder / 'flcn-1' / 'export', 'flcn-1'],
    #     [main_folder / 'flcn-1 300 mm nacl' / 'export', 'flcn-1 300'],
    #     [main_folder / 'fzo-1' / 'export', 'fzo-1'],
    #     [main_folder / 'fzo-1 300 mm nacl' / 'export', 'flcn-1 300'],
    #     [main_folder / 'wt' / 'export', 'wt'],
    #     [main_folder / 'wt 300' / 'export', 'wt 300']
    #     ], r'../processed/2026-04-20 wt vs flcn-1/lugols 17 feb 26 flcn fzo1',
    #     image_directory=r'\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Data\wt vs flcn-1\lugols 17 feb 26 flcn fzo1')

    # main_folder = Path(r'D:\Projects\OIC-262 Worm\QuPath projects\wt vs flcn-1\lugols 02112026 1_20 with m9')

    # analyze_color([
    #     [main_folder / 'FLCN-1' / 'export', 'flcn-1'],
    #     [main_folder / 'FLCN-1 300' / 'export', 'flcn-1 300'],
    #     [main_folder / 'FZO-1' / 'export', 'fzo-1'],
    #     [main_folder / 'FZO-1 300' / 'export', 'flcn-1 300'],
    #     [main_folder / 'WT' / 'export', 'wt'],
    #     [main_folder / 'WT 300' / 'export', 'wt 300']
    #     ], r'../processed/2026-04-20 wt vs flcn-1/lugols 17 feb 26 flcn fzo1',
    #     image_directory=r'\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Data\wt vs flcn-1\lugols 02112026 1_20 with m9')
    
    # main_folder = Path(r'D:\Projects\OIC-262 Worm\QuPath projects\wt vs etc mutants\lugols 02132026 1_20')

    # analyze_color([
    #     [main_folder / 'germline isp-1 300' / 'export', 'germline isp-1 300'],
    #     [main_folder / 'gsy1' / 'export', 'gsy-1'],
    #     [main_folder / 'isp1' / 'export', 'isp-1'],
    #     [main_folder / 'isp1 germline' / 'export', 'germline isp-1'],
    #     [main_folder / 'nduf7' / 'export', 'nduf7'],
    #     [main_folder / 'nduf7 300' / 'export', 'nduf7 300'],
    #     [main_folder / 'wt 300' / 'export', 'wt 300'],
    #     [main_folder / 'wt con 02132026' / 'export', 'wt']
    #     ], r'../processed/2026-04-20 wt vs etc mutants/lugols 02132026 1_20',
    #     image_directory=r'\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Data\wt vs etc mutants\lugols 02132026 1_20')
    
    # main_folder = Path(r'D:\Projects\OIC-262 Worm\QuPath projects\wt vs etc mutants\lugols 02172026')

    # analyze_color([
    #     [main_folder / 'isp 300mM 02172026' / 'export', 'isp 300'],
    #     [main_folder / 'isp con 02172026' / 'export', 'isp'],
    #     [main_folder / 'isp1 germline  300 mm 02172026' / 'export', 'germline isp-1 300'],
    #     [main_folder / 'isp1 germline con 02172026' / 'export', 'germline isp-1'],
    #     [main_folder / 'nduf7 300 02172026' / 'export', 'nduf7 300'],
    #     [main_folder / 'nduf7 con 02172026' / 'export', 'nduf7'],
    #     [main_folder / 'wt 300mM 02172026' / 'export', 'wt 300'],
    #     [main_folder / 'wt con 02172026' / 'export', 'wt']
    #     ], r'../processed/2026-04-20 wt vs etc mutants/lugols 02172026',
    #     image_directory=r'\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Data\wt vs etc mutants\lugols 02172026')
    
    main_folder = Path(r'D:\Projects\OIC-262 Worm\QuPath projects\Germline RNAi 3rd repeat')

    analyze_color([
        [main_folder / 'ctrl RNAi' / 'export', 'ctrl RNAi'],
        [main_folder / 'ctrl RNAi 300' / 'export', 'ctrl RNAi 300'],
        [main_folder / 'gsy-1 RNAi' / 'export', 'gsy-1 RNAi'],
        [main_folder / 'gsy-1 RNAi 300' / 'export', 'gsy-1 RNAi 300'],
        [main_folder / 'pygl-1 RNAi' / 'export', 'pygl-1 RNAi'],
        [main_folder / 'pygl-1 RNAi 300' / 'export', 'pygl-1 RNAi 300']
        ], r'../processed/2026-04-20 Germline RNAi 3rd repeat',
        image_directory=r'\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Data\Germline RNAi 3rd repeat')
