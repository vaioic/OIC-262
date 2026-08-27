"""'
Color analysis of RGB images.

LugolQuant provides utilities for the processing of images to obtain chromatic (color)
information. This module was originally written to quantify color variations in
brightfield images of Lugol-stained C. elegans.

Please see the README.md for more details about expected file types and directory structure.

Examples
--------
>>> from shared import lugolquant
>>> lugolquant.process_directory(r"path/to/data", r"path/to/output")
"""

import json
import os
import re
from pathlib import Path

from oic_toolkit import display

os.environ["OPENBLAS_NUM_THREADS"] = "24"

from warnings import deprecated

import numpy as np
import xarray as xr
from skimage import color, io, measure, segmentation
from sklearn import metrics
from tqdm import tqdm


def process_directory(base_input_path, base_output_path):
    """
    Process image datasets from a base path.

    The function will identify and process all valid datasets in folders below the
    ``base_input_path``. It is expected that each directory will contain a folder named
    ``export`` that contains the exported masks from the images.

    The function will call the ``process_files`` for the valid folder. The output of
    this function is a single CSV-file with all measurements.

    Parameters
    ----------
    base_input_path : str or Path
        Path to the top-level data directory
    base_output_path : str or Path
        Path to save data to
    """

    # Validate the inputs
    base_input_path = Path(base_input_path)
    full_base_input_path = base_input_path.resolve()

    base_output_path = Path(base_output_path)
    full_base_output_path = base_output_path.resolve()

    # Find all folders named "export" in the main input path
    folder_list = [p for p in base_input_path.rglob("export") if p.is_dir()]

    # Initialize a list to hold measured data
    all_data = []

    for folder in tqdm(folder_list, desc="Overall progress"):
        # Get the source and output folder paths relative to the base_input_path
        source_folder_relative = (
            folder.resolve().relative_to(full_base_input_path).parent
        )
        output_path = full_base_output_path / source_folder_relative

        # Process the images in the folder
        data = process_files(
            folder, output_path, save_data=False, exp_label=str(source_folder_relative)
        )

        # Store data in main list
        all_data.append(data)

    # Combine all the data in an xarray dataset. This format is helpful for plotting in
    # Python.
    combined_ds = xr.concat(all_data, dim="id")
    combined_ds.to_netcdf(output_path / "overall_results.nc")

    # Convert to a pandas DataFrame
    combined_df = combined_ds.to_dataframe()

    # Reorder the columns
    front = ["image", "dataset", "exp_label"]
    remaining = [col for col in combined_df.columns if col not in front]

    combined_df = combined_df[front + remaining]

    combined_df.to_csv(base_output_path / "overall_results.csv", index=False)


@deprecated("This function may no longer work. Use `process_directory()` instead.")
def process_dataset(input_path, output_path):
    """
    Process all images in a directory.

    This function is intended to be used to process a dataset, defined as the top-level directory containing sub-directories with experimental data. The output of the function will return a single CSV-file.

    Parameters
    ----------
    input_path : str or Path
        Path to the dataset folder
    output_path : str ot Path
        Path to write output
    """

    # Validate the inputs
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Get sub-directories containing experimental data
    dir_list = [item for item in input_path.iterdir() if item.is_dir()]

    # List to hold data
    all_data = []

    with tqdm(dir_list, leave=True) as pbar:
        for item in pbar:
            pbar.set_description(f"Processing {item.name}")

            data = process_files(item, output_path, save_data=False)
            all_data.append(data)

    # Merge the datasets
    combined_ds = xr.concat(all_data, dim="id")
    combined_ds.to_netcdf(output_path / "results.nc")

    # Convert to DataFrame and save
    combined_df = combined_ds.to_dataframe()

    try:
        combined_df.to_csv(output_path / "results.csv")

    except PermissionError:
        print(
            f"ERROR: The main results file {output_path / 'results.csv'!s} is currently locked (likely opened in Excel)."
        )

        # Base configuration for emergency backup
        backup_base = "results_backup"
        backup_ext = ".csv"
        counter = 0

        # Infinite loop to find a filename that is completely free and unlocked
        while True:
            # Determine filename based on counter
            suffix = f"_{counter}" if counter > 0 else ""
            backup_file = Path(f"{backup_base}{suffix}{backup_ext}")

            # Scenario A: File doesn't exist yet, so it is safe to create
            if not backup_file.exists():
                try:
                    all_data.to_csv(output_path / backup_file, index=False)
                    print(f"Results saved to '{backup_file}' to prevent data loss.")
                    break
                except PermissionError:
                    # Edge case: The file was created by someone else right as we checked
                    counter += 1
                    continue

            # Scenario B: File exists. Let's see if we can overwrite it (check if unlocked)
            try:
                with open(backup_file, "a") as f:
                    pass  # It's unlocked! We can safely overwrite it next.
                all_data.to_csv(output_path / backup_file, index=False)
                print(f"Results saved to '{backup_file}' to prevent data loss.")
                break
            except PermissionError:
                # File exists but it's locked by Excel too! Increment and try next slot.
                counter += 1


def process_files(
    input_export_path, output_path, save_data=True, exp_label="auto", cluster=False
):
    """
    Process all images in a directory.

    This function processes images from an experimental group. The function assumes that
    each image to be analyzed has labels which have been exported using QuPath, in the
    path "exp_dir/export". Additionally, the function reads in the QuPath project
    (QPPROJ) file in each directory to determine the original image path.

    The expected labels are:
      1 - Cells (oocytes or embryos)
      3 - Start location (used in oocyte images to determine oocyte rank)
      4 - Baseline (used to determine color baseline)

    If the Start location is provided, the code will label oocyte position based on
    distance to that location (i.e., the closest oocyte will be "M-1"). If a Baseline
    region is provided, the code will also include a color and intensity differences to
    the baseline. If these labels are missing, the function will only calculate the base
    color metrics.

    If the Cells label is missing, the function should skip the images entirely. This
    way, the user can skip annotating the images that have issues (e.g., duplicated worms, out of focus, etc.).

    The function creates an overlay image for each processed image showing the cell
    outlines and the "dark regions".

    Parameters
    ----------
    input_path : str or Path
        Path to the QuPath project folder
    output_path : str or Path
        Folder to save output (only used if save_data=True)
    save_data : bool, optional
        If True, a CSV_file containing the data will be saved in the ``output_path`` directory, by default True.
    exp_label : str, optional
        Sets the experiment label for the dataset. If set to "auto", the code determines
        this using the parent directory name. By default, "auto".
    cluster : bool, optional
        If True, will include clustering of color values. This is an alternative method
        to calculate the color statistics (e.g., cluster centers as mean color and the
        silhouette or Calinski-Harabsz score gives a measure of variation). By default, False.

    Returns
    -------
    list of xarray.Dataset
        A list containing the processed experimental data, where each row corresponds to an analyzed image file. The list is only returned if save_data is False. Otherwise, the information is exported as a CSV-file.

    Raises
    ------
    FileNotFoundError
        A label file was not found in the input directory.
    FileNotFoundError
        A QuPath project (*.qpproj) file was not found in the directory.
    """

    # Validate the inputs
    input_export_path = Path(input_export_path)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Look for exported labels
    label_list = list(input_export_path.glob("*.png"))

    if not label_list:
        raise FileNotFoundError(
            f"Could not find any label files (*.png) in the directory {input_export_path}."
        )

    # Get the QuPath project file in the parent directory
    input_export_path_parent = input_export_path.parent
    project_file = next((input_export_path_parent).glob("*.qpproj"), None)

    if project_file is None:
        raise FileNotFoundError(
            f"Could not find a QuPath project (.qpproj) file in {input_export_path_parent}"
        )

    # Read QuPath project and append the image files to a dictionary. The image file
    # name is used as a key so it is easy to match with the label filename later.
    image_uri = {}
    with open(project_file, "r", encoding="utf-8") as f:
        project_data = json.load(f)

        for entry in project_data.get("images", []):
            image_name = entry.get("imageName")
            server_builder = entry.get("serverBuilder", {})
            uri = server_builder.get("uri")

            # Only map it if both the name and URI exist
            if image_name and uri:
                image_name_clean = Path(image_name).stem
                image_uri[image_name_clean] = uri

    # Declare a list for storing data
    results = []

    with tqdm(label_list, leave=False) as img_pbar:
        for label_file in img_pbar:
            img_pbar.set_description(f"Processing file: {label_file.stem}")

            labels = io.imread(label_file)

            if 1 not in labels:
                # If no cells present, skip image
                continue

            # Find the corresponding image file
            target_filename = (label_file.stem).split("-")[0]
            image_path = image_uri.get(target_filename)

            image = io.imread(image_path)
            # plt.imshow(image)
            # plt.show()
            # exit()

            # Get the cell mask and re-label
            cell_labels, nCells = measure.label(labels == 1, return_num=True)

            # Convert image to HSV and LAB color space for color analysis
            image_hsv = color.rgb2hsv(image)
            image_lab = color.rgb2lab(image)

            # Convert to grayscale (uses luminance equation)
            image_gray = color.rgb2gray(image)

            # Find dark regions
            mask_dark_regions = image_lab[..., 0] < 30

            # Get the start label if it exists (should be value = 3)
            if 3 in labels:
                measure_oocyte_position = True
                start_label = measure.label(labels == 3)
                start_props = measure.regionprops_table(
                    start_label, properties=("centroid",)
                )
                start_centroid = np.array(
                    [[start_props["centroid-0"][0], start_props["centroid-1"][0]]]
                )

                cell_props = measure.regionprops_table(
                    cell_labels,
                    properties=(
                        "label",
                        "centroid",
                    ),
                )
                cell_centroids = np.column_stack(
                    (cell_props["centroid-0"], cell_props["centroid-1"])
                )

                cell_distances = np.linalg.norm(cell_centroids - start_centroid, axis=1)
                sorted_ranks = np.argsort(np.argsort(cell_distances))

                label_to_position = {
                    label: f"M-{rank}"
                    for label, rank in zip(cell_props["label"], sorted_ranks)
                }

            else:
                measure_oocyte_position = False
                label_to_position = {}

            # Set up baseline measurements if present
            if 4 in labels:
                measure_baseline = True

                baseline_grayscale_intensity = np.mean(image_gray[labels == 4])

                tmp_hue = image_hsv[..., 0]
                baseline_hue = np.mean(tmp_hue[labels == 4])

                tmp_sat = image_hsv[..., 1]
                baseline_saturation = np.mean(tmp_sat[labels == 4])

                tmp_val = image_hsv[..., 2]
                baselinev_value = np.mean(tmp_val[labels == 4])

                tmp_l = image_lab[..., 0]
                baseline_lightness = np.mean(tmp_l[labels == 4])

                tmp_a = image_lab[..., 1]
                baseline_a_star = np.mean(tmp_a[labels == 4])

                tmp_b = image_lab[..., 2]
                baseline_b_star = np.mean(tmp_b[labels == 4])

            # Get the experiment label from the input path
            if exp_label == "auto":
                input_directory_name = input_export_path.parent
                exp_label = get_experiment_label(input_directory_name)

            # Initialize a dict of lists to store data from this image
            cell_data = {
                "cell_label": [],
                "cell_area_pixels": [],
                "cell_ratio_area_dark": [],
                "mean_intensity": [],
                "mean_hue": [],
                "mean_saturation": [],
                "mean_value": [],
                "mean_lightness": [],
                "mean_A": [],
                "mean_B": [],
            }

            if measure_oocyte_position:
                cell_data |= {"position": []}

            if measure_baseline:
                cell_data |= {
                    "baseline_gray_intensity": [],
                    "baseline_hue": [],
                    "baseline_saturation": [],
                    "baseline_value": [],
                    "baseline_lightness": [],
                    "baseline_astar": [],
                    "baseline_bstar": [],
                    "difference_hue": [],
                    "ratio_gray_intensity": [],
                }

            if cluster:
                cell_data |= {
                    "kmeans_centroid1_L": [],
                    "kmeans_centroid1_A": [],
                    "kmeans_centroid1_B": [],
                    "kmeans_centroid2_L": [],
                    "kmeans_centroid2_A": [],
                    "kmeans_centroid2_B": [],
                    "centroid_distance": [],
                    "silhouette_score": [],
                    "calinski_harabasz_score": [],
                }

            # Get a list of unique cell labels
            unique_labels = np.unique(cell_labels)
            unique_labels = unique_labels[unique_labels > 0]

            # num_cells = len(unique_labels)

            # Save the dark region mask
            overlay = display.overlay_mask(
                image, mask_dark_regions, mask_color=(1.0, 0.0, 1.0)
            )
            overlay = segmentation.mark_boundaries(
                overlay, cell_labels, mode="thick", color=(0, 1, 0)
            )
            overlay = (overlay * 255).astype(np.uint8)
            fn = Path(image_path).stem
            io.imsave(output_path / f"{fn}_dark_region.png", overlay)

            # Process each cell
            for idx, curr_label in enumerate(unique_labels):
                cell_data["cell_label"].append(curr_label)

                if measure_oocyte_position:
                    position_str = label_to_position.get(curr_label, "NA")
                    cell_data["position"].append(position_str)

                gray_values = image_gray[cell_labels == curr_label]
                cell_data["mean_intensity"].append(np.mean(gray_values))

                # Get the HSV values for current cell. Data is N x 3 where N is the pixel
                hsv_values = image_hsv[cell_labels == (curr_label)]
                mean_HSV = np.mean(hsv_values, axis=0)

                cell_data["mean_hue"].append(mean_HSV[0])
                cell_data["mean_saturation"].append(mean_HSV[1])
                cell_data["mean_value"].append(mean_HSV[2])

                lab_values = image_lab[cell_labels == (curr_label)]
                mean_LAB = np.mean(lab_values, axis=0)
                cell_data["mean_lightness"].append(mean_LAB[0])
                cell_data["mean_A"].append(mean_LAB[1])
                cell_data["mean_B"].append(mean_LAB[2])

                if cluster:
                    # Calculate k-means clustering using the LAB color space. Input should be N x 3, where each row is data from a single pixel.
                    kmeans = cluster.KMeans(n_clusters=2, n_init="auto")
                    labels = kmeans.fit_predict(lab_values)
                    centers = kmeans.cluster_centers_

                    cell_data["kmeans_centroid1_L"].append(centers[0, 0])
                    cell_data["kmeans_centroid1_A"].append(centers[0, 1])
                    cell_data["kmeans_centroid1_B"].append(centers[0, 2])

                    cell_data["kmeans_centroid2_L"].append(centers[0, 0])
                    cell_data["kmeans_centroid2_A"].append(centers[0, 1])
                    cell_data["kmeans_centroid2_B"].append(centers[0, 2])

                    cell_data["centroid_distance"].append(
                        np.linalg.norm(centers[0, :] - centers[1, :])
                    )

                    cell_data["silhouette_score"].append(
                        metrics.silhouette_score(lab_values, labels, sample_size=3000)
                    )
                    cell_data["calinski_harabasz_score"].append(
                        metrics.calinski_harabasz_score(lab_values, labels)
                    )

                # Calculate percentage of "dark" region vs cell area
                num_pixels_dark_region = np.count_nonzero(
                    mask_dark_regions[cell_labels == (curr_label + 1)]
                )

                cell_data["cell_area_pixels"].append(
                    np.count_nonzero(cell_labels == curr_label)
                )
                cell_data["cell_ratio_area_dark"].append(
                    num_pixels_dark_region / np.count_nonzero(cell_labels == curr_label)
                )

                # Add baseline measurements

                if measure_baseline:
                    # Calculate hue difference
                    hue_diff = calculate_hue_difference(baseline_hue, mean_HSV[0])

                    cell_data["baseline_gray_intensity"].append(baseline_grayscale_intensity),
                    cell_data["baseline_hue"].append(baseline_hue),
                    cell_data["baseline_saturation"].append(baseline_saturation),
                    cell_data["baseline_value"].append(baselinev_value),
                    cell_data["baseline_lightness"].append(baseline_lightness),
                    cell_data["baseline_astar"].append(baseline_a_star),
                    cell_data["baseline_bstar"].append(baseline_b_star),
                    cell_data["difference_hue"].append(hue_diff),
                    cell_data["ratio_gray_intensity"].append(np.mean(gray_values)/baseline_grayscale_intensity),
                }

            # Generate an xarray dataset
            num_cells = len(cell_data["cell_label"])

            # Sort the output columns
            sorted_cols = ["dataset", "image", "exp_label", "cell_label"]

            curr_ds = xr.Dataset(
                data_vars={
                    k: (("id"), v) for k, v in cell_data.items() if k not in sorted_cols
                },
                coords={
                    "exp_label": (
                        "id",
                        [str(input_export_path_parent.name)] * num_cells,
                    ),
                    "image": ("id", [image_path] * num_cells),
                    "dataset": (
                        "id",
                        [exp_label] * num_cells,
                    ),
                },
            )
            results.append(curr_ds)

    combined_ds = xr.concat(results, dim="id")

    if save_data:
        # Merge the datasets
        combined_ds.to_netcdf(output_path / "results.nc")

        # Convert to DataFrame and save
        combined_df = combined_ds.to_dataframe()
        combined_df.to_csv(output_path / "results.csv")

    return combined_ds


def get_experiment_label(input):

    # Match anything before the 8-digit date
    exp_label = re.match(r"^(.+)\s\d{8}$", input)
    if exp_label:
        return exp_label.group(1)

def calculate_hue_difference(hue1, hue2):
    """
    Returns hue angle difference in degrees.

    Parameters
    ----------
    hue1 : float
        Hue 1 in degrees
    hue2 : float
        Hue 2 in degrees
    """

    diff = (hue1 - hue2 + 180) % 360 - 180

if __name__ == "__main__":
    pass
