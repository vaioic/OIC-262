# OIC-262 Colorimetric analysis of Lugol's staining

The goal of this project is to perform a color analysis of brightfield images of _C. elegans_ stained using Lugol's iodine solution. In particular, we are interested in quantifying the color change of the iodine stain in the oocytes with worms of different genotypes and grown in different media. 

This repository contains both the methodology and code for the analysis.

## Getting started

These instructions assume that the images are collected and exported as TIFF files, with the same color profile and white balance applied. Additionally, you should ensure that all images are taken with the same staining protocol and illumination intensity. The images should brightfield color images.

### Prerequisites

The analysis relies on both QuPath for annotation and Python for measurement and analysis.

- [Python](https://www.python.org/downloads/) version 3.13.7 or higher
- [QuPath](https://qupath.github.io/) version 0.6.0 - 0.7.0


### Download code

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-262.git
   cd OIC-262
   ```

### Annotating the cells in QuPath

Images should be separated into different folders for each experimental condition.

1. For each experimental condition, create a QuPath project in a subdirectory of the image folder. To keep it simple, I usually name the subfolder ``qupath``.
2. Click on **Add Images...** then click **Choose files** in the dialog box that opens up
3. Select all the image files in the directory, then click **Open**
4. In the **Set Image Type**, select **Brightfield (other)**
5. Click **Import**
6. (First time only) After the images have been imported, click on the **Annotations** tab.
7. Select each class and click on the **-** button to remove it (Note: You cannot remove the ``None`` class)
8. Click on the **+** symbol and add three classes named (spelling is critical):
    - Cell
    - Gray Patch
    - Start
9. Click on the **Project** tab, then double-click an image to open it.
10. Annotate the cells using the Brush tool. You can find this tool on the toolbar or by pressing (B). Make sure you leave a small gap between each cell - this is critical for the Python script to identify individual cells.
11. Use the Circle tool (O) draw a small circle close to the M-1 oocyte. The exact position is not very important - the Python script will label the oocytes based on position to this circle.
12. Finally, going back to the Annotation tab, select the cell objects (CTRL + Click) then select **Cells** in the class list. Click on **Set Selected** to label the cells.
13. Do the same of the circle, labelling it with **Start**.
14. Repeat this process on all remaining images.
15. When you are done, save and close the final image: **View** > **Multi-view** > **Close viewer**.
16. To export the labels, select **Automate** > **Script Editor**. 
17. In the dialog box, select **File** > **Open**, then select the file ``export_cell_mask.groovy`` from this repository.
18. Click on **Run**  **Run for project**. This will export the labels into a folder called ``export`` under the ``qupath`` directory. 
19. Copy all the label files to the same directory as the original images.
20. Repeat for all experimental conditions.

### Python setup

If running the code for the first time, you will need to create a Python virtual environment and install the necessary packages. 

1. Open a terminal and navigate to the directory where you unzipped the files.

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   
   Linux:
   ```bash
   source venv/scripts/activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

2. You can run the code by calling the analyze_color function with the following syntax:
   ```python
   main_folder = Path('D:\\Projects\\OIC-262\\data\\single_images')

   analyze_color([
        [main_folder / 'daf2 300 02112026_ mislabled as nduf7', 'daf2 300'],
        [main_folder / 'daf2 con 1_20 02112026', 'daf2 con'],
        [main_folder / 'gsy1 300mM 1_20 02112026', 'gsy1 300']
        ], '..\\2026-03-03')
   ```

   Each entry should be a list containing the path to the directory containing  the images and labels, as well as a string that describes the experimental condition. The final entry is the path to the output folder.

   As an alternative, you can also edit the lines under ``if __name__ == "__main__":`` to point to the correct directories and labels.

### Analyzing the data

See [``analyze_data.py``](./analyze_data.py) for an example of a script to analyze the data.

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-244/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Contributors
<a href="https://github.com/vaioic/OIC-262/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/OIC-262" />
</a>

### Dependencies

This project relies on the following packages:

**Note:** For full dependency list, see [requirements.txt](requirements.txt).