# Burton lab - Lugol stain color analysis

The goal of this project is to perform a color analysis of brightfield images of _C. elegans_ stained using Lugol's iodine solution. In particular, we are interested in quantifying the color change of the iodine stain in the oocytes with worms of different genotypes and grown in different media. 

This repository contains both the methodology and code for the analysis.

## Usage

### Images

The original dataset for this project were brightfield color images, collected using a
dissecting microscope. The images should be saved as TIFF files, with the same color 
profile and white balance applied. 

Experimentally, it is important that all images are taken with the same staining 
protocol and illumination intensity.

Note that there are two different types of images: one set shows the developing oocytes
in the germline. For these images, a manual marker is added during the annotation to
indicate the first oocyte. The second set of images shows the developing embryos.

### Prerequisites

The analysis relies on both QuPath for annotation and Python for measurement and 
analysis.

- [Python](https://www.python.org/downloads/) version 3.13.7 or higher
- [QuPath](https://qupath.github.io/) version 0.6.0 or higher

### Download code

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-262.git
   cd OIC-262
   ```

### Annotating the cells in QuPath

#### Folder structure

#### Annotation methodology

1. For each experimental condition, create a QuPath project in an empty directory.
2. Click on **Add Images...** then click **Choose files** in the dialog box that opens up
3. Select all the image files in the directory, then click **Open**
4. In the **Set Image Type**, select **Brightfield (other)**
5. Click **Import**
6. (First time only) Define the following classes in QuPath:
   1. After the images have been imported, click on the **Annotations** tab. This only
      needs to be done once as QuPath should remember these classes going forward.   
   2. Select each class and click on the **-** button to remove it (Note: You cannot 
      remove the ``None`` class)
   3. Click on the **+** symbol and add the two classes named (spelling is critical):
      - Cell
      - Start
7. Click on the **Project** tab, then double-click an image to open it.
8. Annotate the cells using the Brush tool. You can find this tool on the toolbar or by
   pressing (B). Make sure you leave a small gap between each cell - this is critical
   for the Python script to identify individual cells.
9. In the Annotation tab, select the cell objects (CTRL + Click) then select **Cells**
   in the class list. Click on **Set Selected** to label the cells.
9. For the germline images, use the Circle tool (O) draw a small circle **close to the
   M-1 oocyte**. The exact position is not very important - the Python script will label
   the oocytes based on relative positions to this circle.
11. In the Annotation tab, select the circle object (CTRL + Click) then select **Start**
   in the class list. Click on **Set Selected**.
12. Repeat this process on all remaining images. When you are done, save and close the
    final image: **View** > **Multi-view** > **Close viewer**.

#### Exporting the labels

1. To export the labels, select **Automate** > **Script Editor**. 
2. In the dialog box, select **File** > **Open**, then select the file
   ``export_masks.groovy`` in the ``qupath_script`` subfolder from this repository.
3. Click on **Run**  **Run for project**. 
4. Add all images , then click **Run**. This will export the labels into a folder called
   ``export`` directory. You can visualize the masks using QuPath or Fiji.

### Setup and installation of the Python script

#### Using uv (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) to manage virtual environments and dependencies. 

1. Install ``uv``
    * **macOS or Linux:** ``curl -LsSf https://astral.sh/uv/install.sh | sh``
    * **Windows:** ``powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"``
    
    To check if you have ``uv`` installed, open a terminal and run ``uv --version``.

2. Clone the repository
   ```bash
   git clone git@github.com:vaioic/burton-lab-lugols-analysis.git
   cd burton-lab-lugols-analysis
   ```

3. Sync the environment (this will setup the correct virtual environment and dependencies)
   ```bash
   uv sync
   ```

4. Run the analysis
   ```bash
   uv run analysis/20260806_Dataset02_Run01.py
   ```

#### Using venv and pip

1. Clone the repository
   ```bash
   git clone git@github.com:vaioic/burton-lab-lugols-analysis.git
   cd burton-lab-lugols-analysis
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the environment
   ```bash
   # macOS/Linux
   source ./venv/bin/activate

   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

4. Install the repository as an editable module
   ```bash
   python -m pip install -e .
   ```

5. Run the analysis script
   ```bash
   python -m analysis.20260806_Dataset02_Run01

   # or
   python analysis/20260806_Dataset02_Run01.py
   ```

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/burton-lab-lugols-analysis/issues) or send an email to opticalimaging@vai.org. If you are reporting a bug, please include any error messages to aid with troubleshooting.

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Citing & Acknowledgements

This repository is publicly available for open-source use, but it is developed and maintained by the Optical Imaging Core at the Van Andel Institute. If code from this repository contributed to data used in a publication, abstract, or presentation, please cite and acknowledge our work based on your affiliation:

### For External Users
Please cite this repository and acknowledge the author(s) in your publication's materials, methods, or acknowledgements section:
> "Image analysis pipelines were adapted from open-source tools developed by the Optical Imaging Core at the Van Andel Institute (GitHub:[burton-lab-lugols-analysis](https://github.com/vaioic/burton-lab-lugols-analysis))."

If you require custom adjustments or advanced analysis support, please contact us at opticalimaging@vai.org.

### For Internal Users & Close Collaborators
If you are an internal researcher or an external collaborator working directly with our staff, please include our Research Resource Identifier (RRID) in your materials and methods section:
> "Image analysis and data processing were performed in collaboration with the Optical Imaging Core at the Van Andel Institute (RRID:SCR_021968)."

Please review the Acknowledgement and Authorship Guidelines on [VAI's Core Technology and Services website](https://vanandelinstitute.sharepoint.com/sites/Cores/SitePages/Acknowledgements-and-Authorship.aspx)

### Contributors
<a href="https://github.com/vaioic/burton-lab-lugols-analysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/burton-lab-lugols-analysis" />
</a>

## Changelog

## v1.0.0 (2026-08-06)
* Updated the code to match latest standards
* Moved the quantification code into the ``shared`` folder
* Added analysis scripts for June 2026 dataset
  ([OIC-263](https://varioic.atlassian.net/browse/OIC-263))
* Changed the input to accept a more 

### v0.2.0 (2026-06-15)
* Merged the oocyte and embryo analysis code
  ([OIC-263](https://varioic.atlassian.net/browse/OIC-263))