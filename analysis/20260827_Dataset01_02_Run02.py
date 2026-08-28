# 2026-08-27 Full re-run of the datasets using the new metrics and baselines
import os
import shutil

from shared import lugolquant

# lugolquant.process_files(
#     r"\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\QuPath projects\New data for June 2026\19052026\daf2 germline RNAi\ctrl RNAi\export",
#     r"../processed/20260828 Missing data",
# )

# lugolquant.process_files(
#     r"\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\QuPath projects\Timecourse_Feb 2026\lugols NaCl timecourse 02192026\qupath_wt_3h\export",
#     r"../processed/20260828 Missing data",
# )


# exit()


lugolquant.process_directory(
    r"\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\QuPath projects",
    r"../processed/20260828 All",
)

source_folder = r"../processed/20260827 All"
destination_folder = r"\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\Processed Data\2026-08-28 New Data"

try:
    # Check if the source actually exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder does not exist: {source_folder}")
    else:
        # shutil.copytree copies the entire directory tree.
        # dirs_exist_ok=True (Python 3.8+) allows copying into an existing folder without throwing an error.
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f"Successfully copied '{source_folder}' to '{destination_folder}'")

except PermissionError:
    print("Error: Permission denied. Check your network drive access rights.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Had to re-run these directories because the initial processing threw errors


# lugolquant.process_files(
#     r"\\pn.vai.org\projects\burton\VARI CORE GENERATED DATA\OIC\Oocyte glycogen staining\QuPath projects\New data for June 2026\19052026\daf2 germline RNAi\ctrl RNAi\export",
#     r"../processed/20260828 Missing data",
# )
