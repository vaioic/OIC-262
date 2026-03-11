import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

ds1 = xr.open_dataset("..\\processed\\2026-03-11b lugols 02182026 NaCl timecourse\\results.nc")
ds1 = ds1.assign_coords(source="Dataset 1")

print(ds1)

ds2 = xr.open_dataset("..\\processed\\2026-03-11b lugols NaCl timecourse 02192026\\results.nc")
ds2 = ds2.assign_coords(source="Dataset 2")

ds3 = xr.open_dataset("..\\processed\\2026-03-11b lugols NaCl timecourse 02202026\\results.nc")
ds3 = ds3.assign_coords(source="Dataset 3")

ds_list = [ds1, ds2, ds3]

# Combine the datasets while keeping their sources separate
combined = xr.concat(ds_list, dim="id")

# # Save the combined dataset
# combined.to_netcdf('../processed/2026-03-11 Timelapse/combined.nc')

# df_combined = combined.to_dataframe()
# df_combined.to_csv('../processed/2026-03-11 Timelapse/combined.csv', index=True)

order = ['wt 0h', 'wt 1h', 'wt 3h', 'wt 6h', 'wt 24h']

df_combined = combined.mean_lightness.to_dataframe()

# sns.boxplot(data=df_combined, y='mean_lightness', x='exp_label', order=order, hue='source')
# plt.title('Lightness')
# plt.ylabel('Mean Lightness (by dataset)')
# plt.show()


# sns.boxplot(data=df_combined, y='mean_lightness', x='exp_label', order=order)
# plt.title('Lightness')
# plt.ylabel('Mean Lightness (combined)')
# plt.show()

combined.plot.scatter(x="mean_B", y="mean_A", hue="exp_label", cmap="Set1")
plt.title('Mean color')
plt.ylabel('A* (Red-Green)')
plt.xlabel('B* (Yellow-Blue)')
plt.show()


# df = ds.mean_lightness.to_dataframe()
# df.boxplot(column='mean_lightness', by='cell_position')
# plt.title('Lightness')
# plt.ylabel('Mean Lightness')
# plt.show()

# ds.plot.scatter(x="mean_B", y="mean_A", hue="cell_position")
# plt.title('Mean color')
# plt.ylabel('A* (Red-Green)')
# plt.xlabel('B* (Yellow-Blue)')
# plt.show()

ds.plot.scatter(x="cell_position", y="mean_lightness", hue="exp_label", cmap="Set1")
plt.title('Mean color')
plt.ylabel('Mean lightness')
plt.xlabel('Cell position')
plt.show()





# meanL_per_dataset = subset.mean_lightness.dropna("cell_position")

# print(meanL_per_dataset.size)

# meanL_per_dataset.to_series().plot.bar()

# plt.show()

