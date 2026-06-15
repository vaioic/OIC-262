'''
This script is used to analyze data for the germline RNAi dataset.
'''

import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

ds1 = xr.open_dataset("..\\processed\\2026-03-12 Germline RNAi\\20feb26\\results.nc")
ds1 = ds1.assign_coords(source="Dataset 1")

print(ds1)

ds2 = xr.open_dataset("..\\processed\\2026-03-12 Germline RNAi\\lugols 02142026 1_20\\results.nc")
ds2 = ds2.assign_coords(source="Dataset 2")

ds_list = [ds1, ds2]

# Combine the datasets while keeping their sources separate
combined = xr.concat(ds_list, dim="id")

# Save the combined dataset
combined.to_netcdf('../processed/2026-03-12 Germline RNAi/combined.nc')

df_combined = combined.to_dataframe()
df_combined.to_csv('../processed/2026-03-12 Germline RNAi/combined.csv', index=True)

exit()
# order = ['wt 0h', 'wt 1h', 'wt 3h', 'wt 6h', 'wt 24h']

df_combined = combined.mean_lightness.to_dataframe()

sns.boxplot(data=df_combined, y='mean_lightness', x='exp_label', hue='source')
plt.title('Lightness')
plt.ylabel('Mean Lightness (by dataset)')
plt.show()


sns.boxplot(data=df_combined, y='mean_lightness', x='exp_label')
plt.title('Lightness')
plt.ylabel('Mean Lightness (combined)')
plt.show()

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



# meanL_per_dataset = subset.mean_lightness.dropna("cell_position")

# print(meanL_per_dataset.size)

# meanL_per_dataset.to_series().plot.bar()

# plt.show()

