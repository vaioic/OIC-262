import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

ds = xr.open_dataset("..\\processed\\2026-03-04\\results.nc")

# for coord in ds.coords:
#     print(f"Coordinate: {coord}")
#     print(ds[coord].values)
#     print("-" * 20)

df = ds.mean_lightness.to_dataframe()
# print(df)

order = ['wt 0h', 'wt 1h', 'wt 6h', 'wt 24h']

sns.boxplot(data=df, y='mean_lightness', x='exp_label', order=order)
plt.title('Lightness')
plt.ylabel('Mean Lightness')
plt.show()

ds.plot.scatter(x="mean_B", y="mean_A", hue="exp_label", cmap="Set1")
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

