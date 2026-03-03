import xarray as xr
from matplotlib import pyplot as plt

ds = xr.open_dataset("..\\testresults.nc")

# for coord in ds.coords:
#     print(f"Coordinate: {coord}")
#     print(ds[coord].values)
#     print("-" * 20)

df = ds.mean_lightness.to_dataframe()
df.boxplot(column='mean_lightness', by='exp_label')
plt.title('Lightness')
plt.ylabel('Mean Lightness')
plt.show()

ds.plot.scatter(x="mean_B", y="mean_A", hue="exp_label")
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

ds.plot.scatter(x="cell_position", y="mean_lightness", hue="exp_label")
plt.title('Mean color')
plt.ylabel('Mean lightness')
plt.xlabel('Cell position')
plt.show()





# meanL_per_dataset = subset.mean_lightness.dropna("cell_position")

# print(meanL_per_dataset.size)

# meanL_per_dataset.to_series().plot.bar()

# plt.show()

