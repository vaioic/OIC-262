from matplotlib import pyplot as plt
import skimage
import numpy as np
import sklearn

image_file = 'C:\\Users\\jian.tay\\Documents\\Projects\\OIC-263_264\\data\\WT-300\\P5-Image.tiff'

mask_file = 'C:\\Users\\jian.tay\\Documents\\Projects\\OIC-263_264\\data\\WT-300\\qupath\\export\\P5-Image-labels.png'

image = skimage.io.imread(image_file)
mask = skimage.io.imread(mask_file)

print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")

image = (image - np.min(image)) / (np.max(image) - np.min(image))

# plt.imshow(image)
# plt.show()

# Note: Mask is in RGB
mask_gray = skimage.color.rgb2gray(mask)
mask_gray = skimage.util.img_as_ubyte(mask_gray)
# plt.imshow(mask_gray)
# plt.show()

#Get values: [ 0 91 171 193]
unique = np.unique(mask_gray)
unique = np.delete(unique, np.where(unique == 0))
print(unique)

image_hsv = skimage.color.rgb2hsv(image)

# Quantify the HSV in each mask
for obj_id in unique:

    hue_channel = (image_hsv[..., 0]).squeeze()
    sat_channel = (image_hsv[..., 1]).squeeze()
    value_channel = (image_hsv[..., 2]).squeeze()

    # Calculate average of each metric
    mean_hue = np.mean(hue_channel[np.where(mask_gray == obj_id)])

    print(mean_hue)

    # KMeans expects each row to be a pixel and the coluns to be H, S, V
    X = image_hsv.reshape((-1, 3))
    
    # Try k-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init="auto")
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Pick a random selection of points
    rng = np.random.default_rng()
    pts = rng.integers(low=0, high=X.shape[0], size=200)

    # Scatter plot of the data points, colored by their assigned cluster label
    ax.scatter(X[pts, 0], X[pts, 1], X[pts, 2], c=labels[pts], cmap='viridis', s=50, alpha=0.6)

    # Plot the centroids as well
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=200, c='red', label='Centroids')

    ax.set_title('K-Means Clustering in 3D')
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    ax.set_zlabel('Z Dimension')
    ax.legend()

    # Display the plot
    plt.show()

    exit()