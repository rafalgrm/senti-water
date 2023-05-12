import rasterio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

SHOW_IMAGES = True
SHOW_FINAL_IMAGE = True

def visual_analysis(SHOW_IMAGES, SHOW_FINAL_IMAGE):
    file = open("filepath.txt", "r")
    filepath_10m = file.read()
    filepath_20m = filepath_10m.replace("/R10m/", "/R20m/")

    band_green_10m = rasterio.open(filepath_10m + '_B03_10m.jp2')
    band_nir_10m = rasterio.open(filepath_10m + '_B08_10m.jp2')

    band_green_20m = rasterio.open(filepath_20m + '_B03_20m.jp2')
    band_swir_20m = rasterio.open(filepath_20m + '_B11_20m.jp2')
    # band_swir_20m = rasterio.open(filepath_20m + '_B12_20m.jp2') # alternative

    image_band_green_10m = band_green_10m.read(1)
    image_band_nir_10m = band_nir_10m.read(1)
    image_band_green_20m = band_green_20m.read(1)
    image_band_swir_20m = band_swir_20m.read(1)

    if SHOW_IMAGES:
        plt.imshow(image_band_green_10m)
        plt.imshow(image_band_nir_10m)
        plt.imshow(image_band_green_20m)
        plt.imshow(image_band_swir_20m)

    image_band_green_10m_norm = image_band_green_10m / np.max(np.abs(image_band_green_10m))
    image_band_nir_10m_norm = image_band_nir_10m / np.max(np.abs(image_band_nir_10m))

    image_ndwi = (image_band_green_10m_norm - image_band_nir_10m_norm) // (image_band_green_10m_norm + image_band_nir_10m_norm + np.ones((image_band_green_10m_norm.shape[0], image_band_green_10m_norm.shape[1]))) + np.ones((image_band_green_10m_norm.shape[0], image_band_green_10m_norm.shape[1]))

    if SHOW_IMAGES:
        plt.imshow(image_ndwi.astype('uint16'))

    image_band_green_20m_norm = image_band_green_20m / np.max(np.abs(image_band_green_20m))
    image_band_swir_20m_norm = image_band_swir_20m / np.max(np.abs(image_band_swir_20m))

    image_mndwi = (image_band_green_20m_norm - image_band_swir_20m_norm) // (image_band_green_20m_norm + image_band_swir_20m_norm + np.ones((image_band_green_20m_norm.shape[0], image_band_green_20m_norm.shape[1]))) + np.ones((image_band_green_20m_norm.shape[0], image_band_green_20m_norm.shape[1]))

    if SHOW_IMAGES:
        plt.imshow(image_mndwi.astype('uint16'))
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        ax[0].imshow(image_ndwi.astype('uint16'))
        ax[1].imshow(image_mndwi.astype('uint16'))

    def load_sentinel_image(img_folder, filename, bands, scale):
        image = {}
        for band in bands:
            file = img_folder + 'R' + scale + 'm/' + filename + '_' + band + '_' + scale + 'm.jp2'
            print(f'Opening file {file}')
            ds = rasterio.open(file)
            image.update({band: ds.read(1)})

        return image

    def display_rgb(img, b_r, b_g, b_b, alpha = 1., figsize = (10, 10)):
        rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis = -1)
        rgb = rgb / rgb.max() * alpha
        plt.figure(figsize = figsize)
        plt.imshow(rgb)

    img = load_sentinel_image(filepath_10m[:filepath_10m.index("IMG_DATA") + 9], filepath_10m[filepath_10m.index("IMG_DATA") + 14:], ['B02', 'B03', 'B04'], '10')

    if SHOW_IMAGES:
        fig, ax = plt.subplots(1, 3, figsize = (15, 5))
        ax[0].imshow(img['B02'], cmap = 'Blues_r')
        ax[1].imshow(img['B03'], cmap = 'Greens_r')
        ax[2].imshow(img['B04'], cmap = 'Reds_r')

        display_rgb(img, 'B04', 'B03', 'B02', alpha = 5.)

    def new_water_mask_downscale(old_mask, old_shape, new_shape):
        new_mask = np.full((new_shape, new_shape), False)
        for i in range(new_shape):
            for j in range(new_shape):
                new_mask[i][j] = old_mask[(old_shape // new_shape) * i][(old_shape // new_shape) * j]
        
        return new_mask

    rgb = np.stack([img['B04'], img['B03'], img['B02']], axis = -1)
    rgb = rgb / rgb.max() * 5.

    water_mask = image_ndwi > 0.1

    if (water_mask.shape[0] != rgb.shape[0]):
        water_mask = new_water_mask_downscale(water_mask, water_mask.shape[0], rgb.shape[0])

    rgb[water_mask] = [0.1, 0.1, 0.9]

    if SHOW_IMAGES:
        plt.figure(figsize = (7, 7))
        plt.imshow(rgb)

    water_indexes = np.transpose(water_mask.nonzero())
    clusters = DBSCAN(eps = 5.0, min_samples = 10, algorithm='kd_tree', n_jobs = -1).fit(water_indexes)
    # clusters.labels_

    cluster_threshold = 8000
    unique, counts = np.unique(clusters.labels_, return_counts = True)
    cluster_indexes = dict(zip(unique, counts))
    cluster_indexes_above_thre = {k: v for k, v in cluster_indexes.items() if v > cluster_threshold and k != -1}
    # cluster_indexes_above_thre

    cluster_mask = [idx in cluster_indexes_above_thre for idx in clusters.labels_]
    water_indexes_image_coords = water_indexes[cluster_mask]

    rgb = np.stack([img['B04'], img['B03'], img['B02']], axis=-1)
    rgb = rgb/rgb.max() * 5

    X = water_indexes_image_coords[:, 0]
    Y = water_indexes_image_coords[:, 1]
    mask_matrix = np.zeros((10980, 10980), dtype='bool')
    mask_matrix[X, Y] = True
    mask_matrix
    rgb[mask_matrix] = [0.1, 0.1, 0.9]

    if SHOW_FINAL_IMAGE:
        plt.figure(figsize = (7,7))
        plt.imshow(rgb)

    data_transform = band_green_10m.transform

    move_to_real_coords = lambda water_idx: data_transform * water_idx

    zipped_water_clusters = zip(water_indexes, clusters.labels_)
    water_cluster_points = {}
    for point, cluster_idx in zipped_water_clusters:
        if cluster_idx not in water_cluster_points and cluster_idx in cluster_indexes_above_thre:
            water_cluster_points[cluster_idx] = point

    water_indexes_real_coords = np.array([move_to_real_coords(np.array([xi[1], xi[0]])) for xi in water_cluster_points.values()])

    cluster_one_points = []

    zipped_water_clusters = zip(water_indexes, clusters.labels_)
    for point, cluster_idx in zipped_water_clusters:
        if cluster_idx == 0:
            cluster_one_points.append([point[1], point[0]]) 

    cluster_one_points = np.array(cluster_one_points)

    rgb = np.stack([img['B04'], img['B03'], img['B02']], axis=-1)
    rgb = rgb/rgb.max() * 5

    X = water_indexes_image_coords[:, 0]
    Y = water_indexes_image_coords[:, 1]
    mask_matrix = np.zeros((10980, 10980), dtype='bool')
    mask_matrix[X, Y] = True
    rgb[mask_matrix] = [0.1, 0.1, 0.9]

    plt.figure(figsize = (12, 12))
    plt.imshow(rgb)

    for i in cluster_indexes_above_thre:
        print(str(i))
        cluster_one_points = []

        zipped_water_clusters = zip(water_indexes, clusters.labels_)
        for point, cluster_idx in zipped_water_clusters:
            if cluster_idx == i:
                cluster_one_points.append([point[1], point[0]]) 

        cluster_one_points = np.array(cluster_one_points)


        hull = ConvexHull(cluster_one_points)

        cx = np.mean(cluster_one_points[:, 0])
        cy = np.mean(cluster_one_points[:, 1])

        for simplex in hull.simplices:
            plt.plot(cluster_one_points[simplex, 0], cluster_one_points[simplex, 1], 'r-')

        plt.plot(cx, cy, 'rx', ms = 6)

    plt.show()