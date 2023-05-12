import numpy as np
import rasterio
from pyproj import transform
from rasterio.crs import CRS
from rasterio.warp import transform
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from ibmcloudant.cloudant_v1 import CloudantV1, Document
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import time

CLUSTER_THRESHOLD = 8000
SHOW_IMAGES = True
SHOW_DATA = False

SAVE_TO_CLOUDANT = False
APIKEY = ""
URL = ""
DATABASE_NAME_PREFIX = 'mazury_2022_' # region_year_

def process_img(CLUSTER_THRESHOLD, SHOW_IMAGES, SHOW_DATA, SAVE_TO_CLOUDANT, APIKEY, URL, DATABASE_NAME_PREFIX):
    file = open("filepath.txt", "r")
    filepath = file.read()

    dataset_band_1 = rasterio.open(filepath + '_B03_10m.jp2')
    dataset_band_2 = rasterio.open(filepath + '_B08_10m.jp2')

    image_band_1 = dataset_band_1.read(1)
    image_band_2 = dataset_band_2.read(1)
    image_band_1_norm = image_band_1 / np.max(np.abs(image_band_1))
    image_band_2_norm = image_band_2 / np.max(np.abs(image_band_2))
    image_ndwi = (image_band_1_norm - image_band_2_norm) // (image_band_1_norm + image_band_2_norm + np.ones((image_band_1_norm.shape[0], image_band_1_norm.shape[1]))) + np.ones((image_band_1_norm.shape[0], image_band_1_norm.shape[1]))
    water_mask = image_ndwi > 0.1

    water_indexes = np.transpose(water_mask.nonzero())
    clusters = DBSCAN(eps = 5.0, min_samples = 10, algorithm = 'kd_tree', n_jobs = -1).fit(water_indexes)
    unique, counts = np.unique(clusters.labels_, return_counts = True)
    cluster_indexes = dict(zip(unique, counts))
    cluster_indexes_above_thre = {k: v for k, v in cluster_indexes.items() if v > CLUSTER_THRESHOLD and k != -1}
    cluster_mask = [idx in cluster_indexes_above_thre for idx in clusters.labels_]
    water_indexes_image_coords = water_indexes[cluster_mask]

    hulls_pixels = []
    centroids = []
    surface_areas = []

    for index in cluster_indexes_above_thre:
        zipped_water_clusters = zip(water_indexes, clusters.labels_)
        pixels_in_cluster = []
        for point, cluster_idx in zipped_water_clusters:
            if cluster_idx == index:
                pixels_in_cluster.append([point[1], point[0]]) 

        pixels_in_cluster = np.array(pixels_in_cluster)

        hull = ConvexHull(pixels_in_cluster)
        hull_pixels = []
        for s in hull.simplices:
            hull_pixels.append([pixels_in_cluster[s, 0], pixels_in_cluster[s, 1]])
        hulls_pixels.append(hull_pixels)

        cx = np.mean(pixels_in_cluster[:, 1])
        cy = np.mean(pixels_in_cluster[:, 0])
        centroids.append([cx, cy])

        surface_area = len(pixels_in_cluster) * 100
        surface_areas.append(surface_area)

    real_coords_hulls_pixels = []
    real_coords_centroids = []

    new_crs = CRS.from_epsg(4326)

    data_transform = dataset_band_1.transform
    move_to_real_coords = lambda water_idx: data_transform * water_idx

    for n in hulls_pixels:
        real_coords_hulls_pixels_first = []
        real_coords_hulls_pixels_second = []
        real_coords_hulls_pixels_connected = []

        transformed_hulls_pixels_first = np.array([move_to_real_coords(np.array([xi[0][0], xi[1][0]])) for xi in n])
        for hull_pixels in transformed_hulls_pixels_first:
            real_coords_hull_pixels = transform(dataset_band_2.crs, new_crs, xs=[hull_pixels[0]], ys=[hull_pixels[1]])
            real_coords_hulls_pixels_first.append([real_coords_hull_pixels[1][0], real_coords_hull_pixels[0][0]])

        transformed_hulls_pixels_second = np.array([move_to_real_coords(np.array([xi[0][1], xi[1][1]])) for xi in n])
        for hull_pixels in transformed_hulls_pixels_second:
            real_coords_hull_pixels = transform(dataset_band_2.crs, new_crs, xs=[hull_pixels[0]], ys=[hull_pixels[1]])
            real_coords_hulls_pixels_second.append([real_coords_hull_pixels[1][0], real_coords_hull_pixels[0][0]])

        index = 0
        side = 1 # 1 - first to second, 2 - second to first
        while (len(real_coords_hulls_pixels_connected) < len(real_coords_hulls_pixels_first)):
            if len(real_coords_hulls_pixels_connected) == len(real_coords_hulls_pixels_first) - 1:
                real_coords_hulls_pixels_connected.append(real_coords_hulls_pixels_first[0])
            else:
                if side == 1:
                    real_coords_hulls_pixels_connected.append(real_coords_hulls_pixels_first[index])
                else:
                    real_coords_hulls_pixels_connected.append(real_coords_hulls_pixels_second[index])


                for i in range(len(real_coords_hulls_pixels_first)):
                    if side == 1:
                        if real_coords_hulls_pixels_second[index] == real_coords_hulls_pixels_first[i]:
                            index = i
                            break
                        elif real_coords_hulls_pixels_second[index] == real_coords_hulls_pixels_second[i] and index != i:
                            index = i
                            side = 2
                            break
                    else:
                        if real_coords_hulls_pixels_first[index] == real_coords_hulls_pixels_first[i] and index != i:
                            index = i
                            side = 1
                            break
                        elif real_coords_hulls_pixels_first[index] == real_coords_hulls_pixels_second[i]:
                            index = i
                            break

        real_coords_hulls_pixels.append(real_coords_hulls_pixels_connected)

    transformed_centroid = np.array([move_to_real_coords(np.array([xi[1], xi[0]])) for xi in centroids])
    for centroid in transformed_centroid:
        real_coords_centroid = transform(dataset_band_2.crs, new_crs, xs=[centroid[0]], ys=[centroid[1]])
        real_coords_centroids.append([real_coords_centroid[1][0], real_coords_centroid[0][0]])

    if (SHOW_IMAGES):
        for n in range(len(real_coords_hulls_pixels)):
            plt.plot(real_coords_centroids[n][1], real_coords_centroids[n][0], 'rx', ms = 6)
            for i in range(len(real_coords_hulls_pixels[n])):
                plt.plot([real_coords_hulls_pixels[n][i][1], real_coords_hulls_pixels[n][(i + 1) % len(real_coords_hulls_pixels[n])][1]], [real_coords_hulls_pixels[n][i][0], real_coords_hulls_pixels[n][(i + 1) % len(real_coords_hulls_pixels[n])][0]], 'r-')
        
        plt.figure(figsize = (8, 8))
        plt.show()

    if SHOW_DATA:    
        print('surface_areas (m^2):')
        print(surface_areas)
        print('real_coords_centroids:')
        print(real_coords_centroids)
        print('real_coords_hulls_pixels:')
        print(real_coords_hulls_pixels)