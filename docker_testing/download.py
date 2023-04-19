from datetime import date
from sentinelsat import SentinelAPI, make_path_filter
from datetime import date
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


CLOUD_COVERAGE_RANGE = (0, 10)
COORDINATES = [20.860291, 53.633239, 22.118225, 54.059388]
TIME_PEROID_START = date(2022, 7, 1)
TIME_PEROID_END = date(2022, 7, 10)


DOWNLOAD_ALL = True
SHOW_ALL = True

SPECIFIC_IMAGE_INDEX = -1

def download_images(COORDINATES, TIME_PEROID_START, TIME_PEROID_END, CLOUD_COVERAGE_RANGE, DOWNLOAD_ALL, SHOW_ALL, SPECIFIC_IMAGE_INDEX):
    LOGIN = os.getenv('SENTINEL_LOGIN')
    PASS = os.getenv('SENTINEL_PASSWORD')
    api = SentinelAPI(LOGIN, PASS)

    def getQuery(down_left, top_right, start_date, end_date):
        footprint = "POLYGON(("
        footprint += down_left[0] + " " + down_left[1] + ","
        footprint += down_left[0] + " " + top_right[1] + ","
        footprint += top_right[0] + " " + top_right[1] + ","
        footprint += top_right[0] + " " + down_left[1] + ","
        footprint += down_left[0] + " " + down_left[1]
        footprint += "))"

        return api.query(footprint,
                        date = (start_date, end_date),
                        platformname = 'Sentinel-2',
                        processinglevel = 'Level-2A',
                        cloudcoverpercentage = CLOUD_COVERAGE_RANGE)

    products = getQuery((str(COORDINATES[0]), str(COORDINATES[1])), (str(COORDINATES[2]), str(COORDINATES[3])), TIME_PEROID_START, TIME_PEROID_END)

    products_df = api.to_dataframe(products)
    products_df


    if (DOWNLOAD_ALL):
        for product in products:
            try:
                api.download(product, nodefilter = make_path_filter("*B02_60m.jp2", exclude = False), directory_path='downloads')
                api.download(product, nodefilter = make_path_filter("*B03_60m.jp2", exclude = False), directory_path='downloads')
                api.download(product, nodefilter = make_path_filter("*B04_60m.jp2", exclude = False), directory_path='downloads')
            except Exception as e:
                print(e)
                continue



    def get_path(path_idx, scale='10'):
        path = 'downloads/'
        path += products_df.to_dict('split')['data'][path_idx][0] 
        path += '.SAFE/GRANULE/L2A_'
        path += products_df.to_dict('split')['data'][path_idx][0].split('_')[5]
        path += '_'
        path += products_df.to_dict('split')['data'][path_idx][22].split('_')[7]
        path += '_'
        path += products_df.to_dict('split')['data'][path_idx][38].split('_')[7][1:]
        path += '/IMG_DATA/R' + scale + 'm/'
        path += products_df.to_dict('split')['data'][path_idx][0].split('_')[5]
        path += '_'
        path += products_df.to_dict('split')['data'][path_idx][0].split('_')[2]
        return path

    def load_sentinel_image(img_folder, filename, bands, scale):
        image = {}
        for band in bands:
            file = img_folder + 'R' + scale + 'm/' + filename + '_' + band + '_' + scale + 'm.jp2'
            ds = rasterio.open(file)
            image.update({band: ds.read(1)})

        return image

    def display_rgb(img, b_r, b_g, b_b, alpha = 1., figsize = (10, 10), title=''):
        rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis=-1)
        rgb = rgb/rgb.max() * alpha
        plt.figure(figsize = figsize)
        plt.title(title)
        plt.imshow(rgb)

    def show_all_images():
        for i in range(len(products_df.to_dict('split')['data'])):
            try:
                filepath = get_path(i, '60')
                img = load_sentinel_image(filepath[:filepath.index("IMG_DATA") + 9], filepath[filepath.index("IMG_DATA") + 14:], ['B02', 'B03', 'B04'], '60')
                display_rgb(img, 'B04', 'B03', 'B02', alpha = 5., figsize = (4, 4), title = str(i))
                print(str(i) + ': good')
            except:
                print(str(i) + ': error')

    if (SHOW_ALL):
        show_all_images()

    if (SPECIFIC_IMAGE_INDEX != -1):
        path_filter = make_path_filter('*B??_??m.jp2', exclude = False)
        api.download(products_df.to_dict('split')['data'][SPECIFIC_IMAGE_INDEX][41], directory_path='downloads', nodefilter = path_filter)

        print(get_path(SPECIFIC_IMAGE_INDEX))

        file = open("filepath.txt", "w")
        file.write(get_path(SPECIFIC_IMAGE_INDEX))
        file.close()
        
        filepath = get_path(SPECIFIC_IMAGE_INDEX, '60')
        img = load_sentinel_image(filepath[:filepath.index("IMG_DATA") + 9], filepath[filepath.index("IMG_DATA") + 14:], ['B02', 'B03', 'B04'], '60')
        display_rgb(img, 'B04', 'B03', 'B02', alpha = 5., figsize = (5, 5), title = str(SPECIFIC_IMAGE_INDEX))


download_images(COORDINATES, TIME_PEROID_START, TIME_PEROID_END, CLOUD_COVERAGE_RANGE, DOWNLOAD_ALL, SHOW_ALL, SPECIFIC_IMAGE_INDEX)