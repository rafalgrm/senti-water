# Senti-water
## Basics
Our project processes satellite images in order to create a database of water reservoirs, as well as changes that occur on their surfaces.
We use data from the Sentinel-2 satellite, provided for free by The Copernicus Open Access Hub using the SentinelAPI tool.
## How to use our project
### To install packages:
```bash
pip install -r requirements.txt
```
### To access the SentinelAPI data:
1. Create Copernicus account [here](https://scihub.copernicus.eu/userguide/SelfRegistration)
2. Create the `.netrc` file in your home directory (typically `C:/Users/<User>`) that looks like this:
```bash
machine apihub.copernicus.eu
login <your username>
password <your password>
```
### To launch our project:
1. Open `download.ipynb` file. Before you run the program, you can change the variables - the area (we recommend [boundingbox](https://boundingbox.klokantech.com/) in CSV RAW representation), the time period in which the satellite image was taken and cloud coverage percentage. Make sure you typed the variable values correctly. Be aware that too high or too low cloud coverage value, too short a time period or too small area may result in no image being found.
2. If you want to run the file for the first time, with no data downloaded previously, make sure that both variables for preview images are set to true. Now you are ready to launch the notebook.
3. After a while, the satellite images will be displayed in lower quality. Select the photo you are interested in to download in higher quality (we recommend the one that contains as few clouds as possible). 
4. Now run the same file, you can skip downloading and showing preview images (just set both download variables to false), but remember to change ```download_by_index``` and ```path_to_image_by_index``` values to the index of the photo you are interested in.
5. After a few minutes, the high-quality image and filepath will be displayed. The filepath will be saved in `filepath.txt` file and will be used in other `.ipynb` files.
6. Open `analysis.ipynb` file and run the program. Wait for it to analyze the image.
### To 