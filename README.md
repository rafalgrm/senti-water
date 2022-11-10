# Senti-water

### Basics
Our project processes satellite images in order to create a database of water reservoirs, as well as changes that occur on their surfaces.
We use data from the Sentinel-2 satellite, provided for free by The Copernicus Open Access Hub using the SentinelAPI tool.

### How to use project
To install packages:
```bash
pip install -r requirements.txt
```
To access the SentinelAPI data you need to:
1. Create Copernicus account [here](https://scihub.copernicus.eu/userguide/SelfRegistration)
2. Create the ```.netrc``` file in `C:/Users/<User>` that looks like this:
```bash
machine apihub.copernicus.eu
login <your username>
password <your password>
```
