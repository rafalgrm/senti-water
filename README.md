# senti-water

To install packages:
```bash
pip install -r requirements.txt
```

To access the SentinelAPI data you need to:
1. Create Copernicus account [here](https://scihub.copernicus.eu/userguide/SelfRegistration)
2. Create the ```.netrc``` file in C:/Users/<User> that looks like this:
```bash
machine apihub.copernicus.eu
login <your username>
password <your password>
```