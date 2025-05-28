ftw data download --countries BurkinaFaso
 
ftw inference download \
--win_a S2B_MSIL2A_20180113T104409_R008_T30PUT_20201014T053206 \
--win_b S2B_MSIL2A_20181010T104019_R008_T30PUT_20201009T101517 \
-f -o BurkinaFaso.tif --bbox=12.7,48.4,12.97,48.57

ftw inference run BurkinaFaso.tif -f -o BurkinaFaso-inf.tif -m 3_Class_FULL_FTW_Pretrained.ckpt
ftw inference run BurkinaFaso.tif -f -o BurkinaFaso-inf_2_class.tif -m 2_Class_FULL_FTW_Pretrained.ckpt



ftw inference run BurkinaFaso.tif -f -o BurkinaFaso-inf.tif \
--gpu 0 -m 3_Class_FULL_FTW_Pretrained.ckpt

ftw inference polygonize BurkinaFaso-inf.tif -o BurkinaFaso_poly.gpkg
ftw inference polygonize BurkinaFaso-inf.tif -o BurkinaFaso_poly.geojson

from pystac_client import Client
import planetary_computer as pc

# Search against the Planetary Computer STAC API
catalog = Client.open(
  "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Define your area of interest
aoi = {
  "type": "Polygon",
  "coordinates": [
    [
      [-4.447772929038479, 11.362393590122196],
      [-4.362446317545647, 11.362393590122196],
      [-4.362446317545647, 11.40977859855056],
      [-4.447772929038479, 11.40977859855056],
      [-4.447772929038479, 11.362393590122196]
    ]
  ]
}

# Define your search with CQL2 syntax
search = catalog.search(filter_lang="cql2-json", filter={
  "op": "and",
  "args": [
    {"op": "s_intersects", "args": [{"property": "geometry"}, aoi]},
    {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]},
    {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10]}
  ]
})

# Grab the first item from the search results and sign the assets
first_item = next(search.get_items())
pc.sign_item(first_item).assets
