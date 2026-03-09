from shapely.geometry import Polygon
from terracatalogueclient import Catalogue

# Authenticate to the Terrascope platform (registration required)
catalogue = Catalogue().authenticate()
# with username and password
username = "USERNAME"
password = "PASSWORD"
catalogue = catalogue.authenticate_non_interactive(username = username, password = password)

# Define the bounding box
bounds = (25.668510437000105, 35.81543000000018, 44.83476000000008, 42.10757000000004)
geometry = Polygon.from_bounds(*bounds)
## TCI 2021
# Search for products in the WorldCover 2021 collection, filtered by the geometry
products = catalogue.get_products(
    "urn:eop:VITO:ESA_WorldCover_S2RGBNIR_10m_2021_V2", # or "urn:eop:VITO:ESA_WorldCover_S2RGBNIR_10m_2020_V1" for 2020
    geometry=geometry
)

# Download the products to the specified directory
catalogue.download_products(products, "S2RGB_2021")


## Worldcover 2021
# Search for products in the WorldCover 2020 collection, filtered by the geometry
products = catalogue.get_products(
    "urn:eop:VITO:ESA_WorldCover_10m_2021_V2", # or "urn:eop:VITO:ESA_WorldCover_10m_2020_V1" for 2020
    geometry=geometry
)

# Download the products to the specified directory
catalogue.download_products(products, "worldcover_2021")


