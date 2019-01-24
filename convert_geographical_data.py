import dbfread
import shapefile


filename = "data/geo_map/Ancillary/DEMS_TileIndex.shp"

# dbf = dbfread.DBF(filename)

shape = shapefile.Reader(filename)
