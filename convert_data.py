from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal_array

import dbfread


def make_row(x_pix, y_pix, padfTrans, field):
    x = padfTrans[0] + x_pix * padfTrans[1] + y_pix * padfTrans[2]
    y = padfTrans[3] + x_pix * padfTrans[4] + y_pix * padfTrans[5]

    return x, y, field[x_pix, y_pix]


# load data
mines_filepath = "data/mines/copper_containted_resource.dbf"
gravity_filepath = "data/gravity/SA_GRAV.ers"
magnetic_filepath = "data/magnetic/SA_TMI_RTP.ers"

mines_dbf = dbfread.DBF(mines_filepath)

# convert mines to an array of [ [long, lat] ] coords
mines = None
for mine in mines_dbf:

    if mines is None:
        print("hello")
        mines = [[mine["LONGITUDE"], mine["LATITUDE"]]]
    else:
        mines = np.append(mines, [[mine["LONGITUDE"], mine["LATITUDE"]]], axis=0)

gravity = gdal_array.LoadFile(gravity_filepath).T
magnetic = gdal_array.LoadFile(magnetic_filepath).T


# plt.ion()
#
# plt.contourf(gravity)
# plt.show()
#
# plt.contourf(magnetic)
# plt.show()
#
#
# Open the file:


# converty the gravity stuff into triples
gravity_raster = gdal.Open(gravity_filepath)

x_pixels = np.arange(gravity_raster.RasterXSize)
y_pixels = np.arange(gravity_raster.RasterYSize)

# now convert to lon lat
padfTransform = gravity_raster.GetGeoTransform()


xy_gravity = np.zeros((gravity_raster.RasterXSize * gravity_raster.RasterYSize, 3))

count = 0
for x_pixel in x_pixels:
    for y_pixel in y_pixels:
        xy_gravity[count, :] = make_row(x_pixel, y_pixel, padfTransform, gravity)
        count += 1

# create magnetic triple
magnetic_raster = gdal.Open(magnetic_filepath)

x_pixels = np.arange(magnetic_raster.RasterXSize)
y_pixels = np.arange(magnetic_raster.RasterYSize)

# now convert to lon lat
padfTransform = magnetic_raster.GetGeoTransform()

xy_magnetic = np.zeros((magnetic_raster.RasterXSize * magnetic_raster.RasterYSize, 3))

count = 0
for x_pixel in x_pixels:
    for y_pixel in y_pixels:
        xy_magnetic[count, :] = make_row(x_pixel, y_pixel, padfTransform, magnetic)
        count += 1


# just values
np.savetxt("data/csv_files/gravity_matrix.csv", gravity, delimiter=",")
np.savetxt("data/csv_files/magnetic_matrix.csv", magnetic, delimiter=",")


# long lat coords and value
np.savetxt("data/csv_files/mines.csv", mines, delimiter=",")
np.savetxt("data/csv_files/gravity.csv", xy_gravity, delimiter=",")
np.savetxt("data/csv_files/magnetic.csv", xy_magnetic, delimiter=",")
