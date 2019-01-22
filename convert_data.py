from osgeo import gdal
import io
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal_array

import dbfread


def make_row(x_pix, y_pix, padfTrans, field):
    x = padfTrans[0] + x_pix * padfTrans[1] + y_pix * padfTrans[2]
    y = padfTrans[3] + x_pix * padfTrans[4] + y_pix * padfTrans[5]

    return [[x, y, field[x_pix, y_pix]]]


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

gravity = gdal_array.LoadFile(gravity_filepath)
magnetic = gdal_array.LoadFile(magnetic_filepath)


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

x_pixels = np.arange(10)
y_pixels = np.arange(10)

xy_gravity = None

for x_pixel in x_pixels:
    for y_pixel in y_pixels:
        if xy_gravity is None:
            xy_gravity = make_row(x_pixel, y_pixel, padfTransform, gravity)
        else:
            print(x_pixel, y_pixel)
            xy_gravity = np.append(
                xy_gravity, make_row(x_pixel, y_pixel, padfTransform, gravity), axis=0
            )

# create magnetic triple
magnetic_raster = gdal.Open(magnetic_filepath)

x_pixels = np.arange(magnetic_raster.RasterXSize)
y_pixels = np.arange(magnetic_raster.RasterYSize)

# now convert to lon lat
padfTransform = magnetic_raster.GetGeoTransform()

x_pixels = np.arange(10)
y_pixels = np.arange(10)

xy_magnetic = None

for x_pixel in x_pixels:
    for y_pixel in y_pixels:
        if xy_magnetic is None:
            xy_magnetic = make_row(x_pixel, y_pixel, padfTransform, magnetic)
        else:
            print(x_pixel, y_pixel)
            xy_magnetic = np.append(
                xy_magnetic, make_row(x_pixel, y_pixel, padfTransform, magnetic), axis=0
            )


# just values
np.savetxt("data/csv_files/gravity_matrix.csv", gravity, delimiter=",")
np.savetxt("data/csv_files/magnetic_matrix.csv", magnetic, delimiter=",")


# long lat coords and value
np.savetxt("data/csv_files/mines.csv", mines, delimiter=",")
np.savetxt("data/csv_files/gravity.csv", xy_gravity, delimiter=",")
np.savetxt("data/csv_files/magnetic.csv", xy_magnetic, delimiter=",")


# x = np.arange(gravity_raster.RasterXSize)
# y = np.arange(gravity_raster.RasterYSize)
# posX = px_w * x + rot1 * y + xoffset
# posY = rot2 * x + px_h * y + yoffset
#
# # shift to the center of the pixel
# posX += px_w / 2.0
# posY += px_h / 2.0
#
# # coords = gravity_raster.GetProjectionRef()
# #
# # print(coords)
#
# print(xoffset, px_w, rot1, yoffset, px_h, rot2)

# print(posX)

#
# # Check type of the variable 'raster'
# type(raster)
#
# # Projection
# raster.GetProjection()

# # Dimensions
# n_x = raster.RasterXSize
# n_y = raster.RasterYSize
#
# # Number of bands
# raster.RasterCount
#
# # Metadata for the raster dataset
# raster.GetMetadata()
#
# # Read the raster band as separate variable
# band = raster.GetRasterBand(1)
#
# # Check type of the variable 'band'
# type(band)
#
# # Data type of the values
# gdal.GetDataTypeName(band.DataType)
#
#
# # Compute statistics if needed
# if band.GetMinimum() is None or band.GetMaximum()is None:
#     band.ComputeStatistics(0)
#     print("Statistics computed.")
#
# # Fetch metadata for the band
# band.GetMetadata()
#
# # Print only selected metadata:
# print("[ NO DATA VALUE ] = ", band.GetNoDataValue())  # none
# print("[ MIN ] = ", band.GetMinimum())
# print("[ MAX ] = ", band.GetMaximum())


# rasterArray = raster.ReadAsArray()
#
# print(rasterArray)
# x = np.arange(n_x)
# y = np.arange(n_y)
# X, Y = np.meshgrid(x, y)