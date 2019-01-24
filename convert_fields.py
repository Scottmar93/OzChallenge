from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal_array


def make_row(x_pix, y_pix, padfTrans, field):
    x = padfTrans[0] + x_pix * padfTrans[1] + y_pix * padfTrans[2]
    y = padfTrans[3] + x_pix * padfTrans[4] + y_pix * padfTrans[5]

    return x, y, field[y_pix, x_pix]


def make_array(filepath):
    raster = gdal.Open(filepath)
    field = gdal_array.LoadFile(filepath)

    x_pixels = np.arange(raster.RasterXSize)
    y_pixels = np.arange(raster.RasterYSize)
    padfTransform = raster.GetGeoTransform()

    xy_field = np.zeros((raster.RasterXSize * raster.RasterYSize, 3))

    count = 0
    for y_pixel in y_pixels:
        for x_pixel in x_pixels:
            xy_field[count, :] = make_row(x_pixel, y_pixel, padfTransform, field)
            count += 1

    return xy_field


# load data
gravity_filepath = "data/gravity/SA_GRAV.ers"
magnetic_filepath = "data/magnetic/SA_TMI_RTP.ers"
magnetic_1VD_filepath = "data/magnetic_1VD/SA_TMI_RTP_1VD.ers"


gravity = gdal_array.LoadFile(gravity_filepath).T
magnetic = gdal_array.LoadFile(magnetic_filepath).T
magnetic_1VD = gdal_array.LoadFile(magnetic_1VD_filepath).T

# plt.ion()

xy_gravity = make_array(gravity_filepath)
# xy_magnetic = make_array(magnetic_filepath)
# xy_magnetic_1VD = make_array(magnetic_1VD_filepath)

plt.contourf(gravity)
plt.show()


# just values
# np.savetxt("data/csv_files/gravity_matrix.csv", gravity, delimiter=",")
# np.savetxt("data/csv_files/magnetic_matrix.csv", magnetic, delimiter=",")
# np.savetxt("data/csv_files/magnetic_1VD_matrix.csv", magnetic_1VD, delimiter=",")
#
#
# # long lat coords and value
# np.savetxt("data/csv_files/mines.csv", mines, delimiter=",")
# np.savetxt("data/csv_files/gravity.csv", xy_gravity, delimiter=",")
# np.savetxt("data/csv_files/magnetic.csv", xy_magnetic, delimiter=",")
# np.savetxt("data/csv_files/magnetic_1VD.csv", xy_magnetic_1VD, delimiter=",")

# plotting of gravity on coordinates
# x = xy_gravity[:, 0]
# y = xy_gravity[:, 1]
# grav = xy_gravity[:, 2]
#
# plt.contourf(x, y, gravity)
# plt.show()
