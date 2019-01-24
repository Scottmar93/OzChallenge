import dbfread
import numpy as np


def convert_mine(filepath):

    dbf = dbfread.DBF(filepath)
    mines = np.zeros([len(dbf), 2])

    # convert mines to an array of [ [long, lat] ] coords
    for i, mine in enumerate(dbf):
        mines[i, :] = mine["LONGITUDE"], mine["LATITUDE"]

    return mines


# copper mines
copper_filepath = "data/mines/copper_containted_resource.dbf"
copper = convert_mine(copper_filepath)
np.savetxt("data/csv_files/copper_contained.csv", copper, delimiter=",")

# gold contained
gold_filepath = "data/mines/gold_containted_resource.dbf"
gold = convert_mine(gold_filepath)
np.savetxt("data/csv_files/gold_contained.csv", gold, delimiter=",")

# iron contained
iron_filepath = "data/mines/iron_containted_resource.dbf"
iron = convert_mine(iron_filepath)
np.savetxt("data/csv_files/iron_contained.csv", iron, delimiter=",")

# graphite
graphite_filepath = "data/mines/graphite/graphite_containted_resource.dbf"
graphite = convert_mine(graphite_filepath)
np.savetxt("data/csv_files/graphite_contained.csv", graphite, delimiter=",")

# heavy_minerals
heavy_minerals_filepath = (
    "data/mines/heavy_minerals/heavyminerals_containted_resource.dbf"
)
heavy_minerals = convert_mine(heavy_minerals_filepath)
np.savetxt("data/csv_files/heavy_minerals_contained.csv", heavy_minerals, delimiter=",")

# lead
lead_filepath = "data/mines/lead/lead_containted_resource.dbf"
lead = convert_mine(lead_filepath)
np.savetxt("data/csv_files/lead_contained.csv", lead, delimiter=",")

# silver
silver_filepath = "data/mines/silver/silver_containted_resource.dbf"
silver = convert_mine(silver_filepath)
np.savetxt("data/csv_files/silver_contained.csv", silver, delimiter=",")

# uranium
uranium_filepath = "data/mines/uranium/uranium_containted_resource.dbf"
uranium = convert_mine(uranium_filepath)
np.savetxt("data/csv_files/uranium_contained.csv", uranium, delimiter=",")

# zinc
zinc_filepath = "data/mines/zinc/zinc_containted_resource.dbf"
zinc = convert_mine(zinc_filepath)
np.savetxt("data/csv_files/zinc_contained.csv", zinc, delimiter=",")

# something contained
something = np.append(copper, gold, axis=0)
something = np.append(something, iron, axis=0)
something = np.append(something, graphite, axis=0)
something = np.append(something, heavy_minerals, axis=0)
something = np.append(something, lead, axis=0)
something = np.append(something, silver, axis=0)
something = np.append(something, uranium, axis=0)
something = np.append(something, zinc, axis=0)
np.savetxt("data/csv_files/something_contained.csv", something, delimiter=",")
