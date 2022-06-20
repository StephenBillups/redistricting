"""Preprocessing functions for the county school data."""

# =============================================================================
# Conda packages used:  
#    (version numbers are probably wrong now, but this combination worked
#     at some point)
# 
#    python 3.7.11
#    geopandas 0.6.1
#    gdal 2.3.3
#    fiona 1.8.4
#
#    contextily 1.0.1
#    descartes 1.1.0
#    geopy 2.2.0
#    libpysal 4.6.2
#    matplotlib 3.4.3
#    numpy 1.21.2
#    pandarallel 1.5.5
#    pandas 1.3.5
#    shapely 1.6.5  
#
# To install each package, type following in ipython console:  
#    conda install -c conda-forge pkg-name 
#  for example
# conda install -c conda-forge contextily=1.0.1

# conda install -c gurobi gurobi



import csv
import itertools
from typing import Dict

#import shapely
# import fiona
import geopandas
# import scipy
import libpysal

import numpy
#import matplotlib.pyplot as plt

from pandarallel import pandarallel
pandarallel.initialize()


def load_data(year: str = "2017_2018", epsg: int = 3857) -> Dict[str,geopandas.GeoDataFrame]:
    """Load the county's school datasets into a dictionary.

    Args:
        year: A string formatted as "yyyy_yyyy" for a given school year.
        epsg: The CRS to use on the GIS data, which defaults to web mercator.

    Returns:
        A dictionary with the three datasets loaded as Geopandas tables.
    """ 
    year = "2017_2018"
    schools = geopandas.read_file("data/school_data/LCPS_Sites_%s.shp" % year)
    spas = geopandas.read_file("data/school_data/PlanningZones_%s.shp" % year)
    students = geopandas.read_file("data/school_data/Students_%s.shp" % year)
    data = {"schools": schools, "spas": spas, "students": students}

    for key, df in data.items():
        data[key] = df.to_crs(epsg=epsg)

    data["schools"] = clean_schools_df(data["schools"])
    data["students"] = clean_students_df(data["students"], data["spas"])

    return data


def clean_schools_df(schools: geopandas.GeoDataFrame) -> Dict[str, geopandas.GeoDataFrame]:
    """Clean the school dataset by removing invalid sites.

    Args:
        schools: The schools dataset.

    Returns:
        A dictionary splitting the dataset into elementary, middle, and high.
    """
    # First take out school sites that aren't schools.
    subset = schools.CLASS == "ELEMENTARY"
    subset = subset | (schools.CLASS == "MIDDLE")
    subset = subset | (schools.CLASS == "HIGH")
    schools = schools[subset]

    # Two schools didn't have a reported capacity.
    # Both were schools such as a vocational center, so they can be dropped.
    schools = schools[schools.CAPACITY != 0]

    es = schools[schools.CLASS == "ELEMENTARY"]
    ms = schools[schools.CLASS == "MIDDLE"]
    hs = schools[schools.CLASS == "HIGH"]

    return {"es": es, "ms": ms, "hs": hs}


def clean_students_df(students: geopandas.GeoDataFrame, spas: geopandas.GeoDataFrame) \
        -> Dict[str, geopandas.GeoDataFrame]:
    """Clean the students dataset by removing unqualifying students.

    Args:
        students: The students dataset.
        spas: The SPAs dataset.

    Returns:
        A dictionary splitting students into elementary, middle, and high.
    """
    # Grade "14" is a collection of Pre-K grades not for redistricting.
    students = students[students.GRADE != 14]

    # A few dozen students don't have coordinates inside of the county.
    county_polygon = spas.unary_union
    indices = students.geometry.parallel_map(county_polygon.contains)
    students = students[indices]

    es = students[(students.GRADE <= 5) | (students.GRADE == 13)]
    ms = students[(students.GRADE >= 6) & (students.GRADE <= 8)]
    hs = students[(students.GRADE >= 9) & (students.GRADE <= 12)]

    return {"es": es, "ms": ms, "hs": hs}


def write_spa_data(spas: geopandas.GeoDataFrame, epsg: int) -> None:
    """Format the SPA data and write it to a CSV file.

    Args:
        spas: The SPAs dataset.
        epsg: The final projection for Gurobi to use as coefficients.
    """
    population_table = spas.loc[:, 
        ["TOTAL_KG_5", "TOTAL_6_8", "TOTAL_9_12", "ELEM_CODE", "INT_CODE", "HIGH_CODE"]]

    spa_centroids = spas.centroid
    spa_centroids = geopandas.GeoSeries(spa_centroids).to_crs(epsg=epsg)
    population_table["x"] = spa_centroids.x
    population_table["y"] = spa_centroids.y

    cols = ["x", "y", "TOTAL_KG_5", "TOTAL_6_8", "TOTAL_9_12","ELEM_CODE", "INT_CODE", "HIGH_CODE"]
    population_table = population_table.reindex(columns=cols)

    population_table.to_csv("data/spas.csv")


def write_schools_data(schools: Dict[str, geopandas.GeoDataFrame], X: str,
                       spas: geopandas.GeoDataFrame, epsg: int) -> None:
    """Format the school data needed and write it to a CSV file.

    Args:
        schools: The dictionary containing filtered school tables.
        X: X as in ${}_X n_s$ for grade levels "es", "ms", and "hs".
        spas: The SPAs dataset.
        epsg: The final projection for Gurobi to use as coefficients.
    """
    x_schools = schools[X]
    x_schools = x_schools.to_crs(epsg=epsg)
    spas = spas.to_crs(epsg=epsg)

    schools_table = x_schools.loc[:, ["SCH_CODE","CAPACITY"]]
    schools_table["x"] = x_schools.geometry.x
    schools_table["y"] = x_schools.geometry.y
    schools_table = schools_table.reindex(columns=["SCH_CODE","x", "y", "CAPACITY"])
    schools_table.rename(columns={"CAPACITY": "capacity"}, inplace=True)

    xs_setbits = []
    for i, j in itertools.product(range(len(x_schools)), range(len(spas))):
        if spas.iloc[j].geometry.contains(x_schools.iloc[i].geometry):
            xs_setbits.append(j)
    schools_table["spa"] = xs_setbits

    schools_table.to_csv("data/%s.csv" % X)


def write_adjacency_matrix(spas: geopandas.GeoDataFrame) -> None:
    """Create the adjacency matrix and write it to a CSV file.

    Args:
        spas: The SPAs dataset.
    """
    adj_mat = libpysal.weights.Rook.from_dataframe(spas, id_order=spas.index)
    adj_matrix = adj_mat.full()[0].astype(numpy.int64)

    with open("data/adj_matrix.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(adj_matrix)


def write_existing_sazs(schools: geopandas.GeoDataFrame, X: str,
                        spas: geopandas.GeoDataFrame) -> None:
    """Build and write the SAZ assignment matrix for a school level.

    Args:
        schools: The dictionary containing filtered school tables.
        X: X as in ${}_X n_s$ for grade levels "es", "ms", and "hs".
        spas: The SPAs dataset.
    """
    x_schools = schools[X]
    sch_codes = dict(zip(x_schools.SCH_CODE, range(len(x_schools))))
    grade_col = {"es": "ELEM", "ms": "MID", "hs": "HIGH"}
    spa_to_saz = list(map(lambda spa: sch_codes[spa], spas[grade_col[X]]))
    x_W = numpy.zeros((len(x_schools), len(spas)), dtype=numpy.int)
    for j, i in enumerate(spa_to_saz):
        x_W[i, j] = 1
    x_W.dump("data/existing_%s_sazs.pkl" % X)


def main():
    """Preprocess school data for input to Gurobi."""
    # Computing centroids requires a projection, but Gurobi requires
    # smaller coefficients, so Web Mercator is first used, then switched
    # to WGS84 lat/long before writing to file.
    crs = 4326  # WGS84 projection
    data = load_data(year="2017_2018", epsg=3857)
    schools, spas = data["schools"], data["spas"]


    # output data needed for the optimization model to csv files
    # and/or pickle files
    write_spa_data(spas, crs)
    for grade in ["es", "ms", "hs"]:
        write_schools_data(schools, grade, spas, crs)
        write_existing_sazs(schools, grade, spas)
    write_adjacency_matrix(spas)

if __name__ == "__main__":
    main()
