"""A collection of utilities for output handling and analysis."""



import json
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import geodesic

import preprocessing

# load adjacency matrix for contiguity graph.

G = np.loadtxt("data/adj_matrix.csv", delimiter=",", dtype=np.int64)  
levels = ["es", "ms", "hs"]
taus = {"hs": 0.2, "ms": 0.2, "es": 0.25}  # tolerances for capacity constraints
                                           # this must agree with the tolerances
                                           # used in the optimization model

def load_solution(path: str, data: dict) -> pd.DataFrame:
    """Form a SPA to school mapping from the given solution matrix.

    Args:
        path: The pickle file under results/ to read from.
        data: The preprocessed datasets.

    Returns:
        A DataFrame: schools on row indices, SPAs on column indices.
    """
    W = np.load("results/" + path, allow_pickle=True)
    x_S, spas = data["schools"][path[:2]], data["spas"]
    df = pd.DataFrame(data=W, index=x_S.SCH_CODE, columns=spas.STDYAREA)

    # Require one 1 in each column of the matrix so that the mapping works.
    assert (df.sum(axis=0) == np.ones((df.shape[1],))).all()
    return df


def is_feasible(solution: pd.DataFrame, level: str, data: dict,
                verbose: bool = False) -> bool:
    """Check solution feasibility of the given districting plan.

    Args:
        solution: The solution matrix in DataFrame format. The rows and
            columns are named after the school codes and SPA names,
            respectively.
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
        verbose: Whether to print individual constraints to stdout.

    Returns:
        True when each of the model's constraints are satisfied.
    """
    feasible = True

    # Check that there's exactly one 1 in each column.
    check1 = (solution.sum(axis=0) == np.ones((solution.shape[1],))).all()
    feasible = feasible and check1
    if verbose:
        print("One 1 in each column: ", "Pass" if check1 else "Fail")

    # Check that SPAs with a school are assigned to that school.
    check2 = True
    schools_csv = pd.read_csv("data/%s.csv" % level)
    for index, school in schools_csv.iterrows():
        if not(solution.iat[index, int(school.spa)] == 1):
            check2 = False
            break
    feasible = feasible and check2
    if verbose:
        print("SPAs with schools:    ", "Pass" if check2 else "Fail")

    # Check that SAZs are connected.
    check3 = True
    matrix_form = solution.to_numpy(dtype=np.int64)
    for i in range(solution.shape[0]):
        saz_index_set = np.where(matrix_form[i, :] == 1)[0]
        subgraph = G[saz_index_set, :][:, saz_index_set]
        subgraph = nx.from_numpy_matrix(subgraph)
        if not nx.is_connected(subgraph):
            check3 = False
            break
    feasible = feasible and check3
    if verbose:
        print("SAZs are connected:   ", "Pass" if check3 else "Fail")

    # Check that schools are within tau of their student capacities.
    p_col = {"es": "TOTAL_KG_5", "ms": "TOTAL_6_8", "hs": "TOTAL_9_12"}[level]
    differences = []
    for i in range(solution.shape[0]):
        attendance = data["spas"][p_col] @ matrix_form[i, :]
        capacity = data["schools"][level]["CAPACITY"].iloc[i]
        differences.append(float(attendance - capacity) / capacity)
    max_diff = max(differences, key=abs)
    check4 = abs(max_diff) <= taus[level]
    feasible = feasible and check4
    if verbose:
        print("SAZs within capacity: ", "Pass" if check4 else "Fail",
              "(%f largest)" % max_diff)

    return feasible


def df_to_mapping(matrix: pd.DataFrame) -> pd.Series:
    """Convert DataFrame solution format into a SPA -> school mapping.

    Args:
        matrix: The given solution in pandas.DataFrame format.

    Returns:
        A pandas.Series with assigned schools; SPAs on the indices.
    """
    assignment = matrix.apply(lambda col: np.where(col == 1)[0][0], axis=0)
    return assignment.map(lambda i: matrix.index.values[i])


def get_student_spa_map(level: str, data: dict) -> pd.Series:
    """Return a Series with the SPA that each student lives in.

    Args:
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
    """
    with open("data/%s_spa_groups.json" % level, "r") as fp:
        # Lists of student indices keyed by SPA name.
        groups = json.load(fp)
    index = data["students"][level].index
    spa_map = index.map(lambda id: list(filter(lambda tup: id in tup[1],
                                               groups.items()))[0][0])
    return pd.Series(spa_map, index)


def get_student_soln(soln: pd.DataFrame, level: str, data: dict) -> pd.Series:
    """Return the assigned school for each student according to solution.

    Args:
        soln: The given solution in pandas.DataFrame format.
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
    """
    soln_map = df_to_mapping(soln)
    spa_map = get_student_spa_map(level, data)
    return spa_map.parallel_apply(lambda spa: soln_map.loc[spa])


def get_nearest_schools(level: str, data: dict) -> pd.Series:
    """Returns the nearest school for each student.

    Args:
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
    """
    student_locs = data["students"][level].geometry
    schools = data["schools"][level].geometry
    schools.index = data["schools"][level].SCH_CODE

    def nearest_school(student):
        """Finds a student's nearest school by WGS84 ellipsoid distance."""
        t = list(student.coords)[0]  # Convert from shapely to geopy...
        dists = schools.map(lambda s: geodesic(list(s.coords)[0], t).meters)
        return data["schools"][level].iloc[dists.argmin()].SCH_CODE

    return student_locs.parallel_map(nearest_school)


def solution_quality(solution: pd.DataFrame, level: str,
                     data: dict) -> None:
    """Write to stdout statistics of the given districting solution.

    Args:
        soln: The given solution in pandas.DataFrame format.
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
    """
    feasible = is_feasible(solution, level, data)
    print("Does the solution satisfy model constraints?  ", feasible)

    new_student_schools = get_student_soln(solution, level, data)

    # Report number of students displaced by this districting plan.
    current_schools = data["students"][level].Current_S
    staying = (current_schools == new_student_schools).value_counts()
    percent = (staying[False] / len(data["students"][level])) * 100
    print("Students displaced from their current schools: %5d, %05.2f%%"
          % (staying[False], percent))

    nearest_schools = get_nearest_schools(level, data)

    # Report number of students assigned to their nearest school.
    attending_nearest = (new_student_schools == nearest_schools).value_counts()
    percent = (attending_nearest[True] / len(data["students"][level])) * 100
    print("Students assigned to their nearest school:     %5d, %05.2f%%"
          % (attending_nearest[True], percent))

    # Report on walking distances of students to assigned schools.
    student_locs = data["students"][level].geometry
    school_geoms = data["schools"][level].geometry
    school_geoms.index = data["schools"][level].SCH_CODE
    school_geoms = new_student_schools.apply(lambda s: school_geoms.loc[s])
    student_school = zip(student_locs, school_geoms)
    student_school = tuple(map(lambda t:
                               (list(t[0].coords)[0], list(t[1].coords)[0]),
                               student_school))
    distances = [geodesic(t[0], t[1]).meters for t in student_school]
    distances = np.array(distances)
    print("Geodesic distance of students to schools:      "
          "[%4.2fm min, %4.2fm avg, %4.2fm std, %4.2fm max]"
          % (distances.min(), distances.mean(), distances.std(),
             distances.max()))

    # Print a more detailed report of population/capacity in schools.
    attendances = new_student_schools.value_counts()
    capacities = data["schools"][level].CAPACITY
    capacities.index = data["schools"][level].SCH_CODE
    capacity_diff = (abs(attendances - capacities) / capacities) * 100
    print("School attendance-to-capacity difference:      "
          "[%05.2f%% min, %05.2f%% avg, %05.2f%% std, %05.2f%% max]"
          % (capacity_diff.min(), capacity_diff.mean(), capacity_diff.std(),
             capacity_diff.max()))

    # TODO: Implement more diversity statistics.


def plot_solution(solution: pd.DataFrame, level: str, data: dict,
                  filename: str) -> None:
    """Plot the districting solution geographically.

    Args:
        soln: The given solution in pandas.DataFrame format.
        level: One of "es", "ms", or "hs".
        data: The preprocessed datasets.
        filename: The path and name for the matplotlib output.
    """
    spa_map = df_to_mapping(solution)
    spas = data["spas"].copy()
    myspas = spas.geometry
    spas.index = data["spas"].STDYAREA
    sazs = spas.groupby(by=spa_map).apply(lambda saz: saz.unary_union)
    ax = gpd.GeoSeries(sazs).plot(figsize=(10, 10), alpha=0.5, edgecolor="k",
                                  cmap="tab20")
# =============================================================================
#     ax = gpd.GeoSeries(sazs).plot(figsize=(20, 20), alpha=0.5, edgecolor="k",
#                                   facecolor="none")
# =============================================================================
    gpd.GeoSeries(myspas).plot(ax=ax, alpha=0.5, edgecolor="b",
                                  facecolor="none")
    data["schools"][level].plot(ax=ax, color="blue")
    ax.set_axis_off()
    plt.savefig(filename)
    print("School districting map plotted at:", filename)


def main():
    data = preprocessing.load_data("2017_2018", epsg=4326)

    for level in levels:
        print("\n")
        filename = "%s_solution.pkl" % level
        solution_df = load_solution(filename, data)
        print("Reporting feasibility and quality of %s." % filename)
        is_feasible(solution_df, level, data, verbose=True)

        print()

        # solution_quality(solution_df, level, data)

        plot_solution(solution_df, level, data, "plots/ilp_%s.pdf" % level)


if __name__ == "__main__":
    main()
