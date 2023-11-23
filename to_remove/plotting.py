
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
# from osgeo import gdalconst, ogr  #--import-mode=importlib

from src.faster.custom_typing import Coordinate
from src.faster.node_types import NodeType
# from src.faster.simplified_aokp import pre_AOKP
from src.user_inputs import path_subcatchments


def plot_with_graph(graph: nx.DiGraph) -> Figure:
    """
    The indices in `connections` need to match `coordinates`.
    """
    coordinates: list[Coordinate] = [coordinate for _,
                                     coordinate in graph.nodes.data("coordinates")]

    sc2outlet = pre_AOKP(coordinates)
    outlet2sc = {val: key for key, val in sc2outlet.items()}

    ds_polygonized = ogr.Open(path_subcatchments, gdalconst.GA_ReadOnly)
    layer_polygonized = ds_polygonized.GetLayer()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig = plt.figure()

    sc_centroids = []
    # for i_outlet, i_sc in zip(idxs_outlet, idxs_sc):
    for data_node in graph.nodes().values():
        idx_node = data_node["idx_node"]
        idx_sc = outlet2sc[idx_node]

        polygon = layer_polygonized.GetFeature(idx_sc)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            boundary = polygon_geometry.GetBoundary()
            boundary_points = boundary.GetPoints()
            boundary_points_arr = np.array(boundary_points)
            boundary_centroid = boundary.Centroid().GetPoint()[:2]
            sc_centroids.append(boundary_centroid)
            plt.plot(boundary_points_arr[:, 0], boundary_points_arr[:,
                     1], color=colors[idx_node % len(colors)])
            plt.text(*boundary_centroid, f"{idx_node}")

    node_type_colors = {
        NodeType.NOTHING: "gray",
        NodeType.WWTP: "r",
        NodeType.WWPS: "b",
    }

    arrow_styles = {
        NodeType.NOTHING: "--",
        NodeType.WWPS: "-",
    }

    for _, node_data in graph.nodes().items():
        node_color = node_type_colors[node_data["node_type"]]
        plt.scatter(*node_data["coordinates"], s=50, c=node_color)

    nx.draw(graph, {id_node: attrs["coordinates"]
            for id_node, attrs in graph.nodes().items()}, with_labels=True)

    plt.axis('equal')
    plt.show()

    return fig
