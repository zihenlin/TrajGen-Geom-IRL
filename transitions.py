from typing import List

import torch as t
import pandas as pd
import networkx as nx 
import osmnx as ox
import numpy as np
import geopandas as gpd

import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib_map_utils.core.scale_bar import ScaleBar, scale_bar
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow



def simple_gravity (buildings : pd.DataFrame):
    beta = 2
    
    coor = t.tensor(buildings.loc[:, ["x", "y"]].to_numpy())
    vol = t.tensor(buildings.floorspace.to_numpy())
    distance = t.cdist(coor, coor, 2).pow(beta)
    distance[distance == 0] = 1e5
    scores = vol.div(distance)
    scores = scores.fill_diagonal_(0)
    transition = scores.div(t.sum(scores, 1).view(-1, 1)) 

    return transition

def inverse_euclidean(buildings : pd.DataFrame):
    
    coor = t.tensor(buildings.loc[:, ["x", "y"]].to_numpy())
    vol = t.tensor(buildings.floorspace.to_numpy())
    eu_distance = t.cdist(coor, coor, 2)
    eu_distance /= eu_distance.sum(1).view(-1,1)
    eu_distance = 1 - eu_distance
    eu_distance.fill_diagonal_(0)
    
    transition = eu_distance.div(t.sum(eu_distance, 1).view(-1, 1))

    return transition

def get_transition(buildings : pd.DataFrame, use_simple_gravity : bool, density : float):

    transition = simple_gravity(buildings) if use_simple_gravity == True else inverse_euclidean(buildings) 
    threshold = transition.quantile(1-density)
    transition[transition < threshold] = 0
    out_D_inv = (1/ (transition.sum(1) + 1))
    transition = out_D_inv.view(-1, 1) * transition # Left Laplacian

    return transition 
    
def plot_buildings_and_connectivity(gdf_buildings, policy, name,ax=None, show_legends=False):
    if ax is None:
        ax = plt.gca()

    sparse_transitions = policy.numpy()
    sparse_transitions[sparse_transitions < np.quantile(policy,1-0.005)] = 0
    G = nx.from_numpy_array(sparse_transitions, create_using=nx.DiGraph)
    edge_size = [edge[2]["weight"] for edge in G.edges(data=True)]
    usg_code_str = gdf_buildings.replace({"USG_CODE": {1:"Residential", 2:"Mixed R/C", 3:"Commercial", 4:"Public", 5:"Industry", 6:"Retirement"}}).USG_CODE.to_list()
    legend = {val:key for key,val in {1:"Residential", 2:"Mixed R/C", 3:"Commercial", 4:"Public", 5:"Industry", 6:"Retirement"}.items()}
    for n, usg in zip(G.nodes(), usg_code_str):
        G.nodes[n]["usg"] = usg 

    coor_4326 = gdf_buildings.loc[:, ["y", "x"]].T.to_numpy()

    node_cm = plt.get_cmap('tab10')
    # edge_cm = plt.get_cmap('Greys')
    values = gdf_buildings.USG_CODE
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=max(values))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=node_cm)
    
    # Using a figure to use it as a parameter when calling nx.draw_networkx
    for label in legend:
        ax.plot([coor_4326[1].mean()],[coor_4326[0].mean()],color=scalarMap.to_rgba(legend[label]),label=label)
    
    nx.draw(G, dict(zip(G.nodes, np.column_stack((coor_4326[1], coor_4326[0])))), vmin=0, vmax= max(values), node_size=3, width=edge_size, arrowsize=edge_size, node_color=values, cmap=node_cm, ax=ax)
    
    # Get building patches for legend (but don't add legend yet)
    building_patches = ax.collections[-1].legend_elements()[0]
    
    # Configure plot appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}")
    
    return ax, building_patches

def create_visualization_grid_connectivity(gdf_buildings, transitions,model_types, ncols=2):
    assert len(transitions) == len(model_types)
    nrows = (len(transitions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, dpi=300)
    buildings_osmnx = ox.features.features_from_bbox(31.781228258516897, 31.789756905743264, 35.20582386130486, 35.21960771722701, tags = {'building':True} )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
    
    building_types = ["Residential", "Mixed R/C", "Commercial", "Public", "Industry", "Elderly Home"]
    
    # Create subplots
    for idx, (name,transition) in enumerate(zip(model_types, transitions)):
        row, col = idx // ncols, idx % ncols
        buildings_osmnx.plot(alpha=0.2, color="grey", ax=axes[row,col])
        ax, building_patches = plot_buildings_and_connectivity(
            gdf_buildings, transition, name, ax=axes[row, col]
        )
    
    # Add figure legends
    building_legend = fig.legend(
        building_patches, 
        building_types,
        title="Land Use",
        loc='center left',
        bbox_to_anchor=(0.87, 0.55),  # Positioned at top right
        title_fontsize=10,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
    )

    x_max = axes.shape[0] - 1
    y_max = axes.shape[1] - 1
    
    north_arrow(axes[x_max,y_max], scale=.2, shadow=False, location="lower right", rotation={"crs": gdf_buildings.crs, "reference": "center"}, label={"position": "bottom", "text": "North", "fontsize": 8}, aob={"bbox_to_anchor":(1.2,0.0), "bbox_transform":axes[x_max,y_max].transAxes})
    scale_bar(axes[x_max,y_max], location="lower right", style="boxes", bar={"projection": gdf_buildings.crs, "major_div": 1, "max":200,
        "minor_div": 2,},labels={"loc": "below", "style": "major","fontsize":10},units={"loc": "bar", "fontsize": 10}, aob={"bbox_to_anchor":(1.5,0.0), "bbox_transform":axes[x_max,y_max].transAxes})

    fig.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return fig, axes
    
def plot_transition(buildings : pd.DataFrame, transitions : List[t.Tensor], ncols=1, model_types= ["Linear", "MLP", "GCN", "GraphConv"]):
    gdf = gpd.GeoDataFrame(buildings,crs='EPSG:2039', geometry=gpd.points_from_xy(x=buildings.x, y=buildings.y)).to_crs(4326)
    x, y = pyproj.Transformer.from_crs("EPSG:2039", 4326).transform(gdf.loc[:, "x"],gdf.loc[:, "y"])
    gdf.loc[:, "x"] = y
    gdf.loc[:, "y"] = x
    
    fig, axs = create_visualization_grid_connectivity(gdf, transitions, model_types,ncols)
    plt.show()