import torch as t
from torch_geometric.nn import Sequential, GCNConv, GraphConv
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib_map_utils.core.scale_bar import ScaleBar, scale_bar
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow



class Linear(t.nn.Module):
    def __init__(self, n_features,seed=0):
        super().__init__()
        t.manual_seed(seed)
        self.weights = t.nn.parameter.Parameter(t.ones((n_features)))

    def forward(self, features):
        return features @ self.weights


class Perceptron(t.nn.Module):
    def __init__(self, n_features,seed=0):
        super().__init__()
        t.manual_seed(seed)
        self.linear = t.nn.Sequential(t.nn.Linear(n_features, 1))

    def forward(self, features):
        return self.linear(features)


class MLP(t.nn.Module):
    def __init__(self, n_features,seed=0):
        super().__init__()
        t.manual_seed(seed)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(n_features, 32),
            t.nn.ReLU(inplace=True),
            t.nn.BatchNorm1d(32),
            
            t.nn.Linear(32, 32),
            t.nn.ReLU(inplace=True),
            t.nn.BatchNorm1d(32),
            
            t.nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.mlp(features)


class GCN(t.nn.Module):
    def __init__(self, n_features, p_transition,seed=5):
        super().__init__()
        t.manual_seed(seed)
        self.gcn = Sequential(
            "x, edge_index, edge_weight",
            [
                (GCNConv(n_features, 32, normalize=False, improved=True), "x, edge_index, edge_weight -> x"),
                t.nn.BatchNorm1d(32),
                t.nn.ReLU(inplace=True),

                (GCNConv(32, 32, normalize=False, improved=True), "x, edge_index, edge_weight -> x"),
                t.nn.BatchNorm1d(32),
                t.nn.ReLU(inplace=True),

                (GCNConv(32, 1, normalize=False, improved=True), "x, edge_index, edge_weight -> x")
            ],
        )

        self.coo_transition = p_transition.to_sparse_coo()

    def forward(self, features):
        return self.gcn(
            # X,
            features.float(),
            self.coo_transition.indices(),
            edge_weight=self.coo_transition.values().float())
        

class GRAPHCONV(t.nn.Module):
    def __init__(self, n_features, p_transition,seed=5):
        super().__init__()
        t.manual_seed(seed)
        self.conv = Sequential(
            "x, edge_index, edge_weight",
            [
                (GraphConv(n_features, 32, aggr="add"), "x, edge_index, edge_weight -> x"),
                t.nn.BatchNorm1d(32),
                t.nn.ReLU(inplace=True),

                (GraphConv(32, 32, aggr="add"), "x, edge_index, edge_weight -> x"),
                t.nn.BatchNorm1d(32),
                t.nn.ReLU(inplace=True),

                (GraphConv(32, 1, aggr="add"), "x, edge_index, edge_weight -> x")
            ],
        )

        self.coo_transition = p_transition.to_sparse_coo()

    def forward(self, features):
        return self.conv(
            # X,
            features.float(),
            self.coo_transition.indices(),
            edge_weight=self.coo_transition.values().float())

def plot_reward_correlation(buildings, factor, name, R_values):
    fig, axs = plt.subplots(2, 2, dpi=300, sharex=True, sharey=True)

    categories = ["Residential", "Mixed R/C", "Commercial", "Public", "Industry", "Retirement"]
    cmap = plt.cm.tab10
    
    # For first subplot, plot each category separately to create legend
    for i, category in enumerate(categories):
        mask = buildings.USG_CODE == i+1  # Assuming USG_CODE values are 0,1,2,3,4,5
        axs[0,0].scatter(factor[mask], buildings.mlp_reward[mask], 
                         c=cmap(i), s=5, label=category)
        axs[0,1].scatter(factor[mask], buildings.linear_reward[mask], c=cmap(i), cmap="tab10", s=5)
        axs[1,0].scatter(factor[mask], buildings.gcn_reward[mask], c=cmap(i), cmap="tab10", s=5)
        axs[1,1].scatter(factor[mask], buildings.graphconv_reward[mask], c=cmap(i), cmap="tab10",s=5)
    
    axs[0,0].set_ylabel("Normalized Rewards")
    axs[1,0].set_ylabel("Normalized Rewards")
    axs[1,0].set_xlabel(f"{name}")
    axs[1,1].set_xlabel(f"{name}")
    
    for r, ax, title in zip(R_values, axs.flatten(), ["MLP", "Linear", "GCN", "GraphConv"]):
        ax.annotate(f"r={str(np.round(r, 2))}", xy=(0.8,0.9))
        ax.set_title(title)
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Land Use', bbox_to_anchor=(1.02, 0.5), 
               loc='center left', borderaxespad=0)
    fig.tight_layout()
    plt.show()

def plot_reward_spatial_patterns(buildings):
    gdf = gpd.GeoDataFrame(buildings,crs='EPSG:2039', geometry=gpd.points_from_xy(x=buildings.x, y=buildings.y)).to_crs(4326)
    x, y = pyproj.Transformer.from_crs("EPSG:2039", 4326).transform(gdf.loc[:, "x"], gdf.loc[:, "y"])
    gdf.loc[:, "x"] = y
    gdf.loc[:, "y"] = x
    fig, axs = plt.subplots(4,3, figsize=(9,9), dpi=300)

    title = ["Initial Rewards", "Final Rewards", "Differences", "STD"]
    cols = ["Linear Reward","MLP Rewards", "GCN Rewards", "GraphConv Rewards"]
    names = ["linear", "mlp", "gcn", "graphconv"]
    buildings_osmnx = ox.geometries.geometries_from_bbox(31.78122826, 31.78975691, 35.20582386, 35.21960772, tags = {'building':True} )
    
    for idx, ax in enumerate(axs):
        gdf.plot(ax=ax[0], x="x", y="y", kind="hexbin", C="initial_reward", cmap="coolwarm")
        gdf.plot(ax=ax[1], x="x", y="y", kind="hexbin", C=names[idx] + "_reward", cmap="coolwarm")
        gdf.plot(ax=ax[2], x="x", y="y", kind="hexbin", C=names[idx] + "_diff", cmap="coolwarm")
        for i in ax:
            buildings_osmnx.plot(ax=i,alpha=0.2, color="grey")
            '''Remove ColorBar'''
            cb = i.collections[0].colorbar
            cb.remove()
    for idx, ax in enumerate(axs[0,:]):
        ax.set_title(title[idx])
    for idx, ax in enumerate(axs.flatten()):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("")
        if (idx % 3) == 0:
            ax.set_ylabel(cols[int(idx/3)], fontsize=12)
        else:
            ax.set_ylabel("")
        
    north_arrow(
        axs[0,2], scale=.2, shadow=False, location="upper right", rotation={"crs": gdf.crs, "reference": "center"}, label={"position": "bottom", "text": "North", "fontsize": 8}, aob={"bbox_to_anchor":(1.2,1.0), "bbox_transform":axs[0,2].transAxes})
    # ax.add_artist(ScaleBar(1))
    scale_bar(axs[3,1], location="lower center", style="boxes", bar={"projection": gdf.crs, "major_div": 1, "max":200,
            "minor_div": 2,},labels={"loc": "below", "style": "major","fontsize":10},units={"loc": "bar", "fontsize": 10}, aob={"bbox_to_anchor":(0.5,-0.3), "bbox_transform":axs[3,1].transAxes})
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
    fig.tight_layout()
    plt.show()