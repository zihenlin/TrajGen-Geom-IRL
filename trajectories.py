from typing import Tuple

import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely import Point, LineString, Polygon

import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib_map_utils.core.scale_bar import ScaleBar, scale_bar
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow
from matplotlib.patches import FancyArrowPatch 
from adjustText import adjust_text
import seaborn as sns



def generate_one_traj(policy, terminal_states, discount, buildings):
    n_states = len(buildings)
    start = np.random.choice(range(n_states))
    # final = np.random.choice(terminal_states)  # config 3, random terminal state
    # final = terminal_states # config 2, terminal state set
    final = start # config 1, start as terminal state
    state = start
    trajectory = [start]
    trans_prob = policy[state].clone()
    while state == start:
        next_state = np.random.choice(range(n_states), p=trans_prob.numpy())
        if next_state != start:
            trajectory.append(next_state)
            state = next_state
        
    # while state not in final:  # config 2, terminal state set
    while state != final:  # config 1 & 3
        trans_prob = policy[state].clone()
        
        trans_prob[trajectory[1:]] = 0  # config 1
        # trans_prob[trajectory] = 0  # config 2 & 3
        
        idx = len(trajectory) + 1
        if idx >= 25: 
            discount_factor = np.zeros(n_states)
            discount_factor[:] = 1e-3 / (n_states - len(trajectory) - 1)  # survival probability
            discount_factor[final] = 1 - 1e-3
        else:
            discount_factor = np.zeros(n_states)
            discount_factor[:] = discount[idx][0] / (n_states - len(trajectory) - 1)  # survival probability
            discount_factor[final] = discount[idx][1]

        trans_prob *= discount_factor
        if trans_prob.sum() == 0: # trajectory dies and reset
            trajectory = [] # restart
            start = np.random.choice(range(n_states))
            
            # comment both final for config 2, terminal set
            # final = np.random.choice(terminal_states) # config 3, random terminal state
            final = start # config 1, start as terminal state
            trajectory.append(start)
            state = np.random.choice(range(n_states), p=policy[start].numpy())
            trans_prob = policy[state].clone()
            while state == start:
                next_state = np.random.choice(range(n_states), p=trans_prob.numpy())
                if next_state != start:
                    trajectory.append(next_state)
                    state = next_state
        
            trajectory.append(state)
            continue
        trans_prob /= trans_prob.sum()
        next_state = np.random.choice(range(n_states), p=trans_prob.numpy())
        trajectory.append(next_state)
        state = next_state
    if trajectory[-1] == start:
        trajectory = trajectory[:-1]
    return buildings.index[trajectory]

def get_traj_stats(trajs,buildings):
    segment_lengths = []
    convex_hulls = []
    USGs = []

    for traj in trajs:
        if len(traj) < 3:
            continue
        coor = [buildings.loc[item, ["x", "y"]].to_list() for item in traj]
        usg = [buildings.loc[item, "USG_CODE"] for item in traj]

        d = np.diff(coor, axis=0)
        segment_length = np.hypot(d[:,0], d[:,1])
        convex_hull = gpd.GeoDataFrame(geometry=[LineString([Point(*item) for item in coor])], crs="EPSG:2039").convex_hull.area.values
        convex_hulls.append(convex_hull)
        segment_lengths.append(segment_length)
        USGs.append(usg)

    return segment_lengths, convex_hulls, USGs

def get_segment_stat(segment_lengths):
    """Get statistics of empirical trajectory"""
    segment_length_np = np.zeros([len(segment_lengths),len(max(segment_lengths,key = lambda x: len(x)))])
    for idx, segment_length in enumerate(segment_lengths):
        segment_length_np[idx][:len(segment_length)] = segment_length
    total_empirical_length = segment_length_np.sum(1)
    mean_empirical_segment_length = segment_length_np.mean(1)
    return total_empirical_length, mean_empirical_segment_length, segment_length_np

def plot_buildings_and_trajectories(gdf_buildings, routines, agent_idx, ax=None, show_legends=False):
    if ax is None:
        ax = plt.gca()
    
    # Plot buildings with categorical colors (without legend)
    ax = gdf_buildings.plot(
        column="USG_CODE",
        categorical=True,
        cmap="Pastel2",
        ax=ax,
        markersize=1,
        legend=False  # Remove subplot legend
    )
    
    # Get building patches for legend (but don't add legend yet)
    building_patches = ax.collections[-1].legend_elements()[0]
    
    # Define trajectory visualization parameters
    colors = ["Reds", "Purples", "Oranges", "Blues"]
    traj_types = ["Linear", "MLP", "GCN", "GraphConv"]
    annotations = []
    trajectory_lines = []  # Store line objects for legend
    
    # Plot trajectories for each model
    for color, routine, traj_type in zip(colors, routines, traj_types):
        sample_routine = routine[agent_idx][np.nonzero(routine[agent_idx])[0].flatten()].astype(int)
        norm = mpl.colors.TwoSlopeNorm(1, vmin=0, vmax=len(sample_routine))
        cmap = mpl.colormaps[color]
        
        source = gdf_buildings.loc[sample_routine[0], ["x", "y"]]
        for idx in range(1, len(sample_routine)):
            target = gdf_buildings.loc[sample_routine[idx], ["x", "y"]]
            color = cmap(norm(idx))
            arrow = FancyArrowPatch(
                posA=source.tolist(),
                posB=target.tolist(),
                arrowstyle="->",
                linewidth=0.9,
                mutation_scale=2,
                color=color
            )
            ax.add_patch(arrow)
            if len(sample_routine) > 3:
                annotations.append(
                    ax.annotate(f"{idx}", xy=((source+target)/2).tolist(), color=color)
                )
            source = target
        
        # Create line for legend but don't show subplot legend
        line = ax.plot(source[0], source[1], markersize=0, color=color, label=traj_type)[0]
        trajectory_lines.append(line)
    
    # Adjust annotation positions to avoid overlap
    adjust_text(annotations, ax=ax)
    
    # Configure plot appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Agent {agent_idx}")
    
    return ax, building_patches, trajectory_lines

def create_visualization_grid(gdf_buildings, routines, agent_indices, ncols=2):
    nrows = (len(agent_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, dpi=300)
    buildings_footprint = ox.features.features_from_bbox(north=gdf_buildings.y.max(), east=gdf_buildings.x.max(), south=gdf_buildings.y.min(), west=gdf_buildings.x.min(), tags={"building":True})
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
    
    building_types = ["Residential", "Mixed R/C", "Commercial", "Public", "Industry", "Elderly Home"]
    traj_types = ["Linear", "MLP", "GCN", "GraphConv"]
    
    # Create subplots
    for idx, agent_idx in enumerate(agent_indices):
        row, col = idx // ncols, idx % ncols
        buildings_footprint.plot(alpha=0.1, color="grey", ax=axes[row,col])
        ax, building_patches, trajectory_lines = plot_buildings_and_trajectories(
            gdf_buildings, routines, agent_idx, ax=axes[row, col]
        )
    
    # Remove empty subplots if any
    for idx in range(len(agent_indices), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        fig.delaxes(axes[row, col])
    
    # Add figure legends
    building_legend = fig.legend(
        building_patches, 
        building_types,
        title="Building Types",
        loc='center left',
        bbox_to_anchor=(0.87, 0.65),  # Positioned at top right
        title_fontsize=10,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
    )
    
    # Add trajectory types legend at bottom right
    trajectory_legend = fig.legend(
        trajectory_lines, 
        traj_types,
        title="Trajectory Types",
        loc='center left',
        bbox_to_anchor=(0.87, 0.33),  # Positioned at bottom right
        title_fontsize=10,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
    )
    # fig.suptitle('Initial to all residential/mixed-use buildings')
    north_arrow(axes[0,1], scale=.2, shadow=False, location="upper right", rotation={"crs": gdf_buildings.crs, "reference": "center"}, label={"position": "bottom", "text": "North", "fontsize": 8}, aob={"bbox_to_anchor":(1.25,1.1), "bbox_transform":axes[0,1].transAxes})
    scale_bar(axes[1,1], location="lower right", style="boxes", bar={"projection": gdf_buildings.crs, "major_div": 1, "max":200,
        "minor_div": 2,},labels={"loc": "below", "style": "major","fontsize":10},units={"loc": "bar", "fontsize": 10}, aob={"bbox_to_anchor":(1.35,-0.2), "bbox_transform":axes[1,1].transAxes})

    fig.tight_layout()
    # Add space for legends
    plt.subplots_adjust(right=0.85)
    
    return fig, axes
    
def plot_trajectories(buildings, routines, agent_indices, ncols=2):
    gdf = gpd.GeoDataFrame(buildings,crs='EPSG:2039', geometry=gpd.points_from_xy(x=buildings.x, y=buildings.y)).to_crs(4326)
    x, y = pyproj.Transformer.from_crs("EPSG:2039", 4326).transform(gdf.loc[:, "x"], gdf.loc[:, "y"])
    gdf.loc[:, "x"] = y
    gdf.loc[:, "y"] = x
    fig, axes = create_visualization_grid(gdf, routines, agent_indices, ncols=2)
    plt.show()
    
def load_empirical():
    trajectories = pd.read_csv("data/stops_buildings_Jerusalem.csv", usecols=[1, 8,9,10,14,16]).dropna()
    trajectories = trajectories.drop(trajectories.loc[trajectories["HI_PNT_X"] == 0].index)

    empirical_segment_lengths = []
    empirical_convex_hulls = []
    empirical_USGs = []
    empirical_pids = []
    curr_pid = trajectories.loc[0, "pid"]
    coor = []
    usg = []
    curr_uniq_id = -1
    for agent in trajectories.iterrows():
        if agent[1].pid != curr_pid:
            if len(coor) > 1: # skip trajectories invalid
                # store
                # segment_length = [math.dist(coor[idx-1],coor[idx]) for idx in range(1, len(coor[1:]))]
                d = np.diff(coor, axis=0)
                segment_length = np.hypot(d[:,0], d[:,1])
                convex_hull = gpd.GeoDataFrame(geometry=[LineString([Point(*item) for item in coor])], crs="EPSG:2039").convex_hull.area
                empirical_convex_hulls.append(convex_hull.values)
                empirical_segment_lengths.append(segment_length)
                empirical_USGs.append(usg)
                empirical_pids.append(curr_pid)
    
            # reset
            curr_pid = agent[1].pid
            coor = []
            usg = []
            curr_uniq_id = -1
            
        if (curr_uniq_id == agent[1].UNIQ_ID) or (agent[1][["HI_PNT_X", "HI_PNT_Y"]] == 0).all():
            continue # de-duplicate buildings
        else:
            curr_uniq_id = agent[1].UNIQ_ID
            
        try:
            coor.append(agent[1][["HI_PNT_X", "HI_PNT_Y"]].to_list())
            usg.append(agent[1][["USG_CODE"]].to_list())
        except: # skip out-of-city building 
            continue
    return empirical_segment_lengths, empirical_convex_hulls, empirical_USGs, empirical_pids

def flatten_nan(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.flatten()
    x[x == 0] = np.nan
    return x

def get_building_count(x):
    return x.astype(bool).astype(int).sum(1) + 1
    
def plot_trajectories_stats(trajs : Tuple[np.ndarray], buildings : pd.DataFrame, title : str):
     
    gdf = gpd.GeoDataFrame(buildings,crs='EPSG:2039', geometry=gpd.points_from_xy(x=buildings.x, y=buildings.y)) # .to_crs(4326)
    
    empirical_segment_lengths, empirical_convex_hulls, _, _ = load_empirical()

    linear_traj, mlp_traj, gcn_traj, graphconv_traj = trajs
    linear_traj_stats = get_traj_stats(linear_traj, gdf)
    mlp_traj_stats = get_traj_stats(mlp_traj, gdf)
    gcn_traj_stats = get_traj_stats(gcn_traj, gdf)
    graphconv_traj_stats = get_traj_stats(graphconv_traj, gdf)

    empirical_total, empirical_mean, empirical_raw = get_segment_stat(empirical_segment_lengths)
    linear_total, linear_mean, linear_raw = get_segment_stat(linear_traj_stats[0])
    mlp_total, mlp_mean, mlp_raw = get_segment_stat(mlp_traj_stats[0])
    gcn_total, gcn_mean, gcn_raw = get_segment_stat(gcn_traj_stats[0])
    graphconv_total, graphconv_mean, graphconv_raw = get_segment_stat(graphconv_traj_stats[0])

    total_df = pd.DataFrame([np.log(empirical_total), np.log(linear_total), np.log(mlp_total), np.log(gcn_total), np.log(graphconv_total)]).T
    total_df.columns = ["Empirical", "Linear", "MLP", "GCN","GraphConv"]
    
    mean_df = pd.DataFrame([np.log(empirical_mean) ,np.log(linear_mean), np.log(mlp_mean), np.log(gcn_mean), np.log(graphconv_mean)]).T
    mean_df.columns = ["Empirical", "Linear", "MLP", "GCN","GraphConv"]
    
    raw_df = pd.DataFrame([np.log(flatten_nan(empirical_raw)) ,np.log(flatten_nan(linear_raw)) ,np.log(flatten_nan(mlp_raw)) , np.log(flatten_nan(gcn_raw)) , np.log(flatten_nan(graphconv_raw)) ]).T
    raw_df.columns = ["Empirical", "Linear", "MLP", "GCN","GraphConv"]
    
    count_df = pd.DataFrame([get_building_count(empirical_raw) ,get_building_count(linear_raw),get_building_count(mlp_raw), get_building_count(gcn_raw), get_building_count(graphconv_raw)]).T
    count_df.columns = ["Empirical", "Linear", "MLP", "GCN","GraphConv"]
    
    convex_df = pd.DataFrame([np.log(flatten_nan(empirical_convex_hulls)) ,np.log(flatten_nan(linear_traj_stats[1])) ,np.log(flatten_nan(mlp_traj_stats[1])) , np.log(flatten_nan(gcn_traj_stats[1])) , np.log(flatten_nan(graphconv_traj_stats[1])) ]).T
    convex_df.columns = ["Empirical", "Linear", "MLP", "GCN","GraphConv"]

    fig, axes = plt.subplots(5,1, figsize=(12,15), dpi=300)
    sns.boxplot(total_df, ax=axes[0], notch=True)
    sns.boxplot(mean_df, ax=axes[1], notch=True)
    sns.boxplot(raw_df, ax=axes[2], notch=True)
    sns.boxplot(count_df, ax=axes[3], notch=True)
    sns.boxplot(convex_df, ax=axes[4], notch=True)
    
    axes[0].set_title("Distribution of Total Trajectory Length")
    axes[0].set_ylabel("Log of Total Trajectory Length", fontsize=10)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].tick_params(axis='both', which='minor', labelsize=10)
    
    axes[1].set_title("Distribution of Mean Trajectory Segment Length")
    axes[1].set_ylabel("Log of Mean Trajectory Segment Length", fontsize=10)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].tick_params(axis='both', which='minor', labelsize=10)
    
    axes[2].set_title("Distribution of Trajectory Segment Length")
    axes[2].set_ylabel("Log of Trajectory Segment Length", fontsize=10)
    axes[2].set_xlabel("")
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].tick_params(axis='both', which='minor', labelsize=10)
    
    axes[3].set_title("Distribution of Trajectory Building Count")
    axes[3].set_ylabel("Number of Buildings in Trajectory", fontsize=10)
    axes[3].set_xlabel("")
    axes[3].tick_params(axis='both', which='major', labelsize=12)
    axes[3].tick_params(axis='both', which='minor', labelsize=10)
    
    axes[4].set_title("Distribution of Trajectory Convex Hull Area")
    axes[4].set_ylabel("Log of Trajectory Convex Hull Area", fontsize=10)
    axes[4].set_xlabel("Trajectories with Length < 3 are excluded.", fontsize=10)
    axes[4].tick_params(axis='both', which='major', labelsize=12)
    axes[4].tick_params(axis='both', which='minor', labelsize=10)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()