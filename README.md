# TrajGen-Geom-IRL

Trajectory Generation and Geometric Inverse Reinforcement Learning for Urban Environments

## Overview

**TrajGen-Geom-IRL** is a research codebase for generating and analyzing movement trajectories in urban environments using geometric Inverse Reinforcement Learning (IRL). The project focuses on modeling, learning, and simulating human movement patterns (trajectories) between buildings or locations based on real-world urban data and inferred reward structures.

The code implements and compares several IRL models, including:
- Linear reward models
- Multi-Layer Perceptron (MLP) models
- Graph Convolutional Network (GCN) models
- GraphConv models

It enables the analysis of spatial patterns of rewards, correlations with building attributes, and scenario-based trajectory simulations under different hypothetical urban interventions.

## Features

- **Data Preparation:** Loads urban data (buildings, translation tables, discount factors).
- **Transition Modeling:** Computes transition matrices via gravity models or distance-based models.
- **IRL Algorithms:** Trains and evaluates multiple IRL models to infer reward functions from observed data.
- **Trajectory Generation:** Simulates trajectories using the learned rewards and transition dynamics.
- **Statistical Analysis:** Analyzes and visualizes spatial reward distributions and their relationship with urban features.
- **Scenario Analysis:** Supports experiments with reduced boundaries and random removal.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zihenlin/TrajGen-Geom-IRL.git
   cd TrajGen-Geom-IRL
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The main requirements include:
   - Python 3.10+
   - numpy, pandas, matplotlib, seaborn, PyTorch
   - geopandas, shapely, Cartopy, folium, networkx
   - jupyter, notebook, scikit-learn, osmnx, GDAL, etc.

   *(See `requirements.txt` for the full list.)*

## Usage

The main workflow is contained in the Jupyter notebook `IRL_deep.ipynb`. To get started:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `IRL_deep.ipynb` and follow the notebook cells sequentially. The steps include:
   - Loading and preprocessing data
   - Computing transition matrices
   - Defining initial and terminal states
   - Training IRL models
   - Generating and visualizing trajectories
   - Running scenario analyses

**Note:** The code expects urban data files (e.g., building attributes, feature matrices) in the `data/` directory. Please ensure these files are present.

## Project Structure

```
├── IRL_deep.ipynb         # Main analysis notebook
├── requirements.txt       # Python dependencies
├── data/                  # Urban datasets 
├── transitions.py         # Transition matrix utilities
├── trajectories.py        # Trajectory generation and plotting
...
```

## Example Analyses

- **Reward Spatial Patterns:** Visualizes spatial distribution of inferred rewards for different models.
- **Correlation Analysis:** Measures correlations between rewards and urban features like floorspace and connectivity.
- **Scenario Experiments:** Simulates hypothetical interventions (e.g., boundary reductions, random removals) and their effects on movement.

## Citation

If you use this codebase for your research, please cite appropriately or acknowledge the authors.

## License

This project is released under the MIT License.

---

**Contact:** For questions or collaborations, open an issue or contact the repository owner.
