# **Stage 4.1 — Interactive Output Visualization**

**Target File:** cochem\_geom\_reporter\_ui.py

### **1\. Purpose & Scope**

Build visual telemetry loops and diagnostic widgets for the Jupyter Notebook. Offloads heavy rendering entirely to the browser.

### **2\. Required Imports & Dependencies**

`import plotly.graph_objects as go`  
`import py3Dmol`  
`import ipywidgets as widgets`  
`from IPython.display import display, HTML`

### **3\. Interactive Widgets**

**Method: render\_correlation\_heatmap(self, cov\_matrix: np.ndarray, param\_labels: list, threshold: float \= 0.85)**

> * **Logic:** Plots a Plotly heatmap (go.Heatmap) of the normalized covariance matrix.  
> * **Filter:** Applies a diverging color scale (e.g., RdBu). Any square where absolute correlation \\vert{}r\\vert{} \> \\text{threshold} is highlighted bright red, allowing users to visually spot over-parameterization.

**Method: render\_3d\_displacement(self, start\_xyz: str, end\_xyz: str)**

> * **Logic:** Instantiates a py3Dmol.view.  
> * Loads start\_xyz with a blue carbon stick model.  
> * Calculates the Cartesian difference vectors: \\vec{v} \= R\_{\\text{end}} \- R\_{\\text{start}}.  
> * Uses view.addCylinder to draw red 3D arrows from the start position to the end position, visually scaling the arrow length to magnify subtle (mÅ) structural shifts between the theoretical prediction and the experimental fit.