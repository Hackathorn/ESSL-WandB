# %%
# import modules

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import umap

# %%
# Create edges and its weights using UMAP

def create_edges_with_UMAP(points_df, n_neighbors, min_dist, n_dim, wgt_cutoff, verbose):
    
    ##### train model using latent_dim hparam
    if verbose: 
        print(f'>>> CREATING EDGES with n_neighbors={n_neighbors}, min_dist={min_dist}, ')

    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,      
        min_dist=min_dist,  
        n_components=n_dim, 
        metric='euclidean',
        random_state=42)
        
    z_pos = np.stack(points_df['pt_encoded'])   # TODO should be vstack ???
    umap_object.fit(z_pos)
    
    coo_graph = sp.triu(umap_object.graph_).tocoo()  
    edges = np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T

    n_total = edges.shape[0]
    n_delete = 'NONE'
    if wgt_cutoff > 0: # ignore edges with weight less/equal than cutoff
        edges = edges[edges[:,2] >= wgt_cutoff]
        n_delete = n_total - edges.shape[0]

    edges_df = pd.DataFrame(edges, columns=("source", "target", "weight"))
    edges_df["source"] = edges_df.source.astype(np.int32)
    edges_df["target"] = edges_df.target.astype(np.int32)
    # edges_df['flags'] = ''
    edges_df.sort_values(by=(['weight', 'source', 'target']), inplace=True, ascending=False)
    edges_df.reset_index(drop=True, inplace=True)

    if verbose: 
        print(f'    Created {n_total} edges_df from UMAP.graph, but deleted {n_delete}\n', edges_df.tail())
    return edges_df, umap_object

# %%
# Calculate Euclidean distance between two points of an edge

def find_euc_distance(p1, p2):
    sum = 0
    for k in range(p1.shape[0]):
        sum += (p1[k] - p2[k])**2
    return math.sqrt(sum)

def add_Euclidean_distance_to_edges(edges_df, points_df, verbose):
    
    pos = np.vstack(points_df.pt_pos)
    for i in range(edges_df.shape[0]):
        s = edges_df.at[i, "source"]
        p1 = pos[s, :]
        t = edges_df.at[i, "target"]
        p2 = pos[t, :]
        dist = find_euc_distance(p1, p2)
        edges_df.at[i, "euclidean"] = dist

    if verbose: print("*** Added Distance to Edges")
    return edges_df

# %%
#################################################################
# Plot distribution of points degrees (ie, # of edges that involves each point)
def plot_point_degree_distribution(run_folder, pt_degrees):
    # using https://stackoverflow.com/questions/27083051/matplotlib-xticks-not-lining-up-with-histogram
    # with np.bincount and plt.bar (instead of plt.hist) -- for integer histrograms starting at zero

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_subplot()
    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder
    fig.suptitle(f'Point Degree Distribution -- {title_str}', fontsize=12, fontweight='bold')

    n_bins = 30     # TODO larger if lowest degree when n_neighbor > 30
    counts = np.bincount(pt_degrees, minlength=n_bins)
    n_pt = pt_degrees.shape[0]
    n_edge = pt_degrees.sum()
    max_deg = pt_degrees.max()
    ax.set_title(f'{n_pt:,d} points having max degree of {max_deg} linking {n_edge:,d} edges')    

    # if counts is too big => max+1 > n_bins
    if counts.shape[0] > n_bins:  
        counts = counts[:n_bins]

    plt.grid(axis='y')
    # plt.ylim(0, 2000)
    # plt.xlim(-1, 20)
    plt.xlabel("Point Degree (No of Edges)")
    plt.ylabel("Count of Points with this Degree")
    ax.set(xticks=range(n_bins), xlim=[-1, n_bins])

    # fig, ax = plt.subplots()
    try:    # TODO problem child....
        ax.bar(range(n_bins), counts, width=0.8, align='center')
    except Exception as e:
        print(f'    ERROR: Plot Degree Distribution ignored for {title_str} with exception {e}')
        return

    # plt.show()
    plt.savefig(run_folder+'/Point_Degree_Dist.png')
    plt.close()

# %%
# Plot distribution of points degrees (ie, # of edges that involves each point)
def plot_point_weightsum_distribution(run_folder, pt_weightsum):    # >>>>>>> TODO eliminate???

    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.suptitle(f'Point Sum-of-Edge-Weights Distribution -- {title_str}', fontsize=12, fontweight='bold')
    bins = np.arange(11)        # - 0.5
    plt.hist(pt_weightsum, bins=bins, color='c', edgecolor='k', alpha=0.65)
    plt.ylim(0, 1800)
    plt.xlim(-1, 10)
    plt.xticks(range(10))
    plt.grid()
    plt.xlabel("Point Weight Sum")
    plt.ylabel("Count of Points")

    # plt.show()
    plt.savefig(run_folder+'/Point_WeightSum_Dist.png')
    plt.close()


# %%
# Add point degree & weight sum to points_df columns

def add_pt_degreeweights_to_points(points_df, edges_df, run_folder, verbose):

    edges = edges_df.to_numpy()
    n_points = len(points_df.index)
    pt_degrees = np.zeros((n_points,)).astype(int)
    pt_weightsum = np.zeros((n_points,))

    for s, t, w in edges:
        s = int(s); t = int(t)
        pt_degrees[s] += 1
        pt_degrees[t] += 1
        pt_weightsum[s] += w
        pt_weightsum[t] += w

    points_df['pt_degrees'] = pt_degrees
    points_df['pt_weightsum'] = pt_weightsum

    if verbose: 
        plot_point_degree_distribution(run_folder, pt_degrees)
        plot_point_weightsum_distribution(run_folder, pt_weightsum)
    
    return points_df
