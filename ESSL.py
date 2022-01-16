'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TODO add comments
'''
# %%
# import modules

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sample import  create_samples

from point import   create_points, \
                    add_posLowD_to_points, plot_pos2D_points

from edge import    create_edges_with_UMAP, \
                    add_pt_degreeweights_to_points, \
                    add_Euclidean_distance_to_edges
# from points import load_previous_points
# from STEP5_Subgraphs import find_subgraph_cover_from_cliques
from subgraph import create_covers_from_edges, plot_covers, create_subgraph_from_covers
from mapping import create_mappings
# from STEP6_Mappings import find_peak_step_in_cover 
# from STEP6_Mappings import plot_density_of_sg_class_map
from experiment import create_experiment, get_dict_from_arg, save_dataframe, load_dataframe

# %%
def set_params():
    """set_params [summary]

    Returns:
        dict: Key/value for each parameter for process_run. If value is list,
                param is a hyper-parameter, for which each combination generates a run
    """
    
    dsn = 'FASHION'          # ['MNIST','FASHION",'DCAI','CHECKER']   # MNIST and/or DCAI 
    
    tag = 'COMBO_base' if isinstance(dsn, list) else dsn + '_base'
        
    params = {  
            ########### parameters for experiment.py - scalar values only
            'exper_tag':        # experiment name
                tag,
            'verbose': 
                1, 
            'filelog':          # print log to file
                1,
            ########### parameters for process_run
            #   Use 3-char abrevation as dict key. Key values can be scalar or list.
            #   If list, it is a hyper-parameter, which appears in Run folder name.
            #   If list len > 1, every combination results in an unique run.
            
            'dsn':              # dataset_path= MNIST, DCAI, CHECKER >>> TODO path to structured dataset folder
                dsn,
            'pts':              # points_path= ''; if not null, use path for points_df to bypass VAE training
                '', 
            'img':              # img_size= 32, 64, 96, 128; pixel width/height of square image
                32,
            'ncl':              # no of classes in samples (CHECKER)
                [10],
            'nbk':              # no of pixel blocks in image to vary info density (CHECKER)
                [1],
            'lr':               # learning rate during training
                [0.0005],
            'dim':              # latent_dim = 8, 16, 32, 64, or any value; dim = 2 & 3 auto perform
                # [8,16],
                [16],
            'epo':              # epoches= 10, 100, 200; no of epoch training cycles
                40, 
            'n_n':              # nearest_neighbor= 2 to 25% of sample size; UMAP  param 15 default
                # [2,15,100],
                15,
            'dis':              # min_dist= 0.0 to 0.99; fine clumpiest to global structure - 0.1 default
                # [0.1,0.99],
                0.1,
            'cut':              # wgt_cutoff= 0.01; ignore edges with weights below cutoff 
                0.01,
        }
    
    nRuns = np.prod([len(v) for v in params.values() if isinstance(v, list) ])
    hparams_keys = [k for k, v in params.items() if isinstance(v, list) ]
    if nRuns > 1: print(f'No of Runs = {nRuns} for hparams key = {hparams_keys}')
    
    return params 


# %%
def process_run(run_folder, params):
    """process_run routine is executed by create_experiment for each run

        Args:
        run_folder (str):   path to run folder
        params (dict):      See set_params method for param key definitions

        Returns:
        [dict]: key/value for metrics from this run
        """

    ##### set param variables from params arg
    global verbose
    verbose = params['verbose'] if 'verbose' in params else True    # >>> TODO test verbose from ESSL.py

    ##### set hparam variables from param arg
    dataset_path = params['dsn'] if 'dsn' in params else None
    # preprocess = params['pre'] if 'pre' in params else True  >>> TODO remove! samples_df is responsible
    n_classes = params['ncl'] if 'ncl' in params else 10
    n_blk_size = params['nbk'] if 'nbk' in params else 2
    points_path = params['pts'] if 'pts' in params else ''
    img_size = params['img'] if 'img' in params else 32
    nEpochs = params['epo'] if 'epo' in params else 10
    latent_dim = params['dim'] if 'dim' in params else 16
    n_neighbors = params['n_n'] if 'n_n' in params else 5
    min_dist = params['dis'] if 'dis' in params else 0.1
    wgt_cutoff = params['cut'] if 'cut' in params else 0 # accept ALL as default
    pct_node_covered = params['pct_node_covered'] if 'pct_node_covered' in params else 0.9
    
    ##### STEP1 - Create samples by preprocessing initial dataset
    samples_df = create_samples(dataset_path, img_size, n_classes=n_classes, 
                                n_blk_size=n_blk_size, verbose=verbose)
    # sample_df = convert_ds_to_df(dataset_folder, img_invert=True)
    # print('preprocess_images_to_array img_size = ', img_size)
    # # sample_df = preprocess_images_to_array(sample_df, img_size)
    # sample_df = preprocess_images_to_array(sample_df, 256)      # save as many pixels as possible TODO flags!!!
    save_dataframe(run_folder, 'samples_df', samples_df)

    ##### STEP2 - Create points by embedding samples into D-dim Latent Space
    if points_path == '':   # if no previous points_df, create a new points_df
        # points_df = create_points(samples_df, img_size, preprocess, latent_dim, nEpochs, run_folder) >>> TODO remove! samples_df is responsible
        points_df = create_points(samples_df, img_size, latent_dim, nEpochs, run_folder, verbose=verbose)
        save_dataframe(run_folder, 'points_df', points_df)
    else:
        points_df = load_dataframe(points_path, 'points_df')

    points_df = add_posLowD_to_points(points_df, n_neighbors, min_dist)
    save_dataframe(run_folder, 'points_df', points_df)
    plot_pos2D_points(points_df, n_neighbors, min_dist, run_folder)

    ##### STEP3 -  Create edges by calculating UMAP weights among points >>>TODO add more distance metrics
    edges_df, _ = create_edges_with_UMAP(points_df, n_neighbors, min_dist, latent_dim, wgt_cutoff, verbose)
    edges_df = add_Euclidean_distance_to_edges(edges_df, points_df, verbose) 
    # edges_df = add_labels_to_edges(edges_df, points_df)   # TODO FLAG edge if different labels?
    save_dataframe(run_folder, 'edges_df', edges_df)

    # points_df = add_pt_degreeweights_to_points(points_df, edges_df, run_folder, verbose)
    # save_dataframe(run_folder, 'points_df', points_df)
    
    ##### STEP4 - Create Subgraphs from Edges
    # create covers by merging edges, strongest first
    covers_df = create_covers_from_edges(edges_df, verbose)
    # save_dataframe(run_folder, 'covers_df', covers_df)    # >>> TODO HUGE!!! need to save?
    # plot results of cover builds
    plot_covers(covers_df, run_folder)
    # create the subgraph based on % of points/nodes covered
    subgraphs_df = create_subgraph_from_covers(covers_df, run_folder, [70,50,40,20,10])
    save_dataframe(run_folder, 'subgraphs_df', subgraphs_df)    

    # ##### STEP4 - Create cliques among points with strong edges
    # cliques_df = find_cliques_from_edges(points_df, edges_df, umap_object, wgt_cutoff)
    # plot_cliq_weight_by_K(cliques_df, wgt_cutoff, run_folder)
    # save_dataframe(run_folder, 'cliques_df', cliques_df)

    # #### STEP4 - Create subgraphs by merging edges in Graph Space
    # covers_df = find_subgraph_cover_from_cliques(cliques_df, wgt_cutoff, run_folder)
    # save_dataframe(run_folder, 'covers_df', covers_df)
    # subgraphs_df = create_subgraphs(edges_df, pct_node_covered, run_folder)
    # save_dataframe(run_folder, 'subgraphs_df', subgraphs_df)

    ##### STEP5 - Create mappings of subgraphs to pre/post class labels
    # mappings_df = create_mappings(subgraphs_df, points_df, run_folder)   # >>>>>>>> TODO
    # save_dataframe(run_folder, 'mappings_df', mappings_df)


    ##### Create CSV/JSON for Nodes/Cliqeus to ESSL Workshop    >>>>>> TODO 
    # save_nodes_for_run(run_folder, points_df, node_lim=None)
    # save_cliques_for_run(run_folder, cliques_df, cliq_lim = None, k_lim=None)

    # save_nodes_cliques_as_CSV(run_folder, points_df, cliques_df)
    # save_nodes_as_json(run_folder, points_df, node_lim=None)
    # save_cliques_as_json(run_folder, points_df, Kcliq_lim=None)

    ####################### Update results from this runs using metrics from sample_df, etc
    ##### metrics from sample_df
    m_flagged_images = sum([1 for s in samples_df['flags'] if s != ''])

    ##### metrics from points_df
    m_flagged_points = sum([1 for s in points_df['flags'] if s != ''])
    
    mse = np.vstack(points_df.pt_mse_loss).flatten()
    m_mse_loss = mse.mean()
    z_mse = (mse - mse.mean()) / mse.std()
    mse_out = mse[z_mse > 3]
    m_mse_out_pct = 100 * len(mse_out) / len(mse)
    
    m_pos_std = np.vstack(points_df.pt_std).mean()
    # m_degree = points_df['pt_degrees'].mean()
    # m_weightsum = points_df['pt_weightsum'].mean()

    ##### metrics from edges_df
    # # m_flagged_edges = sum([1 for s in edges_df['flags'] if s != ''])  >>>>>>>>>>> TODO add 'flags' column
    # m_edge_weight = edges_df['weight'].mean()
    # m_pct_edge_weight_one = 100 * sum([1 for w in edges_df['weight'] if w == 1.0]) / len(edges_df.index)

    ##### metrics from cliques_df
    # m_flagged_cliques = sum([1 for s in cliques_df['flags'] if s != ''])  >>>>>>>>>>> TODO add 'flags' column
    # m_cliq_weight = cliques_df['weight'].mean()
    # m_cliq_ksize = cliques_df['ksize'].mean()          # >>>>>>>>>>>>>> TODO ignore 1-cliques ???
    # m_cliq_kmax = cliques_df['ksize'].max()   
    # # m_cliq_size = mean[s for s in cliques_df['ksize'] if s > 1])
    # m_pct_cliq_weight_one = 100 * sum([1 for w in cliques_df['weight'] if w == 1.0]) / len(cliques_df.index)

    ##### metrics from cover_df
    # m_flagged_covers = sum([1 for s in covers_df['flags'] if s != ''])  >>>>>>>>>>> TODO add 'flags' column
    # m_max_sg_cnt, m_max_sg_step = find_peak_step_in_cover(covers_df)

    run_metrics = { 'm_flagged_images': m_flagged_images, 
                    'm_flagged_points': m_flagged_points,
                    'm_mse_loss': m_mse_loss,
                    'm_mse_out_pct': m_mse_out_pct,
                    'm_pos_std': m_pos_std,
                    # 'm_degree': m_degree,
                    # 'm_weightsum': m_weightsum,
                    # 'm_edge_weight': m_edge_weight, 
                    # 'm_pct_edge_weight_one': m_pct_edge_weight_one,
                    # 'm_cliq_weight': m_cliq_weight, 
                    # 'm_cliq_ksize': m_cliq_ksize,
                    # 'm_cliq_kmax': m_cliq_kmax,
                    # 'm_pct_cliq_weight_one': m_pct_cliq_weight_one,
                    # 'm_max_sg': m_max_sg_cnt,
                    # 'm_max_sg_step': m_max_sg_step,
                }

    return run_metrics

# %%
######### Create Experiment #########
def ESSL(params):
    """Create experiment by executing process_run for each run

    Parameters
    ----------
    params : dict 
        key as param_name with value as list for a hyper-param or scalar 
        NOTE: each permutation of hyper-params generate unique run of process_run

    Returns
    -------
    str 
        file path to folder containing data about that run
    """
    
    experiment_path = create_experiment(process_run, params)
    return experiment_path

# %%
################ MAIN #################
if __name__ == "__main__":
    import sys, json        # TODO use argparse for dict arg

    if len(sys.argv) > 1:
        # get params from command-line
        arg = str(sys.argv[1]).replace("'", '"')
        params = json.loads(arg)
    else:
        # get params via set_params function
        params = set_params()

    ESSL(params)
