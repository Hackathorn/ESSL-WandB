# %% [markdown]
####################################################################################################
#                       STEP6 - Create mappings of subgraphs to pre/post class labels
####################################################################################################
'''

'''

# %%
############################################# map cover to nodes
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

# %%
############################################# map cover to nodes
def generate_node_cover_for_build_step(covers_df, step_id):
    '''
    param   covers_df       df      subgraph covers for all build steps
    param   step_id         int     index into covers_df for specific build step
    return  cover_sg_nodes  list    list-of-lists of nodes, grouped by their subgraph
    '''
    # collect all nodes for this cover step, grouped by their subgraph
    cover_sg_nodes = covers_df.loc[step_id].sg_nodes

    return cover_sg_nodes
    
# %%
############################################# find unque & duplicate nodes in cover
def find_unique_and_duplicate_nodes(cover_sg_nodes):
    '''
    param   cover_sg_nodes  list    list-of-lists of nodes, grouped by their subgraph
    return  nodes_set       list    list of unique node indices for all subgraphs
    return  nodes_dup       list    list of node indices that are part of multiple subgraphs
    '''
    # flatten into single list of nodes
    nodes_flat = [node for sg_nodes in cover_sg_nodes for node in sg_nodes]
    # eliminate duplicate nodes with set
    nodes_set = set(nodes_flat)    
    # find duplicate nodes in nodes_flat
    nodes_dup = [x for n, x in enumerate(nodes_flat) if x in nodes_flat[:n]]

    return nodes_set, nodes_dup

# %%
############################################# find peak step in build cover
def find_peak_step_in_cover(covers_df):
    '''
    find peak/max no of subgraphs in cover-build  
    choose last if multiple peaks during build  
    param:   cover_df       df      subgraph covers for all build steps
    return:  max_sg_cnt     int     list of unique node indices for all subgraphs
    return:  max_sg_step    int     list of node indices that are part of multiple subgraphs
    '''
    max_sg_cnt = int(covers_df.sg_cnt.max())
    max_sg_id_list = covers_df[covers_df.sg_cnt == max_sg_cnt].index.values
    # choose last sg_id in list of 'peak' subgraphs
    max_sg_step = int(max_sg_id_list[-1])

    return max_sg_cnt, max_sg_step

# %%
############################################# map peak subgraphs to node classes
def map_sg_to_node_classes(covers_df, points_df, sg_step='peak', verbose=True):
    '''
    param   covers_df       df      subgraph covers for all build steps
    param   points_df       df      point dataframe
    param   sg_step         int     build step no (c_id in covers_df) or...
                                    'peak' = step for peak subgraphs (default)
                                    'knee'  TODO: adsf
                                    'end'   TODO
    param   points_df       df      point dataframe
    return  sg_class_map    array   shape(# classes, # subgraphs)
    return  class_labels    array   print labels for classes
    '''
    class_labels = np.array(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'])

    # find peak in build steps with max subgraphs if sg_step='peak'
    if sg_step == 'peak':
        sg_cnt = covers_df.sg_cnt.max()
        # get the last peak step  ...or first peak at [0]
        target_step = covers_df[covers_df.sg_cnt == sg_cnt].index.values[-1]
        peak_text = ' (peak max)'
    else: 
        target_step = sg_step
        sg_cnt = covers_df.loc[target_step].sg_cnt
        peak_text = ''

    cover_sg_nodes = generate_node_cover_for_build_step(covers_df, target_step)

    nodes_set, nodes_dup = find_unique_and_duplicate_nodes(cover_sg_nodes)
    total_nodes = points_df.shape[0]
    nodes_covered = len(nodes_set) + len(nodes_dup)
    ratio_total_to_cover = nodes_covered / total_nodes

    # create node class array with shape(# nodes)
    nodes_class = points_df.pt_class.to_numpy()
    class_set = set(nodes_class)

    # create array shape with subgraphs as row and class value as columns
    sg_class_map = np.zeros((len(class_set), len(cover_sg_nodes)), dtype=int)

    # create array mapping subgraph to point counts for each class
    for j, sg_nodes in enumerate(cover_sg_nodes):   # loop thru subgraphs
        for node in sg_nodes:
            i = nodes_class[node]
            sg_class_map[i, j] += 1

    # calculate stats
    max_cnt = sg_class_map.max()
    zero_cnt = np.count_nonzero(sg_class_map==0)
    n_total = sg_class_map.shape[0] * sg_class_map.shape[1]
    zero_ratio = zero_cnt / n_total

    # print stats
    if verbose:
        print(f'>>> Cover Map of {sg_cnt:,d} subgraphs at step {target_step:,d} {peak_text}')
        print(f'     Nodes covered are {nodes_covered:,d} ({ratio_total_to_cover:2.1%} of {total_nodes:,d} total)')
        print(f'     These {len(nodes_dup)} nodes duplicate cover in multiple subgraphs')
        print(f'     Shape of sg_class_map (classes by subgraphs) is {sg_class_map.shape}')
        print(f'     Total cells are {n_total} with {zero_cnt} ({zero_ratio:2.1%}) zero-cells, max cell count = {max_cnt}')
        print(f'     Labels of {len(class_labels)} classes are {class_labels}')

    return sg_class_map, target_step, class_labels

# %%
############################################# map peak subgraphs to node classes
def shuffle_sg_class_density(sg_class_map, low_cutoff=0):
    '''
    param   sg_class_map        array   shape(# classes, # subgraphs)
    param   low_cutoff          int     ignore sg_class cnt < cutoff
    return  class_sg_cnt        list    list of (class, subgraph, count)
    return  shuffled_sg_class_map array same as sg_class_map except columns are shuffled
    '''
    # scan all cells & create list of (i_class, j_sg, cnt) where cnt>0
    class_sg_cnt = []
    for i_class in range(sg_class_map.shape[0]):
        for j_sg in np.argsort(-sg_class_map[i_class, :]):
            cnt = sg_class_map[i_class, j_sg]
            if cnt > low_cutoff:
                class_sg_cnt.append((i_class, j_sg, cnt))
            else: break

    sg_list = []
    for (i_class, i_sg, cnt) in class_sg_cnt:
        if i_sg not in sg_list:
            sg_list.append(i_sg)

    shuffled_sg_class_map = sg_class_map.copy()
    shuffled_sg_class_map = shuffled_sg_class_map[:, sg_list]

    return class_sg_cnt, shuffled_sg_class_map

# %%
############################################# map peak subgraphs to node classes
def plot_density_of_sg_class_map(sg_class_map, class_labels, run_folder, shuffle=False):
    '''
    ##### display the density matrix of classes by subgraphs
    param   sg_class_map        array   shape(# classes, # subgraphs)
    param   class_labels        array   print labels for classes, starting with outlier class 'OUT'
    return  sorted_sg_class_map array   same sg_class_map except sg axis for each class sorted 
    '''
    if shuffle:
        _, data = shuffle_sg_class_density(sg_class_map, low_cutoff=0)
    else:
        data = sg_class_map

    n_nonzero_cells = np.count_nonzero(sg_class_map)
    ratio_nonzero_cells = n_nonzero_cells / sg_class_map.size
    n_nodes = sg_class_map.sum()
    
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_subplot(111)
    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder
    shuffle_text = "Shuffled - " if shuffle else ''
    fig.suptitle(f'Subgraph Cover Density by Classes -- {title_str}', 
            # y=0.90, fontsize=10, fontweight='bold')
            fontsize=10, fontweight='bold')

    shp = sg_class_map.shape    # (11, 260)
    ax.set_title(f'{shuffle_text}{shp[0]} Classes by {shp[1]} Subgraphs with {n_nonzero_cells:,d} nonzero cells ' + 
            f'({ratio_nonzero_cells:2.1%} of total) covering {n_nodes:,d} nodes', fontsize=10)
    ax.grid(axis='y')

    ext = [0, shp[1], 0, 22*shp[0]]
    im = ax.matshow(data, cmap=plt.cm.BuGn, extent=ext, aspect='auto', origin='lower')  # BuGn, Reds, Greens, 

    # ref: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)       # Similar to fig.colorbar(im, cax = cax)
    im.set_clim(0, 10)
    # plt.colorbar(cax)
    # tic_loc = [plt.Text(0, 5*i, lab) for i, lab in enumerate(class_labels)]
    # ax.yaxis.set_major_locator(plt.FixedLocator(tic_loc))
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    ax.set_yticks([22*i+12 for i in range(0, shp[0])])
    ax.set_yticklabels(class_labels)

    # plt.show()
    plt.savefig(run_folder+'/Subgraph_Cover_Density.png')
    plt.close()

# %%
############################################# map peak subgraphs to node classes
def get_nodeids_for_subgraph(covers_df, target_step, sg_id):
    '''
    Get node_id list for specific subgraph sg_id  
    param:   covers_df       df          covers_df with entire cover build of cliques into subgraphs  
    param:   target_step     int         build step no, same as c_id for clique merged into covers  
    param:   sg_id           int         index into sg_nodes in covers_df into node_list for sg_id  
    return:  node_list       int list    same sg_class_map except sg axis for each class sorted  
    '''
    # select the target_step with c_id
    df = covers_df[covers_df.c_id == target_step]

    # get sg_nodes and select subgraph for sg_id 
    # >>>> TODO sg_nodes in covers_df has EXTRA level; needs '[0]'
    node_list = df.sg_nodes.tolist()[0][sg_id]

    return node_list

################# display images

def show_node_images(node_list, sample_df, title='', file_name='', run_folder='./', img_type='PNG'):
    '''
    Display small images from dataframe as (N_PER_ROW, N_MAX/N_PER_ROW) = max (10, 12)

    param:	node_list   list    node_id (same as img_id) 
    param:	sample_df   df      sample_df with path to original GT image file 
    param:	title       str		title for entire figure
    param:	file_name   str 	save image figure to this file, if not null
    param:	run_folder  str     path to folder to save image figure
    param:	img_type    str     image filetype, default PNG 
    '''
    N_MAX = 120                 # max no of plots - 2 rows of 10 images
    N_PER_ROW = 12				# max images per row

    # select images from sample_df
    if len(node_list) > N_MAX:				# trim if too many images
        node_list = node_list[:N_MAX]
        
    df = sample_df.loc[node_list]
    class_list = df.class_folder.tolist()
    if img_type == 'PNG':
        file_list = (df['dataset_name']+'\\'+df['status_folder']+'\\'+df['class_folder']+'\\'+df['img_name']).tolist()
        # image_list = [img for img in Image.open(file_list)]
        image_list = [Image.open(f) for f in file_list]
    else:
        print(f'ERROR: Only supporting PNG image files')
        
    n = len(node_list) 		# no of images in df to be plotted 
    n_rows = int(math.ceil(n / N_PER_ROW))

    plt.figure(figsize=(26, 18))
    plt.suptitle(title, fontsize=16)

    for i, img in enumerate(image_list):

        ax = plt.subplot(n_rows, N_PER_ROW, i + 1)
        # if len(img.shape) == 1:					# if large 1D image
        # 	n = int(math.sqrt(img.shape[0]))	#    reshape to 2D
        # 	img = img.reshape((n, n))
        
        # img = (1 - img)		# invert grayscale assuming normalized [0..1] pixels
        plt.imshow(img, cmap='binary_r')
        ax.set_title(f'#{str(node_list[i])} - {class_list[i]}')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if file_name != '': 
        plt.savefig(run_folder + file_name + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

# %%
############################################# map cover to nodes
def create_mappings(covers_df, points_df, run_folder):
    '''
    param   covers_df       df      subgraph covers for all build steps
    param   run_folder      str     path to run folder
    return  mappings_df     df      mappings of subgraphs to pre/post class labels
    '''

    sg_class_map, target_step, class_labels = map_sg_to_node_classes(covers_df, points_df, sg_step='peak')

    plot_density_of_sg_class_map(sg_class_map, class_labels, run_folder, shuffle=True)

    ####### TODO What are mappings????? 

    mappings_df = pd.DataFrame()     # >>>>> TODO complete!!!!!!!!!!!!!!

    return mappings_df
    
