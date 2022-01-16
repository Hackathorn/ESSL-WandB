# %%
'''
    Protype for 
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg.decomp_svd import null_space

from experiment import save_dataframe, load_dataframe

# %%

def plot_covers(covers_df, run_folder):
    '''
    covers_df = pd.DataFrame(columns=['step', 'e_source', 'e_target', 'e_weight', 
                    'build_type', 'sg_id', 'pts_covered', 'n_uniqSG', 'n_zeroSG', 'pt_to_sg'])
    '''

    fig = plt.figure(figsize=(9.6, 5.4), dpi=150)
    ax = fig.add_subplot()
    title_str = run_folder[run_folder.find('RUN') :]	# Get run# & hparm from run_folder
    fig.suptitle(f'Subgraph Cover Build -- {title_str}', fontsize=12, fontweight='bold')

    end_weight = covers_df.e_weight.to_numpy()[-1]
    ax.set_title(f'SG Collapse at Step #{len(covers_df):,} with Edge Weight {end_weight:.3f}')
    # ax.set_xscale('log')  NOT - smashes the tail
    ax.grid()
    # ax.set_xscale("log")
    # plt.ylim(0, 800)    # ???
    plt.xlabel("Build Step Adding Next Edge")
    plt.ylabel("Count of Subgraphs at Each Step")

    # x = range(1, len(covers_df.index)+1)
    x = covers_df.step
    y = covers_df.n_uniqSG
    plt.plot(x, y, linewidth=2)
    # also plot the pct of pts covered over build steps NOT NEEDED!!!
    # ax2 = ax.twinx()
    # y2 = covers_df.pct_pts_covered
    # ax2.plot(x, y2, linewidth=1, c='g')

    # https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
    fsize = 7

    # mark where peak # of SGs occurs 
    max_sg = int(covers_df.n_uniqSG.max())
    x = covers_df[ covers_df.n_uniqSG == max_sg ].index.values[-1]    # choose last peak
    y = covers_df.iloc[x]['n_uniqSG']
    sc = ax.scatter(x, y, marker='o', s=50, edgecolors='k', linewidth=1, alpha=1)
    sc.set_facecolor('none')
    cov = covers_df.iloc[x]['pct_pts_covered']
    txt = f'Max SGs with {y:,} SGs & {cov:.0%} pts covered'
    ax.text(x+50, y+50, txt)
    # ax.annotate(f'Max SGs at {x} with {y} SGs & {cov:.0%} pts covered', 
    #             (x+50, y), xytext=(x+500, y+50), fontsize=fsize, )
                # arrowprops=dict(arrowstyle="-|>",
                # arrowprops=dict(arrowstyle="-",
                # connectionstyle="angle3,angleA=0,angleB=-90"))

    # mark where ALL points are covered
    x = covers_df[ covers_df.pct_pts_covered == 1.0 ].index.values[0]
    y = covers_df.iloc[x]['n_uniqSG']
    sc = ax.scatter(x, y, marker='s', s=50, edgecolors='k', linewidth=1, alpha=1)
    sc.set_facecolor('none')
    wgt = covers_df.iloc[x]['e_weight']
    txt = f'All points covered with {y:,} SGs & {wgt:0.2f} weight'
    ax.text(x+50, y+50, txt)
    # ax.annotate(f'All points covered at {x} with {y} SGs & {wgt:0.2f} weight', 
    #             (x+50, y), xytext=(x+500, y+20), fontsize=fsize, )
                # arrowprops=dict(arrowstyle="-",
                # connectionstyle="angle3,angleA=0,angleB=-90"))

    # mark where edge weight is less than...
    for wgt in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:

        x = covers_df[ covers_df.e_weight < wgt ].index.values[0]
        y = covers_df.iloc[x]['n_uniqSG']
        sc = ax.scatter(x, y, marker='^', s=50, edgecolors='k', linewidth=1, alpha=1)
        sc.set_facecolor('none')
        txt = f'Weight={wgt:0.1f} with {y:,} SGs'
        ax.text(x+50, y+50, txt)
        # ax.annotate(f'Weight<{wgt:0.1f} at {x} with {y} SGs', 
        #             (x+50, y), xytext=(x+500, y+10), fontsize=fsize, )
                    # arrowprops=dict(arrowstyle="-",
                    # # connectionstyle="angle3,angleA=0,angleB=-90"))
                    # connectionstyle="angle3,angleA=0,angleB=90"))

    fig.tight_layout()
    # plt.show()
    plt.savefig(run_folder+'/Subgraph_Cover_Build.png')
    plt.close()    

# %%
# Build subgraphs by merging edges together
def create_covers_from_edges(edges_df, verbose):
    
    ##### train model using latent_dim hparam
    if verbose: 
        print(f'>>> CREATING SUBGRAPHS by building all covers with adding strongest edges first')

    df = edges_df.sort_values(by=['weight', 'source', 'target'], 
                                ascending=[False, True, True])
    s_pts = df.source.to_numpy()
    t_pts = df.target.to_numpy()
    wgts = df.weight.to_numpy()   #.dtype=np.float32

    n_edges = len(edges_df.index)
    n_points = len( set(s_pts).union(set(t_pts)) )
    pt_to_sg = np.zeros((n_points), dtype=np.int32)
    flag10 = True; flag05 = True; flag_zeroSG = True; flag_uniq = True
    cover_data = []

    for i in range(n_edges): 
        s = s_pts[i]        # PT id for edge source & target
        t = t_pts[i]
        w = wgts[i]
        js = pt_to_sg[s]    # SG id for edge s & t
        jt = pt_to_sg[t]

        if js == 0:
            if jt == 0:
                # both zero (both pts are NOT in SG) => create new SG (id on build step)
                pt_to_sg[s] = i
                pt_to_sg[t] = i
                bt = 'N'
                j = i
                # print(f'{i:3d} create new SG {s:4d}-{t:4d} sg = {i}')
                # sg_build.append((i, s, t, w, 'N', i))
            else:
                # target nonzero => extend target SG
                pt_to_sg[s] = jt
                bt = 'XT'
                j = jt
                # print(f'{i:3d} extend target {s:4d}-{t:4d} sg = {jt}')
                # sg_build.append((i, s, t, w, 'X', jt))
        else:
            if jt == 0:
                # source nonzero => extend source SG
                pt_to_sg[t] = js
                bt = 'XS'
                j = js
                # print(f'{i:3d} extend source {s:4d}-{t:4d} sg = {js}')
                # sg_build.append((i, s, t, w, 'X', js))
            else:
                # both nonzero => merge SG into earlier SG (jx is smaller)
                if js < jt:
                    pt_to_sg[pt_to_sg == jt] = js
                    bt = 'MS'
                    j = js
                    # print(f'{i:3d} merge both    {s:4d}-{t:4d} sg = {jt} to {js}')
                    # sg_build.append((i, s, t, w, 'M', js))
                else:
                    pt_to_sg[pt_to_sg == js] = jt
                    bt = 'MT'
                    j = jt
                    # print(f'{i:3d} merge both    {s:4d}-{t:4d} sg = {js} to {jt}')
                    # sg_build.append((i, s, t, w, 'M', jt))

        n_pts_covered = np.count_nonzero(pt_to_sg)
        n_zeroSG = n_points - n_pts_covered
        pct_pts_covered = n_pts_covered / n_points
        n_uniqSG = len(np.unique(pt_to_sg))

        cover_data.append(dict( step =  i, 
                                e_source =  s, 
                                e_target =  t, 
                                e_weight =  w, 
                                build_type =  bt, 
                                sg_id =  j, 
                                pct_pts_covered =  pct_pts_covered, 
                                n_uniqSG =  n_uniqSG, 
                                n_zeroSG =  n_zeroSG, 
                                pt_to_sg =  np.copy(pt_to_sg),  # copy since pt_to_sg object gets reused! 
                            ))

        if w < 1.0 and flag10: 
            print(f'{i:3d} >>>>> hit wgt < 1.0 {s:4d}-{t:4d}'); flag10 = False
        if w < 0.5 and flag05: 
            print(f'{i:3d} >>>>> hit wgt < 0.5 {s:4d}-{t:4d}'); flag05 = False
        if n_zeroSG == 0 and flag_zeroSG: 
            print(f'{i:3d} >>>>> all pts cover {s:4d}-{t:4d}'); flag_zeroSG = False
        if n_uniqSG == 1 and n_zeroSG == 0 and flag_uniq: 
            print(f'{i:3d} >>>>> hit 1 uniqSG  {s:4d}-{t:4d} BREAK!!!'); flag_uniq = False
            break

    # create cover df to maintain log of cover building
    # covers_df = pd.DataFrame(dict_list, columns=['step', 'e_source', 'e_target', 'e_weight', 'build_type', 
    #         'sg_id', 'pct_pts_covered', 'n_uniqSG', 'n_zeroSG', 'pt_to_sg'])
    covers_df = pd.DataFrame(cover_data)

    return covers_df

# %%
# Animate the SG build process as MP4       >>> TODO maybe...
def animate_sg_build(sg_build, run_path):
    pass

# %%
############# MAIN
def create_subgraph_from_covers(covers_df, run_folder, target_sg_list):
    '''
    covers_df = pd.DataFrame(columns=['step', 'e_source', 'e_target', 'e_weight', 
                    'build_type', 'sg_id', 'pts_covered', 'n_uniqSG', 'n_zeroSG', 'pt_to_sg'])
    
    subgraphs_df = pd.DataFrame(columns=['sg_id', 'pt_list', 'pts_covered'])
    '''
    # consider only the cover builds when all points are covered
    # print('covers_df size = ', len(covers_df))

    # df = covers_df[covers_df.pct_pts_covered >= 1.0]        #.copy()
    # assert not df.empty, f'ERROR: Cover builds did not reach 100% coverage!'
    df = covers_df
    # print('df size = ', len(df))
    
    sg_list = []
    # then find cover builds with # of subgraphs = 5x, 4x...1x to the target_sg
    for i, t_sg in enumerate(target_sg_list):
        step = df[df.n_uniqSG == t_sg].index.values
        # print('step =', step, type(step), 't_sg =', t_sg)
        assert step.shape[0] != 0, 'ERROR: No build step has this # of subgraphs'
        step = step[-1]     #  # choose last build step
        e_weight = df.loc[step].e_weight
        pct_pts_cov = df.loc[step].pct_pts_covered
        pt_to_sg = df.loc[step].pt_to_sg
        # n_uniq_sg = len(set([sg for sg in pt_to_sg]))
        n_uniq_sg = len(set(pt_to_sg.flatten()))
        uniq_sg, cnt_sg = np.unique(pt_to_sg, return_counts=True)
        
        print(80*'-')
        print(f'step = {step} t_sg = {t_sg} e_weight = {e_weight:.3f} pct_pt_cov = {pct_pts_cov:.1%} max_cnt_sg = {max(cnt_sg)}')
        print(f'unique_sg = ', uniq_sg[:10])
        print(f'cnt_uniq_sg = ', cnt_sg[:10])
        
        sg_list.append(dict(sg_id = step, 
                            target_sg = t_sg,
                            n_uniq_sg = n_uniq_sg,
                            e_weight = e_weight,
                            pt_to_sg = pt_to_sg,
        ))

    # create subgraph_df results
    subgraphs_df = pd.DataFrame(sg_list)
    
    return subgraphs_df
