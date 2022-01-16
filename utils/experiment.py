####################################################################################################
#									EXPERIMENT TRACKING ROUTINES
####################################################################################################

import numpy as np
import imageio
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import image as mpimg
import os, sys, json
import glob
import pickle
import pandas as pd
import itertools as it
from time import strftime, localtime, time, perf_counter

################### Global Parameters
EXPERIMENT_PATH = r'/experiments'     # path where new experiments will be created


################### create_experiment
def create_experiment(process_run, params):
    '''
    Create experiment by processing runs over all combinations of hparam key values

    param:  process_run	func    function variable for run processing logic
    param:  params      dict    dict of parameters that are NOT hyper-parameters
    param:  hparams		dict    dict of hyperparameters whose value  are all combinations per run
    param:  verbose		int     0 = no output, 1 = only print, 2 = print + plots (as PNGs)

    return: str exper_path	returns the experiment path

    ** Assumes process_run method whose args match hparam keys. 
    For example...
        from util_experiment import create_experiment
        ...
        my_process_run(run_folder, params):
            data_shape = hparam['data_shape']       # will be 32 on first run, 64 on second...
            ...
            return(run_summary)    where run_summary is dict of metrics

        my_hparams = {data_shape: [32, 64, 128], ...}    # will process_run for all combinations
        start_experment('my_exp_name', my_process_run, my_hparams, verbose=2)
    '''
    # set base path to exerimentS folder (needs lots of space)

    # set params variables for all experiment runs
    exper_tag = params['exper_tag'] if 'exper_tag' in params else 'Unknown'
    verbose = bool(params['verbose']) if 'verbose' in params else True  # print everything instead of minimal
    filelog = bool(params['filelog']) if 'filelog' in params else True  # print to file instead of terminal

    # create experiment folder under /logs
    dt = get_datetime_string()
    exper_path = EXPERIMENT_PATH + exper_tag + '-'+ dt
    if not os.path.isdir(exper_path):
        os.makedirs(exper_path)

    # if filelog:			# all stdout print to file?     >>>>>>>>>>>> ignore filelog for Colab WandB version
    #     sys.stdout = open(exper_path + '/' + exper_tag + '-print.txt', 'w')	

    if verbose:
        nRuns = np.prod([len(v) for v in params.values() if isinstance(v, list) ])
        print(f'>>> START EXPERIMENT: path="{exper_path}" with {nRuns} runs using...\nparams = {params}')

    # generate list of all hparams and its combinations
    hparam_list = []
    h_keys = []
    
    # find all params that are hyper-params, having a list of values, rather than single scalar
    for key, value in params.items():
        # print('key = ', key, 'value = ', value, type(value))    # DEBUG: remove
        if isinstance(value, list):
            h_keys.append(key)

    # generate all combinations among hyper-params list
    combinations = it.product(*(params[key] for key in h_keys))

    for tup in combinations:
        s = '{ '
        for i, key in enumerate(h_keys):
            if isinstance(tup[i], str):
                s += f"'{key}': '{tup[i]}', "
            else:
                s += f"'{key}': {tup[i]}, "
        s = s[:-2] + ' }'   # remove comma and add closing bracket
        # print('tup = ', tup, 's = ', s)                 # >>>> TODO debug remove
        hparam_list.append(eval(s))
    
    # execute process_run for each hparam_list combination
    result_list = []
    for i, hparam in enumerate(hparam_list):
        if verbose:
            print('=' * 80)
            print(f'>>> START RUN{i:02d} with hparams = {hparam}')
            print(f'>>> INFO: START RUN{i:02d} with hparam = {hparam}', file=sys.stderr)

        run_log = hparam
        params.update(hparam)
        # print('params = ', params)
        
        # create new run folder with hparam value appended
        run_folder = exper_path + f'/runs/RUN{i:02d}_' + dt[-4:] + '_' +  \
                ''.join([f'{str(key)[:3]}{str(value).replace(".", "")}_' for (key, value) in hparam.items()])[:-1] 
        if not os.path.isdir(run_folder):
            os.makedirs(run_folder)
        
        start_time = perf_counter()

        ############# execute PROCESS_RUN function in calling experiment
        run_metrics = process_run(run_folder, params)
        run_log.update(run_metrics)

        # get elapse time of this run and save
        elapse_time = perf_counter() - start_time
        run_log.update({'run_time': elapse_time})

        # save run_log ndarray plus scalars as exper_results rows
        run_dict = _save_run_log(run_folder, run_log, verbose)

        result_list.append(run_dict)

        if verbose:
            # print(f'>>> END RUN{i:02d} elapse={elapse_time:0.3f} sec with run_log={list(run_log.keys())}')
            print(f'>>> END RUN{i:02d} elapse={elapse_time:0.3f} sec run_log={",".join([str(key) for key in run_log.keys()])}')
    
    # create run-results df with rows of metrics from all runs
    run_results = pd.DataFrame(result_list)
    total_run_time = run_results['run_time'].sum()
    run_results.to_json(exper_path + '/run_results.json')
    print(f'>>> INFO: Saving run results with total run time = {total_run_time/60:0.1f} min', file=sys.stderr)

    # create animated GIF for all PNG plots across all runs
    if len(hparam_list) > 1:    # only if there is more than one PNG to convert to video
        create_all_MP4(exper_path, nImagesPerSec=1.0)
        print(f'>>> INFO: Generated MP4 video from all run PNGs', file=sys.stderr)

    if verbose:
        print('Experiment Results... ', run_results)
        print(f'>>> END EXPERIMENT: elapse={(total_run_time/60):3.1f} min')
    sys.stdout.close()

    return exper_path

##########################################################################################
#                        Internal Subroutines
##########################################################################################

####################### _create_all_MP4
def create_all_MP4(exper_path, nImagesPerSec=1.0):
    '''
        find all PNG file in RUN01; then call _create_MP4 for each

        exper_path	    str		path to experiment folder
        nImagesPerSec	float	sets images-per-second, so 0.5 is PNG image every 2 sec
    '''
    
    filenames = glob.glob(exper_path + '/runs/RUN00*/*.png')
    if len(filenames) == 0:
        print(f'    WARNING: create_all_MP4 found no PNG plot files in first run folder')
        return

    for filename in filenames:
        plot_name = filename[filename.rfind('\\') :]	# find last slash before PNG filename
        plot_name = plot_name[1 : -4]					# trim slash in front and '.PNG' at end
        # print('plot_name = ', plot_name)
        _create_MP4(exper_path, plot_name, nImagesPerSec=nImagesPerSec)

####################### _create_aminGIF  ...used by _create_all_aminGIF
def _create_MP4(exper_path, plot_name, nImagesPerSec):
    '''
        create aminated GIF from PNG plot in each run

        exper_path	str		path to experiment folder
        plot_name	str		name of plot PNG file (without '.png' ext)
        fps			float	sets frames-per-second
    '''

    MP4_file = exper_path + '/' + plot_name + '.mp4'

    filenames = glob.glob(exper_path + '/runs/RUN*' + '/' + plot_name + '.png')
    filenames = sorted(filenames)
    images = [mpimg.imread(f) for f in filenames]
    print(f'    INFO: generating {plot_name} MP4 with {len(images)} PNG images')
    
    if len(filenames) == 0:
        print(f'    WARNING: Create_MP4 found no PNG files for {plot_name} plot')

    # set/calculate video parameters
    nFramesPerSec = 30
    nImages = len(images)
    nFramesPerImage = int(nFramesPerSec * nImagesPerSec)
    nFrames = nFramesPerImage * nImages

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    plt.axis('off')
    a = images[0]
    # im = plt.imshow(a, interpolation=None, aspect='auto', vmin=0, vmax=1)
    im = plt.imshow(a, interpolation=None, aspect='equal')

    def animate(i):
        ii = i // nFramesPerImage   # repeat each image nFramesPerImage times
        im.set_array(images[ii])
        return [im]

    # anim = animation.FuncAnimation(fig, animate, frames=len(images), 
    anim = animation.FuncAnimation(fig, animate, frames=nFrames, repeat=False)
    # anim = animation.FuncAnimation(fig, animate, frames=nFrames, blit=True, repeat=False)
                            # interval=10, blit=True, repeat=False)

    anim.save(MP4_file, fps=nFramesPerSec, extra_args=['-vcodec', 'libx264'])

####################### _create_all_aminGIF
def _create_all_aminGIF(exper_path, fps=0.5, repeat_first=5):
    '''
        find all PNG file in RUN01; then call _create_aminGIF for each

        exper_path	str		path to experiment folder
        fps			float	sets frames-per-second, so 0.5 is frame every 2 sec
        repeat_first int	repeats the first PNG as start of loop
    '''
    
    filenames = glob.glob(exper_path + '/runs/RUN00*/*.png')
    print(f'    INFO: generating {len(filenames)} anim-GIFs')
    if len(filenames) == 0:
        print(f'    WARNING: create_aminGIF found no PNG plot files in first run folder')
        return

    for filename in filenames:
        plot_name = filename[filename.rfind('\\') :]	# find last slash before PNG filename
        plot_name = plot_name[1 : -4]					# trim slash in front and '.PNG' at end
        # print('plot_name = ', plot_name)
        _create_aminGIF(exper_path, plot_name, fps=fps, repeat_first=repeat_first)

####################### _create_aminGIF  ...used by _create_all_aminGIF
def _create_aminGIF(exper_path, plot_name, fps=0.5, repeat_first=5):
    '''
        create aminated GIF from PNG plot in each run

        exper_path	str		path to experiment folder
        plot_name	str		name of plot PNG file (without '.png' ext)
        fps			float	sets frames-per-second
        repeat_first int	repeats the first PNG as start of loop
    '''

    anim_file = exper_path + '/' + plot_name + '.gif'

    with imageio.get_writer(anim_file, mode='I', fps=fps) as writer:

        # collect filenames for plot_name across all runs
        # filenames = glob.glob(exper_path + '/runs/RUN*' + '/' + plot_name + '.png')
        filenames = glob.glob(exper_path + '/runs/RUN*' + '/' + plot_name + '.png')
        filenames = sorted(filenames)
        print(f'    INFO: generating anim-GIF {plot_name} with {len(filenames)} PNGs')
        
        if len(filenames) == 0:
            print(f'    WARNING: Create_AminGIF found no PNG file for {plot_name} plot')

        # append each PNG together
        for i, filename in enumerate(filenames):
            if i == 0:	# repeat the first PNG several times
                for ii in range(repeat_first):
                    image = imageio.imread(filename)
                    writer.append_data(image)
            else:
                image = imageio.imread(filename)
                writer.append_data(image)

            # write anim GIF
            image = imageio.imread(filename)
            writer.append_data(image)

####################### _save_run_log
def _save_run_log(run_path, run_log, verbose):
    '''
        save all run_log items in run_path folder, depending on object type

        run_path	str		path to current run folder
        run_log		dict	keyvalue of result variables to be saved for this run
        verbose		int		level of print verbosity
    '''

    print(f'>>> SAVE RUN LOG: run_path={run_path} with type={type(run_path)}')
    # if (type(run_log) is dict):		# TODO: fix by defining 'dict' type
    # 	print(f'ERROR: run_log is not type DICT')


    dict = {}
    for key, value in run_log.items():	# TODO - save model, history as ndarray
        value_str = ''

        if isinstance(value, int) or isinstance(value, np.int32):
            dict.update({key: value})
            value_str = str(value)
        elif isinstance(value, float) or isinstance(value, np.float32):
            dict.update({key: value})
            value_str = str(value)
        elif isinstance(value, np.ndarray):
            np.save(run_path + '/' + str(key), value)
        elif isinstance(value, list):
            with open(run_path + '/' + str(key + '.pkl'), 'wb') as f:	
                # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file/37795053#37795053 TODO:
                pickle.dump(value, f)
            # value.to_pickle(run_path + '/' + str(key))
        else:
            print(f'    SAVE ERROR: var {key} with value {value} not saved')
            
        if verbose: print(f'    {key}: {type(value)} = {value_str}')
    return dict

################### get_datetime_string

def get_datetime_string():
    '''
    Return datetime string like "YYYYMMDD-HHMMSS" 
    Note: add '%z' for '-ZZZZ' is GMT offset, like MST = -7000
    '''
    return strftime("%Y%m%d-%H%M%S", localtime())

######### get dict from arg

def get_dict_from_arg(arg_no):
    '''
    Returns dictionary pass in arg# as string 
    '''
    # for arg in sys.argv:
    #     print('arg = ', arg)
    
    data = sys.argv[arg_no]
    # print('data = ', data, type(data))
    data2 = str(data).replace("'", '"')
    # print('data2 = ', data2, type(data))
    dict = json.loads(data2)
    # print('dict = ', dict)
    # for key, value in dict.items():
    #     print(f'key={key} value={value}')

    return dict
    # return json.loads(str(sys.argv[arg]).replace("'", '"'))
    
#################################################################
# Save dataframe to experiment run_path

def save_dataframe(path, df_name, df_table):    #>>>>> TODO use json instead ???
    """Save dataframe to path folder with name 'df_name' using pickle

    Parameters
    ----------
    path : str
        file path to destination
    df_name : str   
    df_table : [type]
        [description]
    """    
    # df_table.to_pickle(f'{path}/{df_name}.pkl')
    df_table.to_json(f'{path}/{df_name}.json')

# Load dataframe from a previous experiment run_path
def load_dataframe(path, df_name):

    # df = pd.read_pickle(path + '\\' + df_name + '.pkl')
    df = pd.read_json(path + '\\' + df_name + '.json')
    return df
