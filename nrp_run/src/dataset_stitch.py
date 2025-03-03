"""
Dataset Stitching
input_paths (str): Comma separated list of input paths URLs to stitch. Must be a complete URL.
output_path (str): Output path URL to save the stitched dataset. Must be a complete URL.
"""
import json, os, sys, h5py, zipfile, torch
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss


def stitch_sets(input_paths: str, output_path: str):
    print(f'\n=== PARAMS ===\nInput Path: {input_paths} | Output Path: {output_path}')

    # Parse comma separated input paths into a list
    input_paths = input_paths.replace(" ", "").split(',')

    fold = "./data/" # This is the working directory just for this file
    data_paths = [] # Local file paths for downloaded datasets
    for dataset in input_paths:
        os.system(f"aws s3 cp {dataset} ./data/")
        data_paths.append("./data/" + dataset.split("/")[-1])

    # Force install the maxwell plugin
    recording = se.read_maxwell(data_paths[0], install_maxwell_plugin=True) 

    # Move the libcompression.so file to the hdf5 plugin directory
    os.system("sh correct_maxwell_plugin.sh")

    # Sort the datapaths by using the h5 metadata timestamp
    data_paths.sort(key=lambda x: h5py.File(x, 'r', libver='latest', rdcc_nbytes=2 ** 25)['data_store/data0000/start_time'][0])

    recording_list = []
    stitch_inds = []
RUN apt-get update
    cur_ind = 0

    for data_path in data_paths:
        print('Appending', data_path)
        recording = se.read_maxwell(data_path, install_maxwell_plugin=True)
        recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
        recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
        recording_list.append(recording_cmr)

        cur_ind += recording_cmr.get_num_frames()
        stitch_inds.append((data_path, cur_ind))

    print('Concatenating')
    multirecording = si.concatenate_recordings(recording_list)
    print('Setting probe')
    multirecording = multirecording.set_probe(recording_list[0].get_probe())
    print(multirecording)
    recording_saved = multirecording.save(folder=fold + "preprocessed")

    # free up the cuda memory
    torch.cuda.empty_cache()

    ks_params = {
        "batch_size": 6000,
        "nblocks": 1,
        "Th_universal": 10,
        "Th_learned": 8,
        "do_CAR": True,
        "invert_sign": False,
        "nt": 61,
        "artifact_threshold": None,
        "nskip": 25,
        "whitening_range": 32,
        "binning_depth": 5,
        "sig_interp": 20,
        "nt0min": None,
        "dmin": None,
        "dminx": None,
        "min_template_size": 10,
        "template_sizes": 5,
        "nearest_chans": 10,
        "nearest_templates": 100,
        "templates_from_data": True,
        "n_templates": 3,
        "n_pcs": 3,
        "Th_single_ch": 6,
        "acg_threshold": 0.1,
        "ccg_threshold": 0.1,
        "cluster_downsampling": 20,
        "cluster_pcs": 64,
        "duplicate_spike_bins": 15,
        "do_correction": True,
        "keep_good_only": False,
        "save_extra_kwargs": False,
        "skip_kilosort_preprocessing": False,
        "scaleproc": None,
    }

    print('Running Kilosort4')
    # Save to the output folder
    multisorting = ss.run_sorter('kilosort4', recording_saved, output_folder=fold + 'results_KS4_stitch',
                             verbose=True, **ks_params)

    # Save the stitch indices and include the file name
    json_object = json.dumps(stitch_inds , indent=4)

    with open(fold + 'stitch_inds.json', 'w') as f:
        f.write(json_object)

    json_object = json.dumps(stitch_inds , indent=4)
    with open('stitch_inds.json', 'w') as f:
        f.write(json_object)
    # Add the stitch inds in
    multisorting.annotate(stitch_inds=stitch_inds)

    # Zip the files in ./data/results_KS_stitch/sorter_output/ folder
    with zipfile.ZipFile('./stitch_phy.zip', 'w') as zip_file:
        # Iterate over the files in the directory
        for filename in os.listdir('./data/results_KS4_stitch/sorter_output/'):
            # Add the file to the ZIP archive
            zip_file.write(os.path.join('./data/results_KS4_stitch/sorter_output/', filename))
    zip_file.close()

    # Upload the /data/ folder to the output path
    print("Upload data to s3...")
    print("current dir: ")
    os.system("ls")
    print("====")

    os.system(f"aws s3 cp ./stitch_phy.zip {output_path}")
    os.system(f"aws s3 cp ./data/stitch_inds.json {output_path}")
    

# Read parameters from sysargs and call the function for commandline usage
input_paths = sys.argv[1]
output_path = sys.argv[2]
stitch_sets(input_paths, output_path)
