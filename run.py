"""
Runs kilosort4 on an input across comet ML.

Modified from https://github.com/braingeneers/ks4paramsweep by https://github.com/OjasPBrahme .
"""
import json
import comet_ml
import sys
import os
import zipfile
import torch
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import pandas as pd


this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, "config.json"), 'r') as f:
    config = json.load(f)

data_dir = os.path.join(this_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(data_dir, 'ks4_results')

try:
    comet_ml.init(api_key=os.environ['COMET_API_KEY'], project_name=os.environ['COMET_PROJECT_NAME'])
except KeyError as e:
    print('Please set COMET_PROJECT_NAME and COMET_API_KEY!', file=sys.stderr)
    raise e
comet_optimizer = comet_ml.Optimizer(config=config)


def kilorun(
    input_path,
    output_path,
    endpoint="https://s3-west.nrp-nautilus.io",
    params=None,
):
    print(
        f"\n=== PARAMS ==="
        f"\nInput Path: {input_path} | Output Path: {output_path} | Endpoint: {endpoint}"
        f"\n==================\n\n\n"
    )

    if input_path.startswith("s3:"):
        local_input_path = os.path.join(data_dir, input_path.split('/')[-1])
        os.system(f"aws --endpoint={endpoint} s3 cp {input_path} {local_input_path}")
        input_path = local_input_path

    recording = se.read_maxwell(input_path, install_maxwell_plugin=True)
    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_cmr = spre.common_reference(recording_f, reference="global", operator="median")

    # find spike interface for
    multirecording = si.concatenate_recordings([recording_cmr])
    print("Setting probe")
    recording = multirecording.set_probe(recording_cmr.get_probe())
    recording_saved = recording.save(folder="./data/preprocessed/")

    torch.cuda.empty_cache()

    print(f"Running Kilosort4 with:\n{params}")
    ss.run_sorter(
        "kilosort4",
        recording_saved,
        output_folder=results_dir,
        verbose=True,
        **params,
    )

    try:
        df = pd.read_csv(os.path.join(results_dir, "cluster_KSLabel.tsv"), sep="\t", lineterminator="\r")
        good_count = (df.iloc[:, 1] == "good").sum()
    except:
        good_count = 0

    with zipfile.ZipFile(os.path.join(results_dir, "stitch_phy.zip"), "w") as zip_file:
        for filename in os.listdir(results_dir):
            zip_file.write(filename)  # Add the file to the ZIP archive
    zip_file.close()

    return good_count


def number_of_good_units(
    batch_size,
    nblocks,
    Th_universal,
    Th_learned,
    do_CAR,
    invert_sign,
    nt,
    artifact_threshold,
    nskip,
    whitening_range,
    binning_depth,
    sig_interp,
    nt0min,
    dmin,
    dminx,
    min_template_size,
    template_sizes,
    nearest_chans,
    nearest_templates,
    templates_from_data,
    n_templates,
    n_pcs,
    Th_single_ch,
    acg_threshold,
    ccg_threshold,
    cluster_downsampling,
    cluster_pcs,
    duplicate_spike_bins,
    do_correction,
    keep_good_only,
    save_extra_kwargs,
    skip_kilosort_preprocessing,
    scaleproc,
):
    number_of_good_units.params = {
        "batch_size": batch_size,
        "nblocks": nblocks,
        "Th_universal": Th_universal,
        "Th_learned": Th_learned,
        "do_CAR": do_CAR,
        "invert_sign": invert_sign,
        "nt": nt,
        "artifact_threshold": artifact_threshold,
        "nskip": nskip,
        "whitening_range": whitening_range,
        "binning_depth": binning_depth,
        "sig_interp": sig_interp,
        "nt0min": nt0min,
        "dmin": dmin,
        "dminx": dminx,
        "min_template_size": min_template_size,
        "template_sizes": template_sizes,
        "nearest_chans": nearest_chans,
        "nearest_templates": nearest_templates,
        "templates_from_data": templates_from_data,
        "n_templates": n_templates,
        "n_pcs": n_pcs,
        "Th_single_ch": Th_single_ch,
        "acg_threshold": acg_threshold,
        "ccg_threshold": ccg_threshold,
        "cluster_downsampling": cluster_downsampling,
        "cluster_pcs": cluster_pcs,
        "duplicate_spike_bins": duplicate_spike_bins,
        "do_correction": do_correction,
        "keep_good_only": keep_good_only,
        "save_extra_kwargs": save_extra_kwargs,
        "skip_kilosort_preprocessing": skip_kilosort_preprocessing,
        "scaleproc": scaleproc,
    }
    input_paths = sys.argv[1]
    output_path = sys.argv[2]
    return kilorun(input_paths, output_path, params=number_of_good_units.params)


def main():
    for exp in comet_optimizer.get_experiments(project_name="kilosort4-optimizer"):
        good_count = number_of_good_units(
            batch_size=exp.get_parameter("batch_size"),
            nblocks=exp.get_parameter("nblocks"),
            Th_universal=exp.get_parameter("Th_universal"),
            Th_learned=exp.get_parameter("Th_learned"),
            do_CAR=eval(exp.get_parameter("do_CAR")),
            invert_sign=eval(exp.get_parameter("invert_sign")),
            nt=exp.get_parameter("nt"),
            artifact_threshold=exp.get_parameter("artifact_threshold"),
            nskip=exp.get_parameter("nskip"),
            whitening_range=exp.get_parameter("whitening_range"),
            binning_depth=exp.get_parameter("binning_depth"),
            sig_interp=exp.get_parameter("sig_interp"),
            nt0min=exp.get_parameter("nt0min"),
            dmin=exp.get_parameter("dmin"),
            dminx=exp.get_parameter("dminx"),
            min_template_size=exp.get_parameter("min_template_size"),
            template_sizes=exp.get_parameter("template_sizes"),
            nearest_chans=exp.get_parameter("nearest_chans"),
            nearest_templates=exp.get_parameter("nearest_templates"),
            templates_from_data=eval(exp.get_parameter("templates_from_data")),
            n_templates=exp.get_parameter("n_templates"),
            n_pcs=exp.get_parameter("n_pcs"),
            Th_single_ch=exp.get_parameter("Th_single_ch"),
            acg_threshold=exp.get_parameter("acg_threshold"),
            ccg_threshold=exp.get_parameter("ccg_threshold"),
            cluster_downsampling=exp.get_parameter("cluster_downsampling"),
            cluster_pcs=exp.get_parameter("cluster_pcs"),
            duplicate_spike_bins=exp.get_parameter("duplicate_spike_bins"),
            do_correction=eval(exp.get_parameter("do_correction")),
            keep_good_only=eval(exp.get_parameter("keep_good_only")),
            save_extra_kwargs=eval(exp.get_parameter("save_extra_kwargs")),
            skip_kilosort_preprocessing=eval(
                exp.get_parameter("skip_kilosort_preprocessing")
            ),
            scaleproc=exp.get_parameter("scaleproc"),
        )
        exp.log_parameters(number_of_good_units.params)
        exp.log_metric("good units", good_count)
        exp.end()


if __name__ == '__main__':
    main()
