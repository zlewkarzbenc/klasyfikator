#!/usr/bin/env python3

import os
import time
import json
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

def process_window(window: str)-> defaultdict:
    """ Compute a 3D dictionary of float values derived from chromVAR z-scores averaged over cell subpopulations
        (with structure: data[loop][motif][profile]).

    Args:
        window (str): timepoint to process (eg. 'hrs06-08')

    Returns:
        data_3D: a defaultdict of float values (average chromVAR scores for motifs in cell populations having a distinct loop activity pattern)

    """
    activity_profiles = ["1-1", "other"]

    loops_path = f"data/new_time/hrs{window}_NNv1_time_matrix_loops.tsv"
    motifs_path = f"data/new_time/hrs{window}_NNv1_time_matrix_motifs.tsv"

    loops_df = pd.read_csv(loops_path, sep='\t')
    motifs_df = pd.read_csv(motifs_path, sep='\t')

    # Preliminary processing files
    # This order (to_numeric(), then dropna()) deletes loop/motif names as well as true missing values.
    for df in [loops_df, motifs_df]:
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(axis=1, inplace=True)

    # Real business
    data_3D = defaultdict(lambda: defaultdict(dict))

    for loop_id in loops_df.index:

        # Identify subpopulations based on loop presence
        loop_values = loops_df.loc[loop_id]
        cells_11 = loop_values[loop_values == 11].index
        cells_other = loop_values.index.difference(cells_11)

        for motif_id in motifs_df.index:
            motif_values = motifs_df.loc[motif_id]
            # Compute mean z-score within each subpopulation
            mean_11 = motif_values.loc[cells_11].mean() if len(cells_11) > 0 else np.nan
            mean_other = motif_values.loc[cells_other].mean() if len(cells_other) > 0 else np.nan

            data_3D[loop_id][motif_id]["1-1"] = mean_11
            data_3D[loop_id][motif_id]["other"] = mean_other

    return data_3D

def convert_2D(data_3D: defaultdict)-> pd.DataFrame:
    """ For each (loop, motif) pair: compute the difference in mean chromVAR scores between populations of cells '1-1' and 'other'. 

    Args:  
        data_3D (defaultdict): output of the process_window() function.

    Returns:
        data_diff: a DataFrame with loops (rows), motifs (columns) and "1-1" - "other" difference (values).
    """
    num_loops, num_motifs = len(data_3D), len(data_3D[0])
    data_2D = defaultdict(lambda: defaultdict(dict))

    records = []
    for loop_id, motif_dict in data_3D.items():
        for motif_id, profile_dict in motif_dict.items():

            mean_11 = profile_dict["1-1"]
            mean_other = profile_dict["other"]
            data_2D[loop_id][motif_id] = np.subtract(mean_11, mean_other)

    data_diff = pd.DataFrame.from_dict(data_2D, orient='index')

    return data_diff

def main():
    #   os.makedirs(OUT_DIR, exist_ok=True)
    windows = ['06-08', '10-12', '14-16']
    activity_profiles = ["1-1", "other"]

    with open('log_data.txt', 'w+') as log:
        for window in windows:
            start = time.time()
            log.write(f"{datetime.datetime.now()}\t Started processing window: hrs{window}...\n")
            os.makedirs(f"results/hrs{window}", exist_ok=True)

            data_3D = process_window(window)
            json.dump(data_3D, open(f"results/hrs{window}/data_3D_hrs{window}.json", 'w+'))

            data_diff = convert_2D(data_3D)
            data_diff.to_csv(f"results/hrs{window}/data_diff_her{window}.csv")
            log.write(f'{datetime.datetime.now()}\t Finished processing window: "hrs{window}" after {time.time()-start:.3f} seconds\n')


if __name__ == '__main__':
    main()