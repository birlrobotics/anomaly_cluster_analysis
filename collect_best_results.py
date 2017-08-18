import os
import re
import glob
import ipdb
import json
import shutil 



def run(
    analysis_folder_path,
):


    all_results = glob.glob(os.path.join(analysis_folder_path, "datatype_*/algorithm_*/config_*/state_*/state * cluster *.png"))
    result_group_by_state = {}


    file_name_parser = re.compile('state (\d+) cluster (\d+).png')
    for path in all_results:
        png_file_name = os.path.basename(path)
        folder_path = os.path.dirname(path)

        cluster_result = json.load(open(os.path.join(folder_path, 'cluster_result.json'), 'r'))


        m = file_name_parser.match(png_file_name)
        if not m:
            raise Exception('path return by glob is problematic:', path)
        state_no = m.group(1)
        cluster_no = m.group(2)

        silhouette_score = cluster_result[cluster_no]['silhouette_score']


        if state_no not in result_group_by_state:
            result_group_by_state[state_no] = []

        result_group_by_state[state_no].append({
            'silhouette_score': silhouette_score,
            'path': path,
        })


    output_path = os.path.join(analysis_folder_path, 'colleted_best_result_state_wise')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for state_no in result_group_by_state:
        list_of_result = result_group_by_state[state_no]
        best = max(list_of_result, key=lambda x:x['silhouette_score'])
        file_name = os.path.basename(best['path'])
        shutil.copyfile(best['path'], os.path.join(output_path, file_name))

