from optparse import OptionParser
import config
import copy
import birl.data_for_or_from_HMM.ML_DATA_Shuangqi.load_data_folder as load_data_folder
import time
import sys


def inform_config(config):
    import json
    config_to_print = [
        'config.config_by_user',
        'config.interested_data_fields',
        'config.result_id',
    ]
    
    for s in config_to_print:
        print '-'*20
        print s, ':'
        print json.dumps(
            eval(s),
            indent=4,
        )
    print '#'*20

def build_parser():
    parser = OptionParser()

    parser.add_option(
        "--cluster",
        action="store_true", 
        dest="cluster",
        default = False,
        help="True if you want to cluster.")

    parser.add_option(
        "--visualizeclusters",
        action="store_true", 
        dest="visualizeclusters",
        default = False,
        help="True if you want to visualizeclusters.")

    parser.add_option(
        "--traverseconfig",
        action="store_true", 
        dest="traverseconfig",
        default = False,
        help="True if you want to traverseconfig.")

    parser.add_option(
        "--collectbestresults",
        action="store_true", 
        dest="collectbestresults",
        default = False,
        help="True if you want to collectbestresults.")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()


    if options.traverseconfig:
        import traverse_config_automatically
        config_iterable = traverse_config_automatically.get_config_generator(config.config_to_traverse)
    else:
        config_iterable = [None]
    
    for _ in config_iterable:
        config.generate_auto_config()
    
        inform_config(config)

        import load_anomaly_data
        anomaly_group_by_state = load_anomaly_data.run(
            config.anomaly_data_path,
            config.interested_data_fields,
            config.data_preprocessing_config,
        )

        if options.cluster:
            if config.cluster_algorithm['name'] == 'kmeans':
                import cluster_by_kmeans
                cluster_by_kmeans.run(
                    copy.deepcopy(anomaly_group_by_state),
                    config.interested_data_fields,
                    config.cluster_algorithm['parameters'],
                    config.result_save_path,
                    config.result_id,
                )
            elif config.cluster_algorithm['name'] == 'gmm':
                import cluster_by_gmm
                cluster_by_gmm.run(
                    copy.deepcopy(anomaly_group_by_state),
                    config.interested_data_fields,
                    config.cluster_algorithm['parameters'],
                    config.result_save_path,
                    config.result_id,
                )
        if options.visualizeclusters:
            df_group_by_foldername = load_data_folder.run(config.anomaly_data_path)

            import visualize_clusters
            visualize_clusters.run(
                copy.deepcopy(anomaly_group_by_state),
                df_group_by_foldername,
                config.result_save_path,
                config.result_id,
            )

    if options.collectbestresults:
        import collect_best_results
        collect_best_results.run(
            config.analysis_folder_path,
        )
