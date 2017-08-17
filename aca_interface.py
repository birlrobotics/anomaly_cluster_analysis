
from optparse import OptionParser
import config

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
    print "press any key to continue."
    raw_input()

def build_parser():
    parser = OptionParser()

    parser.add_option(
        "--cluster",
        action="store_true", 
        dest="cluster",
        default = False,
        help="True if you want to cluster.")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()
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
                anomaly_group_by_state,
                config.interested_data_fields,
                config.cluster_algorithm['parameters'],
                config.result_save_path,
                config.result_id,
            )
        elif config.cluster_algorithm['name'] == 'gmm':
            import cluster_by_gmm
            cluster_by_gmm.run(
                anomaly_group_by_state,
                config.interested_data_fields,
                config.cluster_algorithm['parameters'],
                config.result_save_path,
                config.result_id,
            )
