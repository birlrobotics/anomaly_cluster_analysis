
from optparse import OptionParser
import config

def inform_config(config):
    import json
    config_to_print = [
        'config.config_by_user',
        'config.interested_data_fields',
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
    return parser

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()
    inform_config(config)

    import load_anomaly_data
    list_of_anomaly = load_anomaly_data.run(
        config.anomaly_data_path,
        config.interested_data_fields,
    )
    if config.cluster_algorithm['name'] == 'kmeans':
        import cluster_by_kmeans
        cluster_by_kmeans.run(
            list_of_anomaly,
        )
