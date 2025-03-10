import osm_graph
import network_random_risk
import json
import sys
import os
import csv
import requests
import math
import pandas as pd



def validate_poi(poi_file):
    """
    Validate POI file
    """
    lat_list = []
    lon_list = []
    name_list = []
    with open(poi_file) as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) >= 4, "must at least consists 4 columns: name, lat, lng, type"
            name = row[0]
            cat = row[1]
            lat = float(row[2])
            lon = float(row[3])
            name_list.append(name)
            lat_list.append(lat)
            lon_list.append(lon)

    lat_avg = sum(lat_list) / len(lat_list)
    lon_avg = sum(lon_list) / len(lon_list)
    assert lat_avg <= 90.0, "average lat > 90.0. Something is wrong"
    assert lat_avg >= -90.0, "average lat < -90.0. Something is wrong"
    assert lon_avg <= 180.0, "average lon > 180.0. Something is wrong"
    assert lon_avg >= -180.0, "average lon < -180.0. Something is wrong"

    # check unusual lat/lon
    threshold = 0.3
    for i in range(len(name_list)):
        if abs(lat_list[i]-lat_avg) > threshold or abs(lon_list[i]-lon_avg) > threshold:
            print(f"unusual: {i+1} {name_list[i]} {lat_list[i]} {lon_list[i]}")

input_json = sys.argv[1]

with open(input_json) as f:
    input = json.load(f)
    output_dir = input['output_dir']
    locations_table_name = input['name'] + '_locations'
    subnetwork_json_path = os.path.join(output_dir, input['name'] + '_subnetwork.json')
    subnetwork_pycgr_path = os.path.join(output_dir, input['name'] + '_subnetwork.pycgrc')
    # validate
    validate_poi(input['poi_file'])

    print('Process Begin!')
    # process    
    osm_graph.load_locations_and_routes(input['name'], input['network_pycgr_file'])
    osm_graph.get_nearest_node_ids(locations_table_name, input['poi_file'])
    osm_graph.create_subgraph(input['poi_file'], input['network_json_file'], subnetwork_json_path)
    osm_graph.graph_to_pycgr(subnetwork_json_path, subnetwork_pycgr_path)
    osm_graph.resolve_conflicting_nodes(locations_table_name, input['poi_file'], subnetwork_pycgr_path)
    osm_graph.reindex_graph_locations(input['poi_file'])
    # Calculate Risk From Inarisk
    # if(input['name']+'_subnetwork.pycgrc'):
    #     print('Calculating Network Risk')

    #
    network_random_risk.generate_risk_from_pycgr(input['network_pycgr_file'], input['category_risk']);
    network_random_risk.generate_risk_from_pycgr(subnetwork_pycgr_path, input['category_risk'])

# network_risk_inarisk.generate_risk_from_pycgr(input['network_pycgr_file'], input['risk_layer_file'], input['risk_coordinates_samples'])
        # network_risk_inarisk.generate_risk_from_pycgr(subnetwork_pycgr_path, input['risk_layer_file'], input['risk_coordinates_samples'])

