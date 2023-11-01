from argparse import ArgumentParser
import libsumo as traci
from os import system
import pandas as pd
from sklearn.model_selection import train_test_split

def merge_vehicles_and_pedestrians(fcd:pd.DataFrame):
    merged = pd.DataFrame()
    merged['timestep'] = fcd['timestep_time']
    merged['type'] = fcd['vehicle_type'].apply(lambda e: e if not pd.isna(e) else 'pedestrian')
    merged['lane/edge'] = fcd['vehicle_lane'].fillna(fcd['person_edge'])
    for col in ['id', 'angle', 'pos', 'slope', 'speed', 'x', 'y']:
        merged[col] = fcd[f'vehicle_{col}'].fillna(fcd[f'person_{col}'])
    merged = merged.dropna()
    return merged


parser = ArgumentParser(
    description="Script for Stop Detection Ground-truth Generator")
parser.add_argument("-s","--speed-threshold", type=float, default=0.0,
                    help="Speed Threshold for Stop Classication, egual to or below.")
parser.add_argument("-t","--test-size", type=float, default=0.2,
                    help="Test size, percent")

if __name__ == '__main__':
    args = parser.parse_args()
    # use the osmWebWizard or something else to Generate Scenario
    system('python ./customOsmWebWizard/osmWebWizard.py -o osmOutputDir')

    # run the simulation and save results
    traci.start(['sumo', '-c', './osmOutputDir/osm.sumocfg','--fcd-output.geo','--fcd-output', 'fcd-output.xml'])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

    # turn the the .xml into .csv
    system('python $SUMO_HOME/tools/xml/xml2csv.py ./fcd-output.xml')

    # read .csv file
    fcd_output = pd.read_csv('./fcd-output.csv', sep=';')

    merged = merge_vehicles_and_pedestrians(fcd_output)

    # # Add Stop column, when speed igual to 'speed_threshold'
    speed_threshold = args.speed_threshold
    merged['stop'] = merged['speed'].apply(lambda s: s <= speed_threshold)

    output_file = 'groundTruth'
    system(f'mkdir -p {output_file}/')
    merged.to_csv(f'./{output_file}/stop_dataset.csv', index=False)

    # Split in training and test
    train_ids, test_ids = train_test_split(merged['id'].unique(), test_size=0.2, random_state=0)

    train_data = merged[merged['id'].isin(train_ids)]
    test_data = merged[merged['id'].isin(test_ids)]

    train_data.to_csv(f'./{output_file}/stop_train.csv', index=False)
    test_data.to_csv(f'./{output_file}/stop_test.csv', index=False)