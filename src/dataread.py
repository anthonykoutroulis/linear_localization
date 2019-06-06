import csv;
import os;


def get_nodes():
    import numpy as np;
    # LLA (deg,deg,m)
    xn = np.array(( 37.6133603333333,
                    37.6133198666667,
                    37.6132646,
                    37.61321225,
                    37.6131987833333,
                    37.6132384666667,
                    37.6132954,
                    37.6133465));

    yn = np.array(( -119.021031533333,
                    -119.02096225,
                    -119.020950133333,
                    -119.020994066667,
                    -119.02106895,
                    -119.021139883333,
                    -119.021151716667,
                    -119.0211086));

    zn = np.array(( 2737.456704,
                    2737.276664,
                    2737.345904,
                    2737.602288,
                    2737.702424,
                    2737.897384,
                    2738.013144,
                    2737.757144));

    xn -= min(xn);
    yn -= min(yn)
    nodes = [(x,y,z) for x,y,z in zip(xn,yn,zn)];
    return np.array(nodes);

def get_sensor_logs():
    # constants
    ANCHOR_NODE_COL = 2;
    MAX_NUM_ANCHORS = 8;
    TIME_COL = 0;
    MAX_TIMEOUT = 0.040000;

    # variables
    full_readings_list = [];    # list of lists of 8 contiguous sensor readings from nodes 1-8
    file_list = [];
    temp_list = [];             # holds at most the last 8 contiguous sensor readings

    # get list of all csv files to parse in current directory
    data_dir = os.getcwd() + "/data/sensor_logs";

    #print(os.)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_list.append(data_dir + "/" + file);

    for data_file in file_list:
        # VERSION 2 can cycle through all sensor data files. see krystine for version 2.
        sensor_file = open(data_file, 'r');
        sensor_data_reader = csv.reader(sensor_file);

        # find first instance of node 1, then only pull data for complete
        # sensor sets (all sensors had a value)
        current_sensor = 0;
        expected_sensor_val = 1;    # expected values: 1-8 increasing
        temp_list.clear();

        first_row = True;

        for row in sensor_data_reader:
            current_sensor = row[ANCHOR_NODE_COL];

            if first_row:
                temp_time = row[TIME_COL].split(' ')[1]
                temp_time = temp_time.split(":");
                time_last = int(temp_time[0])*3600 + int(temp_time[1])*60 + float(temp_time[2]);
                first_row = False;

            # convert time to seconds.useconds
            temp_time = row[TIME_COL].split(' ')[1]
            temp_time = temp_time.split(":");
            time_sec = int(temp_time[0])*3600 + int(temp_time[1])*60 + float(temp_time[2]);

            # check that it's the expected sensor val and that the time offset is
            # within a constant value
            if int(current_sensor) == expected_sensor_val and (time_sec - time_last) < MAX_TIMEOUT:
                # add row to temp list if it matches what was expected
                temp_list.append(row);
                expected_sensor_val += 1;

                if int(current_sensor) == MAX_NUM_ANCHORS:
                    # add temp list to good list and clear temp list if full set has been read
                    full_readings_list.append(temp_list);
                    temp_list = [];
                    expected_sensor_val = 1;
            else:
                # clear temp list
                # set expected sensor back to 1
                temp_list.clear();
                expected_sensor_val = 1;

            time_last = time_sec;

        sensor_file.close();

    # test print
    print("Num good sets: " + str(len(full_readings_list)));
    return full_readings_list;

def get_flight_logs():
    # constants
    LAT_COL = 0;
    LON_COL = 1;
    ALT_COL = 2;
    TIME_COL = 7;

    # variables
    file_list = [];
    lat_list = [];
    lon_list = [];
    alt_list = [];
    time_list = [];

    # get list of all csv files to parse in current directory
    data_dir = os.getcwd() + "/data/flight_logs";

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_list.append(data_dir + "/" + file);

    smallest_z = 1000000;

    for data_file in file_list:
        # cycle through all sensor data files
        sensor_file = open(data_file, 'r');
        sensor_data_reader = csv.reader(sensor_file);

        first_row = True;

        for row in sensor_data_reader:
            if not first_row:
                lat_list.append(float(row[LAT_COL]));
                lon_list.append(float(row[LON_COL]));
                time_list.append(row[TIME_COL])

                # convert feet to meters before appending
                alt_meters = float(row[ALT_COL]) * 0.3048;
                alt_list.append(alt_meters);

                if alt_meters < smallest_z:
                    smallest_z = alt_meters;
            else:
                first_row = False;

        sensor_file.close();

    # test print
    print("Num good sets: " + str(len(lat_list)));
    print("Lowest altitude: " + str(smallest_z));
    return lat_list, lon_list, alt_list, time_list;

def get_isensor_logs():
    import numpy as np;
    # constants
    ANCHOR_NODE_COL = 2;
    MAX_NUM_ANCHORS = 8;

    # variables
    full_readings_list = [];    # list of lists of 8 contiguous sensor readings from nodes 1-8
    file_list = [];
    temp_list = [];             # holds at most the last 8 contiguous sensor readings

    # get list of all csv files to parse in current directory
    data_dir = os.getcwd() + "/data/sensor_logs";

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_list.append(data_dir + "/" + file);

    expected_sensor_val = 1;    # expected values: 1-8 increasing

    for data_file in file_list:
        # TODO cycle through all sensor data files
        sensor_file = open(data_file, 'r');
        sensor_data_reader = csv.reader(sensor_file);

        # find first instance of node 1, then only pull data for complete
        # sensor sets (all sensors had a value)
        current_sensor = 0;
        temp_list.clear();

        for row in sensor_data_reader:
            current_sensor = row[ANCHOR_NODE_COL];
            if int(current_sensor) == expected_sensor_val:
                # add row to temp list if it matches what was expected
                temp_list.append(row);
            else:
                # add empty rows to temp list
                while expected_sensor_val < int(current_sensor):
                    temp_list.append([np.nan,str(expected_sensor_val),np.nan]);
                    expected_sensor_val += 1;
                # add the current sensor row
                temp_list.append(row);

            expected_sensor_val += 1;
            if int(current_sensor) == MAX_NUM_ANCHORS:
                    # add temp list to good list and clear temp list if full set has been read
                    full_readings_list.append(temp_list);
                    temp_list = [];
                    expected_sensor_val = 1;
        sensor_file.close();

    # test print
    print("Num good sets: " + str(len(full_readings_list)));
    return full_readings_list;

def get_node_locs():
    # constants
    LAT_COL = 1;
    LON_COL = 2;
    ALT_COL = 3;
    
    # variables
    lat_list = [];
    lon_list = [];
    alt_list = [];
    
    data_file = os.getcwd() + "/data/nod_locs.csv"
    
    # cycle through all sensor data files
    file = open(data_file, 'r');
    reader = csv.reader(file);
    
    # skip header, TODO add sniffer
    next(reader);

    for row in reader:
        print(row);
        lat_list.append(float(row[LAT_COL]));
        lon_list.append(float(row[LON_COL]));
        alt_list.append(float(row[ALT_COL]));

    file.close();
    return lat_list, lon_list, alt_list;