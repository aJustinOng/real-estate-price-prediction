import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(state, bed, bath, sqft):
    try:
        loc_index = __data_columns.index(state.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bed
    x[1] = bath
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    print("Loading save artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/usa_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Arkansas', 2, 2, 1000))
    print(get_estimated_price('Arkansas', 3, 3, 1000))
    print(get_estimated_price('New York', 2, 2, 1000))
    print(get_estimated_price('New York', 3, 3, 1000))