import os
import pickle

def pickle_dump(loadout, file):
    """
    Dump a pickle file. Create the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(str(file)), exist_ok=True)

    with open(file, 'wb') as f:
        pickle.dump(loadout, f)
