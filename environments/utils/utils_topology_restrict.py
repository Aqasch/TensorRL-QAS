import configparser
import json
from itertools import product


def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict



def dictionary_of_actions_hexagon_connectivity(num_qubits):
    """
    Creates dictionary of actions with generalized honeycomb connectivity pattern.
    """
    dictionary = dict()
    i = 0
         
    # Generate all possible control-target pairs
    for c, x in product(range(num_qubits), range(1, num_qubits)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1
   
    # Generate rotation axes configurations
    for r, h in product(range(num_qubits), range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1

    # Hexagon connectivity pattern
    if num_qubits == 6:
        hexagon_connections = [(0,1), (0,2), (0,3), (3,4), (4,5)]
    elif num_qubits == 8:
        hexagon_connections = [(0,1), (1,0), (0,2), (2,0), (0,3), (3,0), (3,4), (4,3),\
                               (4,5), (5,4), (4,6), (6,4), (6,7), (7,6)]
    elif num_qubits == 10:
        hexagon_connections = [(0,1), (0,2), (0,3), (3,4), (4,5), (4,6), (6,7), (7,8), (7,9)]

    # print(hexagon_connections)
    # Filter valid actions using honeycomb pattern
    valid_actions = []
    for k in dictionary.keys():
        act = dictionary[k]
        ctrl = act[0]
        targ = (act[0] + act[1]) % num_qubits
        tup = (ctrl, targ)

        if tup in hexagon_connections:
            valid_actions.append(act)

    # Create final action dictionary
    return {len(valid_actions)-1-val_act_no: val_act 
            for val_act_no, val_act in enumerate(valid_actions)}
    

def dictionary_of_actions_hexagon_connectivity_reverted(num_qubits):
    """
    Creates dictionary of actions with generalized honeycomb connectivity pattern.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1

    # Hexagon connectivity pattern
    if num_qubits == 6:
        hexagon_connections = [(0,1), (0,2), (0,3), (3,4), (4,5)]
    elif num_qubits == 8:
        hexagon_connections = [(0,1), (0,2), (0,3), (3,4), (4,5), (4,6), (6,7)]
    elif num_qubits == 10:
        hexagon_connections = [(0,1), (0,2), (0,3), (3,4), (4,5), (4,6), (6,7), (7,8), (7,9)]

    # print(hexagon_connections)
    # Filter valid actions using honeycomb pattern
    valid_actions = []
    for k in dictionary.keys():
        act = dictionary[k]
        ctrl = act[0]
        targ = (act[0] + act[1]) % num_qubits
        tup = (ctrl, targ)

        if tup in hexagon_connections:
            valid_actions.append(act)

    # Create final action dictionary
    return {len(valid_actions)-1-val_act_no: val_act 
            for val_act_no, val_act in enumerate(valid_actions)}

   
