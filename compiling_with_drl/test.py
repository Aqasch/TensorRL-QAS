import torch
from itertools import product

num_qubits = 2
state_MLP = torch.zeros(2*4**num_qubits)
state_phasor = torch.zeros(2*2**num_qubits)
print(state_MLP, state_phasor)

basis_type = 'rotations'


def action_to_gate_converter(num_qubits, basis_type):
    dictionary = dict()
    i = 0
    if basis_type == 'rotations':
        for c, x in product(range(3),
                        range(2)):
            dictionary[i] = [c,x]
            i +=1

    print(dictionary)

action_to_gate_converter(1, basis_type)