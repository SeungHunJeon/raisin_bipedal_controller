import sys
import torch
from ruamel.yaml import YAML, dump, RoundTripDumper

cfg = YAML().load(open("./../config/params.yaml", 'r'))
iteration = cfg['model_number']['value']
full = torch.load('full_'+str(iteration)+'.pt', map_location='cpu')  # full_*.pt file


def save_graph_to_txt(key: str, name: str):
    model = full[key]
    f = open(name + '_' + str(iteration) + ".txt", 'w')

    content = ''

    for w in model.items():
        if w[0].split('.')[0] == name:
            w = w[1]  # weight tensor
            if w.ndim == 2:
                w = w.transpose(0, 1).flatten()
            for i in w:
                content += str(i.item()) + ", "  # tensor to float

    f.write(content[:-2])


save_graph_to_txt('actor_architecture_state_dict', 'GRU')
save_graph_to_txt('actor_architecture_state_dict', 'MLP')
