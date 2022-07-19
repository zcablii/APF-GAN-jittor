import pickle
import os
import numpy as np
import sys


def merge_ckpt(path,ckpts_to_merge):
    ckpt_root_path = path
    save_path = os.path.join(path, 'avg_' + '_'.join(ckpts_to_merge) + '_net')
    model_nets = ['E','G'] # 'E' or 'G'

    print(save_path)

    for model in model_nets:
        model_weights = []
        for ep in ckpts_to_merge:
            net_name = '%s_net_%s.pkl'%(ep,model)
            model_path = os.path.join(ckpt_root_path,net_name)
            print(model_path)
            model_weights.append(pickle.load(open(model_path,'rb+')))
        
        weight_avg = {}
        for key in model_weights[0].keys():
            ws = np.zeros(model_weights[0][key].shape)
            for model_weight in model_weights:
                ws+=model_weight[key]
            weight_avg[key] = ws/len(model_weights)

        with open('%s_%s.pkl'%(save_path, model),'wb') as fo:
            pickle.dump(weight_avg,fo)

if __name__ == "__main__":
    path = sys.argv[1]
    ckpts_to_merge = []
    for i in range(2, len(sys.argv)):
        ckpts_to_merge.append(str(sys.argv[i]))
    merge_ckpt(path,ckpts_to_merge)