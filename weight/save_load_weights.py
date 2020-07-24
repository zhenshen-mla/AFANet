import torch


def load_pretrained_model(model, path, name):
    item = path + name
    pretrain_dict = torch.load(item)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k[:8] == 'backbone':
            if k in state_dict:
                print('load: ', k)
                model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    print('load pretrained model finished.')
    return model


def save_pretrained_model(model, path, name):
    torch.save(model.state_dict(), path + name)
    print('save pretrained model finished.')