import torch


def load_pretrained_model_pixel(model, path, name):
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
    print('load pretrained pixel model finished.')
    return model


def load_pretrained_model_image(model, path, name):

    item = path + name
    pretrain_dict = torch.load(item)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k[2:4] != 'fc':
            if k in state_dict:
                print(k)
                model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    print('load pretrained image model finished.')
    return model


def save_pretrained_model(model, path, name):
    torch.save(model.state_dict(), path + name)
    print('save pretrained model finished.')