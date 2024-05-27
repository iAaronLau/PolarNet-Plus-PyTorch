import torch


def int2tuple(i: int):
    return (i, i) if isinstance(i, int) else i


def str_size(input_size: tuple, input_size_b: tuple = None):
    input_size = int2tuple(input_size)
    str_input_size = "{}x{}".format(input_size[0], input_size[1])

    if input_size_b:
        input_size_b = int2tuple(input_size_b)
        str_input_size += "&{}x{}".format(input_size_b[0], input_size_b[1])

    return str_input_size


class ConfigBase():
    token = "PDU293TXLCWZlAT894b2ghLOH675pSf1pAU83OD"
    num_classes = 2
    pretrained = False
    is_use_cam = False
    is_save_model = True
    n_fold = [1, 2, 3, 4, 5]
    cam_save_epochs = -1
    base_lr = 2e-5
    weight_decay = 5e-5

    # prior_k = None

    vis_servers = [
        "http://172.18.188.241",
        "http://172.18.188.240",
        # "http://172.18.12.179",
        # "http://172.18.12.104",
    ]
    vis_port = 20001

    batch_size = 28
    # rate = 14
    resize_to_ori = (224, 224)
    # resize_to_polar = (16 * 6 * rate, 16 * rate)
    num_epochs = 200
    stop_epoch = 120

    layers = ["浅层血管复合体", "深层血管复合体", "脉络膜毛细血管层"]
    in_channel = len(layers)
    # layer_index_polar_ori = [[0, 1, 2], [0, 1, 2]]
    dataset_name = "Combine"
    weight = [.11, .21]
    input_size = str_size(resize_to_ori)

    def __getitem__(self, key):
        return self.__getattribute__(key)


class UniversalConfig(ConfigBase):
    # layers = ["浅层血管复合体", "深层血管复合体", "脉络膜毛细血管层"]
    layers = ["深层血管复合体"]
    # layers = ["浅层血管复合体"]
    # layers = ["深层血管复合体"]
    # layers = ["脉络膜毛细血管层"]
    in_channel = len(layers)
    layer_index_polar_ori = [[0, 1, 2], [0, 1, 2]]
    resize_to = (224, 224)
    # scale = 1.3


# class MUCOConfig(ConfigBase):
#     layers = ["浅层血管复合体", "深层血管复合体", "脉络膜毛细血管层"]
#     in_channel = len(layers)

# class OurConfig(ConfigBase):
#     input_size = str_size(ConfigBase.resize_to_ori, ConfigBase.resize_to_polar)
#     # layer_index_polar_ori = [[0, 1, 2], [0, 1, 2]]
#     # in_channel = (len(layer_index_polar_ori[0]), len(layer_index_polar_ori[1]))
#     is_cam: bool = True
#     use_pk: bool = True

#     prior_k: torch.Tensor = None
