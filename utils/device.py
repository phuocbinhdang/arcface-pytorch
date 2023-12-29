import torch


def select_device(device):
    # device = None or 'cpu' or '0' or '1' or '2'
    device = device.strip().lower().replace("cuda:", "").replace("none", "")
    cpu = device == "cpu"

    if not cpu and torch.cuda.is_available():
        if device.isnumeric() and (int(device) < torch.cuda.device_count()):
            arg = "cuda:" + device
        else:
            arg = "cuda:0"
    else:
        arg = "cpu"

    return torch.device(arg)
