import torch


def send_batch_to(batch, device, unsqueeze=False):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
            if unsqueeze:
                batch[key] = batch[key].unsqueeze(0)
        else:
            send_batch_to(value, device, unsqueeze=unsqueeze)
