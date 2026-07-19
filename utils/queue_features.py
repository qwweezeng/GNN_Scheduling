import torch


def relative_queue_feature(queues):
    """Return link queues normalized by the largest queue in the snapshot."""
    queues = torch.clamp(torch.as_tensor(queues, dtype=torch.float32), min=0.0)
    if queues.numel() == 0:
        return queues
    return queues / queues.max().clamp_min(1.0)


def local_relative_queue(queues, edge_index):
    """Normalize each queue by the largest one-hop-neighborhood queue."""
    queues = torch.as_tensor(
        queues, dtype=torch.float32
    ).reshape(-1).clamp_min(0.0)
    local_max = queues.clone().clamp_min(1.0)
    if edge_index.numel() > 0:
        source, target = edge_index
        local_max.scatter_reduce_(0, target, queues[source], reduce="amax")
        local_max.scatter_reduce_(0, source, queues[target], reduce="amax")
    return queues / local_max.clamp_min(1.0)
