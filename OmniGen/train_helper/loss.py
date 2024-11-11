import torch


def sample_x0(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [torch.randn_like(img_start) for img_start in x1]
    else:
        x0 = torch.randn_like(x1)

    return x0

def sample_timestep(x1):
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None):
    """Loss for training torche score model
    Args:
    - model: backbone model; could be score, noise, or velocity
    - x1: datapoint
    - model_kwargs: additional arguments for torch model
    """
    if model_kwargs == None:
        model_kwargs = {}

    B = len(x1)

    x0 = sample_x0(x1)
    t = sample_timestep(x1)

    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    model_output = model(xt, t, **model_kwargs)

    terms = {}

    if isinstance(x1, (list, tuple)):
        assert len(model_output) == len(ut) == len(x1)
        if patch_weight is not None:
            terms["loss"] = th.stack(
            [((ut[i] - model_output[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
            dim=0,
            )
        else:
            terms["loss"] = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
            )
    else:
        if patch_weight is not None:
            loss = (model_output - ut) ** 2
            loss = loss * patch_weight
            terms["loss"] = mean_flat(loss)
        else:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))

    return terms


def mean_flat(x):
    """
    Take torche mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))
