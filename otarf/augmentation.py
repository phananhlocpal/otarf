"""Past-only view masking: the full blackout uses context-preceding anchor."""


def corrupt(x, mark, prefix, config, generator=None):
    import torch

    b, t, c = x.shape
    altered = x.clone()
    marks = mark.clone()
    rand = lambda: torch.rand((b, c), device=x.device, generator=generator)
    selected = rand() < config["augmentation"]["channel_fraction"]
    first = selected & (rand() < 0.5)
    for length, chosen in [(24, first), (96, selected & ~first)]:
        start = t - length
        if start:
            value = altered[:, start - 1, :]
            age = marks[:, start - 1, c : 2 * c]
            valid = marks[:, start - 1, 2 * c : 3 * c]
            anchor = marks[:, start - 1, 3 * c : 4 * c]
        else:
            value, age, valid, anchor = (prefix[:, i, :] for i in range(4))
        w = chosen[:, None, :]
        altered[:, start:, :] = torch.where(w, value[:, None, :], altered[:, start:, :])
        marks[:, start:, :c] = torch.where(
            w, torch.zeros_like(marks[:, start:, :c]), marks[:, start:, :c]
        )
        marks[:, start:, c : 2 * c] = torch.where(
            w,
            age[:, None, :]
            + torch.arange(1, length + 1, device=x.device)[None, :, None],
            marks[:, start:, c : 2 * c],
        )
        marks[:, start:, 2 * c : 3 * c] = torch.where(
            w, valid[:, None, :], marks[:, start:, 2 * c : 3 * c]
        )
        marks[:, start:, 3 * c : 4 * c] = torch.where(
            w, anchor[:, None, :], marks[:, start:, 3 * c : 4 * c]
        )
    return altered, marks
