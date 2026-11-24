from torch import nn

class FrameRefiner(nn.Sequential):
    """Predict a metric depth correction from D0, optionally conditioned on Y."""

    def __init__(self, mode):
        if mode not in ('depth', 'measurement'):
            raise ValueError(f'Unknown refinement mode {mode!r}')
        channels = 2 if mode == 'measurement' else 1
        super().__init__(
            nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        self.mode = mode
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)
