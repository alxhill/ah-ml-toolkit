import torch

from ah_ml_toolkit.toolkit.midas.external.dpt_depth import DPTDepthModel


class MidasModel(torch.nn.Module):
    def __init__(self, model_weights_path: str, size: tuple[int, int]):
        super().__init__()
        self.size = size
        self.depth_model = DPTDepthModel(model_weights_path, True, )
