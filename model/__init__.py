from .transformer.trans_wsvos_tracker import build_trans_wsvos_tracker
from .transformer.trans_wsvos import build_trans_wsvos

def build_model(cfg):
    if cfg.MODEL.TYPE == 'TRANSFORMER':
        return build_trans_wsvos(cfg)
    else:
        raise Exception("Unkown model!")

def build_tracker(cfg, net):
    if cfg.MODEL.TYPE == 'TRANSFORMER':
        return build_trans_wsvos_tracker(cfg, net)
    else:
        raise Exception("Unkown tracker!")
