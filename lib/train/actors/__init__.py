from .base_actor import BaseActor
from .tpn_st import SiamTPNSTActor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from lib.utils.box_ops import giou_loss, ciou_loss
from torch.nn.functional import l1_loss
from lib.utils.focal_contrastive_loss import focal_contrastive_loss

def build_actor_st(cfg, net):
    if cfg.TRAIN.ACTOR == "TPNSTACTOR":
        # Changed to BCEWithLogitsLoss to match single-neuron output
        objective = {'giou': giou_loss, 'l1': l1_loss, 'ce': BCEWithLogitsLoss(), 'center': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'ce': cfg.TRAIN.CLS_WEIGHT, 'center': cfg.TRAIN.CENTER_WEIGHT}
        actor = SiamTPNSTActor(net=net, objective=objective, loss_weight=loss_weight, cfg=cfg)
        return actor          

    raise RuntimeError(F"Actor not implemented")
