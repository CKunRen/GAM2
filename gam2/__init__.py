"""GAM2: gated adaptive multi-fidelity metamodelling.

Reference implementation of the network and training procedure described in
the paper.  The package contains no benchmark examples; the model is built from
the class below and trained with :class:`gam2.training.trainer.GAM2Trainer`.
"""
from .models.dg_moe_bnn import DGMoEBNN
from .training.trainer import GAM2Trainer
from .training.trainer import GAM2Trainer as Trainer
from .active_learning.gasa import GASA

__all__ = ['DGMoEBNN', 'GAM2Trainer', 'Trainer', 'GASA']
