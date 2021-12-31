#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from idiaptts.src.neural_networks.pytorch.loss.AtomLoss import atom_loss
from idiaptts.src.neural_networks.pytorch.loss.DiscretizedMixturelogisticLoss import DiscretizedMixturelogisticLoss
from idiaptts.src.neural_networks.pytorch.loss.L1WeightedVUVMSELoss import L1WeightedVUVMSELoss
from idiaptts.src.neural_networks.pytorch.loss.OneHotCrossEntropyLoss import OneHotCrossEntropyLoss
from idiaptts.src.neural_networks.pytorch.loss.WeightedNonzeroMSELoss import WeightedNonzeroMSELoss
from idiaptts.src.neural_networks.pytorch.loss.WeightedNonzeroWMSEAtomLoss import WeightedNonzeroWMSEAtomLoss
from idiaptts.src.neural_networks.pytorch.loss.WMSELoss import WMSELoss
from idiaptts.src.neural_networks.pytorch.loss.VAEKLDLoss import VAEKLDLoss
from idiaptts.src.neural_networks.pytorch.loss.UnWeightedAccuracy import UnWeightedAccuracy
