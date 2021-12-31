#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# Constants need to go before imports that need them.
FIXED_ATTENTION = "FixedAttention"
ATTENTION_GROUND_TRUTH = "ground_truth_durations"

from .EncDecDyn import EncDecDyn
from .Config import Config
from .attention.FixedAttention import FixedAttention
