#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Tuple, Any

import torch


class Attention(torch.nn.Module):
    """
    Base class for all attention mechanisms.
    """

    def allows_batched_forward(self) -> bool:
        """
        Whether the attention block can be run once and not incremental.

        :raises NotImplementedError: Implement in subclass.
        """
        raise NotImplementedError()

    def get_go_frame(self) -> Any:
        """
        Returns the attention state for the very first frame. The state
        can be of any type, and can be used in the forward functions in
        stateful attention.

        :raises NotImplementedError: Implement in subclass.
        :return: If true, forward_batched will be used.
        :rtype: boolean
        """
        raise NotImplementedError()
        return attention_state

    def forward_batched(self,
                        encoder_input: torch.Tensor,
                        attention_state: Any):
        """
        A forward call for the entire sequence at once. Return the
        attention context for the entire sequence and the used
        attention matrix to compute it.

        :param encoder_input: Computed sequence of encoder.
        :type encoder_input: torch.Tensor
        :param attention_state: Last return attention state
        :type attention_state: Anything
        :raises NotImplementedError: Implement in subclass.
        """
        raise NotImplementedError()
        return attention_context, attention_matrix

    def forward_incremental(self,
                            current_frame_idx: int,
                            encoder_input: torch.Tensor,
                            decoder_input: torch.Tensor,
                            attention_state) \
            -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Incremental call of attention mechanism.

        :param current_frame_idx: Index of current frame in sequence.
        :type current_frame_idx: int
        :param encoder_input: Computed sequence of encoder.
        :type encoder_input: torch.Tensor
        :param decoder_input: Last decoder output.
        :type decoder_input: torch.Tensor
        :param attention_state: Last return attention state
        :type attention_state: Anything
        :raises NotImplementedError: [description]
        :return: attention context (results of weighted sum), weights
                 used to compute the attention context
        :rtype: Tuple
        """
        raise NotImplementedError()
        return attention_context, attention_weights, attention_state
