#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Union, Any, List, Optional, Tuple, cast, Dict
import copy
import logging
from functools import partial, reduce

import torch
from torch.nn.utils.rnn import pad_sequence

from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule
from idiaptts.src.neural_networks.pytorch.models.NamedForwardWrapper import NamedForwardWrapper
from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
from idiaptts.src.neural_networks.pytorch.models.RNNDyn import RNNDyn
from idiaptts.src.neural_networks.pytorch.utils import tensor_pad


class WindowingWrapper(torch.nn.Module):

    class Config:
        def __init__(self,
                     batch_first: bool,
                     output_merge_type: str,
                     step_size: int,
                     window_size: int,
                     wrapped_model_config: ModelConfig,
                     zero_padding: bool = True):

            self.batch_first = batch_first
            self.output_merge_type = output_merge_type
            self.step_size = step_size
            self.window_size = window_size
            self.wrapped_model_config = wrapped_model_config
            self.zero_padding = zero_padding

        def create_model(self):
            return WindowingWrapper(self)

    def __init__(self, config: Config):
        super().__init__()

        self.batch_first = config.batch_first
        self.output_merge_type = config.output_merge_type
        self.step = config.step_size
        self.window_size = config.window_size
        self.zero_padding = config.zero_padding

        # self.config = copy.deepcopy(config)

        if config.wrapped_model_config is not None:  # Allows easier testing.
            self.model = config.wrapped_model_config.create_model()

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            # TODO: Test speed impact.
            if item != "model":
                return getattr(self.model, item)
            else:
                raise e

    def forward(
            self,
            input_: Union[Tuple[torch.Tensor, ...], List[torch.Tensor],
                          torch.Tensor],
            lengths: torch.LongTensor,
            max_length: torch.LongTensor,
            target: Union[Tuple[torch.Tensor, ...], List[torch.Tensor],
                          torch.Tensor] = None,
            return_kwargs: bool = True,
            **kwargs):

        org_lengths = lengths
        batch_size = len(lengths)

        num_valid_chunks = self._length_to_num_chunks(lengths)
        lengths_win = self._window_lengths(lengths)
        max_length_win = torch.tensor(self.window_size, dtype=lengths.dtype,
                                      device=lengths.device)

        if type(input_) in [tuple, list]:
            input_win = [self._window(i) for i in input_]
        else:
            input_win = self._window(input_)

        if target is not None:
            if type(target) in [tuple, list]:
                target_win = [self._window(t) for t in target]
            else:
                target_win = self._window(target)
            kwargs["target"] = target_win

        output, kwargs = self.model(input_win, lengths_win, max_length_win,
                                    **kwargs)
        lengths_win = kwargs['lengths']
        max_length_win = kwargs['max_length']

        if self.output_merge_type == ModelConfig.MERGE_TYPE_CAT:
            assert lengths.max() == max_length, "This module does not support "\
                "parallel training with MERGE_TYPE_CAT."
            # NOTE: The problem is that max_length_win will be longer than
            #       lengths_win on GPUs with the shorter sequences. In that case
            #       the last element of max_length_win is never passed to the
            #       model, thus we do not know how it is changed in length.

        lengths = self._merge_lengths(lengths_win, batch_size)

        # At this point the merge operation either reduces all chunks to a fixed
        # number of frames (e.g. mean, mul, etc.) or lengths.max() and
        # max_length are equal, so we can use it as the max_length.
        max_length = lengths.max()

        output = self._merge_outputs(output, num_valid_chunks)

        kwargs['lengths'] = lengths
        kwargs['max_length'] = max_length
        if return_kwargs:
            return output, kwargs
        else:
            return output

    def _length_to_num_chunks(self, length):
        num_chunks = torch.clamp(length - self.window_size, min=0) // self.step + 1
        if self.zero_padding:
            length = length.clone()
            overlap = torch.clamp(length - self.window_size, min=0) % self.step
            padding_mask = overlap > 0
            num_chunks[padding_mask] += 1
            # length[multi_chunk_mask] += padding[multi_chunk_mask]
        return num_chunks

    def _window_lengths(self, lengths: torch.LongTensor) -> torch.LongTensor:
        lengths = lengths.clone()
        if lengths.ndim == 0:
            lengths.unsqueeze(0)

        lengths_win = []
        while True:
            sub_lengths = torch.clamp(lengths, max=self.window_size)
            lengths = torch.clamp(lengths - self.step, min=0)
            if sub_lengths.max() <= self.window_size - self.step:
                # Missing frames were only in overlap so part of last chunk.
                break

            lengths_win.append(sub_lengths)

            if not self.zero_padding and lengths.max() < self.window_size:
                break
        if len(lengths_win) == 0:
            # Edge case where all lengths are in zeroes overlap.
            return sub_lengths
        else:
            return torch.stack(lengths_win, dim=1).view(-1)

    def _window(self, tensor: torch.Tensor) -> torch.Tensor:

        time_dim = 1 if self.batch_first else 0
        if self.zero_padding:
            pad_length = self.step - (tensor.shape[time_dim] % self.step)
            if pad_length > 0 and pad_length < self.step:
                if self.batch_first:
                    padding = [0, 0, 0, pad_length] + [0, 0] * (tensor.ndim - 2)
                else:
                    padding = [0, pad_length] + [0, 0] * (tensor.ndim - 1)
                padding = tuple(padding)
                tensor = torch.nn.functional.pad(tensor, padding)

        if tensor.shape[time_dim] <= self.window_size:
            return tensor
        # batch_first: B x T x C -> B x num_chunks x C x window_size
        # otherwise: T x B x C -> num_chunks x B x C x window_size
        tensor = tensor.unfold(dimension=time_dim, size=self.window_size,
                               step=self.step)

        if self.batch_first:
            # B x num_chunks x C x window_size -> B x num_chunks x window_size x C
            tensor = tensor.transpose(2, 3)
            # B x num_chunks x window_size x C -> B*num_chunks x window_size x
            return tensor.reshape(-1, *tensor.shape[2:])
        else:
            # Rotate window dimension to the front:
            # num_chunks x B x C x window_size -> window_size x num_chunks x B x C
            tensor = tensor.permute(tensor.ndim - 1, *range(tensor.ndim - 1))
            # window_size x num_chunks x B x C -> window_size x num_chunks*B x C
            return tensor.reshape(self.window_size, -1, *tensor.shape[3:])

        # Only select the rows which contain data and not only zero padding.
        all_non_zero_rows = []
        batch_size = len(num_chunks)
        for batch_idx in range(batch_size):
            if self.batch_first:
                non_zero_rows = tensor[batch_idx, :num_chunks[batch_idx]]
            else:
                non_zero_rows = tensor[:num_chunks[batch_idx], batch_idx]
            non_zero_rows = non_zero_rows.transpose(1, 2)  # n x window_size x C
            all_non_zero_rows.append(non_zero_rows)

        tensor = torch.cat(all_non_zero_rows, axis=0)  # N_nonzero x window_size x C

        return tensor

    def _merge_lengths(self, lengths: torch.Tensor, batch_size: int) \
            -> torch.Tensor:

        lengths = lengths.view(batch_size, -1)

        if self.output_merge_type == ModelConfig.MERGE_TYPE_CAT:
            lengths = lengths.sum(-1)
        elif self.output_merge_type == ModelConfig.MERGE_TYPE_ATTENTION:
            raise NotImplementedError("Not sure what this should do.")
        else:
            # For all other merges use the first one, as they all have
            # to be the same length.
            lengths = lengths[:, 0]

        return lengths

    def _merge_outputs(self, outputs: Union[Tuple[torch.Tensor, ...],
                                            List[torch.Tensor], torch.Tensor],
                       num_chunks: torch.LongTensor) \
            -> Union[torch.Tensor, List[torch.Tensor]]:
        if type(outputs) in [tuple, list]:
            outputs = [self._merge_output(o, num_chunks) for o in outputs]
        else:
            outputs = self._merge_output(outputs, num_chunks)

        return outputs

    def _merge_output(self, output: torch.Tensor, num_chunks: torch.LongTensor)\
            -> torch.Tensor:

        if self.output_merge_type == ModelConfig.MERGE_TYPE_CAT:
            batch_size = len(num_chunks)
            if self.batch_first:
                # B*num_chunks x W' x C' -> B x num_chunks*W' x C'
                return output.view(batch_size, -1, *output.shape[2:])
            else:
                # W' x num_chunks*B x C' -> W'*num_chunks x B x C'
                return output.transpose(0, 1).view(-1, batch_size, *output.shape[2:])

        if self.batch_first:
            return self._merge_output_batch_first(output, num_chunks)
        else:
            return self._merge_output_batch_second(output, num_chunks)

    def _merge_output_batch_first(self, output: torch.Tensor,
                                  num_chunks: torch.LongTensor) -> torch.Tensor:
        batch_size = len(num_chunks)
        # B*num_chunks x W' x C' -> B x num_chunks x W' x C'
        output_view = output.view(batch_size, -1, *output.shape[1:])

        out_tensor = torch.zeros((batch_size, *output_view.shape[2:]),
                                 dtype=output.dtype, device=output.device,
                                 requires_grad=output.requires_grad)

        for batch_idx, valid_chunks in enumerate(num_chunks):
            valid_output = output_view[batch_idx, :valid_chunks]

            if self.output_merge_type == ModelConfig.MERGE_TYPE_ADD:
                out_tensor[batch_idx] = valid_output.sum(0)
            elif self.output_merge_type == ModelConfig.MERGE_TYPE_MEAN:
                out_tensor[batch_idx] = valid_output.mean(0)
            elif self.output_merge_type == ModelConfig.MERGE_TYPE_MUL:
                # num_chunks x W' x C' -> num_chunks times 1 x W' x C'
                valid_output = valid_output.split(1, dim=0)
                out_tensor[batch_idx] = reduce(lambda x, y: x * y,
                                               valid_output).squeeze(0)
            else:
                raise NotImplementedError()

        return out_tensor

    def _merge_output_batch_second(self, output: torch.Tensor,
                                   num_chunks: torch.LongTensor) -> torch.Tensor:

        batch_size = len(num_chunks)
        # W' x num_chunks*B x C' -> W' x num_chunks x B x C'
        output_view = output.view(output.shape[0], -1, batch_size,
                                  *output.shape[2:])

        output_length = output_view.shape[0]
        out_tensor = torch.zeros(
            (output_length, batch_size, *output_view.shape[3:]),
            dtype=output.dtype, device=output.device,
            requires_grad=output.requires_grad)

        for batch_idx, valid_chunks in enumerate(num_chunks):
            valid_output = output_view[:, :valid_chunks, batch_idx]

            if self.output_merge_type == ModelConfig.MERGE_TYPE_ADD:
                out_tensor[:, batch_idx] = valid_output.sum(1)
            elif self.output_merge_type == ModelConfig.MERGE_TYPE_MEAN:
                out_tensor[:, batch_idx] = valid_output.mean(1)
            elif self.output_merge_type == ModelConfig.MERGE_TYPE_MUL:
                # W' x num_chunks x C' -> num_chunks times W' x 1 x C'
                valid_output = valid_output.split(1, dim=1)
                out_tensor[:, batch_idx] = reduce(lambda x, y: x * y,
                                                  valid_output).squeeze(1)
            else:
                raise NotImplementedError()

        return out_tensor


def main():
    tests = [
        torch.LongTensor([9, 14]),
        torch.LongTensor([4, 4]),
        torch.LongTensor([8, 90]),
        torch.LongTensor([8, 94]),
        torch.LongTensor([8, 95]),
        torch.LongTensor([8, 96]),
        torch.LongTensor([4, 51]),
        torch.LongTensor([5, 51]),
        torch.LongTensor([6, 51]),
        torch.LongTensor([5, 50]),
        torch.LongTensor([11, 50]),
        torch.LongTensor([8, 94, 94]),
        torch.LongTensor([8, 94, 95]),
        torch.LongTensor([8, 94, 96]),
        torch.LongTensor([8, 95, 95]),
        torch.LongTensor([8, 95, 96]),
        torch.LongTensor([8, 96, 96])
    ]

    chunks = [  # Chunks for window_size=10, step=5
        torch.LongTensor([1, 2]),
        torch.LongTensor([1, 1]),
        torch.LongTensor([1, 17]),
        torch.LongTensor([1, 18]),
        torch.LongTensor([1, 18]),
        torch.LongTensor([1, 19]),
        torch.LongTensor([1, 10]),
        torch.LongTensor([1, 10]),
        torch.LongTensor([1, 10]),
        torch.LongTensor([1, 9]),
        torch.LongTensor([2, 9]),
        torch.LongTensor([1, 18, 18]),
        torch.LongTensor([1, 18, 18]),
        torch.LongTensor([1, 18, 19]),
        torch.LongTensor([1, 18, 18]),
        torch.LongTensor([1, 18, 19]),
        torch.LongTensor([1, 19, 19])
    ]

    def dummy(input_win, lengths, max_lengths, **kwargs):
        kwargs['lengths'] = lengths
        kwargs['max_length'] = max_lengths
        return input_win, kwargs

    wrapper = WindowingWrapper.Config(
        batch_first=True,
        output_merge_type=ModelConfig.MERGE_TYPE_MEAN,
        step_size=5,
        zero_padding=True,
        window_size=10,
        wrapped_model_config=None).create_model()
    wrapper.model = dummy
    for lengths, target_chunk in zip(tests, chunks):
        num_chunks = wrapper._length_to_num_chunks(lengths)
        assert (num_chunks == target_chunk).all(), \
            "Wrong number of chunks for {}.\nExpected {}\nfound {}".format(
            lengths, target_chunk, num_chunks)

    # (step, window_size)
    wrapper_configs = [
        (5, 10),
        (45, 50),
        (10, 10)
    ]

    for step, window_size in wrapper_configs:
        wrapper = WindowingWrapper.Config(
            batch_first=True,
            output_merge_type=ModelConfig.MERGE_TYPE_MEAN,
            step_size=step,
            zero_padding=True,
            window_size=window_size,
            wrapped_model_config=None).create_model()
        wrapper.model = dummy

        for lengths in tests:
            batch_size = len(lengths)
            max_length = max(lengths)
            feature_dim = 4

            # Prepare the input.
            input_ = torch.arange(batch_size * max_length).view(
                batch_size, max_length, 1).repeat(1, 1, feature_dim).float()
            for batch_idx, seq_length in enumerate(lengths):
                input_[batch_idx, seq_length:] = 0.0

            # Check the windowing.
            input_win = wrapper._window(input_)

            # Check the merging.
            output, kwargs = wrapper(input_, lengths, lengths.max())
            merged_length = kwargs["lengths"]

            remaining_frames = max_length % step
            if remaining_frames != 0:
                pad_length = max_length + step - remaining_frames
            pad_length = max(window_size, pad_length)
            input_padded = tensor_pad(input_, target_length=pad_length, dim=1)
            target = input_padded.unfold(dimension=1, size=window_size,
                                         step=step).transpose(2, 3)
            target_mean = torch.empty((batch_size, *target.shape[2:]))
            num_chunks = wrapper._length_to_num_chunks(lengths)
            for batch_idx, valid_chunks in enumerate(num_chunks):
                target_mean[batch_idx] = target[batch_idx, :valid_chunks].mean(dim=0)

            assert (output == target_mean).all(), "Wrong output for lengths {}.\n" \
                "Expected {}\n found {}".format(lengths, target_mean, output)

            continue
            num_chunks = wrapper._length_to_num_chunks(length)
            valid_chunks_mask = wrapper._get_valid_chunks_mask(num_chunks)
            length_win = wrapper._window_lengths(length)
            assert len(valid_chunks_mask) == len(length_win), "Size missmatch"

            input_ = pad_sequence([torch.arange(start=0, end=l) for l in length],
                                  batch_first=True).float().unsqueeze(-1)
            input_win = wrapper._window(input_, num_chunks)

            if input_.shape[1] > window_size:
                for batch_idx in range(num_chunks[0]):
                    expected_output = torch.zeros((window_size, 1))
                    expected_start = step * batch_idx
                    expected_length = min(length[0] - expected_start, window_size)
                    expected_output[:expected_length, 0] = torch.arange(
                        start=expected_start, end=expected_start + expected_length)
                    assert (input_win[batch_idx] == expected_output).all(), \
                        "Windowing error"

            output, kwargs = wrapper(input_, length, length.max())
            merged_length = kwargs["lengths"]

            if step == 0:
                expected_output = torch.zeros((len(length), window_size, 1))
                for batch_idx, chunk_len in enumerate(num_chunks):
                    non_padded_input = input_[batch_idx,
                                              :chunk_len * window_size]
                    if chunk_len > 1:
                        non_padded_input = non_padded_input.split(window_size,
                                                                  dim=0)
                        last_input_len = len(non_padded_input[-1])
                        if last_input_len < window_size:
                            padded_last = torch.zeros((window_size, 1))
                            padded_last[:last_input_len] = non_padded_input[-1]
                            non_padded_input = (*non_padded_input[:-1],
                                                padded_last)
                        non_padded_input_win = torch.stack(non_padded_input)
                        expected_output[batch_idx] = non_padded_input_win.mean(
                            dim=0)
                    else:
                        expected_output[batch_idx] = non_padded_input
                assert (expected_output == output).all(), \
                    "Input did not remain unchanged."
                expected_length = torch.clamp(length, max=window_size)
                assert (expected_length == merged_length).all(), \
                    "Lengths did not remain unchanged"
        continue


if __name__ == "__main__":
    main()
