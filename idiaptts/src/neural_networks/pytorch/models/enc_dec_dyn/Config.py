#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
from typing import Union, List
from types import MethodType

from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
from .SubModule import SubModule
from .DecoderModule import DecoderModule
from idiaptts.src.neural_networks.pytorch.models.NamedForwardCombiner import NamedForwardCombiner
from idiaptts.src.neural_networks.pytorch.models.NamedForwardSplitter import NamedForwardSplitter
from idiaptts.src.neural_networks.pytorch.models.NamedForwardWrapper import NamedForwardWrapper


no_op = lambda *a, **k: None


class Config(ModelConfig):

    class ModuleConfig(ModelConfig):
        def __init__(self,
                     input_names: List[str],
                     config=None,
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     name: str = None,
                     process_group: int = 0,
                     output_names: List[str] = None,
                     **kwargs):
            """
            Submodule configuration.

            :param input_names: Names of input features (The first one
                determines the sequence length for RNNDyn).
            :type input_names: List[str]
            :param config: Config to build internal model, defaults to
                None
            :type config: [type], optional
            :param input_merge_type: How to merge input features,
                defaults to ModelConfig.MERGE_TYPE_CAT
            :type input_merge_type: str, optional
            :param name: Name of the sub module (parameters will contain
                the name), defaults to None
            :type name: str, optional
            :param process_group: Determines the computational order of
                modules, defaults to 0
            :type process_group: int, optional
            :param output_names: Names for each of the internal model's
                outputs, defaults to None
            :type output_names: List[str], optional
            """
            super().__init__(input_names=input_names,
                             batch_first=True,
                             input_merge_type=input_merge_type,
                             name=name,
                             output_names=output_names,
                             **kwargs)
            self.config = config
            self.process_group = process_group

        def create_model(self):
            return SubModule(self)

    class ProjectionConfig(ModuleConfig):
        def __init__(self, out_dim: int, is_autoregressive_input: bool = False,
                     **kwargs):
            super().__init__(input_names=None, **kwargs)
            self.is_autoregressive_input = is_autoregressive_input
            self.out_dim = out_dim

        def create_model(self):
            projector = SubModule(self)
            projector.is_autoregressive_input = self.is_autoregressive_input
            projector.out_dim = self.out_dim
            return projector

    class DecoderConfig(ModuleConfig):
        def __init__(self,
                     attention_config=None,
                     attention_args: dict = {},  # LEGACY support, also remove in DecoderModule
                     teacher_forcing_input_names: List[str] = None,
                     n_frames_per_step: int = 1,
                     p_teacher_forcing: float = 1.0,
                     pre_net_config=None,
                     projection_configs=None,
                     **kwargs):
            super().__init__(**kwargs)
            self.attention_config = attention_config
            self.attention_args = attention_args
            assert teacher_forcing_input_names is None \
                or len(teacher_forcing_input_names) > 0,\
                "Empty list not allowed for teacher_forcing_input_names, use" \
                " None instead."
            self.teacher_forcing_input_names = teacher_forcing_input_names
            self.n_frames_per_step = n_frames_per_step
            self.p_teacher_forcing = p_teacher_forcing
            self.pre_net_config = pre_net_config
            self.projection_configs = projection_configs

        def create_model(self):
            return DecoderModule(self)

    # TODO: Should this inherit from NamedForwardCombiner.Config?
    class CombinerConfig(ModuleConfig):
        def __init__(self,
                     input_names: List[str],
                     output_names: str,
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     process_group: int = 0,
                     **kwargs):
            super().__init__(input_names=input_names,
                             input_merge_type=input_merge_type,
                             name="Combiner[{}->{}]".format(
                                 " ".join(input_names), output_names),
                             process_group=process_group,
                             output_names=[output_names],
                             **kwargs)

        def create_model(self):
            combiner = NamedForwardCombiner(self)
            combiner.init_hidden = MethodType(no_op, combiner)
            return combiner

    class SplitterConfig(NamedForwardSplitter.Config):
        def __init__(self,
                     input_names: List[str],
                     output_names: Union[str, List[str]],
                     split_sizes: Union[int, List[int]],
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     process_group: int = 0,
                     split_dim: int = -1,
                     **kwargs):
            super().__init__(input_names=input_names,
                             batch_first=True,
                             output_names=output_names,
                             split_sizes=split_sizes,
                             input_merge_type=input_merge_type,
                             split_dim=split_dim,
                             **kwargs)
            self.process_group = process_group

        def create_model(self):
            splitter = NamedForwardSplitter(self)
            splitter.init_hidden = MethodType(no_op, splitter)
            return splitter

    class WrapperConfig(NamedForwardWrapper.Config):
        def __init__(self,
                     wrapped_model_config,
                     input_names: List[str],
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     name: str = None,
                     output_names: List[str] = None,
                     process_group: int = 0):
            super().__init__(wrapped_model_config=wrapped_model_config,
                             input_names=input_names,
                             batch_first=True,
                             input_merge_type=input_merge_type,
                             name=name,
                             output_names=output_names)
            self.process_group = process_group

        def create_model(self):
            return NamedForwardWrapper(self)

    def __init__(self, modules: List[ModuleConfig]):
        self.process_groups = self._sort_by_process_group(modules)

    def _sort_by_process_group(self, modules: List[ModuleConfig]):
        max_process_group = max([m.process_group for m in modules])
        process_groups = [[] for _ in range(max_process_group + 1)]  # Process_group is 0 based.

        for module in modules:
            process_groups[module.process_group].append(module)

        return process_groups

    def create_model(self):
        from .EncDecDyn import EncDecDyn  # Import here to break cyclic dependency.
        return EncDecDyn(self)

    def __getattr__(self, item):
        if item != "process_groups":
            for group in self.process_groups:
                for module in group:
                    if module.name == item:
                        return module
        raise AttributeError("%r object has no attribute %r" % (
            self.__class__.__name__, item))

    # TODO: Should override __dir__ as well?
