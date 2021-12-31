#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import copy
import logging

import jsonpickle

from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule


class EncDecDyn(NamedForwardModule):

    logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(NamedForwardModule, self).__init__()
        self.config = copy.deepcopy(config)

        self.process_groups = []
        for idx, group in enumerate(config.process_groups):
            submodules = []
            for config in group:
                submodule = config.create_model()
                if submodule.name is None:
                    msg = "Every module needs a name, but name of {} is None."\
                        .format(submodule)
                    self.logger.warn(msg, DeprecationWarning)
                # TODO: Should there be an automatic name?
                # assert submodule.name is not None, msg

                module_id = "{}".format(submodule.name)
                if hasattr(self, module_id):
                    raise ValueError("{} module defined twice.".format(
                        module_id))
                self.logger.info("Adding {} to {}".format(
                    module_id, "EncDecDyn"))

                self.add_module(module_id, submodule)  # TODO: Handle duplicates.
                submodules.append(submodule)

            self.process_groups.append(submodules)

    def init_hidden(self, batch_size=1):
        for process_group in self.process_groups:
            for module in process_group:
                module.init_hidden(batch_size)

    def forward(self, data_dict, lengths, max_lengths):
        for process_group in self.process_groups:
            for module in process_group:
                module(data_dict, lengths, max_lengths)
        return data_dict

    def inference(self, data_dict, lengths, max_lengths):
        for process_group in self.process_groups:
            for module in process_group:
                module.inference(data_dict, lengths, max_lengths)
        # data_dict["pred_acoustic_features"] = data_dict["pred_intermediate_acoustic_features"]
        return data_dict

    def get_config_as_json(self):
        return jsonpickle.encode(self.config, indent=4)
