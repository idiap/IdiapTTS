#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import random
import re
import logging
from types import MethodType

from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig


class IntercrossDataReaderConfig(DataReaderConfig):

    def __init__(self, category_regexes, id_list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._id_list = id_list
        self._category_regexes = category_regexes

    def create_reader(self):
        reader = super().create_reader()
        self._category_regexes = [re.compile(regex) for regex in self._category_regexes]
        reader._ids_per_categories = self._create_ids_per_regex(self._id_list)

        reader.change_id_name = MethodType(self.change_id_name, reader)
        return reader

    def _create_ids_per_regex(self, id_list):
        ids_per_categories = {regex: set() for regex in self._category_regexes}
        for id_name in id_list:
            for regex in self._category_regexes:
                if regex.match(id_name):
                    ids_per_categories[regex].add(id_name)

        for regex in self._category_regexes:
            assert len(ids_per_categories[regex]) > 0, "No id_name found for regex {}.".format(regex)

        return ids_per_categories

    def get_named_item(self_config, self, id_name, output_names, get_item_fn, chunk_size, pad_fn):
        new_id_name = self.change_id_name(id_name)
        output = super(IntercrossDataReaderConfig, self_config).get_named_item(
            self, new_id_name, output_names, get_item_fn, chunk_size=chunk_size, pad_fn=pad_fn)
        output["_id_list"] = id_name
        return output

    @staticmethod
    def change_id_name(self, id_name):
        for regex, regex_ids in self._ids_per_categories.items():
            if regex.match(id_name):
                return random.sample(regex_ids, k=1)[0]
        logging.warning("{} does not match any regex, thus remains unchanged.")
        return id_name
