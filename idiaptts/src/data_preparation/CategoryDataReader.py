#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from idiaptts.src.data_preparation.NpzDataReader import DataReader
from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig


class CategoryDataReader(DataReader):

    class Config(DataReader.Config):
        def __init__(self, name, id_to_category_fn):
            super().__init__(name,
                             output_names=DataReaderConfig._str_to_list(name))
            self.id_to_category_fn = id_to_category_fn

        def create_reader(self):
            return CategoryDataReader(self)

    def __init__(self, config: Config):
        super().__init__(config)
        self.id_to_category_fn = config.id_to_category_fn

    def __getitem__(self, id_name: str):
        return {self.name: self.id_to_category_fn(id_name)}
