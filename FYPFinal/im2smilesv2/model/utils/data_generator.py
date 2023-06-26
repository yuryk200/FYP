import time
import os
import numpy as np
import cv2
import re


from .text import load_formulas
from .general import init_dir


class DataGeneratorFile(object):

    def __init__(self, filename):

        self._filename = filename

    def __iter__(self):
        with open(self._filename) as f:
            for line in f:
                line = line.strip().split(' ')
                path_img, id_formula = line[0], line[1]
                yield path_img, id_formula


class DataGenerator(object):


    def __init__(self, path_formulas, dir_images, path_index, bucket=False,
                 form_prepro=lambda s: s.strip().split(""), iter_mode="data",
                 img_prepro=lambda x: x, max_iter=None, max_len=None,
                 bucket_size=20, bg_rem=False, ED=False):

        self._path_formulas  = path_formulas
        self._dir_images     = dir_images
        self._path_index     = path_index
        self._img_prepro     = img_prepro
        self._form_prepro    = form_prepro
        self._max_iter       = max_iter
        self._max_len        = max_len
        self._iter_mode      = iter_mode
        self._bucket         = bucket
        self._bucket_size    = bucket_size
        
        self._length         = None
        self._formulas       = self._load_formulas(path_formulas)

        self._set_data_generator()

    def _set_data_generator(self):

        self._data_generator = DataGeneratorFile(self._path_index)

        if self._bucket:
            self._data_generator = self.bucket(self._bucket_size)

    def bucket(self, bucket_size):

        print("Bucketing the dataset...")
        bucketed_dataset = []
        old_mode = self._iter_mode  # store the old iteration mode
        self._iter_mode = "full"

        # iterate over the dataset in "full" mode and create buckets
        data_buckets = dict()  # buffer for buckets
        for idx, (img, formula, img_path, formula_id) in enumerate(self):
            s = img.shape
            if s not in data_buckets:
                data_buckets[s] = []
            # if bucket is full, write it and empty it
            if len(data_buckets[s]) == bucket_size:
                for (img_path, formula_id) in data_buckets[s]:
                    bucketed_dataset += [(img_path, formula_id)]
                data_buckets[s] = []

            data_buckets[s] += [(img_path, formula_id)]

        # write the rest of the buffer
        for k, v in data_buckets.items():
            for (img_path, formula_id) in v:
                bucketed_dataset += [(img_path, formula_id)]

        self._iter_mode = old_mode
        self._length = idx + 1

        print("- done.")
        return bucketed_dataset

    def _load_formulas(self, filename):

        formulas = load_formulas(filename)
        return formulas

    def _get_raw_formula(self, formula_id):
        try:
            formula_raw = self._formulas[int(formula_id)]
        except KeyError:
            print("Tried to access id {} but only {} formulas".format(
                formula_id, len(self._formulas)))
            print("Possible fix: mismatch between index file and formulas")
            raise KeyError

        return formula_raw

    def _process_instance(self, example):

        img_path, formula_id = example
        img = cv2.imread(self._dir_images + "/" + img_path)

        img = self._img_prepro(img)


        formula = self._form_prepro(self._get_raw_formula(formula_id))

        if self._iter_mode == "data":
            inst = (img, formula)
        elif self._iter_mode == "full":
            inst = (img, formula, img_path, formula_id)

        # filter on the formula length
        if self._max_len is not None and len(formula) > self._max_len:
            skip = True
        else:
            skip = False

        return inst, skip

    def __iter__(self):

        n_iter = 0
        for example in self._data_generator:
            if self._max_iter is not None and n_iter >= self._max_iter:
                break
            result, skip = self._process_instance(example)
            if skip:
                continue
            n_iter += 1
            yield result

    def __len__(self):
        if self._length is None:
            print("First call to len(dataset) - may take a while.")
            counter = 0
            for _ in self:
                counter += 1
            self._length = counter
            print("- done.")

        return self._length
