from __future__ import annotations

import re
from typing import List

import nrrd
import numpy as np


class NRRDReader:
    def __init__(self, filepath: str):
        self.data, self.header = nrrd.read(filepath)
        if len(self.data.shape) == 3:
            self.data = self.data.reshape((1, self.header["sizes"][0], self.header["sizes"][1], 1))
        self.shape = (self.data.shape[1:3])

    # YXC
    def extract_segments(self, segment_names: List[str]):
        arr = np.zeros((*self.shape, len(segment_names)), dtype=np.int8)
        seg_info = self.get_segments()
        for i, segment in enumerate(segment_names):
            # if it is a tuple, handle with same i, iterate and assign same i.
            if type(segment) is tuple:
                for seg in segment:
                    extracted = self.data[seg_info[seg]["layer"], :, :, 0] == seg_info[seg]["value"]
                    arr[extracted.T, i] = 1
            else:
                extracted = self.data[seg_info[segment]["layer"], :, :, 0] == seg_info[segment]["value"]
                arr[extracted.T, i] = 1

        return arr

    def get_segments(self):
        segs = {}
        for key in self.header.keys():
            res = re.search("^Segment([0-9]+)_Name$", key)
            if not res is None:
                idx = res[1]  # first capture group, [0] is full str.
                segs[self.header[f"Segment{idx}_Name"]] = {
                    "index": int(idx),
                    "layer": int(self.header[f"Segment{idx}_Layer"]),
                    "value": int(self.header[f"Segment{idx}_LabelValue"]),
                    "name": self.header[f"Segment{idx}_Name"],
                }
        return segs
