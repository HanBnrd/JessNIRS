# fNIRS LSL reader
# © Johann Benerradi

import logging
import nirsimple as ns
import numpy as np
import pandas as pd
import re
import warnings

from pylsl import StreamInlet, resolve_stream
from random import random
from time import sleep


GYRO = ['HEADING', 'ROLL', 'PITCH']


class fNIRSReader:
    def __init__(self):
        # Establish all parameters for fNIRS stream
        self.indices_keep = None
        self.indices_gyro = None
        self.ch_names = None
        self.ch_wls = None
        self.ch_distances = None
        self.sfreq = None

        # Set it logging
        logging.info('fNIRS reader ready')
        self.fnirs = False

    def start(self):
        # Instantiate the board reading
        print("Starting fNIRS LSL stream...")

        # Data stream
        data_stream = resolve_stream("name", "OxySoft")[0]
        self.data_inlet = StreamInlet(data_stream)
        info = self.data_inlet.info()
        name, self.sfreq, stream_type = info.name(), info.nominal_srate(), info.type()
        print(f"Recording 10 sec baseline for {stream_type}...")
        all_chs = []
        all_ch_wls = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            label = ch.child_value("label")
            if re.match(r"\[\d*\] .* \[\d*nm\]", label):
                all_chs.append(label.split("] ")[1].split(" [")[0])
                all_ch_wls.append(label.split(" [")[1].split("nm]")[0])
            else:
                all_chs.append(label)
                all_ch_wls.append(None)
            ch = ch.next_sibling()

        # Get channel data of interest
        keep = pd.read_csv(f"./modules/optodes_brite.csv")
        ch_rls = keep.columns
        ch_sds = keep.values[0]
        self.indices_keep = [all_chs.index(ch_keep) for ch_keep in ch_rls]
        self.ch_names = [ch.split(' ')[0] for ch in ch_sds]
        self.ch_wls = [all_ch_wls[i] for i in self.indices_keep]
        self.ch_dpfs = [6.0 for _ in ch_sds]
        self.ch_distances = keep.values[1].astype(float)
        self.indices_gyro = [all_chs.index(ch_gyro) for ch_gyro in GYRO
                             if ch_gyro in all_chs]

        self.fnirs = True

        # Read 10 sec of data to fill in buffers and refs
        od_buffer = np.empty((len(self.indices_keep), 0))
        for _ in range(int(10*self.sfreq)):
            # Get new sample
            sample, timestamp = self.data_inlet.pull_sample()
            sample_array = np.array(sample)
            od = sample_array[self.indices_keep]
            od = od[:, np.newaxis]

            # Append new sample
            od_buffer = np.append(od_buffer, od, axis=1)

        self.od_refs = np.mean(od_buffer, axis=1).tolist()

    def read(self, num_points):
        if self.fnirs:
            # Get new sample
            sample, timestamp = self.data_inlet.pull_sample()
            sample_array = np.array(sample)
            od = sample_array[self.indices_keep]
            od = od[:, np.newaxis]
            gyro = sample_array[self.indices_gyro]

            # Preprocessing
            warnings.simplefilter("ignore")
            dod = ns.od_to_od_changes(od, refs=self.od_refs)
            mbll_data = ns.mbll(dod, self.ch_names, self.ch_wls, self.ch_dpfs,
                                self.ch_distances, "cm")
            dc, ch_names, ch_types = mbll_data
            warnings.resetwarnings()
            self.data = dc.squeeze().tolist() + gyro.tolist()

        else:
            # Get dummy data instead of fNIRS stream
            data_len = len(self.indices_keep + self.indices_gyro)
            self.data = [random() for _ in range(data_len)]

        logging.debug(f"fNIRS data = {self.data}")
        return self.data

    def terminate(self):
        if self.fnirs:
            pass


if __name__ == "__main__":
    fr = fNIRSReader()
    fr.start()
    while True:
        data = fr.read(1)
        print(data)
        sleep(1)
