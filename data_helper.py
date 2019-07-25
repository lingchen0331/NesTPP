# -*- coding: utf-8 -*-
# Author: chen
# Created at: 6/12/19 10:09 AM

import pandas as pd
import numpy as np
import json
import datetime, time, warnings
import matplotlib.pyplot as plt

def load_json(filename):
    """Load json files in a given directory"""
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    return data


def calculate_duration(time_before, time_after):
    """Calculate the time duration between two time in the unit of hours"""
    duration = (time_after - time_before) / datetime.timedelta(hours=1)

    return duration


def main_stream_time_lists(main_dataframe):
    """ Getting time series with type of unix timestamp from the main dataframe """
    #time_lists = [pd.to_datetime(i, unit='s') for i in list(main_dataframe.created)]
    time_lists = [int(k) for k in list(main_dataframe.created)]
    
    return time_lists


def reply_stream_time_lists(reply_dataframe):
    """ Getting time series with type of unix timestamp from the reply dataframe """
    reply_stream_time_list = []
    for i, x in enumerate(reply_dataframe):
        try:
            time_lists = [int(k) for k in list(x.created_at)]
            #time_lists = [pd.to_datetime(i, unit='s') for i in list(x.created_at)]
        except:
            time_lists = []
        reply_stream_time_list.append(time_lists)
    return reply_stream_time_list


def normalization_time_list(time_list):
    """ Return the normalized time list for testing """
    return np.array([time_list[i] - time_list[0] for i in range(len(time_list))])


def time_plot(filename, ground_truth, ntpp, mhp, poisson):
    plt.plot(list(range(len(ground_truth))), ground_truth, 'r*-', linewidth=2.1, label='truth')
    plt.plot(list(range(len(ntpp))), ntpp, 'b:', linewidth=2.1, label='ntpp')
    plt.plot(list(range(len(mhp))), mhp, 'g:', linewidth=2.1, label='mhp')
    plt.plot(list(range(len(poisson))), poisson, 'c:', linewidth=2.1, label='poisson')
    plt.legend(fontsize=11)
    plt.xlabel('Simulated Events Number (N = 900)', fontsize=15)
    plt.ylabel('Time', fontsize=15)
    plt.grid(True)
    plt.xticks(list(range(len(ground_truth))))
    plt.tight_layout()
    plt.savefig('paper/figures/{}.pdf'.format(filename), dpi=600)
    plt.show()

    
class TookTooLong(Warning):
    pass


class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
        
        
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            print("Elapsed: %.3f sec" % elapsed)
