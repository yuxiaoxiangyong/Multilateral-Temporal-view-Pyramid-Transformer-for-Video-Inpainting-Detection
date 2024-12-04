#!/usr/bin/env python
import yaml

""" Youtubevos Configuration file."""

import os
import os.path as osp

import sys
from easydict import EasyDict as edict

from enum import Enum
import pdb
class phase(Enum):
    TRAIN    = 'train'
    VAL      = 'val'
    TESTDEV  = 'test-dev'
    TRAINVAL = 'train-val'
    TEST = 'test'

__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 16

# Paths to dataset folders
__C.PATH = edict()

# Dataset resolution: ("480p","1080p")
__C.RESOLUTION="480p"

# Dataset year: ("2018","2019")
__C.YEAR ="2018"

__C.PHASE = phase.TRAIN

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = False

# Root folder of project
__C.PATH.ROOT = osp.abspath('../')

# Data folder
__C.PATH.DATA = osp.abspath('/media/zhangying/Database/YoutubeVOS-2018/train')

# Path to input images
__C.PATH.ORIGINAL = osp.join(__C.PATH.DATA,"inputs")

__C.PATH.SEQUENCES3 = '/media/zhangying/Database/Youtubevos_2018_inpaintinged/IS/train'
__C.PATH.SEQUENCES2 = '/media/zhangying/Database/Youtubevos_2018_inpaintinged/EG2/train'
__C.PATH.SEQUENCES = '/media/zhangying/Database/Youtubevos_2018_inpaintinged/FF/train'

__C.PATH.FLOW = '../../flownet2-pytorch/result/inference/run.epoch-0-flow-field'

# Path to annotations
__C.PATH.ANNOTATIONS = osp.join(__C.PATH.DATA,"Annotations/")

# Color palette
__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'configs/davis/palette.txt'))

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO = osp.abspath(osp.join(__C.PATH.ROOT,"configs/youtube/youtubevos_2018.yaml"))

# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J','F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS= ['mean','recall','decay']

def db_read_info():
	""" Read dataset properties from file."""
	with open(cfg.FILES.DB_INFO,'r') as f:
		return (yaml.safe_load(f))

def db_read_attributes():
	""" Read list of sequences. """
	return db_read_info().attributes

def db_read_years():
	""" Read list of sequences. """
	return db_read_info().years

def db_read_sequences(year=None, db_phase=None):
  """ Read list of sequences. """
  sequences = db_read_info()#.sequences
  if year is not None:
    sequences = filter(
        lambda s:int(s["year"]) <= int(year), sequences)

  if db_phase is not None:
    if db_phase == phase.TRAINVAL:
      sequences = filter(
          lambda s: ((s["set"] == phase.VAL.value) or (s["set"] == phase.TRAIN.value)), sequences)
    elif db_phase == phase.TEST.value:
        sequences = filter(
            lambda s : ((s["set"] == phase.VAL.value)), sequences)
    else:
      sequences = filter(
          lambda s:s["set"] == db_phase and osp.isdir(osp.join(__C.PATH.SEQUENCES, s["video_name"])), sequences)

  return sequences

# Load all sequences
__C.SEQUENCES = dict([(sequence["video_name"], sequence) for sequence in db_read_sequences()])

import numpy as np
__C.palette = np.loadtxt(__C.PATH.PALETTE, dtype=np.uint8).reshape(-1,3)
