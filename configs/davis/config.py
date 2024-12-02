#!/usr/bin/env python
""" Davis Configuration file."""
import yaml
import pdb
import os
import os.path as osp
import sys
from enum import Enum
from easydict import EasyDict as edict

class phase(Enum):
    TRAIN    = 'train'
    VAL      = 'val'
    TESTDEV  = 'test-dev'
    TRAINVAL = 'train-val'

__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 16

# Paths to dataset folders
__C.PATH = edict()

# Dataset resolution: ("480p","1080p")
__C.RESOLUTION="480p"

# Dataset year: ("2016","2017")
__C.YEAR ="2016"

__C.PHASE = phase.TRAIN

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = False

# Root folder of project
__C.PATH.ROOT = osp.abspath('../')

# Data folder
__C.PATH.DATA = osp.abspath('/media/zhangying/Datas/gitCode/VIDNet/src/databases/DAVIS_2016_vi/')

# Path to input images
__C.PATH.ORIGINAL = osp.join(__C.PATH.DATA, "Original", __C.RESOLUTION)

__C.PATH.SEQUENCES3 = None
__C.PATH.SEQUENCES2 = '/media/zhangying/Datas/gitCode/VIDNet/src/databases/DAVIS_2016_cp/JPEGImages_224/480p'
__C.PATH.SEQUENCES = '/media/zhangying/Datas/gitCode/VIDNet/src/databases/DAVIS_2016_vi/JPEGImages/480p'

__C.PATH.FLOW = '../../flownet2-pytorch/result/inference/run.epoch-0-flow-field'

# Path to annotations
__C.PATH.ANNOTATIONS = '/media/zhangying/Datas/gitCode/VIDNet/src/databases/DAVIS_2016_vi/Annotations/480p'

# Color palette
__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'configs/davis/palette.txt'))

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO = osp.abspath(osp.join(__C.PATH.ROOT, "configs/davis/db_info.yaml"))

# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J','F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS= ['mean','recall','decay']

def db_read_info():
	""" Read dataset properties from file."""

	with open(cfg.FILES.DB_INFO,'r') as f:
		return edict(yaml.safe_load(f))

def db_read_attributes():
	""" Read list of sequences. """
	return db_read_info().attributes

def db_read_years():
	""" Read list of sequences. """
	return db_read_info().years

def db_read_sequences(year=None,db_phase=None):
  """ Read list of sequences. """
  sequences = db_read_info().sequences
  if year is not None:
    sequences = filter(
        lambda s:int(s.year) <= int(year),sequences)

  if db_phase is not None:
    if db_phase == phase.TRAINVAL:
      sequences = filter(
          lambda s: ((s.set == phase.VAL) or (s.set == phase.TRAIN)), sequences)
    else:
      sequences = filter(
          lambda s:s.set == db_phase and osp.isdir(osp.join(__C.PATH.SEQUENCES,s.name)),sequences)
  return sequences

# Load all sequences
__C.SEQUENCES = dict([(sequence.name,sequence) for sequence in
  db_read_sequences()])

import numpy as np
__C.palette = np.loadtxt(__C.PATH.PALETTE,dtype=np.uint8).reshape(-1,3)
