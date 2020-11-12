# Inspired by: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/sklearn_abalone_featurizer.py
from __future__ import print_function

import time
import sys
from io import StringIO,BytesIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy
import joblib

from numpy import genfromtxt
from sklearn.pipeline import Pipeline

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()
    
    # read in xgboost model file, needs to be provided in script directory when creating the SKLearn model
    xgboost_classifier = xgb.Booster({'nthread':4})
    xgboost_classifier.load_model('xgboost_model.bin')

    xgboost_classifier.save_model(os.path.join(args.model_dir, "xgboost_model.bin"))
    print("saved model!")
    
    
def input_fn(input_data, content_type):
    if (content_type == 'text/csv; charset=utf-8') | (content_type == 'text/csv'):
        return xgb.DMatrix(scipy.sparse.csr_matrix(pd.read_csv(StringIO(input_data.decode('utf-8')),header = None)))
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    if (accept == 'text/csv; charset=utf-8') | (accept == 'text/csv'):
        return worker.Response(response=encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeError("{} accept type is not supported by xgboost_wrapper.".format(accept))


def model_fn(model_dir):
    xgboost_classifier = xgb.Booster({'nthread':4})
    xgboost_classifier.load_model(os.path.join(model_dir, "xgboost_model.bin"))
    return xgboost_classifier
