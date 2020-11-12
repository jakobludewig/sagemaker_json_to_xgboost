# Inspired by: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/sklearn_abalone_featurizer.py
from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
import joblib

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

from json_parser import JSONParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()

    preprocessor = JSONParser()
    
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, content_type):

    if content_type == 'application/json':
        return json.loads(input_data)
    else:
        raise ValueError("{} not supported by json_parser!".format(content_type))
        

def output_fn(prediction, accept):
    if (accept == 'text/csv; charset=utf-8') | (accept == 'text/csv'):
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeError("{} accept type is not supported by json_parser.".format(accept))


def predict_fn(input_data, model):
    features = model.transform(input_data)
    return features
    

def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor