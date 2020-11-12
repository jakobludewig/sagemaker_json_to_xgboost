# Helper class that implements the conversion of the JSON payload to numpy arrays 
import pandas as pd
import numpy as np
import json
import pickle
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator

class JSONParser(BaseEstimator):
        def __init__(self, feature_definitions_file = "features.pkl", separator= "_"):
            self.separator = "_"
            self.dict_vectorizer = DictVectorizer(sparse = False,
                                                 separator = self.separator)
            with open(feature_definitions_file, 'rb') as f:
                feature_definitions = pickle.load(f)
            
            assert 'categorical_variables_values' in feature_definitions.keys()
            
            self.continuous_variables = feature_definitions['continuous_variables']
            self.categorical_variables = feature_definitions['categorical_variables']
            self.categorical_variables_values = feature_definitions['categorical_variables_values']
            self.target_columns = feature_definitions['target_columns']
            
            assert set(feature_definitions['categorical_variables_values']) == set(feature_definitions['categorical_variables'])
            
            # check for potential collisions in output columns
            potential_dummy_columns = [[k + '_' + vv for vv in v] for k,v in self.categorical_variables_values.items()]
            assert len(set([item for sublist in potential_dummy_columns for item in sublist])) == len([item for sublist in potential_dummy_columns for item in sublist])
          
        def fit():
            return self

        def transform(self,data_json):
            output_array = np.empty((0,len(self.target_columns)),float)
            for d in data_json:
                output_series = self.dict_vectorizer.fit_transform({kk:("nan" if (kk in self.categorical_variables and vv is None) else vv) for kk,vv in d.items() }) 
                output_series = pd.Series(output_series[0],index = self.dict_vectorizer.feature_names_)
                output_series = output_series.reindex(self.target_columns,fill_value = 0.0)
                
                # continuous variables that are implicitly missing (ie not in the payload) have to be set explicitly to None
                output_series[[k for k in self.continuous_variables if k not in list(d.keys())]] = None
                
                output_array = np.append(output_array,output_series.to_numpy().reshape([1,-1]),axis = 0)
            return output_array
        