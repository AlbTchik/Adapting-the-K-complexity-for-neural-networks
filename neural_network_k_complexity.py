# -*- coding: utf-8 -*-
"""neural_network_K_complexity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J4cvhoNmEhICtl90pyXtmkoKPyPfT28i
"""

import os
import sys
import numpy as np
from keras.optimizers import Adam

def complexity_evaluation(model, X_test, Y_test, closeness_to_expected_results_computation="loss", neural_network_size_computation="memory"):
  #computation of the closeness to expected results
  loss, accuracy = model.evaluate(X_test, Y_test,verbose=0)
  if closeness_to_expected_results_computation == "accuracy":
    closeness_to_expected_results = accuracy
  elif closeness_to_expected_results_computation == "loss":
    #as the loss is inversly proportional to the accuracy, the inverse of the loss is taken
    closeness_to_expected_results = 1/loss

  if neural_network_size_computation == "parameters":
    neural_network_size = get_nb_params(model)
  elif neural_network_size_computation == "neurons":
    neural_network_size = get_nb_neurons(model)
  elif neural_network_size_computation == "memory":
    neural_network_size = get_model_memory_size(model)

  return closeness_to_expected_results/neural_network_size

def get_nb_params(model):
  trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
  non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
  return trainable_params + non_trainable_params

def get_nb_neurons(model):
  nb_neurons = 0
  for layer in model.layers:
    nb_neurons += layer.units if hasattr(layer, 'units') else 0
  return nb_neurons

def get_model_memory_size(model):
  model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
  model.save("model.h5")
  file_stats = os.stat('model.h5')
  model_size = file_stats.st_size / (1024 * 1024)
  os.remove('model.h5')
  return model_size