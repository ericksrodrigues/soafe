import numpy as np
import math
import timeit

def gaussian_convolution(vector,sigma=0.05, min_v=0, max_v=1):
  new_vector = []
  for v in vector:
    pertubation = False
    while (not pertubation) or ((v + pertubation) < min_v or (v+pertubation) > max_v):
      pertubation =  np.random.normal(0, sigma)
      print(pertubation)
    v = v + pertubation
    new_vector.append(v)
  return new_vector

print(gaussian_convolution([0.5, 0.5, 0.5,0.5,0.5]))

def mapper(vector, min_v, max_v, integer_vector):
  new_vector = []
  for i,v in enumerate(vector):
    new_vector.append(min_v[i] + vector[i]*(max_v[i] - min_v[i]))
    if integer_vector[i]:
      new_vector[i] = round(new_vector[i])
  return new_vector


def evolutionary_strategy(fit):
  number_of_generations = 30
  sigma=0.5
  # vetor de parametros = [first_shape, second_shape, learning_rate, iteractions, adaline_output, adaline_iteractions,subset_size]
  min_v = [5, 5, 0.01, 100, 5,5,0.1]
  max_v = [15, 15, 0.1, 2000, 15,20,0.3]
  integer_vector = [True, True, False, True, True,True, False]
  vector = [0, 0, 0, 0, 0,0,0]
  real_vector = mapper(vector, min_v,max_v, integer_vector)
  print("real_vector", real_vector)
  best = fit(real_vector)
  array_best = [best]
  array_timer = []
  for i in range(number_of_generations):
    #convolução gaussiana
    start = timeit.default_timer()
    child_vector = gaussian_convolution(vector,sigma)
    real_vector = mapper(child_vector, min_v,max_v, integer_vector)
    print("real_vector", real_vector)

    atual_fit = fit(real_vector)
    print(atual_fit, best)

    if atual_fit > best:
      vector = child_vector
      best = atual_fit
      sigma *= 1.1
    else:
      sigma *= 0.9
    stop = timeit.default_timer()
    print("Time:", stop - start)
    array_timer.append(stop-start)

    array_best.append(best)
    print("best_vector", vector)
    print("child_vector", child_vector)
    print("real_vector",real_vector)
    print("sigma", sigma)
    print(array_best)
  print("TEMPO", array_timer)
  return array_best

