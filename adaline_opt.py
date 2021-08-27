import numpy as np

def tweak(dim, tweak_size=0.1):
  tk = np.random.uniform(-1,1,dim)*tweak_size
  return tk


def initial_weights(dim, outputs):
  w = np.random.uniform(-1,1,dim)
  return w


def aux_function(inputs):
  return sum(inputs)

def fitness_function(results):
  return np.median(results)

def adaline_func(data, target_function, clf, old_features, test_values,w=None,outputs=10, cicles=100):
  dim = (len(data[0]) + 1,outputs) #+1 para o bias
  if(w is None):
    w = initial_weights(dim,outputs)
  best_fitness = -np.Infinity
  for _ in range(0,cicles):
    results = []
    tk = tweak(dim)

    new_w = w + tk
    for inputs in data:
      input_with_bias = np.concatenate((np.asarray([1]), inputs))
      adaline_output = np.tanh(np.matmul(input_with_bias,new_w))
      results.append(adaline_output)
    fitness = target_function(results, old_features, clf, test_values,outputs)
    if(fitness > best_fitness):
      w = new_w
      best_fitness = fitness
  return w

def generate_adaline_features(inputs, w):
  return np.tanh(np.matmul(np.concatenate((np.asarray([1]), inputs)), w))

#adaline_func([[1,1,1],[1,-1, -10]], aux_function)
  # [a,b]
  # [c,d,e]
  # c = w11a + w12b + w13*1
  # d = w21a + w22b + w23*1
  # e = w31a + w32b + w33*1