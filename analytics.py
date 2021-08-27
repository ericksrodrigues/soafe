from matplotlib import pyplot as plt
import json

def get_json(filename):
  f = open(filename + ".txt", 'r')
  obj = [x for x in json.loads(f.read())]
  return obj

def get_mean_json(filename):
  f = open(filename + ".txt", 'r')
  document = json.loads(f.read())
  array_len = len(document[0])
  mean_list = [0 for x in range(0,array_len)]
  for doc in document:
    for i, x in enumerate(doc):
      mean_list[i] += x
  mean_list = [x/len(document) for x in mean_list]
  return mean_list

dataset="spam"
normal = get_json("./resultados/"+dataset+"/normal")
desconstruida = get_json("./resultados/"+dataset+"/desconstruida")


model="rnn"
plt.plot([x[model] for x in normal], label="normal")
plt.plot([x[model] for x in desconstruida], label="desconstruida")
plt.legend()
plt.show()

plt.boxplot([[x[model] for x in normal], [x[model] for x in desconstruida]],labels=["normal", "Desconstruída"])
plt.show()
#rnn da spam