import pandas as pd
import SimpSOM as sps

df = pd.read_csv("./datasets/diabetes.csv")
del df['class']
train = df.to_numpy()
print(train)
net = sps.somNet(2, 3, train, PBC=True)
net.train(0.01, 200)
net.save('flename_weights')
net.nodes_graph(colnum=0)

net.nodes_graph(colnum=0)
net.diff_graph()

#Project the datapoints on the new 2D network map.

#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(train, type='qthresh')	