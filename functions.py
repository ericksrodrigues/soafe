functions.py

#RN
clf = MLPClassifier(solver='lbfgs', alpha=0.001,hidden_layer_sizes=(150,),random_state=1, max_iter=10000)
      clf.fit(new_X_train_pot, y_train_pot)

