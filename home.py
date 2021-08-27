from so_desconstruction import desconstruction
from sklearn import metrics
from EE import evolutionary_strategy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier

import pandas as pd
import numpy as np
import json

def target_function(new_features, old_features,clf, test_values,output_size):
        values_to_predict = []
        for i,nf in enumerate(new_features):
          values_to_predict.append(np.concatenate((nf,old_features[i][output_size:])))
        results = clf.predict(values_to_predict)
        #print(metrics.f1_score(y_train_pot,results))
        return metrics.f1_score(test_values,results)
#/*"diabetes", "hill_valley", "ionosphere", "letters"
#datasets = ["diabetes","hill_valley", "ionosphere","megawatt1", "sonar", "spam"]
datasets = ["ionosphere"]
models_name = ["random_forest","decision_tree", "mlp", "knn"]
models = [RandomForestClassifier, DecisionTreeClassifier,MLPClassifier,lambda: KNeighborsClassifier(n_neighbors=5)]

def traditional_execution():
  
  df_obj = {
    "dataset": [],
    "model": [],
    ###Teste Dimensões
    "size_10": [],
    "size_15": [],
    # "size_5": [],
    # "not_opt_5": []
    ###Teste Normal
    # "score_optimized": [],
    # "score_not_optimized": []
  }

  for dataset in datasets:
    for i,model in enumerate(models):
      # result = []
      # result_not_opt = []
      for x in range(0, 30):
        print(x, models_name[i], dataset)
        opt_5, not_opt_5 = desconstruction(
          target_function=target_function,
          model=model,
          soms_config=[{
            "shape": (5,5),
            "learning_rate": 0.01,
            "iterations": 1000
          }
          ],
          data_path="./datasets/" + dataset + ".csv",
          disableAdversarial=True)
        ### Teste Dimensão
        opt_10, not_opt = desconstruction(
          target_function=target_function,
          soms_config=[{
            "shape": (10,10),
            "learning_rate": 0.01,
            "iterations": 1000
          },
          ],
          model=model,
          data_path="./datasets/" + dataset + ".csv",
          disableAdversarial=True)
        opt_15, not_opt = desconstruction(
          target_function=target_function,
          soms_config=[{
            "shape": (15,15),
            "learning_rate": 0.01,
            "iterations": 1000
          },
          ],
          model=model,
          data_path="./datasets/" + dataset + ".csv",
          disableAdversarial=True)
      

        df_obj["dataset"].append(dataset)
        df_obj["model"].append(models_name[i])
        #Test Normal
        # df_obj["score_optimized"].append(opt_5)
        # df_obj["score_not_optimized"].append(not_opt_5)

        ### Test Dimensoes
        # df_obj["size_5"].append(opt_5)
        # df_obj["not_opt_5"].append(not_opt_5)
        df_obj["size_10"].append(opt_10)
        df_obj["size_15"].append(opt_15)
        #print("opt", opt_5, "not_opt",not_opt_5)
        

    #     break;
    #   break;
    # break;

      # with open('results_' + dataset + "_" + models_name[i] + ".json", 'w') as outfile:
      #   json.dump(result, outfile)
      # with open('results_not_opt_' + dataset + "_" + models_name[i] + ".json", 'w') as outfile:
      #   json.dump(result_not_opt, outfile)
  df = pd.DataFrame(data=df_obj)
  df.to_csv("results_retest_sizes.csv", index=False)

def ee_execution():
  datasets = ["ionosphere", "diabetes", "megawatt1"]
  models_name = ["random_forest","decision_tree", "mlp", "knn"]
  models = [RandomForestClassifier,DecisionTreeClassifier,MLPClassifier, lambda: KNeighborsClassifier(n_neighbors=5)]
  
  # models_name = ["random_forest"]
  # models = [RandomForestClassifier]
  
  for dataset in datasets:
    for i,model in enumerate(models):
      array_bests = []
      def fit(real_vector):
        [first_shape, second_shape, learning_rate, iteractions, adaline_output, adaline_iteractions,subset_size] = real_vector
        fitness, _ = desconstruction(
              target_function=target_function,
              soms_config=[{
                "shape": (first_shape,second_shape),
                "learning_rate": learning_rate,
                "iterations": iteractions
              },
              ],
              adaline_outputs=adaline_output,
              model=model,
              data_path="./datasets/" + dataset + ".csv",
              adversarial_iteractions=adaline_iteractions,
              subset_size=subset_size,
              disableAdversarial=True
              
        )
        return fitness
      for iteraction in range(30):
        print(iteraction, models_name[i])
        array_bests.append(evolutionary_strategy(fit))
      with open('results_'+dataset+'_ASOAFE_bmu_'+models_name[i] + ".json", 'w') as outfile:
        json.dump(array_bests, outfile)


ee_execution()