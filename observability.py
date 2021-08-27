import json
from matplotlib import pyplot as plt

import pandas as pd;

# diabetes = json.loads(open("./results_diabetes_balanced.json", "r").read())
# diabetes_n = json.loads(open("./results_normal_diabetes.json", "r").read())
# diabetes_u = json.loads(open("./results_diabetes.json", "r").read())

# ionosphere = json.loads(open("./results_ionosphere.json", "r").read())
# ionosphere_n = json.loads(open("./results_normal_ionosphere.json", "r").read())
# sonar = json.loads(open("./results_sonar.json", "r").read())
# sonar_n = json.loads(open("./results_normal_sonar.json", "r").read())
# spam = json.loads(open("./results_spam.json", "r").read())
# spam_n = json.loads(open("./results_normal_spam.json", "r").read())
# megawatt = json.loads(open("./results_megawatt_balanced.json", "r").read())
# megawatt_u = json.loads(open("./results_megawatt.json", "r").read())
# megawatt_n = json.loads(open("./results_normal_megawatt.json", "r").read())

# print(max(diabetes))
# fig, ax = plt.subplots()
# ax.set_ylabel("F1-Score")
# plt.boxplot([diabetes,diabetes_u, megawatt, megawatt_u],
#     labels=["Balanced Diabetes", "Unbalanced Diabetes", "Balanced Megawatt1", "Unbalanced Megawatt1"])
# plt.show()

compare = {
  "Dataset": ["diabetes","megawatt1","sonar", "ionosphere","spam"],
  'LFE':[0.762, 0.894, 0.801, 0.932, 0.947 ],
  "SOAFE": [0.715, 0.889, 0.909, 0.978 ,0.929]
}

df = pd.DataFrame(compare)
print(df.to_latex())