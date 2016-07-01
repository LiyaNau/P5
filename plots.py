import sys
import pickle
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
sys.path.append("../tools/")

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    data_dict.pop('TOTAL')

df = pd.DataFrame.from_dict(data_dict, orient = "index")
df = df.replace('NaN',0)

#pl.plot(df["salary"])
#print df["salary"]
feature = 'exercised_stock_options'
limit = 10000000
#plt.figure()
df.boxplot(feature, by = "poi")
plt.show()
full_features_list =  list(df.columns.values)

print df.loc[df[feature]>limit][["poi",feature]]

# for feature in full_features_list:
#     missing = len(df[df[feature]==0])
#     percent = len(df[df[feature]==0])/float(len(df))
#     if percent > 0.6 :
#         print feature, missing, percent
print full_features_list
