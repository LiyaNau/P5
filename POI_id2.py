#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value']# 'bonus', 'salary', 'deferred_income'] # You will need to use more features
features_list_all = ['salary',"bonus","to_messages",'total_payments','exercised_stock_options',
                  'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                  'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
                  'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
features_list = ["poi"]
features_list += features_list_all
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
#clf = SVC(kernel='rbf', class_weight='balanced', C= 1000000000, gamma=0.1)

clf = make_pipeline(PCA(),SVC())
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
clf.set_params(svc__C=1, svc__gamma = 0.001, pca__n_components = 5)
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

clf.fit(features_train,labels_train)

pred = clf.predict(features_test)

print pred, labels_test
print "accuracy: ", clf.score(features_test,labels_test)
print "f1;", f1_score(labels_test,pred)
print "precision:", precision_score(labels_test,pred)
print "recall:", recall_score(labels_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
test_classifier(clf, my_dataset, features_list, folds = 1000)
dump_classifier_and_data(clf, my_dataset, features_list)