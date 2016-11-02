#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Load the dictionary containing the dataset with the new feature 'mentioned_by_poi'
#   See my_feature.py for details about how the feature was created
with open("final_project_dataset_with_new_feature.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop( 'TOTAL', 0 )

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Initialize list of all features
features_list = ['poi', 'salary', 'to_messages', 'mentioned_by_poi', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']


### Extract features and labels from dataset 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Create the chosen models
clf = GaussianNB()
clfDT = tree.DecisionTreeClassifier()

### This was originally ran using all the features: range(0, 19) to find the ideal number
#   Set to 3 features for the final analysis using DecisionTree
for k_index in range(3 ,4):

	### Select top features 
	selector = SelectKBest(f_classif, k=k_index)
	new_features = selector.fit_transform(features, labels)
	actual_features = features_list[1:]
	selected_features = numpy.asarray(actual_features)[selector.get_support()].tolist()
	selected_features.insert(0, 'poi')
	print "Selected top ",k_index," features:", selected_features

	### Display scores
	print "Scores of features:"
	for x in range(len(actual_features)):
		print actual_features[x],"=",selector.scores_[x],"  "

	### Display results for chosen approach (Decision Tree)
	print "================"
	print "With top ", k_index
	print "================"
	test_classifier(clf, my_dataset, selected_features) 	
	#test_classifier(clfDT, my_dataset, selected_features)
	#print "Importances"
	#print clf.feature_importances_

	### Testing the inclusion of the new feature
	'''
	print " "
	print "================"
	print "With top ", k_index, "plus the mentioned_by_poi new feature"
	print "================"
	selected_plus_new_feature = selected_features
	selected_plus_new_feature.append('mentioned_by_poi')
	test_classifier(clf, my_dataset, selected_plus_new_feature)
	#test_classifier(clfDT, my_dataset, selected_features)
	#print "Importances"
	#print clf.feature_importances_
	'''


### Dump Classifier, dataset and features
dump_classifier_and_data(clf, my_dataset, selected_features)



