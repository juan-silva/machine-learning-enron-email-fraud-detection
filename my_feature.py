import sys
import numpy
import pickle
import os.path
sys.path.append( "../tools/" )

from sklearn.feature_extraction.text import CountVectorizer
from parse_out_email_text import parseOutText

'''
This script will add a new feature to the dictionary of Enron Data

The feature consists in the number of times a particular person's name is mentioned in the corpus of emails
sent from email addresses corresponding to known persons of interest. 
'''

def dataWithMentionsByPOIs():

	debug = True
	
	### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
	    data_dict = pickle.load(data_file)
	if(debug):
		print data_dict[data_dict.keys()[0]]

	### Find all the addresses for POIs
	poi_addresses = []
	for name in data_dict.keys():
		if(data_dict[name]['poi'] == True):
			poi_addresses.append(data_dict[name]['email_address'])
	if(debug):
		print poi_addresses		
		#poi_addresses = ['jeff.skilling@enron.com']

	### For each one of the addresses parse the email texts adding to a list for the vectorizer
	poi_words = []
	for address in poi_addresses:
		file_name = "emails_by_address/from_"+address+".txt"
		if(debug):
			print "Opening", file_name
		if(os.path.isfile(file_name)):
			from_person  = open(file_name, "r")
			if(debug):
				print from_person
			for path in from_person:
				path = path.replace("enron_mail_20110402/", "")
				path = os.path.join('..', path[:-1])
				if(os.path.isfile(path)):
					email = open(path, "r")
					txt = parseOutText(email)
					poi_words.append(txt)
				else:
					if(debug):
						print "File not found:",path
	if(debug):
		print len(poi_words)
		print poi_words[0]

	### Lets count the frequency of words in those emails
	vec = CountVectorizer()
	result = vec.fit_transform(poi_words)

	### Transform the result in a dictionary of the form {'word': count}
	freq = zip(vec.get_feature_names(),
	    numpy.asarray(result.sum(axis=0)).ravel())
	freq_dict = {}
	for word, count in freq:
		freq_dict[word] = count

	if(debug):
		#print freq_dict
		pass

	### Iterate the list of people in the data set and add the new feature
	#   Which is the number of times that any of their names are mentioned by POIs
	for name in data_dict.keys():
		nameCount = 0
		for singleName in name.split():
			if (len(singleName) > 2):
				if(singleName.lower() in freq_dict.keys() and freq_dict[singleName.lower()] > 0):
					nameCount = nameCount + freq_dict[singleName.lower()]
		data_dict[name]['mentioned_by_poi'] = nameCount
		if(debug):
			print name, " : ", nameCount

	return data_dict

my_dict = dataWithMentionsByPOIs()

with open('final_project_dataset_with_new_feature.pkl', "w") as dataset_outfile:
        pickle.dump(my_dict, dataset_outfile)

print my_dict['BAXTER JOHN C']['mentioned_by_poi']

