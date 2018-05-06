#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import os
import pandas as pd
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'Size of the Enron dataset: ',len(enron_data)
print 'Number of features for each person: ',len(enron_data['SKILLING JEFFREY K'])

n = 0
for person, features_person in enron_data.items():
    if features_person['poi'] == 1:
        n += 1
    else:
        continue
print 'Number of persons of interest from the Eron data: ', n

poi_df = pd.read_csv('../final_project/poi_names.txt', skiprows=[0,1], names=['poi','first_name','last_name'],sep=' ')
#poi_df['first_name']=poi_df['first_name'].apply(lambda x: x.replace(',',''))

print 'Number of persons of interest in total from names file: ', poi_df.shape[0]

print 'features: ', enron_data['PRENTICE JAMES'].keys()
print 'email messages from Wesley Colwell to poi: ', enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print 'value of stock options exercised by Jeffrey K Skilling: ', enron_data['SKILLING JEFFREY K']['exercised_stock_options']
max_payments=0
for poi in ['SKILLING JEFFREY K', 'LAY KENNETH L','FASTOW ANDREW S']:
    payments=enron_data[poi]['total_payments']
    if payments>max_payments:
        max_payments = payments
        max_poi=poi

print max_poi,max_payments

n_salary = 0
n_email = 0
for person, features_person in enron_data.items():
    if features_person['salary'] != 'NaN':
        n_salary += 1
    if features_person['email_address'] != 'NaN':
        n_email += 1
    else:
        continue

print 'salaires NaN: ',n_salary
print 'emails NaN: ', n_email

n_payments = 0
for person, features_person in enron_data.items():
    if features_person['total_payments'] == 'NaN':
        n_payments += 1
    else:
        continue

print 'percentage of person with payments NaN: ', 100*n_payments/146.0, n_payments

n_poi_payments = 0
for person, features_person in enron_data.items():
    if (features_person['total_payments'] == 'NaN') and (features_person['poi'] == 1):
        n_poi_payments += 1
    else:
        continue
print '% of POI with NaN for total payments: ', 100*n_poi_payments/146.0


