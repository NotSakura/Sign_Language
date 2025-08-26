import pickle

## to train the classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np


data_dict = pickle.load(open('./data.pkl', 'rb'))


data = np.asarray(data_dict['dict']) # to to convert to array because thats what they take it as
labels = np.asarray(data_dict['labels'])


# split into trainng set and test set

x_train, x_test, y_train, y_test =train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # 0.2 = 20 % test set  , startify means same propotion of labels in train set and test set (so x_test and y_test should be proprotianal for all labels)

model = RandomForestClassifier() # super simple algo and super fast
model.fit(x_train, y_train) #MOST IMPORTANTT
y_predict = model.predict(x_test)

## see if the classifier did well

score = accuracy_score(y_predict, y_test)
print('{}% of samples werer classified correctly'.format(score*100)) # should get 100


# save the model
f = open('model.p', 'wb') # save the info
pickle.dump({'model':model}, f)
f.close()


