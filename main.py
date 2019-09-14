
import pandas

from pandas import read_csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def correlation_matrix(df,labels):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    df = pandas.DataFrame.from_records(df)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')

    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


data_train = read_csv('train.csv')
'''
plt.matshow(data_train.corr())
plt.show()
'''

f = plt.figure(figsize=(19, 15))
plt.matshow(data_train.corr(), fignum=f.number)
plt.xticks(range(data_train.shape[1]), data_train.columns, fontsize=4, rotation=45)
plt.yticks(range(data_train.shape[1]), data_train.columns, fontsize=4)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=4)
plt.title('Correlation Matrix', fontsize=16)
plt.show()
#data_test = read_csv('test.csv')

#print "data train keys are ",data_train.keys()

#define normal = 0 Glaucoma = 1

data_train['target'] = data_train[u'target'].map({'Class_1':0,'Class_2':1,'Class_3':3,'Class_4':3,'Class_5':4,'Class_6':5,'Class_7':6,'Class_8':7,'Class_9':8})

#data_test[u'target'] = data_test[u'target'].map({'Class_1':0,'Class_2':1,'Class_3':3,'Class_4':3,'Class_5':4,'Class_6':5,'Class_7':6,'Class_8':7,'Class_9':8})

feat_train = data_train.keys()

feat_labels_train = feat_train.get_values()

dataset_train = data_train.values

'''
feat_test = data_test.keys()

feat_labels_test = feat_test.get_values()

dataset_test = data_test.values
'''

#correlation_matrix(data_train,feat_train)

#Shuffle the dataset

X_train = dataset_train[:,0:94]

Y_train = dataset_train[:,94:]

print ("Y_train is ",Y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.3, random_state=42)

print ("Size of the dataset in MB is ",X_train.nbytes/1e6)
print ("Size of X_train ", X_train.shape)

trees            = 30

max_feat     = 7

max_depth = 30

min_sample = 2

clf = RandomForestClassifier(n_estimators=trees,

max_features=max_feat,

max_depth=max_depth,

min_samples_split= min_sample, random_state=0,

n_jobs=-1)

import time

start = time.time()
clf.fit(X_train, y_train)
end = time.time()

#Lets Note down the model training time

print("Execution time for building the Tree is: %f"%(float(end)- float(start)))

Y_pred = clf.predict(X_test)

print ("Accuracy is ",100*accuracy_score(y_test,Y_pred))


print (clf.feature_importances_)

sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(X_train,y_train)
Xtrain_1 = sfm.transform(X_train)
Xtest_1      = sfm.transform(X_test)

print ("Size of the dataset after feature selection in MB ",Xtrain_1.nbytes/1e6)

print ("Size of Xtrain_1 ", Xtrain_1.shape)

start = time.time()
clf.fit(Xtrain_1, y_train)
end = time.time()

print("Execution time for building the Tree is: %f"%(float(end)- float(start)))

pre = clf.predict(Xtest_1)

print ("Accuracy is ",100*accuracy_score(y_test,pre))



