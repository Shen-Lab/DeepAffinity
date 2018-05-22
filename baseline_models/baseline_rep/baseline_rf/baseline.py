from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import scipy.sparse as sps
import numpy as np
import math
import random
from sklearn.model_selection import GridSearchCV
import pickle
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

random.seed(1234)
def normalize_labels(label):
   x = []
   minprecision = 0.001
   maxprecision = 30
   for i in label:
      if i < minprecision:
         i = minprecision
      elif i > maxprecision:
         i = maxprecision
      m = -math.log(i/maxprecision)
      x.append(m)

   return x

################# reading labels
data_path = "/scratch/user/mostafa_karimi/CPI/final_data/Baseline/"
label_path_train=data_path+"train_IC50"
label_train = []
f = open(label_path_train, "r")
length_train=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_train.append(float(line))
            length_train = length_train+1

f.close()
label_train = normalize_labels(label_train)


label_path_test=data_path+"test_IC50"
label_test = []
f = open(label_path_test, "r")
length_test=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_test.append(float(line))
            length_test = length_test+1

f.close()
label_test = normalize_labels(label_test)

label_path_ER=data_path+"ER_IC50"
label_ER = []
f = open(label_path_ER, "r")
length_ER=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_ER.append(float(line))
            length_ER = length_ER+1

f.close()
label_ER = normalize_labels(label_ER)


label_path_GPCR=data_path+"GPCR_IC50"
label_GPCR = []
f = open(label_path_GPCR, "r")
length_GPCR=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_GPCR.append(float(line))
            length_GPCR = length_GPCR+1

f.close()
label_GPCR = normalize_labels(label_GPCR)


label_path_kinase=data_path+"channel_IC50"
label_kinase = []
f = open(label_path_kinase, "r")
length_kinase=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_kinase.append(float(line))
            length_kinase = length_kinase+1

f.close()
label_kinase = normalize_labels(label_kinase)

##################  reading compound features
feature_ER = sps.lil_matrix((length_ER,17593))
textfile1 = open(data_path+"ER_compound_fingerprint")
textfile2 = open(data_path+"ER_protein_domain")
count=0
while length_ER > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result = np.array([list(map(int, list(x+y)))])
    feature_ER[count,]=result
    count = count+1


feature_GPCR = sps.lil_matrix((length_GPCR,17593))
textfile1 = open(data_path+"GPCR_compound_fingerprint")
textfile2 = open(data_path+"GPCR_protein_domain")
count=0
while length_GPCR > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result = np.array([list(map(int, list(x+y)))])
    feature_GPCR[count,]=result
    count = count+1

feature_kinase = sps.lil_matrix((length_kinase,17593))
textfile1 = open(data_path+"channel_compound_fingerprint")
textfile2 = open(data_path+"channel_protein_domain")
count=0
while length_kinase > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result = np.array([list(map(int, list(x+y)))])
    feature_kinase[count,]=result
    count = count+1

feature_train = sps.lil_matrix((length_train,17593))
textfile1 = open(data_path+"train_compound_fingerprint")
textfile2 = open(data_path+"train_protein_domain")
count=0
while length_train > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result = np.array([list(map(int, list(x+y)))])
    feature_train[count,]=result
    count = count+1


feature_test = sps.lil_matrix((length_test,17593))
textfile1 = open(data_path+"test_compound_fingerprint")
textfile2 = open(data_path+"test_protein_domain")
count=0
while length_test > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result = np.array([list(map(int, list(x+y)))])
    feature_test[count,]=result
    count = count+1

######### Lasoo model
regr = RandomForestRegressor(n_jobs=20) # 20 cores

num_estimator = [50,100,200,500]
max_feature = ["auto","sqrt","log2"]
min_samples_leaf = [10,50,100]

tuned_parameters = [{'n_estimators': num_estimator,'max_features':max_feature,'min_samples_leaf':min_samples_leaf}]
n_folds = 10
clf = GridSearchCV(regr, tuned_parameters, cv=n_folds)
clf.fit(feature_train,label_train)

print("train error:")
y_pred = clf.predict(feature_train)
print(mean_squared_error(label_train,y_pred))

results = sm.OLS(y_pred,sm.add_constant(label_train)).fit()
print(results.summary())

print("test error:")
y_pred = clf.predict(feature_test)
print(mean_squared_error(label_test,y_pred))

results = sm.OLS(y_pred,sm.add_constant(label_test)).fit()
print(results.summary())

print("ER error:")
y_pred = clf.predict(feature_ER)
print(mean_squared_error(label_ER,y_pred))

results = sm.OLS(y_pred,sm.add_constant(label_ER)).fit()
print(results.summary())


print("kinase error:")
y_pred = clf.predict(feature_kinase)
print(mean_squared_error(label_kinase,y_pred))

results = sm.OLS(y_pred,sm.add_constant(label_kinase)).fit()
print(results.summary())


print("GPCR error:")
y_pred = clf.predict(feature_GPCR)
print(mean_squared_error(label_GPCR,y_pred))

results = sm.OLS(y_pred,sm.add_constant(label_GPCR)).fit()
print(results.summary())


#########  Saving model
rf_pkl_filename = 'rf_20182101.pkl'
rf_model_pkl = open(rf_pkl_filename, 'wb')
pickle.dump(clf,rf_model_pkl)
rf_model_pkl.close()

########## Saving prediction test
rf_pkl_test_filename = 'rf_20182101_test_pred.pkl'
rf_test_pkl = open(rf_pkl_test_filename, 'wb')
pickle.dump(y_pred,rf_test_pkl)
rf_test_pkl.close()

######### Saving Real label test
rf_pkl_test_filename = 'rf_20182101_test_real_lable.pkl'
rf_test_pkl = open(rf_pkl_test_filename, 'wb')
pickle.dump(label_test,rf_test_pkl)
rf_test_pkl.close()

########## Saving prediction ER
rf_pkl_ER_filename = 'rf_20182101_ER_pred.pkl'
rf_ER_pkl = open(rf_pkl_ER_filename, 'wb')
pickle.dump(y_pred,rf_ER_pkl)
rf_ER_pkl.close()

######### Saving Real label ER
rf_pkl_ER_filename = 'rf_20182101_ER_real_lable.pkl'
rf_ER_pkl = open(rf_pkl_ER_filename, 'wb')
pickle.dump(label_ER,rf_ER_pkl)
rf_ER_pkl.close()

########## Saving prediction kinase
rf_pkl_kinase_filename = 'rf_20182101_kinase_pred.pkl'
rf_kinase_pkl = open(rf_pkl_kinase_filename, 'wb')
pickle.dump(y_pred,rf_kinase_pkl)
rf_kinase_pkl.close()

######### Saving Real label kinase
rf_pkl_kinase_filename = 'rf_20182101_kinase_real_lable.pkl'
rf_kinase_pkl = open(rf_pkl_kinase_filename, 'wb')
pickle.dump(label_kinase,rf_kinase_pkl)
rf_kinase_pkl.close()

########## Saving prediction GPCR
rf_pkl_GPCR_filename = 'rf_20182101_GPCR_pred.pkl'
rf_GPCR_pkl = open(rf_pkl_GPCR_filename, 'wb')
pickle.dump(y_pred,rf_GPCR_pkl)
rf_GPCR_pkl.close()

######### Saving Real label GPCR
rf_pkl_GPCR_filename = 'rf_20182101_GPCR_real_lable.pkl'
rf_GPCR_pkl = open(rf_pkl_GPCR_filename, 'wb')
pickle.dump(label_GPCR,rf_GPCR_pkl)
rf_GPCR_pkl.close()

