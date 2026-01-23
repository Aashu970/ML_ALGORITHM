      

#RANDOM FOREST :

from google.colab import files
      uploaded = files.upload()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Assuming X_train, y_train are the training data features and targets, respectively
rfc = RandomForestClassifier()
# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 5, 10]
}
# Perform grid search
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
# Print the best hyperparameters and corresponding score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Assuming X_train, y_train, X_test, and y_test are the training and testing data
# features and targets, respectively
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1)
rfc.fit(X_train, y_train)
# Predict the target variable for the test data
y_pred_rfc = rfc.predict(X_test)

#evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Evaluate the performance of the model
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Precision:", precision_score(y_test, y_pred_rfc, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rfc, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_rfc, average='weighted'))

rf_acc = accuracy_score(y_test, y_pred_rfc)
print("Random forest accuracy:", rf_acc)

#DECISION TREE :
from google.colab import files
      uploaded = files.upload()

#import necessary libraries
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      %matplotlib inline
      import seaborn as sns

#read csv file as a pandas dataframe
      ckd_data = pd.read_excel("/content/Training set.xlsx")

ckd_data.head()

ckd_data.tail()

#separate numerical and categorical columns
def separate_numerical_categorical_columns(df):
      numerical_cols = []
      categorical_cols = []
      for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
      return numerical_cols, categorical_cols


#get separate columns
      numerical_cols, categorical_cols =                                                                    separate_numerical_categorical_columns(ckd_data)
     # print the results
     print('Numerical columns:', numerical_cols)
     print('Categorical columns:', categorical_cols)

#check distribution of numerical columns by plotting histogram for numerical columns
     for col in numerical_cols:
      plt.figure()
      sns.histplot(ckd_data[col])
      plt.xlabel(col)
      plt.show()

#*plotting bar plot for categorical columns
      for col in categorical_cols:
       plt.figure(figsize=(8,6))
       ckd_data[col].value_counts().plot(kind='bar', color='blue')
       plt.title(col, fontsize=14)
       plt.show()

target_var = 'Druggability'

# Create a scatter plot for each numerical column against       the target variable
           for col in numerical_cols:
           plt.scatter(ckd_data[col], ckd_data[target_var])
           plt.xlabel(col)
           plt.ylabel(target_var)
           plt.show()

ckd_data['Druggability'].value_counts()

#divide data into attributes and labels
           X = ckd_data.drop('Druggability', axis=1)
           y = ckd_data['Druggability']

from sklearn.model_selection import train_test_split
   # Assuming X contains the features and y contains the    labels/targets
      X_train, X_test, y_train, y_test = train_test_split(X, y,        test_size=0.2, random_state=42)
# The "test_size" parameter specifies the percentage of data to be used for testing.
# The "random_state" parameter ensures that the same random split is obtained every time the code is run.

from sklearn.tree import DecisionTreeClassifier
# Assuming X_train and y_train are the training set features and targets, respectively
dct = DecisionTreeClassifier()
dct.fit(X_train, y_train)
# The fit() method trains the decision tree classifier on the training data.
# The target variable y_train is assumed to be a 1D array with 3 possible values.

y_pred_dct = dct.predict(X_test)

#evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_dct))
print(classification_report(y_test, y_pred_dct))

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred_dct))
print("Precision:", precision_score(y_test, y_pred_dct, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_dct, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_dct, average='weighted'))

dt_acc = accuracy_score(y_test, y_pred_dct)
print("Decision tree accuracy:", dt_acc)

#PERCEPTRON :

from google.colab import files
uploaded = files.upload()
      
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

# Assuming X contains the features and y contains the labels/targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The "test_size" parameter specifies the percentage of data to be used for testing.
# The "random_state" parameter ensures that the same random split is obtained every time the code is run.

perceptron = Perceptron(max_iter=1000, eta0=0.1)

perceptron.fit(X_train, y_train)

# Make predictions
y_pred = perceptron.predict(X_test)
print(y_pred)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#SVM :

from google.colab import files
      uploaded = files.upload()

#for loading data and for performing data analysis operations on it
import pandas as pd
import numpy as np
#for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
#for PCA (feature engineering)
from sklearn.decomposition import PCA
#for data scaling
from sklearn.preprocessing import StandardScaler
#for splitting dataset
from sklearn.model_selection import train_test_split
#for fitting SVM model
from sklearn.svm import SVC
#for displaying evaluation metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#for file operations
import os
print("All required libraries loaded!")

df = pd.read_excel("/content/Training set.xlsx")
df.shape

df.dtypes

df.head()

df.tail()

X = df.iloc[:,2:32]
print (X.shape)
X.head()

y = df.Druggability
print (y.shape)
y.head()

y_num = pd.get_dummies (y)
y_num.tail()

X= df.iloc[:,0:11]
#80:20 train:test data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape ",X_train.shape)
print("y_train shape ",y_train.shape)
print("X_test shape ",X_test.shape)
print("y_test shape ",y_test.shape)

#model fitting
svc = SVC()
svc.fit(X_train, y_train)

#predict values
y_pred_svc =svc.predict(X_test)
y_pred_svc.shape

#print confusion matrix
cm = confusion_matrix(y_test, y_pred_svc)
print("Confusion matrix:\n",cm)

#print classification report
creport = classification_report(y_test, y_pred_svc)
print("Classification report:\n",creport)

#LOGISTIC REGRESSION

#from google.colab import files
uploaded = files.upload()

#import the necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#read csv file as a pandas dataframe
ckd_data = pd.read_excel("/content/Training set.xlsx")
ckd_data.head()
ckd_data.tail()

#separate numerical and categorical columns
def separate_numerical_categorical_columns(df):
      numerical_cols = []
      categorical_cols = []
      for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
      return numerical_cols, categorical_cols

#get separate columns
numerical_cols, categorical_cols = separate_numerical_categorical_columns(ckd_data)

#print the results
print('Numerical columns:', numerical_cols)
print('Categorical columns:', categorical_cols)

ckd_data['Druggability'].value_counts()

#divide data into attributes and labels
X = ckd_data.drop('Druggability', axis=1)
y = ckd_data['Druggability']

#from sklearn.model_selection import train_test_split
# Assuming X contains the features and y contains the labels/targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# The "test_size" parameter specifies the percentage of data to be used for testing.
# The "random_state" parameter ensures that the same random split is obtained every time the code is run.

#LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

#Prediction
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc*100)





#ROC-AUC curve codes

Load libraries
import pandas as pd
#import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

data_set = pd.read_excel("/content/test dataset.xlsx")
#print(data_set.head())
print(data_set.isnull().sum()) #check null value

#Create a LabelEncoder object
le = LabelEncoder()

#Extracting Independent and dependent Variable  
#x= data_set[['EC (Enzyme Commission) Number', 'GO', 'Molecular weight', 'PPI', 'Orthology', 'Druggability']]
x = data_set[['EC (Enzyme Commission) Number', 'GO', 'Molecular weight', 'PPI', 'Orthology', 'Druggability']]

print(x)  
y= data_set[['Druggability']] 
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
y['Druggability']= label_encoder.fit_transform(y['Druggability'])
y['Druggability'].unique() 
#print(y)
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=42)  
#print(x_train, y_train)

#generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

#Fitting Decision Tree classifier to the training set  
from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=42)  
classifier.fit(x_train, y_train)

#predict probabilities
lr_probs = classifier.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Dission Tree: ROC AUC=%.3f' % (lr_auc))

y_pred= classifier.predict(x_test)
print("Y_Prediction --->>",y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred) 
print(cm)


#summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# Write Figure 
plt.savefig('./DecisionTree.png')




