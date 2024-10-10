#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Import Libraries and Load Dataset


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
wine_data = pd.read_csv('WineQT.csv')


# In[4]:


# Basic info about the dataset
print(wine_data.info())


# In[5]:


print(wine_data.head())


# In[7]:


#Data Preprocessing


# In[8]:


# Separate features (X) and target (y)
X = wine_data.drop(['quality', 'Id'], axis=1)  # Features
y = wine_data['quality']  # Target: wine quality


# In[9]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


#Train Classification Models


# In[12]:


# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)


# In[13]:


# Stochastic Gradient Descent (SGD) Classifier
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train_scaled, y_train)
sgd_predictions = sgd_model.predict(X_test_scaled)


# In[14]:


# Support Vector Classifier (SVC)
svc_model = SVC(random_state=42)
svc_model.fit(X_train_scaled, y_train)
svc_predictions = svc_model.predict(X_test_scaled)


# In[15]:


#Evaluate Models


# In[16]:


# Random Forest performance
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)


# In[17]:


# SGD performance
sgd_accuracy = accuracy_score(y_test, sgd_predictions)
sgd_cm = confusion_matrix(y_test, sgd_predictions)


# In[18]:


# SVC performance
svc_accuracy = accuracy_score(y_test, svc_predictions)
svc_cm = confusion_matrix(y_test, svc_predictions)


# In[19]:


# Print accuracy scores
print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'SGD Accuracy: {sgd_accuracy}')
print(f'SVC Accuracy: {svc_accuracy}')


# In[20]:


#Visualize Confusion Matrices


# In[21]:


# Function to plot confusion matrices
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


# In[22]:


# Visualizing the confusion matrices
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
plot_confusion_matrix(sgd_cm, "SGD Confusion Matrix")
plot_confusion_matrix(svc_cm, "SVC Confusion Matrix")


# In[ ]:




