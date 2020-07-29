#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Example of an application of the K nearest neightbours algorithm for prediction
# theres is no exact way to define the number K
# for this small example we will use K = 5 and K = 7
# we will also used a built in dataset, for ilustration purposes, to go around memory issues 
# for KNN requires a lot of memory to work

# let import our data
from sklearn import datasets
wine = datasets.load_wine()


# In[17]:


# lets see what is this data about
print(wine.feature_names)


# In[18]:


print(wine.target_names)


# In[19]:


# lets see what is our data size
print(wine.data.shape)


# In[20]:


# 178 rows or observations and 13 columns or features


# In[21]:


# create our training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=69)


# In[22]:


# Create and apply our KNN model
from sklearn.neighbors import KNeighborsClassifier
knn5 = KNeighborsClassifier(n_neighbors=5) # K = 5

knn5.fit(X_train, y_train)

y_pred5 = knn5.predict(X_test)


# In[23]:


# lets see how accurate this model is
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred5))


# In[24]:


# 68.5% , not a disappointment but not satisfactory either
# lets try K = 7 now 


# In[25]:



from sklearn.neighbors import KNeighborsClassifier
knn7 = KNeighborsClassifier(n_neighbors=7) # K = 5

knn7.fit(X_train, y_train)

y_pred7 = knn7.predict(X_test)


# In[26]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred7))


# In[ ]:


# 72% is better but still not ideal
# if we were to make prediction we would use the knn with 7 neighbours

