#!/usr/bin/env python
# coding: utf-8

# # Import the library

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # objectively more readable
import matplotlib as mpl 
from scipy.sparse import csr_matrix # To create a sparse matrix of compressed sparse row format
from sklearn.model_selection import train_test_split # To split our data into train and test sets
from sklearn.neural_network import MLPClassifier # A popular choice for classification tasks

# confusion matrix and accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import seaborn as sns

# Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# # Import the dataset

# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")


# # Reading the Train and Test Datasets.

# In[5]:


train.head() # printing first five columns of train_data


# In[6]:


test.head() # printing first five columns of test_data


# In[3]:


train.shape # print the dimension or shape of train data


# In[4]:


test.shape # print the dimension or shape of test data


# # Data Cleaning

# In[7]:


# check the missing values in the train_dataset 
train.isnull().sum().head(10)


# In[8]:


# check the missing values in the train_dataset
test.isnull().sum().head(10)


# In[9]:


test.describe()


# In[10]:


train.describe()


# In[11]:


# About the test_dataset

print("Dimensions: ",test.shape, "\n") # dimensions
print(test.info()) # data types


# In[12]:


# About the train_dataset

print("Dimensions: ",train.shape, "\n") # dimensions
print(train.info()) # data types


# In[13]:


print(train.columns)
print(test.columns)


# In[14]:


order = list(np.sort(train['label'].unique()))
print(order)


# # EDA (Explotary Data Analysis)

# In[4]:


#EDA 1
#Displays 4 handwritten digit images
def display_digits(N):
    train = pd.read_csv('C:/Users/LENOVO/Computational Intelligence/Final/train.csv')
    images = np.random.randint(low=0, high=42001, size=N).tolist()
    
    subset_images = train.iloc[images,:]
    subset_images.index = range(1, N+1)
    print("Handwritten picked-up digits: ", subset_images['label'].values)
    subset_images.drop(columns=['label'], inplace=True)

    for i, row in subset_images.iterrows():
        plt.subplot((N//8)+1, 8, i)
        pixels = row.values.reshape((28,28))
        plt.imshow(pixels, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return ""


# In[5]:


display_digits(40)


# In[6]:


#EDA 2
#Distribution of the digits in the dataset
x = train['label'].value_counts().plot(kind='bar')
plt.show()


# # MNIST Classification: EDA, PCA, CNN [ 99.7% score]

# In[7]:


#EDA 3
# Visulaize a single digit with an array
digit_array = train.loc[3, "pixel0":]
arr = np.array(digit_array) 

image_array = np.reshape(arr, (28,28)) #.reshape(a, (28,28))
digit_img = plt.imshow(image_array, cmap=plt.cm.binary)
plt.colorbar(digit_img)
print("IMAGE LABEL: {}".format(train.loc[3, "label"]))


# In[8]:


#EDA 4
# Let's build a count plot to see the count of all the labels.
sns.countplot(x=train.label)
print(list(train.label.value_counts().sort_index()))


# In[9]:


#EDA 5
X_train = train.iloc[:,1:]
Y_train = train.iloc[:,0]

plt.subplot(221)
plt.imshow(np.reshape(np.array(X_train.iloc[0]),(28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(np.reshape(np.array(X_train.iloc[1]),(28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(np.reshape(np.array(X_train.iloc[2]),(28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(np.reshape(np.array(X_train.iloc[3]),(28,28)), cmap=plt.get_cmap('gray'))
plt.show()


# In[10]:


#EDA 6
random_samples = 3
fig, axes = plt.subplots(10, random_samples)
for i in range(10):
    class_samples = train[train['label'] == i].sample(random_samples)
    for j, (_, sample) in enumerate(class_samples.iterrows()):
        digit_pixels = np.array(sample.drop('label')).reshape(28,28)
        axes[i, j].imshow(digit_pixels, cmap='gray')
        axes[i, j].axis('off')

plt.suptitle('Digits')
plt.show()


# # Unsupervised Learning for MNIST with EDA

# In[11]:


#EDA 7
train = train.drop(['label'], axis=1).values.reshape(-1,28,28,1)
train = test.values.reshape(-1,28,28,1)


# In[12]:


num_examples = 15
plt.figure(figsize=(20,20))
for i in range(num_examples):
    plt.subplot(1, num_examples, i+1)
    plt.imshow(train[i], cmap='Greys')
    plt.axis('off')
plt.show()


# # Random Forest

# In[16]:


y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) # Drop 'label' column

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
del train # free some space
y_train.value_counts()


# In[6]:


model = RandomForestClassifier()


# In[7]:


# print('Number of Trees used : ', model.n_estimators) # number of trees used


# In[8]:


model.fit(X_train,y_train)


# In[9]:


predict= model.predict(X_train)
predict


# In[10]:


cm= metrics.confusion_matrix(y_train,model.predict(X_train))
cm


# In[11]:


from sklearn.metrics import accuracy_score
trainaccuracy= accuracy_score(y_train,model.predict(X_train))
trainaccuracy
print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))


# # Submission

# In[9]:


# predict result
result = model.predict(test)
result = pd.Series(result,name="Label")


# In[10]:


# predict result
result = model.predict(test)
result = pd.Series(result,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)

submission.to_csv("RathanaThero-Digit-Recognizer.csv",index=False)


# In[ ]:




