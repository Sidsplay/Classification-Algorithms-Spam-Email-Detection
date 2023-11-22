#!/usr/bin/env python
# coding: utf-8

# # Classification Algorithms: Spam Email Detection
# 

# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)
data = {'EmailText': ['Get rich quick!', 'Hello, how are you?', 'Claim your prize now!', 'Meeting at 3 pm'],
        'IsSpam': [1, 0, 1, 0]}
df = pd.DataFrame(data)

df['ContainsRich'] = df['EmailText'].apply(lambda x: 1 if 'rich' in x.lower() else 0)
df['ContainsPrize'] = df['EmailText'].apply(lambda x: 1 if 'prize' in x.lower() else 0)

X = df[['ContainsRich', 'ContainsPrize']]
y = df['IsSpam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# In[ ]:




