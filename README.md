# My ML learning tracker
In this project, I practice machine learning approaches to classification task. For each approach, I implement the models from different machine learning and deep learning libraries and also, I try to reproduce the results on my own python code.
## Datasets 
### 1. Iris dataset. 
The iris data can be loaded through scikit-learn library:

```python
from sklearn import datasets

iris = datasets.load_iris()
```

The features are considered in iris flowers:
- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)


![iris flower](iris.jpg "Image 1.")

### 2. Credit card default prediction
This a small dataset created by AI Vietnam and it is used for pratice since there are only 14 samples.

```python
import pandas as pd

# Define the attribute names
attribute_names = ['age', 'income', 'student', 'credit_rate', 'default']

# Create the data dictionary
data = {
    'age': ['youth', 'youth', 'middle_age', 'senior', 'senior', 'senior', 'middle_age', 'youth', 'youth', 'senior',
            'middle_age', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'medium'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
    'credit_rate': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair',
                    'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair'],
    'default': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
}

# Create the DataFrame
df = pd.DataFrame(data, columns=attribute_names)

print(df)
```

## 1. K Nearest Neighbors (KNN folder)
For KNN approach, the model configuration is:
- k = 5
- distance_metric='minkowski'
- algorithm_neighbors='brute'
- weight='uniform', p=2
