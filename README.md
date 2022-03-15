---
title: "MYOPY MODEL"
author: "Naroa Legarra Marcos and Ignacio Chang Brahim"
date: "9/3/2022"
output: html_document
---

### DESCRIPTION OF THE MODEL

This model is a classification model based on a Linear SVM algorithm that aims to predict the type of myopy of the patient based on data collected. There are four types of classes in which the patient can be classified: myopy tyoe 1 (M1), myopy type 2 (M2) high myopy (MM) or none (C). 

### REQUIREMENTS

**Programming language:** Python 

**Required libraries:**

- Pandas==1.3.4

- Numpy==1.20.3

- Scikit-learn==1.0.2

- Pickle

 

### HOW TO EXECUTE IT

**1. Change working directory and import libraries:**

All the files included with the model should be located in the same folder.

```
path_to_model_folder=''
path_to_test_data=''

import os
os.chdir(path_to_model_folder) #change working directory to the model folder
import miopia
```

**2. Apply the model:**

```
data=miopia.Model(path_to_test_data) #introduce data to the model
data.predict() #generate the prediction
data.save_prediction() #save the prediction as a csv file
```

