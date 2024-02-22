# CADM-plus
## The flow chart of CADM+ framework

<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/framework.png?raw=true"/></div>


&nbsp;&nbsp;&nbsp;&nbsp;**<em>construction of confusion model</em>** module updates an existing classifier with a new set of labeled data to obtain a new model, aiming to generate a confusion model.  
&nbsp;&nbsp;&nbsp;&nbsp;In **<em>drift detection</em>** module, upon the arrival of a new data chunk, the classifiers before and after incremental update will utilize the unlabeled data within it to calculate the cosine similarity. The resulting cosine similarity is then incorporated into a sliding window to compute the concept drift detection metric and threshold.  
&nbsp;&nbsp;&nbsp;&nbsp;In **<em>model update</em>** module, if the concept drift is detected, all samples in the data chunk are labeled, and the classifier is retrained. Otherwise, part of the samples are selected for annotation to update the model. Subsequently, the detection and update modules iterate in a loop.

## Usage
- Download files, then directly run the **main.py** file. You can also modify parameters and datasets in the main file.  
- Example:


```
import pandas as pd

data = pd.read_csv('xxxxxx/nonlinear_gradual_chocolaterotation_noise_and_redunce.csv')

data = data.values 

X = data[:, 0 : 5] 

Y = data[:, 5] 
``` 








