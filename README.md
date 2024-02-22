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
from CADM_plus_strategy import *
t1 = time.time()

stream = FileStream('datasets/LAbrupt.csv')
CADM_plus = CADM_plus_strategy(q = 0.03, stream=stream, train_size = 200, chunk_size = 100, label_ratio = 0.2,
             class_count = stream.n_classes, max_samples=1000000, k=500, classifier_string="NB")
CADM_plus.main()

t2 = time.time()
print('total time:{}s'.format(t2 - t1))
```

- Output:
```
from CADM_plus_strategy import *
t1 = time.time()

stream = FileStream('datasets/LAbrupt.csv')
CADM_plus = CADM_plus_strategy(q = 0.03, stream=stream, train_size = 200, chunk_size = 100, label_ratio = 0.2,
             class_count = stream.n_classes, max_samples=1000000, k=500, classifier_string="NB")
CADM_plus.main()

t2 = time.time()
print('total time:{}s'.format(t2 - t1))
```
  








