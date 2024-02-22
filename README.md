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
------------------ Result ------------------
The count of correct predicted samples: 922667
The count of all predicted samples: 999800
overall accuracy = 92.28515703140629%
label_cost = 0.22368000000000002
total time:58.81265616416931s
```

## Datasets used in CADM+ 
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/datasets_CADM+2.png?raw=true"/></div>  

### Simulated Datasets
- Data distribution display:
<div align="center">
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/LAbrupt.gif?raw=true" width="240px" height="180px" alt="LAbrupt"/>
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/LSudden.gif?raw=true" width="240px" height="180px" alt="LSudden"/>
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/LGradual.gif?raw=true" width="240px" height="180px" alt="LGradual"/>
</div>

<p align="center">&#8195;(a) LAbrupt &#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; (b) LSudden &#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; (c) LGradual</p>

<div align="center">
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/NLAbrupt.gif?raw=true" width="240px" height="180px" alt="NLAbrupt"/>
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/NLSudden.gif?raw=true" width="240px" height="180px" alt="NLSudden"/>
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/NLGradual.gif?raw=true" width="240px" height="180px" alt="NLGradual"/>
</div>

<p align="center">&#8195;&#8195;(a) NLAbrupt &#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; (b) NLSudden &#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195; (c) NLGradual</p>

<div align="center">
  <img src="https://github.com/songqiaohu/pictureandgif/blob/main/LSudden_3.gif?raw=true" width="320px" height="240px" alt="LSudden_3"/>
</div>

<p align="center">(g) LSudden_3</p>



### Benchmark Datasets
- HYP_05 (from scikit-multiflow):
```
import csv
from skmultiflow.data import HyperplaneGenerator
import numpy as np
stream = HyperplaneGenerator(mag_change=0.5)
X, y = stream.next_sample(1000000)
with open('HYP_05.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(np.column_stack((X, y)))
```
- SEA_a (from MOA):
```
WriteStreamToARFFFile -s (ConceptDriftStream -s generators.SEAGenerator -d (ConceptDriftStream -s (generators.SEAGenerator -f 2) -d (ConceptDriftStream -s generators.SEAGenerator -d (generators.SEAGenerator -f 4) -p 250000 -w 50) -p 250000 -w 50) -p 250000 -w 50) -f (SEA_a.arff) -m 1000000
``` 
### Real-world Dataset (Jiaolong)
Please refer to
```
https://github.com/THUFDD/JiaolongDSMS_datasets
```
  








