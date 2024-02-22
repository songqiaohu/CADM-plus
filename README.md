# CADM-plus
# The flow chart of CADM+ framework
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/framework.png?raw=true"/></div>  
<p>** construction of confusion model ** module updates an existing classifier with a new set of labeled data to obtain a new model, aiming to generate a confusion model. In drift detection module, upon the arrival of a new data chunk, the classifiers before and after incremental update will utilize the unlabeled data within it to calculate the cosine similarity. The resulting cosine similarity is then incorporated into a sliding window to compute the concept drift detection metric and threshold. In model update module, if the concept drift is detected, all samples in the data chunk are labeled,
and the classifier is retrained. Otherwise, part of the samples are selected for annotation to update the model. Subsequently, the detection and update modules iterate in a loop.</p>

