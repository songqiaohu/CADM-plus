from CADM_plus_strategy import *
t1 = time.time()

stream = FileStream('datasets/LAbrupt.csv')
CADM_plus = CADM_plus_strategy(q = 0.03, stream=stream, train_size = 200, chunk_size = 100, label_ratio = 0.2,
             class_count = stream.n_classes, max_samples=1000000, k=500, classifier_string="NB")
CADM_plus.main()

t2 = time.time()
print('total time:{}s'.format(t2 - t1))