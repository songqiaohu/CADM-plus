
# CADM+: Confusion-based Learning Framework With Drift Detection and Adaptation for Real-time Safety Assessment
"""
Created on 2024:02:17

@author: Songqiao Hu, Zeyi Liu, Minyue Li, Xiao He
@School: Tsinghua University
"""

# Required Libraries
from skmultiflow.bayes import NaiveBayes
import numpy as np
import copy
import matplotlib.pyplot as plt
from skmultiflow.data import HyperplaneGenerator, FileStream
from spot import SPOT
import time



class CADM_plus_strategy():
    """Class implementing CADM+ strategy."""

    def __init__(self, stream, q = 0.03, train_size = 200, chunk_size = 100, label_ratio = 0.2,
                 class_count = 2, max_samples=1000000, k=500, classifier_string = "NB"):

        """
                Initializes CADM+ strategy.

                Args:
                    - stream: The stream of data samples.
                    - q: The quantile used in EVT (Extreme Value Theory).
                    - train_size: The size of the initial training set.
                    - chunk_size: The size of the data chunks for incremental learning.
                    - label_ratio: The ratio of labeled samples in each chunk.
                    - class_count: The number of classes in the classification task.
                    - max_samples: The maximum number of samples to process.
                    - k: Turning point.
                    - classifier_string: String indicating the type of base classifier (e.g., "NB" for Naive Bayes).

        """
        self.stream = stream
        self.train_size = train_size
        self.chunk_size = chunk_size
        self.label_ratio = label_ratio
        self.class_count = class_count
        self.max_samples = max_samples
        self.k = k
        self.classifier = classifier_string  # NB, can also add other classifiers

        # Lists to store various metrics and data
        self.cosine_all = []
        self.Y_prediction_all = [] # all predicted samples
        self.Y_real_all = [] # all samples
        self.accuracy_chunk_all = []
        self.drift = []
        self.threshold = []  #to store all thresholds
        self.delta_mean = []
        self.delta_std = []
        self.store = [[] for i in range(self.class_count)]
        self.right = 0
        self.prediction = 0
        self.n_samples =0 # sample count

        self.q = q # quantile of EVT
        self.s = SPOT(self.q)
        self.std_threshold = []

    def Uncertainty_margin(self, x_instance, clf):
        """
        Calculates the uncertainty margin of a given instance.

        Args:
        - x_instance: The input instance for which uncertainty margin is calculated.
        - clf: The classifier model.

        Returns:
        - The difference between the highest and lowest class probabilities.
        """

        proba = clf.predict_proba(x_instance)
        return max(proba[0]) - min(proba[0])

    def data_dividing(self, X, Y, Is_retrain, clf, Is_proba = 0):
        """
                Divides the data into labeled and unlabeled portions.

                Args:
                - X: Input features.
                - Y: True labels.
                - Is_retrain: Flag indicating whether it's a retraining phase.
                - clf: The classifier model.
                - Is_proba: Flag indicating whether to use uncertainty-based sampling (0 for random, 1 for uncertainty).

                Returns:
                - X_labeled: Labeled instances.
                - Y_labeled: Corresponding labels.
                - X_unlabeled: Unlabeled instances.
        """
        size = X.shape[0]
        if Is_retrain:
            label_size = round(size * 1)
        else:
            label_size = round(size * self.label_ratio)
        if Is_proba == 0:  # random
            index = np.random.permutation(np.arange(X.shape[0]))
            X_chunk = X[index]
            Y_chunk = Y[index]
            X_labeled = X_chunk[0:label_size]
            Y_labeled = Y_chunk[0:label_size]
            X_unlabeled = X[label_size::]

        else:             # uncertainty
            proba_list = []
            for i in range(X.shape[0]):
                proba_list.append(self.Uncertainty_margin(np.array([X[i, :]]), clf))
            proba_list = np.array(proba_list)
            index = proba_list.argsort()
            X_labeled = X[index[0:label_size]]
            Y_labeled = Y[index[0:label_size]]
            X_unlabeled = X[index[label_size::]]

        return X_labeled, Y_labeled, X_unlabeled

    def pseudo_label(self, X, classifier, size):  #get hard pseudo labels
        """
                Generates pseudo labels for unlabeled data using a classifier.

                Args:
                - X: Unlabeled instances.
                - classifier: The classifier model.
                - size: The size of the unlabeled dataset.

                Returns:
                - X_pseudo: Unlabeled instances.
                - Y_pseudo: Pseudo labels generated by the classifier.
        """

        Y_pseudo = np.zeros(size)
        for i in range(size):
            Y_pseudo[i]=classifier.predict(np.array([X[i]]))[0]
        return X[0:size,:], Y_pseudo


    def cosine_similarity(self, classifier1, classifier2, X_chunk):
        """
        Calculates the cosine similarity between the probability distributions of two classifiers.

        Args:
        - classifier1: The first classifier model.
        - classifier2: The second classifier model.
        - X_chunk: The data chunk for which similarity is calculated.

        Returns:
        - The average cosine similarity between the classifiers.
        """
        h1 = []
        h2 = []
        for i in range(X_chunk.shape[0]):
            h1.append(classifier1.predict_proba([X_chunk[i, :]])[0])
            h2.append(classifier2.predict_proba([X_chunk[i, :]])[0])

        sum_cos = 0
        for j in range(self.class_count):

            h11 = np.array([])
            h22 = np.array([])
            for i in range(len(h1)):
                if np.abs(h1[i][j]) < 1e-3:
                    h11 = np.append(h11, 1e-3)
                else:
                    h11 = np.append(h11, h1[i][j])
            for i in range(len(h2)):
                if np.abs(h2[i][j]) < 1e-3:
                    h22 = np.append(h22, 1e-3)
                else:
                    h22 = np.append(h22, h2[i][j])
            sum_cos += h11.dot(h22) / (np.linalg.norm(h11) * np.linalg.norm(h22))
        return sum_cos / self.class_count

    def classifier_select(self):
        """
            Selects and initializes the appropriate classifier based on the specified classifier string.

            Returns:
            - The selected classifier object.
        """
        if self.classifier == 'NB':
            return NaiveBayes()



    def anomaly_detection(self, window : list):
        """
            Detects anomalies in a sliding window of data.

            Args:
            - window: A list representing the sliding window of data.

            Returns:
            - delta_mean: The relative change in mean within the window.
            - delta_std: The relative change in standard deviation within the window.
        """
        length = len(window)
        if length < 10:
            return 0, 0 # Robustness operation
        else:
            new_data = window[length - 1]
            mean1 = np.mean(window)
            std1 = np.std(window)
            x = window.pop()
            mean2 = np.mean(window)
            std2 = np.std(window)
            delta_mean = abs(mean2 - mean1) / mean1
            if std1 == 0: # Robustness operation
                if std2 == 0:
                    delta_std = 0
                else:
                    delta_std = 1
            else:
                delta_std = abs(std2 - std1) / std1
            window.append(x)
            return delta_mean, delta_std


    def main(self):
        """Main function implementing CADM strategy."""


        # base classifier generation
        classifier_1 = self.classifier_select()
        X_train, Y_train = self.stream.next_sample(self.train_size)
        for i in range(len(Y_train)):
            self.store[int(Y_train[i])].append(X_train[i])
        X_labeled, Y_labeled, X_unlabeled = self.data_dividing(X_train, Y_train, Is_retrain=True, clf=classifier_1)
        classifier_1.fit(X_labeled, Y_labeled)

        X_update, Y_update = self.stream.next_sample(self.chunk_size)

        Y_prediction = classifier_1.predict(np.array(X_update))

        for i in range(len(Y_prediction)):
            self.prediction += 1
            if Y_prediction[i] == Y_update[i]:
                self.right += 1

        for i in Y_prediction:
            self.Y_prediction_all.append(i)
        for j in Y_update:
            self.Y_real_all.append(j)

        # Construction of Confusion Model Module
        X_labeled, Y_labeled, X_unlabeled = self.data_dividing(X_update, Y_update, Is_retrain=False, clf=classifier_1)
        X_pseudo, Y_pseudo = self.pseudo_label(X_unlabeled, classifier_1, X_labeled.shape[0])
        classifier_2 = copy.deepcopy(classifier_1)
        classifier_2.partial_fit(np.row_stack((X_pseudo, X_labeled)), np.append(Y_pseudo, Y_labeled))


        #######  main loop ###############
        self.n_samples = 2 * self.chunk_size   # 2 * self.chunk_size samples passed
        window = []
        while self.stream.has_more_samples() and self.n_samples < self.max_samples:
            print(self.n_samples)
            self.n_samples += self.chunk_size

            X_chunk, Y_chunk = self.stream.next_sample(self.chunk_size)
            Y_prediction = classifier_2.predict(np.array(X_chunk))

            for i in Y_prediction:
                self.Y_prediction_all.append(i)
            for j in Y_chunk:
                self.Y_real_all.append(j)

            accuracy_chunk = 0

            for i in range(len(Y_prediction)):
                self.prediction += 1
                if Y_prediction[i] == Y_chunk[i]:
                    self.right += 1
                    accuracy_chunk += 1

            self.accuracy_chunk_all.append(accuracy_chunk / len(Y_prediction))

            ################ drift detection ################################
            cosine = self.cosine_similarity(classifier_1, classifier_2, X_chunk)
            self.cosine_all.append(cosine)

            # update window
            window.append(cosine)
            if len(window) > 10 :
                window.pop(0)
            temp_mean, temp_std = self.anomaly_detection(window)
            self.delta_mean.append(temp_mean)
            self.delta_std.append(temp_std)
            self.threshold.append(np.mean(window) - 2 * np.sqrt(np.var(window)))
            flag = False
            if len(self.delta_std) < self.k:
                flag = (cosine < self.threshold[len(self.threshold) - 1])
            elif len(self.delta_std) == self.k:
                flag = (cosine < self.threshold[len(self.threshold) - 1])
                self.s.fit_init_data(np.array(self.delta_std), self.max_samples / self.chunk_size - 2 - len(self.delta_std))
                self.s.initialize()

            else:
                if temp_std > self.s.extreme_quantile:
                    flag = True

                elif temp_std > self.s.init_threshold:
                    self.s.peaks = np.append(self.s.peaks, temp_std - self.s.init_threshold)
                    self.s.Nt += 1
                    self.s.n += 1
                    # and we update the thresholds

                    g, s, l = self.s._grimshaw()
                    self.s.extreme_quantile = self.s._quantile(g, s)
                else:
                    self.s.n += 1
                if self.s.extreme_quantile < 0:
                    self.s.extreme_quantile = 0.15
                self.std_threshold.append(self.s.extreme_quantile)  # thresholds record

            if flag :
                self.drift.append(int(self.n_samples / self.chunk_size))
                print('drift occurs!!')
                index = np.random.permutation(np.arange(X_chunk.shape[0]))
                X_chunk = X_chunk[index]
                Y_chunk = Y_chunk[index]
                ########## select some samples to annotate in new chunk ########################
                X_labeled, Y_labeled, _ = self.data_dividing(X_chunk, Y_chunk, Is_retrain=True, clf=classifier_2)

                ##########    retrain   #########################
                classifier_1 = self.classifier_select()


                # train use all classes, avoid error
                for each_class in range(self.class_count):
                    if each_class not in Y_labeled:
                        for i in range(10):
                            X_labeled = np.row_stack((X_labeled, self.store[each_class][i]))
                            Y_labeled = np.append(Y_labeled, each_class)

                classifier_1.fit(X_labeled, Y_labeled)
                classifier_2 = copy.deepcopy(classifier_1)
                window = []
            else:
                index = np.random.permutation(np.arange(X_chunk.shape[0]))
                X_chunk = X_chunk[index]
                Y_chunk = Y_chunk[index]
                ########## select samples to annotate in new chunk #####################
                X_labeled, Y_labeled, X_unlabeled = self.data_dividing(X_chunk, Y_chunk, Is_retrain=False, Is_proba=1, clf=classifier_2)

                ################# get pseudo labels ###################################
                X_pseudo, Y_pseudo = self.pseudo_label(X_unlabeled, classifier_2, X_labeled.shape[0])#the sizes are equal

                ############ update  ####################################
                classifier_1 = copy.deepcopy(classifier_2)
                classifier_2.partial_fit(np.row_stack((X_labeled, X_pseudo)), np.append(Y_labeled, Y_pseudo))



        # output the result
        self.plot_accuracy()
        self.plot_delta_std()
        self.print_result()
        plt.show()
        return self.Y_prediction_all




    def plot_accuracy(self):
        plt.figure(1)
        plt.plot(range(1, len(self.accuracy_chunk_all) + 1), self.accuracy_chunk_all, color = 'forestgreen')
        plt.xlabel('$chunk$')
        plt.ylabel('$accuracy\ of\ each\ chunk$')
        plt.title('$CADM+-{},\ overall accuracy\ =\ {}\%$'.format(
            self.classifier, round(self.right / self.prediction * 100, 3)))


    def plot_delta_std(self):
        plt.figure(2)
        plt.plot(range(3, len(self.delta_std) + 3), self.delta_std, color = 'forestgreen', label=r"$\sigma_v$")
        plt.plot(range(self.k, self.k + len(self.std_threshold)), self.std_threshold, color = 'r', label='threshold')
        plt.xlabel('$chunk$')
        plt.legend(loc=1)

    def print_result(self):
        print('------------------ Result ------------------')
        print('The count of correct predicted samples: {}'.format(self.right))
        print('The count of all predicted samples: {}'.format(self.prediction))
        print('overall accuracy = {}%'.format(self.right / self.prediction * 100))
        print('label_cost = {}'.format((len(self.drift) * 1 + (self.max_samples / self.chunk_size - len(self.drift)) * self.label_ratio) / (self.max_samples / self.chunk_size)))

if __name__ == "__main__" :
    t1 = time.time()

    stream = FileStream('datasets/LAbrupt.csv')
    CADM_plus = CADM_plus_strategy(q = 0.03, stream=stream, train_size = 200, chunk_size = 100, label_ratio = 0.2,
                 class_count = stream.n_classes, max_samples=1000000, k=500, classifier_string="NB")
    CADM_plus.main()

    t2 = time.time()
    print('total time:{}s'.format(t2 - t1))
