import random
import numpy as np


class DataSet:
    """
    A class representing a data set, which is a collection of Items.

    Each instance (Item) in the data set is a collection of features and an associated label, which has the form
    [[f1, f2, ..., fn], label]
    label is either 1 or -1
    """

    def __init__(self):
        """
        Create an empty data set
        :return: None
        """
        # the whole data set
        self._dataset = []
        # number of instances
        self._size = 0
        # all positive instances -- store indices only
        self._positive = []
        self._size_positive = 0
        # all negative instances -- store indices only
        self._negative = []
        self._size_negative = 0
        # number of features of the instances
        self._n_features = 0

    def __str__(self):
        repr_ = ''
        i = 1
        for item in self._dataset:
            repr_ += str(i) + ': ' + str(item) + '\n'
            i += 1
        return repr_

    def size(self):
        return self._size

    def n_positive(self):
        return self._size_positive

    def n_negative(self):
        return self._size_negative

    def add_inst(self, instance):
        """
        Add the given instance into this data set.

        Raise TypeError if the given instance have incompatible number of features
        :param instance: the given instance (Item), which is of the form [[f1, f2, ..., fn], label]
        :return: None
        """
        if self._size == 0:
            # the first instance
            self._n_features = instance.n_features()
        else:
            # check for feature dimension compatibility
            if instance.n_features() != self._n_features:
                raise TypeError('Incompatible feature dimension!')
        self._dataset.append(instance)
        if instance.label() == 1:
            self._positive.append(self._size)
            self._size_positive += 1
        else:
            self._negative.append(self._size)
            self._size_negative += 1
        self._size += 1

    def remove_instance(self, i):
        """
        Remove the ith instance in this data set.
        Raise IndexError if i<0 or i>=self.size()
        :param i: the index of the instance to be removed
        :return: None
        """
        if i < 0 or i >= self._size:
            raise IndexError('Index out of bound!')
        instance = self._dataset[i]
        if instance.label() == 1:
            del self._positive[self._positive.index(i)]
            self._size_positive -= 1
        else:
            del self._negative[self._negative.index(i)]
            self._size_negative -= 1
        del self._dataset[i]
        self._size -= 1
        pass

    def get_inst(self, i):
        """
        Get the ith instance in this data set.
        Raise IndexError if i<0 or i>=self.size()
        :param i: the index of the instance to be removed
        :return: the ith instance
        """
        if i < 0 or i >= self._size:
            raise IndexError('Index out of bound!')
        return self._dataset[i]

    def divide(self, p):
        """
        Randomly divide this data set into a training set and a test set.

        The size of the training set is 100p% of the original data set
        :param p: the ratio between the size of the training set and the whole set
        :return: (training, test)
        """
        # always construct the smaller one -- the remaining one is the other part
        p_ = p
        if p > 0.5:
            p_ = 1 - p
        # 2 auxiliary data sets
        small = DataSet()
        large = DataSet()
        # size of the smaller part
        small_size = int(p_ * self._size)
        # indices selected
        selected_indices = random.sample(range(self._size), small_size)
        # indicator for selection
        selected = [False for i in range(self._size)]
        # construct the smaller one
        for i in selected_indices:
            small.add_inst(self._dataset[i])
            selected[i] = True
        # then the larger one
        for i in range(self._size):
            if not selected[i]:
                large.add_inst(self._dataset[i])
        # decide which one is which
        if p_ == p:
            # training is the smaller one, that is, p<=0.5
            train = small
            test = large
        else:
            # training is the larger one, that is, p>0.5
            train = large
            test = small
        return train, test

    def random_positive_inst(self):
        """
        :return: a randomly selected positive instance
        """
        i = random.randint(0, self._size_positive-1)
        return self._dataset[self._positive[i]]

    def random_negative_inst(self):
        """
        :return: a randomly selected negative instance
        """
        i = random.randint(0, self._size_negative-1)
        return self._dataset[self._negative[i]]

    def pop_random_positive_inst(self):
        """
        Randomly select a positive instance and remove it from the data set
        :return: a randomly selected positive instance
        """
        i = random.randint(0, self._size_positive-1)
        index = self._positive[i]
        inst = self._dataset[index]
        self.remove_instance(index)
        return inst

    def pop_random_negative_inst(self):
        """
        Randomly select a negative instance and remove it from the data set
        :return: a randomly selected negative instance
        """
        i = random.randint(0, self._size_negative-1)
        index = self._negative[i]
        inst = self._dataset[index]
        self.remove_instance(index)
        return inst

    def feature_label(self):
        """
        :return: all features without label as a matrix (ndarray), and labels (ndarray) separately
        """
        features = []
        labels = []
        for instance in self._dataset:
            features.append(instance.features())
            labels.append(instance.label())
        return np.array(features), np.array(labels)


class Item:
    """
    A class representing an item.

    An Item is of the form [[f1, f2, ..., fn], label]
    label is either 1 or -1
    """

    def __init__(self, features, label):
        """
        Create an item with given features and label.
        :param features: a list of features [f1, f2, ..., fn]
        :param label: a label
        :return: None
        """
        self._features = features
        self._label = label
        # number of features
        self._n_features = len(features)

    def __str__(self):
        return str([self._features, self._label])

    def label(self):
        """
        :return: the label of this instance
        """
        return self._label

    def features(self):
        """
        :return: the features of this instance, which is of the form [f1, f2, ..., fn]
        """
        return self._features

    def n_features(self):
        """
        :return: number of features of this instance
        """
        return len(self._features)