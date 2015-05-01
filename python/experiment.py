import random
from sklearn import linear_model as lm

from util import DataSet, Item
import ringnorm


class Experiment:
    """
    The experiment for comparing Random, Repeated, and IEThresh methods for noisy oracles.
    """

    def __init__(self, dataset, p, oracles):
        """
        Initialize the experiment for the given data set
        :param dataset: the given data set (of type DataSet)
        :param p: the proportion of the dataset used as training ((1-p) is the proportion for test)
        :param oracles: the group (list) of noisy oracles
        :return: None
        """
        # divide the given data set into training and test
        train_, test_ = dataset.divide(p)
        # the initial positive and negative examples
        positive_ = train_.pop_random_positive_inst()
        negative_ = train_.pop_random_negative_inst()
        # the training set -- initially, one positive and one negative
        self._train = DataSet()
        self._train.add_inst(positive_)
        self._train.add_inst(negative_)
        # the test set
        self._test = test_
        # separate the features and labels of the test set in order to test conveniently
        self._test_features, self._test_labels = self._test.feature_label()
        # the model -- use logistic regression
        self._model = lm.LogisticRegression()
        # number of queries made
        self._n_query = 0
        # performance sequence on the test set as labels are acquired
        self._accuracy = []
        # the oracles
        self._oracles = oracles

    def _one_train(self):
        """
        Train the model with the current training set once
        :return: None
        """
        features_, labels_ = self._train.feature_label()
        self._model.fix(features_, labels_)

    def _one_test(self):
        """
        Test the current model on the test set once
        :return: None
        """
        accuracy_ = self._model.score(self._test_features, self._test_labels)
        self._accuracy.append(accuracy_)

    def _acquire_label(self, instance, method='iethresh'):
        """
        Acquire a new label from the oracles, and update the training set correspondingly.
        :param instance: the given instance whose label is to be acquired (in this simulation, we actually know the
                          label), which is an Item (feature and label)
        :param method: the given method to acquire a label from the oracles: 'random', 'repeated', or 'iethresh'
        :return: None
        """
        if method == 'random':
            label_ = self._acquire_label_random(instance)
        elif method == 'repeated':
            label_ = self._acquire_label_repeated(instance)
        else:
            label_ = self._acquire_label_iethresh(instance)
        # update the training set with this new labeled instance
        new_labeled = Item(instance.feature(), instance.label()*label_)
        self._train.add_inst((new_labeled))


    def _acquire_label_random(self):
        """
        Acquire a new label from the noisy oracles via the Random method.
        :return: the label
        """
        # number of oracles
        n_oracles = len(self._oracles)
        # randomly pick an oracle
        o_picked_index = random.randint(0, n_oracles-1)
        o_picked = self._oracles[o_picked_index]
        # ask this oracle to give the label
        label_ = o_picked.assert_label()
        return label_


    def _acquire_label_repeated(self):
        """
        Acquire a new label from the noisy oracles via the Repeated method.
        The label is determined by majority vote
        :return: the label
        """
        # predication by all oracles
        labels_ = [o.assert_label() for o in self._oracles]
        # take the majority -- labels are either 1 or -1
        label_ = sum(labels_)
        if label_ > 0:
            return 1
        elif label_ < 0:
            return -1
        else:
            # tie -- return -1 or 1 randomly
            return 2*(random.randint(0,1)-0.5)

    def _acquire_label_iethresh(self, eps):
        """
        Acquire a new label from the noisy oracles via the IEThresh method.
        :param eps: the cut-off value
        :return: the label
        """
        # predication by all oracles
        labels_ = [o.assert_label() for o in self._oracles]
        # upper bound of confidence interval on current mean performance for all oracles
        scores_ = [o.upper_ie() for o in self._oracles]
        cutoff_ = max(scores_) * eps
        label_ = 0
        # query oracles
        for i in range(len(scores_)):
            # only take oracles with high performance
            if scores_[i] >= cutoff_:
                label_ += labels_[i]
        # the label
        if label_ > 0:
            l_ = 1
        elif label_ < 0:
            l_ = -1
        else:
            # tie -- return -1 or 1 randomly
            l_ = 2*(random.randint(0,1)-0.5)
        # update the histories of the queried oracles
        for i in range(len(scores_)):
            # only take oracles queried
            if scores_[i] >= cutoff_:
                if labels_[i] == l_:
                    # correct
                    self._oracles[i].update_history(1)
                else:
                    # incorrect
                    self._oracles[i].update_history(0)
        return l_

    def _uncertain_inst(self):
        """
        Find the most uncertain instance by the current model (for the logistic regression model, )
        :return:
        """
        pass

    def run_exp(self, n_rounds, method):
        """
        Run n_rounds queries and record the performance
        :param n_rounds: number of queries to make
        :param method: the given method to acquire a label from the oracles: 'random', 'repeated', or 'iethresh'
        :return: None
        """
        for i in range(n_rounds):
            self._one_train()
            self._one_test()
            self._acquire_label()
            pass