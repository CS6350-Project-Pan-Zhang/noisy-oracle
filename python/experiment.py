import random
from sklearn import linear_model as lm

from util import DataSet, Item


class Experiment:
    """
    The experiment for comparing Random, Repeated, and IEThresh methods for noisy oracles.
    """

    def __init__(self, dataset, p, oracles, n_rounds, eps, alpha):
        """
        Initialize the experiment for the given data set
        :param dataset: the given data set (of type DataSet)
        :param p: the proportion of the dataset used as training ((1-p) is the proportion for test)
        :param oracles: the group (list) of noisy oracles
        :param n_rounds: number of rounds to query the oracles
        :param eps: the cut-off value for the ithresh method
        :param alpha: confidence level for iethresh
        :return: None
        """
        # divide the given data set into training and test
        train_, test_ = dataset.divide(p)
        # the initial positive and negative examples
        positive_ = train_.pop_random_positive_inst()
        negative_ = train_.pop_random_negative_inst()
        # the unlabeled data set
        self._unlabeled = train_
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
        # number of queries
        self._rounds = n_rounds
        # cut-off value for iethresh
        self._eps = eps
        # confidence level for iethresh
        self._alpha = alpha

    def _one_train(self):
        """
        Train the model with the current training set once
        :return: None
        """
        features_, labels_ = self._train.feature_label()
        self._model.fit(features_, labels_)

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
        :param method: the given method to acquire a label from the oracles: 'random', 'repeated', 'weighted',
                        or 'iethresh'
        :return: None
        """
        if method == 'random':
            label_ = self._acquire_label_random()
        elif method == 'repeated':
            label_ = self._acquire_label_repeated()
        elif method == 'weighted':
            label_ = self._acquire_label_weighted()
        else:
            label_ = self._acquire_label_iethresh(self._eps)
        # update the training set with this new labeled instance
        new_labeled = Item(instance.features(), instance.label()*label_)
        self._train.add_inst(new_labeled)

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
            return 2 * random.randint(0, 1) - 1

    def _acquire_label_weighted(self):
        """
        Acquire a new label from the noisy oracles via the Weighted method.
        The label is determined by weighted vote
        :return: the label
        """
        # predication by all oracles
        labels_ = [o.vote() for o in self._oracles]
        # take the majority -- labels are either 1 or -1
        label_ = sum(labels_)
        if label_ > 0:
            l_ = 1
        elif label_ < 0:
            l_ = -1
        else:
            # tie -- return -1 or 1 randomly
            l_ = 2 * random.randint(0, 1) - 1
        # update the weight
        for i in range(len(labels_)):
            if labels_[i] * l_ > 0:
                self._oracles[i].update_weight(1)
            else:
                self._oracles[i].update_weight(0)
        return l_

    def _acquire_label_iethresh(self, eps):
        """
        Acquire a new label from the noisy oracles via the IEThresh method.
        :param eps: the cut-off value
        :return: the label
        """
        # predication by all oracles
        labels_ = [o.assert_label() for o in self._oracles]
        # upper bound of confidence interval on current mean performance for all oracles
        scores_ = [o.upper_ie(self._alpha) for o in self._oracles]
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
            l_ = 2 * random.randint(0, 1) - 1
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
        Find the most uncertain instance by the current model (for the logistic regression model on binary labels,
        the most uncertain label is the one with smallest probability difference).
        :return: the most uncertain instance
        """
        # all unlabeled instances
        unlabeled_ = self._unlabeled
        # the index of this most uncertain instance
        uncertain_index = -1
        # minimum probability difference
        min_prob_diff = 2
        for i in range(unlabeled_.size()):
            # the ith instance in the unlabeled data
            inst_ = unlabeled_.get_inst(i)
            probs_ = self._model.predict_proba(inst_.features())
            prob_diff = abs(probs_[0][0] - probs_[0][1])
            if prob_diff < min_prob_diff:
                min_prob_diff = prob_diff
                uncertain_index = i
        # the most uncertain instance
        uncertain_inst = unlabeled_.get_inst(uncertain_index)
        # remove this most uncertain instance from the unlabeled list -- since it will be labeled
        unlabeled_.remove_inst(uncertain_index)
        return uncertain_inst

    def run_exp(self, method):
        """
        Run n_rounds queries and record the performance
        :param method: the given method to acquire a label from the oracles: 'random', 'repeated', or 'iethresh'
        :return: the accuracy vector (accuracy over n_rounds)
        """
        for i in range(self._rounds):
            self._one_train()
            self._one_test()
            self._acquire_label(self._uncertain_inst(), method)
        return self._accuracy