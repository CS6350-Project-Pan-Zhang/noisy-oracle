import copy
import math
import statistics as stats
import scipy.stats as scistat

from oracles import NoisyOracle


class IENoisyOracle(NoisyOracle):
    """
    The noisy oracle used in the IEThresh method.

    "Efficiently Learning the Accuracy of Labeling Sources for Selective Sampling"
    by Pinar Donmez, Jaime Carbonell, and Jeff Schneider

    The noisy oracle keeps a history of its predication against the majority.
    It also computes an upper confidence bound for the mean accuracy of its prediction based on the history.
    """

    def __init__(self, prob):
        # call the base class
        super(IENoisyOracle, self).__init__(prob)
        # history for making correct (agree with the majority) labeling -- 1 being correct and 0 being incorrect
        self._history = []
        # everyone gets a 1 and a 0 initially
        self._history.extend([1, 0])

    def update_history(self, performance):
        """
        Update the history of this oracle after the given performance
        :param performance: 1 for correct and 0 for incorrect
        :return: None
        """
        self._history.append(performance)

    def get_history(self):
        return copy.deepcopy(self._history)

    def upper_ie(self, alpha):
        """
        Compute the upper bound of the 100(1-alpha)% confidence interval for the mean performance of
        this oracle's history.
        :param alpha: the parameter for confidence level -- 100(1-alpha)% confidence level
        :return: the upper bound for the confidence interval
        """
        n_ = len(self._history)
        mean_ = stats.mean(self._history)
        stdev_ = stats.stdev(self._history)
        t_alpha = scistat.t.ppf(1-alpha/2, n_-1)
        return mean_ + t_alpha * stdev_ / math.sqrt(n_)


def _test_ie_noisy_oracle():
    p = 0.6
    ie_noisy_o = IENoisyOracle(p)
    predict = [ie_noisy_o.assert_label() for i in range(10)]
    print(predict)
    for i in predict:
        ie_noisy_o.update_history(i)
    print(ie_noisy_o.get_history())
    print(ie_noisy_o.upper_ie(0.05))


if __name__ == '__main__':
    _test_ie_noisy_oracle()