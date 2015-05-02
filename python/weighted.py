import math

from oracles import NoisyOracle


class WTNoisyOracle(NoisyOracle):
    """
    The noisy oracle used in the Weighted method.

    The noisy oracle keeps a weight based on the history of its predication against the majority.
    """

    def __init__(self, prob):
        # call the base class
        super(WTNoisyOracle, self).__init__(prob)
        self._promote = 1.1
        # weight for the vote
        self._weight = 1

    def update_weight(self, performance):
        """
        Update the history of this oracle after the given performance
        :param performance: 1 for correct and 0 for incorrect
        :return: None
        """
        self._weight *= self._promote**(2 * performance - 1)

    def get_weight(self):
        return self._weight

    def vote(self):
        """
        :return: the weighted vote.
        """
        return self.assert_label() * self._weight


def _test_wt_noisy_oracle():
    p = 0.6
    wt_noisy_o = WTNoisyOracle(p)
    predict = [wt_noisy_o.assert_label() for i in range(10)]
    print(predict)
    wt_noisy_o.update_weight(1)
    print(wt_noisy_o.vote())
    wt_noisy_o.update_weight(0)
    print(wt_noisy_o.vote())


if __name__ == '__main__':
    _test_wt_noisy_oracle()