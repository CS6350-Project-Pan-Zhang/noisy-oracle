import random


class NoisyOracle:
    """
    A noisy oracle.

    A noisy oracle has probability p for giving the correct (binary) label.
    """

    def __init__(self, prob):
        self._p = prob

    def get_prob(self):
        return self._p

    def assert_label(self):
        """
        Generate the true_label with probability p.
        :return: a random label
        """
        r = random.random()
        if r < self._p:
            return 1
        else:
            return -1


def _test_noisy_oracle():
    p = 0.9
    noisy_o = NoisyOracle(p)
    predict = [noisy_o.assert_label(1) for i in range(10)]
    print(predict)


if __name__ == '__main__':
    _test_noisy_oracle()