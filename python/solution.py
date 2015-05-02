import matplotlib.pyplot as plt

from oracles import NoisyOracle
from iethresh import IENoisyOracle
from weighted import WTNoisyOracle
from experiment import Experiment
from data_source import Ringnorm

# the methods
method = ['random', 'repeated', 'iethresh', 'weighted']


def test_experiment(source_data):
    # the oracles
    p_ = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.75, 0.80, 0.85, 0.90]
    oracles_ = [NoisyOracle(pi) for pi in p_]
    ie_oracles_ = [IENoisyOracle(pi) for pi in p_]
    wt_oracles_ = [WTNoisyOracle(pi) for pi in p_]
    # the parameters for the experimentation
    train_p = 0.7
    n_rounds = 200
    eps = 0.9
    alpha = 0.05
    # number of repetitions
    n_rep = 10
    # the final accuracies for Random, Repeated, and IEThresh
    acc_rnd = [0 for i in range(n_rounds)]
    acc_rep = [0 for i in range(n_rounds)]
    acc_iet = [0 for i in range(n_rounds)]
    acc_wei = [0 for i in range(n_rounds)]
    accuracy_ = [acc_rnd, acc_rep, acc_iet, acc_wei]
    # the data set
    data_set = source_data.get_dataset()
    # for each method
    for k in range(len(method)):
        ora_ = oracles_
        # the oracles
        if k == 2:
            ora_ = ie_oracles_
        elif k == 3:
            ora_ = wt_oracles_
        # repeat the experiment n_rep times and take the average for the accuracy
        for i in range(n_rep):
            experiment_ = Experiment(data_set, train_p, ora_, n_rounds, eps, alpha)
            this_accuracy_ = experiment_.run_exp(method[k])
            # update the overall accuracy with this experiment
            for j in range(n_rounds):
                accuracy_[k][j] += this_accuracy_[j]
            print('k=' + str(k) + ', round: ' + str(i))
    # averaging
    for k in range(len(method)):
        for i in range(n_rounds):
            accuracy_[k][i] /= n_rep
    source_data.record_accuracy(acc_rnd, acc_rep, acc_iet, acc_wei)
    # plot
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    for i in range(len(accuracy_)):
        plt.plot(accuracy_[i], label=method[i])
    plt.legend(bbox_to_anchor=(1, 0.225))
    plt.show()


if __name__ == '__main__':
    filename = 'ringnorm_data.csv'
    data_source_ = Ringnorm(filename)
    test_experiment(data_source_)
    # data_source_.plot_accuracy()
