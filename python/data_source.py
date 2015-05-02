import numpy as np
import matplotlib.pyplot as plt

from util import DataSet, Item

# the methods
method = ['random', 'repeated', 'iethresh', 'weighted']


class DataSource:
    """
    An interface for different data sources.
    """
    def __init__(self):
        self._path = 'I:/My Classes/Utah/cs/CS6350spring2015/project/replication/data/'
        self._dataset = None

    def get_dataset(self):
        return self._dataset

    @staticmethod
    def _record_accuracy(filenames, accuracies_):
        """
        Auxiliary method for recording the accuracies.
        """
        for i in range(len(filenames)):
            acc_ = accuracies_[i]
            with open(filenames[i], 'w') as out_:
                for j in range(len(acc_)):
                    out_.write(str(round(acc_[j], 3)) + ',')
                out_.write(str(round(acc_[j], 3)))

    @staticmethod
    def _plot_accuracy(accuracies_):
        """
        Auxiliary method for plotting the accuracies.
        """
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        for i in range(len(accuracies_)):
            plt.plot(accuracies_[i], label=method[i])
        plt.legend(bbox_to_anchor=(1, 0.225))
        plt.show()


class Ringnorm(DataSource):
    """
    Handle the ring norm data set.
    """

    def __init__(self, filename):
        """
        Create the data set from the given file.
        :param filename: the filename
        :return: None
        """
        super(Ringnorm, self).__init__()
        self._folder_path = self._path + 'ringnorm/'
        self._filename = self._folder_path + filename
        self._dataset = DataSet()
        # read the data from the given file
        self._read_data_from_file()
        # the accuracy files
        self._acc_filenames = [self._folder_path + 'acc_rnd.txt',
                               self._folder_path + 'acc_rep.txt',
                               self._folder_path + 'acc_iet.txt',
                               self._folder_path + 'acc_wei.txt']

    def _read_data_from_file(self):
        """
        Create the data set from the given file so that self._dataset is a list of Items (DataSet), where each item
        in the data set is of the form [[f1, f2, ..., fn], label]
        """
        with open(self._filename, 'r') as data_source:
            for line in data_source:
                all_data = line.rstrip('\n').split(',')
                features = [float(all_data[i]) for i in range(1, len(all_data))]
                label = int(float(all_data[0]))
                self._dataset.add_inst(Item(features, label))

    def record_accuracy(self, acc_rnd, acc_rep, acc_iet, acc_wei):
        """
        Record the accuracy for of the experiment of Random (acc_rnd), Repeated (acc_rep), Weighted (acc_wei),
        and IEThresh (acc_iet)
        on this data set.
        :return: None
        """
        accuracies_ = [acc_rnd, acc_rep, acc_iet, acc_wei]
        # output
        super(Ringnorm, self)._record_accuracy(self._acc_filenames, accuracies_)

    def plot_accuracy(self):
        accuracies_ = [[] for i in range(len(method))]
        for i in range(len(method)):
            with open(self._acc_filenames[i], 'r') as in_:
                acc_raw = in_.readline().rstrip('\n').split(',')
                for acc_ in acc_raw:
                    accuracies_[i].append(float(acc_))
        super(Ringnorm, self)._plot_accuracy(accuracies_)


def _test_ringnorm():
    filename = 'ringnorm_data1.csv'
    rn = Ringnorm(filename).get_dataset()
    # test size
    print('number of examples ' + str(rn.size()))

    # test division
    def _test_division():
        print('test division')
        p = 0.7
        train, test = rn.divide(p)
        print('training set (' + str(round(100*p, 1)) + '%)')
        print(train)
        print('test set (' + str(round(100*(1-p), 1)) + '%)')
        print(test)
    # _test_division()

    # test pop positive and negative examples
    def _test_pop_pos_neg():
        print('test pop positive/negative')
        print('a positive example removed')
        print(rn.pop_random_positive_inst())
        print(str(rn))
        print('a negative example removed')
        print(rn.pop_random_negative_inst())
        print(str(rn))
    _test_pop_pos_neg()

    # test separating features and labels
    def _test_separation():
        print('test feature/label separation')
        features, label = rn.feature_label()
        print('features without label')
        print(features)
        print('labels')
        print(label)
    # _test_separation()

    # test deletion
    def _test_deletion():
        print('test deletion')
        features, label = rn.feature_label()
        i = 5
        rn.remove_inst(i)
        # features is an ndarray
        features = np.delete(features, i, 0)
        # label is an ndarray -- np.delete(array, index, axis)
        label = np.delete(label, i, 0)
        for i in range(rn.size()):
            inst = rn.get_inst(i)
            fi = features[i]
            li = label[i]
            # print(str(i+1) + ': ' + str(inst))
            # print(str(fi) + ', ' + str(li) + '\n')
            print((inst.features() == fi))
            print((inst.label() == li))
    # _test_deletion()


if __name__ == '__main__':
    _test_ringnorm()

