import numpy as np

from util import DataSet, Item


path_suffix = 'I:/My Classes/Utah/cs/CS6350spring2015/project/replication/'
source_path = 'data/ringnorm/'


def read_data_from_file(filename):
    """
    Obtain the ringnorm data set.
    :param filename: the filename of the data file
    :return: Return a DataSet instance which contains the ringnorm data. Each item in the data set is of the form
              [[f1, f2, ..., fn], label]
    """
    # the ringnorm data set
    ringnorm = DataSet()
    fname = path_suffix + source_path + filename
    with open(fname, 'r') as data_source:
        for line in data_source:
            all_data = line.rstrip('\n').split(',')
            features = [float(all_data[i]) for i in range(1, len(all_data))]
            label = int(float(all_data[0]))
            ringnorm.add_inst(Item(features, label))
    return ringnorm


def _test_ringnorm():
    filename = 'ringnorm_data1.csv'
    rn = read_data_from_file(filename)
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
