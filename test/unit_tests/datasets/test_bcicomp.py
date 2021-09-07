# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets import BCICompetitionIVDataset4

def mock_loadmat(x):
    file_name = x.split('/')[-1]
    {
        'sub1_comp.mat': {
            'train_data': None,
            'test_data': None
        },
        'sub1_testlabels.mat': {
            'train_data': None,
            'test_data': None
        },
    }