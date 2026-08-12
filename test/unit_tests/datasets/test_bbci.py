# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

from unittest.mock import MagicMock

from braindecode.datasets.bbci import BBCIDataset


def test_determine_sensors():
    # Create a mock BBCIDataset object
    bbci_dataset = BBCIDataset('mock_file')

    # Mock the get_all_sensors method to return 64 sensor names
    bbci_dataset.get_all_sensors = MagicMock(return_value=['ch'+str(i) for i in range(64)])

    # Call the _determine_sensors method
    chan_inds, sensor_names = bbci_dataset._determine_sensors()

    # Check if the output has the correct length
    assert len(chan_inds) == 64
    assert len(sensor_names) == 64
