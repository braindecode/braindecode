# first line: 107
@memory.cache
def get_dummy_sample(preprocessing=["microvolt_scaling", "filtering"]):
    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=[0],
        valid_subjects=[1],
        test_subjects=[2],
        recording=[1],
        crop_wake_mins=0,
        preprocessing=preprocessing)
    # for i in range(len(train_sample)):
    #     train_sample[i] = (train_sample[i][0][:50], train_sample[i][1],
    #                        train_sample[i][2])
    # for i in range(len(test_sample)):
    #     test_sample[i] = (test_sample[i][0][:50],
    #                       test_sample[i][1], test_sample[i][2])
    test_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    valid_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    train_tinying_dict = {0: [350, 1029, 1291, 1650, 1571]}
    test_tinying_dict = {0: valid_choice}
    valid_tinying_dict = {0: test_choice}
    train_sample.tinying_dataset(train_tinying_dict)
    test_sample.tinying_dataset(test_tinying_dict)
    valid_sample.tinying_dataset(valid_tinying_dict)

    return train_sample, valid_sample, test_sample
