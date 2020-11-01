import numpy as np
from .transforms.preliminaries import global_variables_initialization

def augmented_train(subpolicies_list, train_dataset, eeg_model, epochs):
    """This function trains a model on an augmented dataset

    Args:
        subpolicies_list (List[torchvision.transforms.Compose]): The list of subpolicies used to augment the model
        train_dataset (Union[TransformDataset, TransformConcatDataset]): The dataset used to train the model
        eeg_model (Union[EEGClassifier, EEGRegressor]): The model that will be trained on the augmented dataset
        epochs (int): Number of epochs the model will be trained on
    """
    # Augments the dataset with the given policy
    train_dataset.update_augmentation_policy(subpolicies_list)
    # Initializes variables depending on the dataset, which are needed to compute the different transforms
    global_variables_initialization(train_dataset)
    
    # Trains the model
    y_train = np.array([data[1] for data in iter(train_dataset)])
    eeg_model.fit(train_dataset, y=y_train, epochs=epochs)
    
    return eeg_model
