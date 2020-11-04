from .global_variables import label_index_dict
from torch.utils.data.dataset import Subset


def update_label_index_dict(train_dataset):
    """Create a dictionnary, with as key the labels available in the multi-classification process, and as value for a given label all indexes of data that corresponds to its label.

    Args:
        train_dataset (Union[TransformDataset, TransformConcatDataset, Subset]): the train dataset.
    """
    if isinstance(train_dataset, Subset):
        subset_aug_indices = train_dataset.indices
    else:
        subset_aug_indices = list(range(len(train_dataset)))
    subset_aug_labels = [train_dataset[indice][1] for indice in subset_aug_indices]
    global label_index_dict
    list_labels = list(set(subset_aug_labels))
    label_index_dict = {}
    for label in list_labels:
        label_index_dict[label] = []
    for i in range(len(subset_aug_indices)):
        label_index_dict[subset_aug_labels[i]].append(subset_aug_indices[i])


def global_variables_initialization(train_dataset):
    """Compute every global variable (label_index_dict, em_decomposition_dict, etc...)

    Args:
        train_dataset (Union[TransformDataset, TransformConcatDataset, Subset]): the train dataset
    """
    update_label_index_dict(train_dataset)


def augment_dataset(train_dataset, subpolicies_list):
    """Transform the raw dataset into an augmented dataset. The Transform class does most of the job, nevertheless some additional work has to be done to adapt subset

    Args:
        train_dataset (Union[TransformDataset, TransformConcatDataset, Subset]): 
        subpolicies_list (List[Union(Transforms, Compose)]): List of transforms/subpolicies (= composition of transforms) that should be applied on the dataset.
    """
    if isinstance(train_dataset, Subset):
        train_dataset.dataset.update_augmentation_policy(subpolicies_list)
        temp = [list(range(i, i + len(subpolicies_list))) for i in train_dataset.indices]
        train_dataset.indices = [item for sublist in temp for item in sublist]
    else:
        train_dataset.update_augmentation_policy(subpolicies_list)

    return(train_dataset)
