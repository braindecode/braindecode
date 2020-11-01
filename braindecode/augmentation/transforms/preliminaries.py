from .global_variables import label_index_dict

def update_label_index_dict(train_dataset):
    subset_aug_sample = None #TODO
    subset_aug_labels = None #TODO
    global label_index_dict
    list_labels = list(set(subset_aug_labels))
    label_index_dict = {}
    for label in list_labels:
        label_index_dict[label] = []
    for i in range(len(subset_aug_sample)):
        label_index_dict[subset_aug_labels[i]].append(subset_aug_sample[i])

def global_variables_initialization(train_dataset):
    update_label_index_dict(train_dataset)
    