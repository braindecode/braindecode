# %% [markdown]
# 
# # Transfer learning for EEG
# 
# This example shows how to train a neural network with supervision on TUH [2]
# EEG data and transfer the model to NMT [3] EEG dataset. We follow the approach of [1]
# 

# %%
# Authors: MJ Bayazi <mj.darvishi92@gmail.com>
#
# License: BSD (3-clause)


random_state = 2024
n_jobs = 1

# %% [markdown]
# ## Loading and preprocessing the dataset
# 
# ### Loading the raw recordings
# 
# First, we load a few recordings from the Sleep Physionet dataset. Running
# this example with more recordings should yield better representations and
# downstream classification performance.
# 
# 

# %%


# %% [markdown]
# ### Preprocessing
# 
# Next, we preprocess the raw data. We convert the data to microvolts and apply
# a lowpass filter. Since the Sleep Physionet data is already sampled at 100 Hz
# we don't need to apply resampling.
# 
# 

# %%


# %% [markdown]
# ### Extracting windows
# 
# We extract 30-s windows to be used in both the pretext and downstream tasks.
# As RP (and SSL in general) don't require labelled data, the pretext task
# could be performed using unlabelled windows extracted with
# :func:`braindecode.datautil.windower.create_fixed_length_window`.
# Here however, purely for convenience, we directly extract labelled windows so
# that we can reuse them in the sleep staging downstream task later.
# 
# 

# %%


# %% [markdown]
# ### Preprocessing windows
# 
# We also preprocess the windows by applying channel-wise z-score normalization.
# 
# 

# %%


# %% [markdown]
# ### Splitting dataset into train, valid and test sets
# 
# We randomly split the recordings by subject into train, validation and
# testing sets. We further define a new Dataset class which can receive a pair
# of indices and return the corresponding windows. This will be needed when
# training and evaluating on the pretext task.
# 
# 

# %%


# %% [markdown]
# ### Creating samplers
# 
# Next, we need to create samplers. These samplers will be used to randomly
# sample pairs of examples to train and validate our model with
# self-supervision.
# 
# The RP samplers have two main hyperparameters. `tau_pos` and `tau_neg`
# control the size of the "positive" and "negative" contexts, respectively.
# Pairs of windows that are separated by less than `tau_pos` samples will be
# given a label of `1`, while pairs of windows that are separated by more than
# `tau_neg` samples will be given a label of `0`. Here, we use the same values
# as in [1]_, i.e., `tau_pos`= 1 min and `tau_neg`= 15 mins.
# 
# The samplers also control the number of pairs to be sampled (defined with
# `n_examples`). This number can be large to help regularize the pretext task
# training, for instance 2,000 pairs per recording as in [1]_. Here, we use a
# lower number of 250 pairs per recording to reduce training time.
# 
# 

# %%


# %% [markdown]
# ## Creating the model
# 
# We can now create the deep learning model. In this tutorial, we use a
# modified version of the sleep staging architecture introduced in [4]_ -
# a four-layer convolutional neural network - as our embedder.
# We change the dimensionality of the last layer to obtain a 100-dimension
# embedding, use 16 convolutional channels instead of 8, and add batch
# normalization after both temporal convolution layers.
# 
# We further wrap the model into a siamese architecture using the
# # :class:`ContrastiveNet` class defined below. This allows us to train the
# feature extractor end-to-end.
# 
# 

# %%


# %% [markdown]
# ## Training
# 
# We can now train our network on the pretext task. We use similar
# hyperparameters as in [1]_, but reduce the number of epochs and
# increase the learning rate to account for the smaller setting of
# this example.
# 
# 

# %%


# %% [markdown]
# ## Visualizing the results
# 
# ### Inspecting pretext task performance
# 
# We plot the loss and pretext task performance for the training and validation
# sets.
# 
# 

# %%


# %% [markdown]
# We also display the confusion matrix and classification report for the
# pretext task:
# 
# 

# %%


# %% [markdown]
# ### Using the learned representation for sleep staging
# 
# We can now use the trained convolutional neural network as a feature
# extractor. We perform sleep stage classification from the learned feature
# representation using a linear logistic regression classifier.
# 
# 

# %%


# %% [markdown]
# The balanced accuracy is much higher than chance-level (i.e., 20% for our
# 5-class classification problem). Finally, we perform a quick 2D visualization
# of the feature space using a PCA:
# 
# 

# %%


# %% [markdown]
# We see that there is sleep stage-related structure in the embedding. A
# nonlinear projection method (e.g., tSNE, UMAP) might yield more insightful
# visualizations. Using a similar approach, the embedding space could also be
# explored with respect to subject-level features, e.g., age and sex.
# 
# ## Conclusion
# 
# In this example, we used self-supervised learning (SSL) as a way to learn
# representations from unlabelled raw EEG data. Specifically, we used the
# relative positioning (RP) pretext task to train a feature extractor on a
# subset of the Sleep Physionet dataset. We then reused these features in a
# downstream sleep staging task. We achieved reasonable downstream performance
# and further showed with a 2D projection that the learned embedding space
# contained sleep-related structure.
# 
# Many avenues could be taken to improve on these results. For instance, using
# the entire Sleep Physionet dataset or training on larger datasets should help
# the feature extractor learn better representations during the pretext task.
# Other SSL tasks such as those described in [1]_ could further help discover
# more powerful features.
# 
# 
# ## References
# 
# .. [1] Darvishi-Bayazi, M. J., Ghaemi, M. S., Lesort, T., Arefin, M. R., Faubert, J., & Rish, I. (2024). Amplifying pathological detection in EEG signaling pathways through cross-dataset transfer learning. Computers in Biology and Medicine, 169, 107893.
# 
# .. [2] Shawki, N., Shadin, M. G., Elseify, T., Jakielaszek, L., Farkas, T., Persidsky, Y., ... & Picone, J. (2022). Correction to: The temple university hospital digital pathology corpus. In Signal Processing in Medicine and Biology: Emerging Trends in Research and Applications (pp. C1-C1). Cham: Springer International Publishing..
# 
# .. [3] Khan, H. A., Ul Ain, R., Kamboh, A. M., & Butt, H. T. (2022). The NMT scalp EEG dataset: an open-source annotated dataset of healthy and pathological EEG recordings for predictive modeling. Frontiers in neuroscience, 15, 755817.
# 
# 
# 


