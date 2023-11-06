# Contribution Guide

Todo: In case of questions, discord? At least for sprint time?

For this contribution guide, we assume you are working in some Unix-environment like Ubuntu.

## Setup Braindecode for Development

### Create your own Braindecode fork

Register on https://github.com/ if you do not have a Github Account yet.
Go to https://github.com/braindecode/braindecode/ and click on fork to have your personal fork of braindecode for
development.

### Clone your own fork locally

You can either clone via ssh or https. Go to your personal fork on github.com to get the URLs, see
https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository
for a general introduction to cloning a repository from Github.
Here it would work with one of those commands (change <yourusername> to your GitHub user name):

```
git clone git@github.com:<yourusername>/braindecode.git
```

or

```
git clone https://github.com/<yourusername>/braindecode.git
```

See also https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh for how to setup ssh
keys.
In the following, I will assume you are using the first version via ssh.

### Add Upstream Repository

Now we set the git configuration to allow us to interact with the upstream repository
at https://github.com/braindecode/braindecode.

When inside the cloned braindecode folder, we add the upstream remote like this:

```
git remote add upstream git@github.com:braindecode/braindecode.git
```

(change the url for https if you used that in the step before)

The results of the `git remote -v` command should look like this now:

```
origin	git@github.com: <username>/braindecode.git (fetch)
origin	git@github.com: <username>/braindecode.git (push)
upstream	git@github.com:braindecode/braindecode.git (fetch)
upstream	git@github.com:braindecode/braindecode.git (push)
```

### Install Braindecode

From within your local repository, install a virtual python
environment. We will use conda, but feel free to use any other
solution:

```
conda create --name braindecode python=3.10
conda activate braindecode
pip install -e .[moabb,tests,docs]
```

### Test your setup

#### Run the tests

From within your local repository, run

```
pytest test/
```

This may take quite some time on the first run, as some datasets will be downloaded.
Subsequent runs will be faster.

#### Build the documentation

From within your local repository, run

```
cd docs
make html
```

Again, the first run may take longer due to datasets being downloaded.
You can look at the documentation by opening `docs/_build/html/index.html` in your browser.

#### Run Stylecheck

From within your local repository, run

```
pre-commit install
pre-commit run pre -a
```

There should be an empty output as there should be no style mistakes.

## Make a contribution

### Find an open issue

First, you can look at the open issues under https://github.com/braindecode/braindecode/issues and find one that
interests you.
There are now tags showing basic/intermediate/advanced tasks, also a tag for the Braindecode sprint 2021, here you see
the open issues for that:
https://github.com/braindecode/braindecode/issues?q=is%3Aissue+is%3Aopen+label%3Asprint.
Make sure to comment that you try to work on that issue so that other people know about it and also assign it to
yourself (right side on the issue page).

### Activate Conda Braindecode Environment

Don't forget to activate your conda environment that you installed braindecode in!

```
conda activate braindecode
```

### Create a local branch for your contribution

To ensure you are up to date with the upstream master, in your local repository,
run:

```
git checkout master
git pull upstream master
```

Pick a local branch-name and within your local repository, run:

```
git checkout -b <branch-name>
```

### Write Code

Now you can start writing the code to address the issue.
Make sure to follow the model conventions, and to properly document and test your code.

#### Add a model

If you want to contribute by adding a new model,
you should be aware that the Braindecode models must all follow certain conventions,
otherwise, your model will fail some tests.
Please follow the detailed guide on *Adding a model to Braindecode* provided at
[the end of this document](#adding-a-model-to-braindecode).

#### Write Tests

We have unit tests that test a unit like a function or a class method under
`test/unit_tests`. The directories below `test/unit_tests` should mirror the directory structure in braindecode.
To add your test, either insert your test to an existing test file or create a new corresponding test file.
For example, if you modify code in `datautil/preprocess.py`, add the corresponding test
to `test/unit_tests/datautil/test_preprocess.py`.
Note the `test_` at the beginning of the test-filename.
The structure in `test/unit_tests` does not perfectly mirror the module structure at the moment, we may change that in
the future.
Please try to make sure that your tests run reasonably fast (<5 sec).

For more complex functionality like an entire training pipeline, we have tests under `test/acceptance_tests`.
You may add another file there if needed. These tests may also run a bit longer.
Make sure your test passes by running:
`pytest test/<yourtestfilepath>`
Also before pushing, make sure to run all tests with:
`pytest test`

#### Write Documentation

##### Add to API Documentation

In case your code adds some public function/class that can be called by the user, please add it to our documentation as
follows.
First, make sure that you import the function in the `__init__.py` file in case you are in a subdirectory already.
For example, the function `create_from_X_y` in the file `datautil/xy.py` is imported in `datautil.__init__.py` as
follows:

```
from .xy import create_from_X_y
```

Also add your function to `docs/api.rst` under the appropriate submodule.
For example the `create_from_X_y` function is added here:

```
Data Utils
==========

:py:mod:`braindecode.datautil`:

.. currentmodule:: braindecode.datautil

.. autosummary::
   :toctree: generated/

    create_from_X_y
```

##### Add to What's New page

Finally, concisely describe your changes in `docs/whats_new.rst`, for example like this:

```
- Adding support for on-the-fly transforms (:gh:`198` by `Hubert Banville`_)
```

This allows everybody to understand how your code improved Braindecode from the last version! :)
The ```:gh:`198` ``` part refers to the corresponding Pull Request which we will get to further below.
You may leave it out for now.
Don't forget to add yourself to the list of authors at the end of the file, for example like this:

```
.. _Hubert Banville: https://github.com/hubertjb
```

##### Add your name to code file author list

Also ensure to add your name at the top of any code file you edited, for example like:

```
# Authors: Hubert Banville <hubert.jbanville@gmail.com>
[...]
```

##### Add an example file

If you add functionality that could be nicely explained with a code example, add a file to the `examples` folder with
a `plot_` at the beginning of the file name.
For example check `examples/plot_bcic_iv_2a_moabb_trial.py`.
These files will be executed when building of the documentation.

##### Build and check the documentation

As explained above, you can run

```
cd docs
make html
```

and check the built documentation under `docs/_build/html/index.html`.

#### Check Style

As explained above, you can run:

```
flake8
```

and ensure that there are no style errors in your code (output of this command should be empty).

### Commit and push to your fork

Now you can add, commit and push to your local fork.
For example, if you had modified `datautil/xy.py` following all the steps above, it may look like this:

```
git add datautil/xy.py test/unit_tests/datautil/test_xy.py docs/api.rst docs/whats_new.rst
git commit -m '<yourcommitmessage>'
git push -u origin <branch-name>
```

You can use `git diff` and `git status` to see if you missed any changes you made.
You have to repeat these steps once you change your code further and want to push it again.

### Start a Pull Request

Open your fork on https://github.com, after you pushed, there should be a button to make a pull request to the main
braindecode repository from your new branch.
Click that button and describe a bit more what you have changed so others can review your changes.
Now you should see your pull request appearing under https://github.com/braindecode/braindecode/pulls. The continuous
integration will try to run the tests and built the documentation from your pull request.
You can also later add the number of your pull request to `docs/whats_new.rst`.

Congratulations, now you have made your first Braindecode pull request!
In order to keep your pull request up to date when you have made further changes, simply rerun add/commit/push as above.

### Rebasing from upstream master

If you worked for a longer time in your branch and the upstream Braindecode master branch has already changed a lot,
it may become necessary to rebase your code changes on the current Braindecode master.
This will make the git history look as if you just started your changes from the current upstream master state.
To do that, from your local repository, first commit all the changes inside your local branch (without pushing) and then
run the following:

```
git fetch upstream master:master
git checkout <branch-name>
git rebase master
```

You may have to resolve some code conflicts in the rebase process, follow the instructions on the terminal to do so.
Once for your next push,you have to add `--force` like:

```
git push -u origin <branch-name> --force
```

## Adding a model to Braindecode
<a name="add_model"></a>

All models present in braindecode are PyTorch models. You are more than welcome to use any published PyTorch model in
braindecode!
Let's demonstrate how to implement a model for braindecode. Suppose we want to add a model called MyNewModel.

First, we create a new file called `my_new_model.py` in the `braindecode/models` directory.

Then, we add the following template to the file:

```python
from torch import nn
from braindecode.model.base import EEGModuleMixin


class MyNewModel(EEGModuleMixin, nn.Module):
    '''
    My new model.

    Model described in [1]_.

    Parameters
    ----------
    my_parameter : int
        DESCRIBE YOUR PARAMETERS HERE AND WHY YOU SET THESE
        DEFAULT VALUES

    References
    ----------
    .. [1] First Author, Second Author (YEAR).
       Title of the Article that Introduces this Architecture.
       Journal or Conference name, DATE.
       Online: URL/DOI OF THE PUBLICATION
    '''

    def __init__(
            self,
            n_outputs=None,
            n_chans=None,
            chs_info=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
            my_parameter=10,  # ADD YOUR PARAMETERS HERE
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.my_parameter = my_parameter

        # DEFINE YOUR MODEL HERE
        emb_dim = 1  # CHANGE THIS

        self.final_layer = nn.Linear(emb_dim, self.n_outputs)

    def forward(self, X):
        # IMPLEMENT THE FORWARD PASS
        return self.final_layer(X)
```

Explaining the template and internal convention:

1. **Inherit from EEGModuleMixin**: To ensure compatibility and standardization, all models in Braindecode should
   inherit from the `EEGModuleMixin` class. This helps standardize input parameters and ensure your model can seamlessly
   work with `EEGClassifier` or `EEGRegressor`.

   ```python
   from braindecode.model.base import EEGModuleMixin
   class MyNewModel(EEGModuleMixin, nn.Module):
       # ... (model definition)
   ```

   _Note:_ in the parents list, `EEGModuleMixin` must necessarily be placed before `nn.Module` (or `nn.Sequential`) like
   in the template.

2. **Mandatory parameters:** Some parameters are mandatory and must be provided in every braindecode model.
   These parameters ensure compatibility with other Braindecode components. Passing them to the parent class
   using `super().__init__(...)` is necessary.
    - **The required parameters are**:
      + `n_outputs`: Number of outputs of the model. This is the number of classes in the case of classification.
      + `n_chans`: Number of EEG channels.
      + `chs_info`: Information about each individual EEG channel. This should be filled with
        `info["chs"]`. Refer to `mne.Info` for more details.
      + `n_times`: Number of time samples of the input window.
      + `input_window_seconds`: Length of the input window in seconds.
      + `sfreq`: Sampling frequency of the EEG recordings.
    - The values of these parameters should be passed to the parent class `EEGModuleMixin` with `super().__init__(...)`
      as described in the template:

      ```python
      def __init__(
          self,
          n_outputs=None,
          n_chans=None,
          chs_info=None,
          n_times=None,
          input_window_seconds=None,
          sfreq=None,
        my_parameter=10,  # Add your custom parameters here
      ):
          super().__init__(
              n_outputs=n_outputs,
              n_chans=n_chans,
              chs_info=chs_info,
              n_times=n_times,
              input_window_seconds=input_window_seconds,
              sfreq=sfreq,
          )
      ```

    - **Some notes about the inheritance**:

        - The parameters will be saved as attributes by `EEGModuleMixin`. i.e. for example, you can use `self.n_times` (
          windows size) to
          access its value.
        - The non-provided parameters will be inferred by `EEGModuleMixin` when possible. i.e. for example, if the user
          passes values for `n_times`
          and `sfreq`, then `EEGModuleMixin` will infer the value of `self.input_window_seconds`.
        - No need to add a description for the parameters passed to `EEGModuleMixin`. The docstring of your new model
          will
          automatically be updated.

3. **Define the Model Architecture**: Implement your model's architecture within the `__init__` method of your model
   class. Customize the layers, dimensions, and any other model-specific components as needed.

   ```python
   def __init__(self, ...):
       # ... (model definition)
       self.final_layer = nn.Linear(emb_dim, self.n_outputs)
   ```

4. **Be aware of the braindecode input dimensions:** By default, your model needs to handle the input of
   shape (`batch_size`, `n_chans`, `n_times`).
   For instance, if your model was originally designed for input of shape (`batch_size`, `n_times`, `n_chans`), you must
   use one
   transpose layer to change the input shape inside your forward method:

   ```python
   from einops.layers.torch import Rearrange


   def __init__(self, ...):
       # ... (model definition)
       self.dimshuffle = Rearrange("batch C T -> batch T C")


   def forward(self, X):
       X = self.dimshuffle(X)
       # ... (forward pass implementation)
       return
   ```

   Check how you can do this in the `braindecode.models` module.

5. **Output shape:** Your method needs to return a tensor either of
   shape (`batch_size`, `self.n_outputs`), or, in special cases, of shape (`batch_size`, `self.n_outputs`, `n_out_times`).
   These special cases can be, for example, if your model should compute a temporal segmentation of the signal or in a
   cropped-decoding scenario (c.f. example `plot_bcic_iv_4_ecog_cropped.py`).

7. **Define the final layer:** Naming the final classification layer as `self.final_layer` and ensuring it returns
   outputs of shape (`batch_size`, `self.n_outputs`) is important for compatibility with Braindecode's classification
   process.

8. **Avoid Softmax or Log-Softmax:** Not adding softmax or log-softmax layers after the final layer is a good practice
   in general and mandatory for braindecode models.
   Braindecode's `EEGClassifier` class handles the softmax, which allows compatibility with both classification and
   regression tasks.

Alternatively, you can implement your model as a `nn.Sequential`. All the above notes still hold in that case.
