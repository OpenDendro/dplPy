# dplPy Developer Instructions (in progress)

Welcome to the dplPy developer manual.

## Environment setup
To contribute to dplpy, you will need to set up some tools

### 1. GitHub setup

#### 1.1 Create dplPy fork in github

You will need your own copy of dplpy to work on the code. Go to the dplPy github page and click the fork button. Make sure the option to copy only the main branch is unchecked.


#### 1.2 Create local repository
In your local terminal, clone the fork to your computer using the commands shown below. Replace {your-user} with your github username.
```
$ git clone https://github.com/{your-user}/OpenDendro/dplPy.git dplpy-{your-user}
$ cd dplpy-{your-user}
git remote add upstream https://github.com/OpenDendro/dplPy.git
git fetch upstream
```

This creates a github repository in dplPy-{your-user} on your computer and connects the repository to your fork, which is now connected to the main dplPy repository.

#### 1.3 Create feature branch

TBC


### 2. Conda environment

The packages required to run dplPy are all specified in environment.yml. 

#### 2.1\. Create your environment with the required packages installed.

If you're using conda, run

```
$ conda env create -f environment.yml 
```

If you're using mamba, run

```
$ mamba env create -f environment.yml
```

If prompted for permission to install requred packages, select y.

#### 2.2\. Activate your environment. 
You will need to have the conda environment activated anytime you want to test code from the package.

```
conda activate dplpy
```

After running this command, you should see (dplpy) on the left of each new line in the terminal.

#### 2.3\. Run unit and integration tests to ensure that installation was successful.
TBA: Instructions for running tests

### 3. IDE setup

We recommend using VSCode for development. The following instructions show how to set up VSCode to recognize the conda environment and debug tests.

#### 3.1\. Open the dplpy folder in VScode
In VSCode, open the folder containing your local dplpy repository. If you followed the instructions above, this should be a folder named `dplpy-{your-github-username}`. Then, open the file `src/dplpy.py`.

#### 3.2\. Change the python interpreter to use the conda environment's interpreter
In the bottom corner of your IDE display, select the language interpreter.

Choose the interpreter `Python 3.x ('dplpy')`, with a path that ends with `/envs/dplpy/python`.

Now you should be able to run any python files within the currently open folder with the run button in VSCode, instead of running them through the terminal. 

Note: If the terminal is opened after the interpreter has been set to use the conda environment, conda activate dplpy will automatically be run and does not need to be run again.

#### 3.3\. Set up unit testing tools

Go to the testing tab (on the left side of the VSCode display). With your environment set. If the tests are not automatically discovered, open `.vscode/settings.json` and add the following lines inside the curly braces, so that your file looks like this:

```
{
    // any pre-existing configurations (DO NOT ADD THIS, THIS REPRESENTS ANYTHING ALREADY IN THE FILE)

    "python.testing.pytestArgs": [
        "./src/unittests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

If `.vscode/settings.json` has not been created, create it and add the lines shown above.

Go back to the testing tab and verify that the dplpy unit tests are showing. They should look like this:

TBA: Image


Run the tests by clicking the play button on src.


## Overview of dplPy functions

