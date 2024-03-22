# dplPy Developer Instructions (in progress)

Welcome to the dplPy developer manual. We welcome all code contributions, bug reports, bug fixes, documentation improvements, and suggestions.

## Index
- [Environment setup](#environment-setup)
- [Making changes and submitting PRs](#making-changes-and-submitting-a-pull-request)
- [API Reference]()


## Environment setup
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

First, ensure that the main branch of your fork is up-to-date with the main dplpy repository:

```
git checkout main
git pull upstream main --ff-only
```

Then, create a new branch for making your changes, with the command below (Replace {feature_name} with the name of your feature or preferred branch name).

```
git checkout -b {feature_name}
```

### 2. Conda environment

The packages required to run dplPy are all specified in environment.yml, which can be used to install them in Conda ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)) or [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) environments.

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
You will need to have the conda environment activated anytime you want to run or test code from the package.

```
conda activate dplpy
```

After running this command, you should see (dplpy) on the left of each new line in the terminal.

#### 2.3\. Run unit and integration tests to ensure that installation was successful.

With the conda environment activated, run `pytest` in the project's home directory to run all tests. 

To run only the unit tests, specify to run tests in the unit tests folder as shown below

```
pytest tests/unit --cov=dplpy
```
Note: The --cov option displays coverage information for all code files

To run the integration tests, run `pytest` while specifying that only tests from the integ tests folder should be run. Run pytest with -r set to A to include a summary of all tests run at the end, which can be useful for quickly pointing out issues for further analysis.


```
pytest tests/integs -rA
```

### 3. IDE setup

We recommend using VSCode for development. The following instructions show how to set up VSCode to recognize the conda environment and debug tests.

#### 3.1\. Open the dplpy folder in VScode
In VSCode, open the folder containing your local dplpy repository. If you followed the instructions above, this should be a folder named `dplpy-{your-github-username}`. Then, open the file `src/dplpy.py`.

#### 3.2\. Change the python interpreter to use the conda environment's interpreter
In the bottom corner of your IDE display, select the language interpreter.

Choose the interpreter `Python 3.x ('dplpy')`, with a path that ends with `/envs/dplpy/python`.

Now you should be able to run any python files within the currently open folder with the run button in VSCode, instead of running them through the terminal. 

Note: If the terminal is opened after the interpreter has been set to use the conda environment, conda activate dplpy will automatically be run and does not need to be run again.

#### 3.3\. Set up testing tools

Go to the testing tab (on the left side of the VSCode display). With your environment set. If the tests are not automatically discovered, open `.vscode/settings.json` and add the following lines inside the curly braces, so that your file looks like this:

```
{
    // any pre-existing configurations (DO NOT ADD THIS, THIS REPRESENTS ANYTHING ALREADY IN THE FILE)

    "python.testing.pytestArgs": [
        "./tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

If `.vscode/settings.json` has not been created, create it and add the lines shown above.

Go back to the testing tab and verify that the dplpy unit tests are showing.

Run the tests by clicking the play button to the right of `tests`.

#### 3.4\. Updating dev environment with changes from main dplpy repository
It is important to frequently update your local `main` with any changes to the remote dplpy `main` repository. To do this, run the following commands:

```
git checkout main
git merge upstream/main
```


## Making Changes and Submitting a Pull Request
### 1. Commit and Push Changes to Feature Branch
#### 1.1 View and stage changes
Run `git status` in the terminal to view all your uncommitted changes.

Stage changes for commit by running `git add` as shown below:

```
# to add all files in current directory
git add .

# to add specific folder/files from current directory
git add {folder_name}
```

#### 1.2 Commit and push changes
Note: Make sure to test your changes before commiting them.

When you're ready, commit your staged changes as shown below. You must add a commit message.
```
git commit -m "Your commit message goes here"
```

You can push your commits to your forked repository's remote branch by running

```
git push origin {branch_name}
```

Your code is now on GitHub, but for it to be part of the main dplpy repository, you need to create a pull request. 

Note: Before creating a pull request, wait for your changes to be tested by our workflow that runs unit and integ tests in all supported versions of python. You can check on the status of the all workflows by navigating to the Github page of your remote branch, and clicking the status indicator to the right of the most recent commit message. This shows a check mark if all workflows ran successfully, an X if any of them failed, and an orange dot if workflows are still running.

### 2. Creating and Submitting a Pull Request

#### 2.1 Create pull request
Navigate to your forked dplpy repository on Github, and in the pull requests tab, create a new pull request.

Note: If you recently pushed changes to a branch other than main, you will sometimes be shown the option to immediately compare and make a pull request to main. This works too.

#### 2.2 Review Changes and Submit PR
Pull requests allow you to view a side-by-side diff comparison of all changed files. Review all changes to ensure correctness.

If you are satisfied with your changes, give the PR a descriptive title, and specify in the description what changes were made and what (if any) issues were addressed. Then, submit the pull request.

Your request will be reviewed by the repository maintainers.

## API Reference

Here is a list of functions (in alphabetical order) with descriptions:

| Function | Description |
| --- | --- |
| [`ar_func`](#ar_funcdata-max_lag5-source) | Fits series or dataframe to autoregressive (AR) models and performs other operations on data with best model fit. |
| [`autoreg`](#autoregdata-max_lag5-source) | Fits series to autoregressive (AR) models and returns parameters of best model fit. |
| [`chron`](#chronrwi_data-biweighttrue-prewhitenfalse-plottrue-source) | Creates a mean value chronology for a dataset, typically the ring width indices of a detrended series |
| [`detrend`](#detrend) | Detrends a given series or data frame, first by fitting data to curve(s), with spline(s) as the default, and then by calculating residuals or differences compared to the original data. |
| [`help`](#help) | Displays help (alpha). |
| [`plot`](#plot) | Generates line, spaghetti or segment plots.|
| [`rbar`](#rbar) | Finds best interval of overlapping series over a  period of years, and calculating rbar constant for a dataset over period of overlap. |
| [`readers`](#readers) | Reads data from supported file types (*.CSV and *.RWL) and stores them in dataframe. |
| [`readme`](#readme) | Goes to this website. |
| [`report`](#report) | Generates a report about absent rings in the data set. |
| [`series_corr`](#series_corr) |  Crossdating function that focuses on the comparison of one series to the master chronology. |
| [`stats`](#stats) | Generates summary statistics for RWL and CSV format files. |
| [`summary`](#summary) | Generates a summary for RWL and CSV format files. |
| [`xdate`](#xdate) | Crossdating function for dplPy loaded datasets. |


### `ar_func(data, max_lag=5)` [[source]](https://github.com/OpenDendro/dplPy/blob/480973dc5f09f748271fb62a5ebd8ff5c88ac2dd/dplpy/autoreg.py#L36)

Fits a given data to an the best-fit autoregressive model, then returns the residuals of AR fit relative to the original data + the mean of the original data.
- **Required Parameters**:
    - **data**  :   ***pandas.DataFrame or pandas.Series***, a pandas dataframe imported from dpl.readers() or a series extracted from such a dataframe.
- **Optional Parameters**:
    - **lag   :   _int_ default 5**, max lag to consider when selecting the best-fit AR model.
- **Returns:**
    -  **pandas.DataFrame or pandas.Series**, dataframe or series of AR-modeled data, depending on which was given as input.
- **Usage Examples:**
    ```
    # ar_func with series
    >>> dpl.ar_func(ca533["CAM191"], 10)
    Year
    1190    0.711307
    1191   -0.232047
    1192    0.521210
    1193    0.575975
    1194    0.901084
            ...   
    1966    0.296554
    1967    0.384609
    1968    0.397742
    1969    0.427618
    1970    0.383847
    Name: CAM191, Length: 781, dtype: float64

    # ar_func with dataframe
    >>> dpl.ar_func(ca533, 10)
            CAM011    CAM021    CAM031    CAM032    CAM041    CAM042    CAM051    CAM061    CAM062  ...  CAM152  CAM161  CAM162  CAM171  CAM172  CAM181  CAM191  CAM201  CAM211
    Year                                                                                            ...                                                                        
    626        NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    627        NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    628        NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    629        NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    630        NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    ...        ...       ...       ...       ...       ...       ...       ...       ...       ...  ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
    1979  0.424423  0.404787  0.142900  0.378733  0.640022  0.369773  0.369770  0.347996  0.535881  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    1980  0.486215  0.614051  0.658424  0.408298  0.898555  0.568861  0.440974  0.693782  0.661847  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    1981  0.498586  0.505126  0.436690  0.260786  0.419491  0.438934  0.345517  0.544592  0.382856  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    1982  0.455773  0.414212  0.485516  0.448526  0.792929  0.443559  0.261443  0.560291  0.510274  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
    1983  0.666578  0.520679  0.223995  0.277267  0.755711  0.456165  0.252873  0.583766  0.320921  ...     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN

    [1358 rows x 34 columns]
    ```


### `autoreg(data, max_lag=5)` [[source]](https://github.com/OpenDendro/dplPy/blob/480973dc5f09f748271fb62a5ebd8ff5c88ac2dd/dplpy/autoreg.py#L103)

Selects the best AR model with a specified maximum order for the given data, and returns the parameters for the model. The best model is selected based on the AIC value.
- **Required Parameters**:
    - **data**  :   ***pandas.Series***, a pandas series extracted from a pandas dataframe containing tree ring widths (presumably imported from [`readers`]())
- **Optional Parameters**:
    - **lag   :   _int_ default 5**, max lag to consider when selecting the best-fit AR model.
- **Returns:**
    - **array** containing the parameters of best-fit AR model in order.
- **Usage Examples:**
    ```
    >>> dpl.autoreg(ca533["CAM191"], 10)
    const         0.022210
    CAM191.L1     0.503373
    CAM191.L2     0.087230
    CAM191.L3     0.143716
    CAM191.L4     0.020119
    CAM191.L5    -0.027769
    CAM191.L6    -0.010029
    CAM191.L7     0.001373
    CAM191.L8     0.025588
    CAM191.L9     0.042340
    CAM191.L10    0.136916
    dtype: float64
    ```

### `chron(rwi_data, biweight=True, prewhiten=False, plot=True)` [[source]](https://github.com/OpenDendro/dplPy/blob/480973dc5f09f748271fb62a5ebd8ff5c88ac2dd/dplpy/chron.py#L44)

Creates a mean value chronology for a dataset, typically the ring width indices of **a detrended series**.

- **Required Parameters**:
    - **rwi_data**  :   ***pandas.Dataframe***, a pandas dataframe containing (expected to be already detrended) tree ring widths.
- **Optional Parameters**:
    - **biweight   :   _int_ default True**; when `True`, means will be calculated using Tukey's biweight robust mean.
    - **prewhiten   :   _int_ default False**; when `True`, data is prewhitened by fitting to an AR model.
    - **plot   :   _int_ default True**; when `True`, results are plotted.
- **Returns:**
    - **pandas.Dataframe** of years, mean RWIs and sample depths for each year.
- **Usage Examples:**
    ```
    >>> dpl.chron(ca533, plot=False)
          Mean RWI  Sample depth
    Year                        
    626   0.170000             1
    627   0.130000             1
    628   0.140000             1
    629   0.190000             1
    630   0.220000             1
    ...        ...           ...
    1979  0.510581            21
    1980  0.722784            21
    1981  0.568495            21
    1982  0.674211            21
    1983  0.638166            21

    [1358 rows x 2 columns]
    ```






### Loading data using  `readers`

- Description: reads data from supported file types (`csv` and `rwl`) and stores them in a dataframe.
- Options: 
    - `header`: rwl input files often have a header present; Default is `False`, use `True` if input has a header.
- Usage examples:
    ```
    >>> data = dpl.readers("/path/to/file.csv")
    # or
    >>> data = dpl.readers("/path/to/file.rwl", header=True)
    ```

### Loading data from online sources using  `readers_url`
**Note: This function is still in development and has only been tested so far with `rwl` raw data files from the [NCEI website](https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/)**

- Description: reads `rwl` formatted data directly from online sources.
- Options: 
    - `header`: rwl input files often have a header present; Default is `False`, use `True` if input has a header.
- Usage examples:
    ```
    >>> data = dpl.readers_url("http://link/to/file.rwl")
    >>> data = dpl.readers_url("http://link/to/file.rwl", header=True)
    ```

### Data Summary from `summary`

- Description: generates a summary of each series recorded in `rwl`  and `csv` format files
- Usage examples:
    ```
    >>> dpl.summary("/path/to/file.rwl")
    # or
    >>> dpl.summary(data)
    ```

### Data Stastics from `stats`

- Description: generates summary statistics for `rwl`  and `csv` format files
- Usage Example:
    ```
    >>> dpl.stats("/path/to/file.rwl")
    # or
    >>> dpl.stats(data)
    ```

### Data Report from `report`

- Description: generates a report about ring measurements and absent rings in the data set
- Usage Example:
    ```
    >>> dpl.report("/path/to/file.rwl")
    # or
    >>> dpl.report(data)
    ```

### Plotting raw data with `plot`

- Description: generates plots of tree ring with data from dataframes. Currently capable of generating `line`, `spag` (spaghetti) and `seg` (segment, default) plots.
- Options:
    - `type="line"`: creates a line plot (default)
    - `type="spag"`: creates a spaghetti plot
    - `type="seg"`: creates a segment plot
- Usage Example:
    ```
    >>> dpl.report("/path/to/file.rwl")
    # or 
    >>> dpl.plot(data)

    # User is able to select specific series of interests.
    # In the example below, the user selects SERIES_1, SERIES_2, SERIES_3 
    # from the "data" dataset and generates a spaghetti plot
    >>> dpl.plot(data[[SERIES_1, SERIES_2, SERIES_3]], type="spag")
    ```

### Detrending using `detrend`
 
- Description: Detrends a given series or data frame, first by fitting data to curve(s), and then by calculating residuals or differences compared to the original data.
- Options:
    - `fit="spline"`: default detrending method.
    - `fit="ModNegEx"`: detrending using negative exponent method.
    - `fit="Hugershoff"`: detrending using the Hugenshoff method.
    - `fit="linear"`: detrending using the linear method.
    - `fit="horizontal"`: detrending using the horizontal method.
    - `method="residual"`: calculates residuals vs original data (default).
    - `method="difference"`: calculates differences vs original data.
    - `plot=True|False`: whether or not to plot results, default is `True`.
- Usage Example:
    ```
    # detrend with default options
    >>> dpl.detrend(data)
    
    # specify fit to hugershoff curve and detrend with difference
    >>> dpl.detrend(data, fit="Hugershoff", method="difference")

    # detrend only SERIES_1, SERIES_2 and SERIES_3
    >>> dpl.detrend(data[[SERIES_1, SERIES_2, SERIES_3]], fit="Hugershoff", method="difference")
    ```






### Build a variance stabilized chronology with `chron_stabilized`

- Description: Builds a variance stabilized mean-value chronology for a dataset of **detrended** ring width indices, by multiplying the chronology with the square root of the effective independent sample size, $ Neff $.

    Note: where n(t) is the number of series at time t, and rbar is the running interseries correlation, 

    $$ Neff = { n(t) \over 1+(n(t)-1)rbar(t) } $$

- Options:
    - `win_length`: an integer for specifying the window lengths where interseries correlations will be calculated (default `50`). Should not be greater than the number of years in the dataset, recommended to be between 30% and 50% of the number of years.
    - `min_seg_ratio`: the minimum ratio of non-NA values to the window length for a series to be considered in an Neff calculation (default `0.33`).
    - `biweight`: boolean indicating whether or not to use Tukey's bi-weight robust mean when calculating the mean-value chronology; default `True`.
    - `running_rbar`: boolean indicating whether or not to return the running interseries correlations as part of chronology output; default `False`.
- Usage Example:
    ```
    # Detrend data first!
    >>> rwi_data = dpl.detrend(data)

    # Perform chronology with default args
    >>> dpl.chron_stabilized(rwi_data)

    # Specify win_length, min_seg_ratio and running_rbar
    >>> dpl.chron_stabilized(rwi_data, win_length=60, min_seg_ratio=0.5, running_rbar=True)
    ```

### Crossdate with `xdate`
- Description: This function calculates correlation serially between each tree-ring series and a master chronology built from all the other series in the dataset (leave-one-out principle).
- Options:
    - `prewhiten`: default `True`, determines whether or not to prewhiten series using AR modeling
    - `corr`: default `'Spearman'`, the type of correlation to use. Can be `'Pearson'` or `'Spearman'`.
    - `slide_period`: default `50`, the number of years to compare to the master chronology at a time.
    - `bin_floor`: default `100`, determines the minimum bin year. The minimum bin year is calculated as $ \lceil (min\_yr/bin\_floor)\rceil*bin.floor $ where `min_yr` is the first year in the dataset.
    - `p_val`: default `0.05`, determines the critical value below which interseries correlations are flagged.
    - `show_flags`: default `True`, determines whether to show flags in the function output to the console.
- Usage examples:
    ```
    >>> ca533_rwi = dpl.detrend(ca533, plot=False)

    # Crossdating of detrended data with default args
    >>> dpl.xdate(ca533_rwi)

    # Crossdating with Pearson correlation and show flags 
    # (other options set to defaults when not specified).
    >>> dpl.xdate(ca533_rwi, corr="Pearson" show_flags=True)
    ```

### Output data to files using  `writers`

- Description: writes data from dataframe to supported file types (`csv`, `rwl`, `crn`, `txt`).
- Required parameters: 
    - `data`: dataframe with ring widths (presumably one read from `readers` or `readers_url`)
    - `label`: name (can include file path) to give to the created file. **should not include file extension**
    - `format`: extension for file to be created. Can be `'csv'`, `'rwl'`, `'crn'` or `'txt'`.

- Usage examples:
    ```
    # Write data to file_name.csv in current working directory.
    >>> dpl.writers(data, "file_name", "csv")

    # Write data to file_name.csv in ./path/to/ directory.
    >>> dpl.writers(data, "./path/to/file_name", "csv")
    ```