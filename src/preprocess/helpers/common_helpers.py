"""Collection of common preprocessing helpers"""


def check_for_nans(ds):
    """Checks for each variable in data set if it contains nan values and raises Error if it does so"""

    # checks for nan values
    for var in ds.data_vars:
        contains_nans = ds[var].isnull().any()
        if contains_nans:
            raise ValueError("{} contains nan values".format(var))

    print("No nan values detected in this dataset")
