# Data Loader

This module contains functions to load and preprocess data from `.npz` files.

## Functions

### `load_data`

```python
def load_data(input_files):
    """
    Load and preprocess data from a list of input .npz files.

    Parameters:
    - input_files (list of Path): List of paths to .npz files to be loaded.

    Returns:
    - list of np.ndarray: List of loaded and preprocessed arrays.
    """
    # Function implementation