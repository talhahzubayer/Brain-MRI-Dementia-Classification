# Workflow and reproducibility

<!-- Navigation header (not source content). Sections below originate from the README preserved at commit 7895f7e. -->

## Google Colab workflow

This project was developed as a notebook-based Google Colab experiment rather than a standalone Python package. The notebooks are the primary implementation.

My original workflow was:

1. Prepare the supplied dataset locally so that every image had an unambiguous filename.
2. Upload the prepared `AD` dataset directory to Google Drive.
3. Open the experiment notebook in Google Colab.
4. Mount Google Drive in the Colab runtime.
5. Point `DATA_DIR` at the prepared dataset.
6. Run the notebook sequentially using a GPU runtime.
7. Allow the notebook to create the stratified data split, train each experiment, evaluate the models and generate the timestamped result directories.

The final recorded run used an **NVIDIA L4 GPU**. The notebooks contain the Google Drive path used during the original run, but this path is not a requirement: `DATA_DIR` can be changed to wherever the prepared dataset is accessible from Colab.

## Repository contents

The repository preserves both the clean notebooks and their original executed counterparts with cell output. The executed versions are useful because they retain the training logs, model parameter counts, metrics and figures from the recorded experiments.

```text
.
|-- README.md
|-- docs/
|   |-- experimental-development.md
|   |-- limitations-and-lessons.md
|   |-- preprocessing-and-class-imbalance.md
|   |-- project-overview.md
|   |-- testing-and-evaluation.md
|   `-- workflow-and-reproducibility.md
|-- filename_change.py
`-- notebooks/
    |-- dementia_mri_classification_coursework_V1.ipynb
    |-- dementia_mri_classification_coursework_V1_WithCellOutput.ipynb
    |-- dementia_mri_classification_coursework_V2.ipynb
    |-- dementia_mri_classification_coursework_V2_WithCellOutput.ipynb
    |-- dementia_mri_classification_coursework_V3.ipynb
    `-- dementia_mri_classification_coursework_V3_WithCellOutput.ipynb
```

### Which notebook should I open?

- **V3** is the final implementation and the best starting point for reading or rerunning the project.
- **V3 WithCellOutput** contains the final recorded training run and its generated results.
- **V1 and V2** preserve the earlier configurations and make it possible to follow how the approach changed in response to earlier results.

The repository intentionally preserves this notebook history rather than presenting the final model as if it had been designed in a single step.

## Running the final notebook

The intended way to run the project is in Google Colab.

1. Make the prepared three-class dataset accessible to the Colab runtime, for example through Google Drive.
2. Open `dementia_mri_classification_coursework_V3.ipynb` in Google Colab.
3. Mount Google Drive if that is where the dataset is stored.
4. Change `DATA_DIR` to the location of your prepared dataset.
5. Select a GPU runtime. The final recorded experiment used an NVIDIA L4.
6. Run the notebook sequentially.
7. Inspect the generated timestamped output directory for the saved split information, trained model weights and evaluation figures.

The random seeds and cuDNN deterministic settings are configured in the notebook to improve reproducibility. However, Google Colab software and GPU environments change over time, so a future rerun is not guaranteed to reproduce every recorded value bit-for-bit. The executed notebooks preserve the results of the original runs.
