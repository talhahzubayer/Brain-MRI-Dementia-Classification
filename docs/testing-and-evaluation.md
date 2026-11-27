# Testing and evaluation

<!-- Navigation header (not source content). Sections below are copied verbatim from README.md. -->

## Held-out testing strategy and why the filenames were changed

The testing workflow was a deliberate part of the project design and was also the reason for the separate filename-preparation step.

### The original filename problem

The supplied dataset already had the required class-directory structure:

```text
AD/
|-- MildImpairment/
|-- ModerateImpairment/
`-- NoImpairment/
```

However, the images inside different class directories reused the same filename sequences. A basename such as `image10.jpg` could therefore refer to different images depending on the class directory.

I wanted the evaluation workflow to do more than calculate aggregate metrics. I also wanted to retain an auditable record of the held-out test population and be able to select individual test images by filename without ambiguity. A duplicated basename across several classes made that inconvenient and potentially error-prone.

Before moving the dataset to Google Drive, I therefore used `filename_change.py` to prefix each image with its class name. For example:

```text
NoImpairment/image10.jpg
    -> NoImpairment/NoImpairment_image10.jpg

MildImpairment/image10.jpg
    -> MildImpairment/MildImpairment_image10.jpg

ModerateImpairment/image10.jpg
    -> ModerateImpairment/ModerateImpairment_image10.jpg
```

The class directories themselves were not changed. The purpose was to give every image a class-identifiable, unambiguous basename before the experiments were run.

The renaming is not what mathematically separates the training and test populations. That separation is performed using dataset indices. The filename change instead supports the surrounding audit and single-image testing workflow, where referring to an image by basename needs to identify one image unambiguously.

### Creating the split

The prepared dataset is loaded once with `torchvision.datasets.ImageFolder`. The notebook does **not** physically copy the images into separate `train/`, `validation/` and `test/` directories. Instead, it creates stratified subsets from the complete dataset using indices.

Splitting is performed in two stages with `random_state=42`:

1. 15% of the complete dataset is reserved for testing.
2. The remaining population is split again to produce a validation set equal to 15% of the original dataset.
3. The remaining 70% is used for training.

This produces:

| Population | Images | Proportion |
|---|---:|---:|
| Training | 3,552 | 70.0% |
| Validation | 762 | 15.0% |
| Test | 762 | 15.0% |

The held-out test population contains 269 `MildImpairment`, 109 `ModerateImpairment` and 384 `NoImpairment` images, preserving the overall class distribution.

The same deterministic splitting function is used by the experiments, so the model comparisons use the same image-level train, validation and test populations when the dataset ordering is unchanged.

### Recording the split

For the recorded split, the notebook saves the indices and test paths:

```text
data_splits/
|-- train_indices.npy
|-- val_indices.npy
|-- test_indices.npy
`-- test_files.txt
```

`test_files.txt` provides a human-readable record of which files belong to the held-out test population. If the split-saving cell is rerun in the same output directory, the previous `data_splits` directory is replaced so that stale split files are not retained alongside a newly generated set.

### Keeping training and testing behaviour separate

Training uses the weighted random sampler to expose the models more frequently to minority-class images. Validation and testing do not use this sampling strategy and their data loaders are not shuffled.

Augmentation is also restricted to the experiment that explicitly uses it for training. Validation and test images receive only the common deterministic preprocessing used for evaluation.

The three populations also have different roles during model development:

- the **training set** is the population used to update model weights;
- the **validation set** is evaluated during training and its validation loss is used for early stopping and selection of the restored model state; and
- the **test set** is not passed to the training or early-stopping routine. It is evaluated after training to measure classification performance.

In V3, the final ViT is evaluated using `test_loader_experiment4`, which is constructed from the stored `test_indices`. Evaluation switches the model to `eval()` mode and runs inference under `torch.no_grad()`. The predicted class for each image is obtained from the model outputs and compared with its ground-truth class label. Overall accuracy is then calculated with `accuracy_score(y_true, y_pred)`.

### What the 95.54% result means

The reported **95.54% is classification accuracy on the 762-image test split**. It is the proportion of test images whose predicted class matched the ground-truth label. For the final ViT-B/16, the confusion matrix contains 728 correct classifications on its diagonal, so:

```text
728 correct predictions / 762 test images = 95.54% test-set accuracy
```

This is different from the model's per-image **confidence** score. The notebook applies `softmax` to model outputs to obtain class scores used in probability-based evaluation and prediction displays. The largest softmax score may be displayed as the model's confidence for a particular image, but it is **not** used to calculate accuracy. A prediction can have a high softmax score and still be assigned to the wrong class.

No calibration analysis was performed in this coursework. A displayed value such as 94% confidence should therefore not be interpreted as a demonstrated 94% probability that the prediction is correct.

### Single-image testing

Each experiment also includes a single-image prediction demonstration. These examples are not selected arbitrarily from the complete dataset. The helper functions select from the stored `GLOBAL_TEST_SET_INDICES`, meaning the selected image belongs to the same held-out test population used for aggregate evaluation.

V3 supports either:

- selection of a random image from the complete test population; or
- selection of a random test image belonging to a requested class.

This is why the filename preparation, saved test indices, `test_files.txt` and single-image testing functions should be understood as parts of the same testing strategy.

### Boundary of the testing strategy

The strategy guarantees separation at the **individual-image level**: an image assigned to the held-out test indices is not intentionally used as a training image.

It does **not** establish patient-level independence. Patient identifiers were not available, so if several different images originated from the same patient, images from that patient could potentially occur in more than one population. This is an important limitation of the reported performance.

## Evaluation outputs

Each experiment evaluates the held-out test set and generates a consistent collection of outputs, including:

- overall and per-class classification metrics;
- confusion matrices;
- one-vs-rest ROC curves and macro AUC;
- precision-recall curves;
- training and validation histories;
- prediction grids with per-image softmax scores displayed as model confidence; and
- individual predictions drawn from the held-out test population.

Results are organised into timestamped directories such as:

```text
Output_DD_MM_YYYY__HHMM/
|-- exp1/
|-- exp2/
|-- exp3/
|-- exp4/
|-- data_splits/
`-- Experiment_Comparison_DD_MM_YYYY__HHMM.png
```

The naming convention was used to keep different Colab runs and the outputs of the four experiments identifiable rather than mixing generated models and figures together.
