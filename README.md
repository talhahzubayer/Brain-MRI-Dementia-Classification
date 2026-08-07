# Brain MRI Classification for Dementia Severity

An iterative deep learning study using custom convolutional neural networks (CNNs), ResNet50 and a Vision Transformer (ViT-B/16) to classify brain MRI images into three dementia-severity categories. The project was developed and executed as a Google Colab notebook workflow using PyTorch, with the final recorded experiments run on an NVIDIA L4 GPU.

The aim was not simply to train the most complex model available. The project progressed through three notebook versions, with problems found in earlier experiments informing changes to model capacity, medical-image augmentation and transfer-learning strategy. The final ViT-B/16 correctly classified **728 of 762 images in the test split**, giving a **test-set classification accuracy of 95.54%**, compared with **91.99%** for ResNet50 and approximately **47%** for the CNNs trained from scratch.

> **Important:** These results come from an image-level split of the supplied dataset. Patient identifiers were not available, so patient-level independence between training and testing could not be verified. The models are research prototypes developed for an MSc project and are not clinical diagnostic systems.

## Project at a glance

| | |
|---|---|
| Dataset | 5,076 brain MRI images |
| Classes | `NoImpairment`, `MildImpairment`, `ModerateImpairment` |
| Original image size | 176 x 208 pixels |
| Split | 70% training / 15% validation / 15% testing |
| Framework | PyTorch / torchvision |
| Development environment | Google Colab |
| Recorded final-run GPU | NVIDIA L4 |
| Final model | ImageNet-pretrained ViT-B/16 with extended domain adaptation |
| Final test-set classification accuracy | **95.54% (728/762 correct)** |
| Final macro AUC | **0.9947** |

## Dataset

The supplied dataset contained 5,076 MRI images across three classes:

| Class | Images | Proportion |
|---|---:|---:|
| `NoImpairment` | 2,560 | 50.4% |
| `MildImpairment` | 1,792 | 35.3% |
| `ModerateImpairment` | 724 | 14.3% |
| **Total** | **5,076** | **100%** |

This produced a class-imbalance ratio of approximately 3.54:1 between the largest and smallest classes. The training strategy therefore combined stratified splitting, an inverse-frequency weighted cross-entropy loss and weighted random sampling.

The dataset was supplied for the coursework and is not redistributed in this repository.

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

## Image preprocessing

All models ultimately require 224 x 224 inputs. The original 176 x 208 images are therefore resized while preserving their aspect ratio rather than being stretched into a square.

The remaining space is filled with bright magenta padding (`RGB 255, 0, 255`). This was chosen to make the padded region visually distinct from greyscale brain tissue. Images are then converted to tensors and normalised using ImageNet statistics for compatibility with the pretrained ResNet50 and ViT-B/16 weights.

For the final augmentation experiment, the training pipeline avoids rotations and vertical flips because the MRI images have a standard anatomical orientation. The V3 pipeline instead uses translation, scaling, brightness/contrast changes, Gaussian blur and low-probability random erasing, with the first four chosen to better reflect positioning, field-of-view, intensity and acquisition variation than arbitrary image rotation or inversion.

## Handling class imbalance

Class imbalance is addressed at several points in the workflow:

- **Stratified splitting** preserves the original class proportions across training, validation and testing.
- **Weighted cross-entropy** assigns a larger loss contribution to errors on the less common classes.
- **Weighted random sampling** increases the frequency with which minority-class examples are sampled during training.
- Evaluation includes per-class metrics, confusion matrices and macro AUC in addition to overall accuracy.

The inverse-frequency class weights used in the recorded runs were approximately:

| Class | Weight |
|---|---:|
| `MildImpairment` | 0.9442 |
| `ModerateImpairment` | 2.3399 |
| `NoImpairment` | 0.6607 |

## Experimental development

The three notebook versions preserve the development history of the project. They should be read as successive iterations rather than three independent benchmark studies.

### V1: initial implementation

The first version used a custom CNN whose flattened feature representation fed a large fully connected layer. This resulted in **51,475,523 trainable parameters**. It also used generic computer-vision augmentation, including rotations and flips, while the ImageNet-pretrained ResNet50 and ViT backbones were almost entirely frozen.

The 51.5-million-parameter CNN achieved 97.64% test-set classification accuracy, substantially above the transfer-learning models. Rather than treating this result as evidence that the custom architecture was superior, the very large parameter count and image-level nature of the dataset motivated a more constrained baseline design. V1 also showed that simply attaching a new classifier to frozen ImageNet representations was insufficient for this MRI task: frozen ResNet50 and ViT-B/16 achieved only 61.02% and 61.15% accuracy respectively.

### V2: reducing the CNN and adapting pretrained models

V2 replaced the large flattened representation with **Global Average Pooling (GAP)** and expanded the convolutional feature extractor. The resulting baseline contained **1,207,459 parameters**, dramatically reducing the capacity of the final classifier and removing direct dependence on a large flattened spatial representation.

The augmentation strategy was also changed. Rotations and flips were removed in favour of transformations designed to be more appropriate for MRI images.

Most importantly, the pretrained models were allowed to adapt to the target domain:

- ResNet50 kept its earlier feature-extraction layers frozen while `layer3` and `layer4` were fine-tuned.
- ViT-B/16 fine-tuned its final four transformer layers rather than training only the classifier head.
- Lower learning rates were used when updating pretrained representations.

This changed ResNet50 accuracy from 61.02% in V1 to **91.99%** in V2, while ViT-B/16 increased from 61.15% to **87.53%**.

V2 also provides the cleanest augmentation-only comparison in the development history. With the same reduced CNN architecture, the no-augmentation model achieved 46.98% accuracy and the augmented version 46.19%, indicating that augmentation did not resolve the underlying underfitting of the from-scratch model.

### V3: final refinement

V3 retained the reduced CNN baseline but attempted to address its underfitting in the second experiment by changing several factors together: convolutional and classifier dropout were reduced, the classifier head was enlarged, the learning rate was increased and the augmentation pipeline was strengthened. The resulting model achieved 47.90% accuracy. Because several changes were introduced together, this result is **not** treated as an augmentation-only effect.

ResNet50 was unchanged from V2 and reproduced the same 91.99% test-set classification accuracy.

The ViT configuration received the largest final refinement:

- the number of unfrozen transformer layers increased from four to eight;
- the patch-embedding projection was made trainable;
- the class token and positional embeddings were made trainable;
- approximately 67% of the model parameters were trainable; and
- a learning rate of `5e-5` was used for fine-tuning.

The final ViT-B/16 correctly classified **728 of 762 test images**, giving **95.54% test-set classification accuracy**.

## Final V3 results

| Experiment | Accuracy | Weighted F1 | Macro AUC |
|---|---:|---:|---:|
| Baseline CNN | 46.98% | 44.85% | 0.7135 |
| Improved CNN + domain-aware augmentation | 47.90% | 46.57% | 0.7188 |
| ResNet50 + partial fine-tuning | 91.99% | 91.95% | 0.9855 |
| ViT-B/16 + extended domain adaptation | **95.54%** | **95.54%** | **0.9947** |

The accuracy column reports the proportion of correctly classified images in the 762-image test split; it is not an average of the models' per-image confidence scores.

The strongest pattern across the development history is the importance of adapting the pretrained representation to the MRI domain:

| Model progression | Test-set classification accuracy |
|---|---:|
| ResNet50, frozen backbone (V1) | 61.02% |
| ResNet50, partial fine-tuning (V2/V3) | **91.99%** |
| ViT-B/16, frozen backbone (V1) | 61.15% |
| ViT-B/16, final 4 transformer layers fine-tuned (V2) | 87.53% |
| ViT-B/16, 8 layers + embedding adaptation (V3) | **95.54%** |

These version-to-version results document an iterative development process rather than a controlled ablation study. Some configurations changed more than one training decision at a time, so differences should not automatically be interpreted as the causal effect of a single component.

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

## Repository contents

The repository preserves both the clean notebooks and their original executed counterparts with cell output. The executed versions are useful because they retain the training logs, model parameter counts, metrics and figures from the recorded experiments.

```text
.
|-- README.md
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

## Limitations

- **Image-level rather than patient-level splitting.** Patient identifiers were not available. If multiple images originated from the same patient, patient information could potentially be represented across different splits.
- **Single split.** Results come from one deterministic train/validation/test split rather than cross-validation.
- **The same test split was revisited during development.** V1, V2 and V3 evaluate the same deterministic image-level test population when the dataset ordering is unchanged. The test images are not used for gradient updates or early stopping within an experiment, but repeated evaluation of this population during iterative development means the final result should not be treated as performance on a one-time, untouched external benchmark.
- **No external validation.** The final model was not evaluated on an independent dataset from another source, scanner or clinical site.
- **No subgroup analysis.** Performance was not established independently across demographic groups or scanner types.
- **ImageNet pretraining.** ResNet50 and ViT-B/16 used natural-image pretraining rather than weights developed specifically for medical imaging.
- **Experimental comparisons are not all single-variable ablations.** In particular, the V3 second CNN experiment changes regularisation, classifier capacity, learning rate and augmentation together.
- **No confidence calibration analysis.** Softmax scores shown as prediction confidence were not evaluated for calibration and should not be interpreted as calibrated probabilities of correctness.
- **No clinical validation.** High test-set classification accuracy should not be interpreted as evidence that the model is suitable for autonomous diagnosis or clinical decision-making.

## What I learned from the project

The most useful outcome of the project was the iterative development process rather than the final accuracy alone:

- A very large parameter count can produce an impressive result without providing a convincing model design.
- Global Average Pooling can remove the parameter explosion introduced by flattening spatial feature maps into a large fully connected layer.
- Data augmentation should reflect the domain; transformations that are common in natural-image classification can be inappropriate for medical images.
- Augmentation is not a substitute for sufficient model capacity when both training and validation performance indicate underfitting.
- ImageNet-pretrained models can require substantial fine-tuning before their representations transfer effectively to greyscale MRI images.
- Class imbalance needs to be considered in data splitting, optimisation and evaluation rather than addressed through accuracy alone.
- Aggregate test accuracy and per-image softmax confidence answer different questions; a highly confident prediction can still be wrong.
- Holding test images out of training and early stopping is important, but repeatedly evaluating the same test population during iterative development reduces its value as a completely untouched final benchmark.
- A held-out test result is only as meaningful as the independence assumptions behind the split, which is why the absence of patient identifiers remains an important limitation.

## Project context

This project was originally developed as coursework for the AI and Machine Learning component of the **MSc Applied Bioinformatics at King's College London**. This public repository preserves the original notebook development history and recorded outputs while documenting the workflow, results and methodological limitations more explicitly.
