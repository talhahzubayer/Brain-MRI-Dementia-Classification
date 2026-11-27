# Preprocessing and class imbalance

<!-- Navigation header (not source content). Sections below originate from the README preserved at commit 7895f7e. -->

## Image preprocessing

All models ultimately require 224 x 224 inputs. The original images are therefore resized while preserving their aspect ratio rather than being stretched into a square. The dataset exploration in the [V3 recorded notebook](../notebooks/dementia_mri_classification_coursework_V3_WithCellOutput.ipynb) measured a sampled image as 208 x 176 pixels (height x width; zero-based cell 23). Only that sample was measured, not every file. The notebook introduction (cell 1) gives the same dimensions in the opposite order, labelling 176 x 208 as height x width.

The remaining space is filled with bright magenta padding (`RGB 255, 0, 255`). This was chosen to make the padded region visually distinct from greyscale brain tissue. The code comments and preprocessing figure in V3 (cells 10 and 11) go further and state that the padding prevents the model from learning spurious patterns from the padded regions. That effect was not tested: no experiment compared padding choices, so the magenta padding is an implementation choice with a documented intent rather than a demonstrated safeguard. Images are then converted to tensors and normalised using ImageNet statistics for compatibility with the pretrained ResNet50 and ViT-B/16 weights.

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
