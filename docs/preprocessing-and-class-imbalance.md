# Preprocessing and class imbalance

<!-- Navigation header (not source content). Sections below are copied verbatim from README.md. -->

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
