# Limitations and lessons

<!-- Navigation header (not source content). Sections below are copied verbatim from README.md. -->

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
