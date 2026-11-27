# Limitations and lessons

<!-- Navigation header (not source content). Sections below originate from the README preserved at commit 7895f7e. -->

## Limitations

- **Image-level rather than patient-level splitting.** Patient identifiers were not available. If multiple images originated from the same patient, patient information could potentially be represented across different splits.
- **Single split.** Results come from one deterministic train/validation/test split rather than cross-validation.
- **The same split design was revisited during development.** V1, V2 and V3 regenerate the same deterministic image-level split, so their test populations are identical only if the dataset ordering was unchanged; this cannot be verified file by file because the original split files are not in the repository. The test images are not used for gradient updates or early stopping within an experiment, but repeated evaluation under this split design during iterative development means the final result should not be treated as performance on a one-time, untouched external benchmark.
- **No external validation.** The final model was not evaluated on an independent dataset from another source, scanner or clinical site.
- **No subgroup analysis.** Performance was not established independently across demographic groups or scanner types.
- **ImageNet pretraining.** ResNet50 and ViT-B/16 used natural-image pretraining rather than weights developed specifically for medical imaging.
- **Experimental comparisons are not all single-variable ablations.** In particular, the V3 second CNN experiment changes regularisation, classifier capacity, learning rate and augmentation together.
- **No confidence calibration analysis.** Softmax scores shown as prediction confidence were not evaluated for calibration and should not be interpreted as calibrated probabilities of correctness.
- **No clinical validation.** High test-set classification accuracy should not be interpreted as evidence that the model is suitable for autonomous diagnosis or clinical decision-making.

## What I learned from the project

In retrospect, the most useful outcome of the project was the iterative development process rather than the final accuracy alone:

- A very large parameter count can produce an impressive result without providing a convincing model design.
- Global Average Pooling can remove the parameter explosion introduced by flattening spatial feature maps into a large fully connected layer.
- Data augmentation should reflect the domain; transformations that are common in natural-image classification can be inappropriate for medical images.
- Augmentation alone did not resolve the weak training and validation performance observed in the redesigned CNN experiments.
- I began by assuming that sophisticated pretrained architectures would outperform the custom CNN while training only new classification heads. The V1 results did not support that expectation: frozen-backbone ResNet50 and ViT-B/16 both scored below the first CNN. Their recorded accuracies were substantially higher in subsequent configurations that also trained parts of their feature extractors. Architectural sophistication alone was therefore not a reliable guide to performance under the tested conditions, although the comparisons do not isolate the effects of the individual changes.
- Class imbalance needs to be considered in data splitting, optimisation and evaluation rather than addressed through accuracy alone.
- Aggregate test accuracy and per-image softmax confidence answer different questions; a highly confident prediction can still be wrong.
- Holding test images out of training and early stopping is important, but repeatedly evaluating test images from the same split design during iterative development reduces their value as a completely untouched final benchmark.
- A held-out test result is only as meaningful as the independence assumptions behind the split, which is why the absence of patient identifiers remains an important limitation.
