# Experimental development

<!-- Navigation header (not source content). Sections below originate from the README preserved at commit 7895f7e. -->

## Experimental development

The three notebook versions preserve the development history of the project. They should be read as successive iterations rather than three independent benchmark studies.

### V1: initial implementation

The first version used a custom CNN whose flattened feature representation fed a large fully connected layer. This resulted in **51,475,523 trainable parameters**. Its augmented CNN experiment used generic computer-vision augmentation, including rotations and flips, while the ImageNet-pretrained ResNet50 and ViT backbones were almost entirely frozen.

The 51.5-million-parameter CNN achieved 97.64% test-set classification accuracy, substantially above the transfer-learning models. Rather than treating this result as evidence that the custom architecture was superior, the very large parameter count and image-level nature of the dataset motivated a more constrained baseline design. V1 also showed that simply attaching a new classifier to frozen ImageNet representations was insufficient for this MRI task: frozen ResNet50 and ViT-B/16 achieved only 61.02% and 61.15% accuracy respectively.

### V2: reducing the CNN and adapting pretrained models

V2 replaced the large flattened representation with **Global Average Pooling (GAP)** and expanded the convolutional feature extractor. The resulting baseline contained **1,207,459 parameters**, dramatically reducing the capacity of the final classifier and removing direct dependence on a large flattened spatial representation. This is the count reported by the instantiated model (zero-based cell 39 of the [V2 recorded notebook](../notebooks/dementia_mri_classification_coursework_V2_WithCellOutput.ipynb)); the notebook's Markdown estimates of about 300K (cell 30) and 330K (cell 72) parameters differ from it.

The augmentation strategy was also changed. Rotations and flips were removed in favour of transformations designed to be more appropriate for MRI images.

Most importantly, the pretrained models were allowed to adapt to the target domain:

- ResNet50 kept its stem, `layer1` and `layer2` frozen and fine-tuned parts of `layer3` and `layer4`, together with the replacement head. The freezing rule matches any parameter name containing `layer1`, `layer2`, `conv1` or `bn1`. Because `conv1` and `bn1` also occur inside the later bottleneck blocks, those parameters in `layer3` and `layer4` remained frozen too (V2 cell 31; V3 cell 33). The resulting model had 18,519,043 trainable parameters (V2 cell 54; V3 cell 57).
- ViT-B/16 fine-tuned its final four transformer layers rather than training only the classifier head.
- The V2 ViT used a learning rate of `1e-5`, a tenth of the `1e-4` base rate.

Across these configurations, ResNet50 accuracy increased from 61.02% in V1 to **91.99%** in V2, while ViT-B/16 increased from 61.15% to **87.53%**.

V2 also provides the cleanest augmentation-only comparison in the development history. With the same reduced CNN architecture, the no-augmentation model achieved 46.98% accuracy and the augmented version 46.19%, indicating that augmentation did not resolve the underlying underfitting of the from-scratch model.

### V3: final refinement

V3 retained the reduced CNN baseline but attempted to address its underfitting in the second experiment by changing several factors together: convolutional and classifier dropout were reduced, the classifier head was enlarged, the learning rate was increased and the augmentation pipeline was strengthened. The resulting model achieved 47.90% accuracy. Because several changes were introduced together, this result is **not** treated as an augmentation-only effect.

ResNet50 was unchanged from V2 and recorded the same 91.99% test-set classification accuracy.

The ViT configuration received the largest final refinement:

- the number of unfrozen transformer layers increased from four to eight;
- the patch-embedding projection was made trainable;
- the class token and positional embeddings were made trainable;
- approximately 67% of the model parameters were trainable; and
- a learning rate of `5e-5` was used for fine-tuning: five times the `1e-5` used for the V2 ViT and half the `1e-4` base rate. The V3 notebook describes this rate as "lower", which holds only relative to the base rate.

Fine-tuning depth, embedding adaptation and learning rate changed together, so the increase from 87.53% to 95.54% is not attributed to any one of these changes.

The final ViT-B/16 correctly classified **728 of 762 test images**, giving **95.54% test-set classification accuracy**.

## Final V3 results

| Experiment | Accuracy | Weighted F1 | Macro AUC |
|---|---:|---:|---:|
| Baseline CNN | 46.98% | 44.85% | 0.7135 |
| Improved CNN + domain-aware augmentation | 47.90% | 46.57% | 0.7188 |
| ResNet50 + partial fine-tuning | 91.99% | 91.95% | 0.9855 |
| ViT-B/16 + extended domain adaptation | **95.54%** | **95.54%** | **0.9947** |

The accuracy column reports the proportion of correctly classified images in the 762-image test split; it is not an average of the models' per-image confidence scores.

The pretrained models recorded substantially higher accuracies in the configurations that allowed their representations to adapt to the MRI domain:

| Model progression | Test-set classification accuracy |
|---|---:|
| ResNet50, frozen backbone (V1) | 61.02% |
| ResNet50, partial fine-tuning (V2/V3) | **91.99%** |
| ViT-B/16, frozen backbone (V1) | 61.15% |
| ViT-B/16, final 4 transformer layers fine-tuned (V2) | 87.53% |
| ViT-B/16, 8 layers + embedding adaptation (V3) | **95.54%** |

These version-to-version results document an iterative development process rather than a controlled ablation study. Some configurations changed more than one training decision at a time, so differences should not automatically be interpreted as the causal effect of a single component.
