<!-- Navigation header (not source content). Sections below originate from the README preserved at commit 7895f7e. -->

# Brain MRI Classification for Dementia Severity

An iterative deep learning study using custom convolutional neural networks (CNNs), ResNet50 and a Vision Transformer (ViT-B/16) to classify brain MRI images into three dementia-severity categories. The project was developed and executed as a Google Colab notebook workflow using PyTorch, with the final recorded experiments run on an NVIDIA L4 GPU.

The project followed a progressive, course-style sequence of experiments rather than a controlled test of architectural complexity. I began by expecting ViT-B/16 to outperform ResNet50, and both to outperform the custom CNN, with only new classification heads trained on their frozen ImageNet-pretrained backbones; the first recorded results did not follow that ordering. The project then progressed through three notebook versions, with problems found in earlier experiments informing changes to model capacity, medical-image augmentation and transfer-learning strategy. The final ViT-B/16 correctly classified **728 of 762 images in the test split**, giving a **test-set classification accuracy of 95.54%**, compared with **91.99%** for ResNet50 and approximately **47%** for the two CNNs trained from scratch in V3. Earlier CNN configurations recorded different results, including 97.64% for the first V1 CNN; see [experimental development](experimental-development.md).

> **Important:** These results come from an image-level split of the supplied dataset. Patient identifiers were not available, so patient-level independence between training and testing could not be verified. The models are research prototypes developed for an MSc project and are not clinical diagnostic systems.

## Project at a glance

| | |
|---|---|
| Dataset | 5,076 brain MRI images |
| Classes | `NoImpairment`, `MildImpairment`, `ModerateImpairment` |
| Original image size | 208 x 176 pixels (height x width, sampled image) |
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

## Project context

This project was originally developed as coursework for the AI and Machine Learning component of the **MSc Applied Bioinformatics at King's College London**. This public repository preserves the original notebook development history and recorded outputs while documenting the workflow, results and methodological limitations more explicitly.
