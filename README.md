# Brain Tumor MRI Classification

**Comparing a custom CNN with VGG16 transfer learning for binary MRI image classification.**

**Author:** Abdelrahman Mohamed  
**Framework:** TensorFlow / Keras · **Environment:** Google Colab · **Image size:** 128 × 128 RGB

[Open the notebook](project_deep_learninig_Brain_Tumor_MRI.ipynb)

## Overview

This project classifies MRI images into the dataset's `yes` (tumor-present) and `no` (tumor-absent) categories. It includes data loading, image preprocessing, augmentation, a custom convolutional neural network, a frozen-backbone VGG16 model, evaluation, and model export.

The project performs binary image classification, not tumor localization, segmentation, or subtype prediction.

## Recorded Performance

| Model | Test accuracy |
| --- | ---: |
| Custom CNN | 87.56% |
| VGG16 transfer learning | 91.56% |

The numbers come from the notebook's saved output. VGG16 is **4.00 percentage points higher** on the recorded test split. The test report contains **450 images**, with **225 per class**. These are single-run results; this documentation update did not retrain either model.

### Custom CNN class metrics

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `no` | 0.89 | 0.86 | 0.87 | 225 |
| `yes` | 0.87 | 0.89 | 0.88 | 225 |

The notebook includes the CNN confusion matrix and accuracy/loss curves. It prints VGG16 test accuracy but does not generate a separate VGG16 class report or confusion matrix.

## Dataset and Preprocessing

The notebook downloads `brain_tumor.zip` using the Google Drive file ID in its code and extracts it to `/content/`. The extracted archive must provide:

| Path | Class |
| --- | --- |
| `/content/yes` | Tumor-present label |
| `/content/no` | Tumor-absent label |

Each readable image is resized to 128 × 128, converted to RGB, and normalized by dividing by 255. Labels are encoded and converted to categorical targets. Unreadable files are silently skipped.

Data is split into approximately **70% training, 15% validation, and 15% test**, using two stratified splits with `random_state=42`. Training augmentation includes rotation, zoom, width/height shifts, and horizontal flips. Validation and test arrays are not passed through the augmentation generator.

Dataset authorship, patient counts, acquisition details, licensing, and the total successfully loaded image count are not explicitly documented in the supplied notebook.

## Models

| Setting | Custom CNN | VGG16 |
| --- | --- | --- |
| Feature extractor | Two convolution/pooling blocks, 32 then 64 filters | Frozen ImageNet-pretrained backbone |
| Aggregation | Flatten | Global average pooling |
| Dense layer | 128 units, ReLU | 128 units, ReLU |
| Dropout | 0.5 | 0.3 |
| Output | 2-class softmax | 2-class softmax |
| Optimizer | Adam | Adam |
| Loss | Categorical cross-entropy | Categorical cross-entropy |
| Batch size | 32 | 32 |
| Maximum epochs | 20 | 10 |

The custom CNN summary reports **7,392,578 trainable parameters**. VGG16's backbone stays frozen; this is classifier-head training rather than backbone fine-tuning.

Both fits use the same callback list: early stopping with best-weight restoration, learning-rate reduction, and best-model checkpointing.

## Run in Google Colab

1. Open `project_deep_learninig_Brain_Tumor_MRI.ipynb` in Google Colab.
2. Ensure the runtime can download the configured Google Drive archive and VGG16 ImageNet weights.
3. Run the implementation cell. It installs gdown, downloads and extracts data, trains both models, and generates evaluation outputs.
4. Download the final `.keras` files from the runtime to retain the trained models.

A GPU runtime can accelerate training. The code uses notebook shell syntax and Colab-style `/content/` paths; local execution requires environment/path adaptation.

### Dependencies

TensorFlow/Keras, NumPy, Matplotlib, seaborn, Pillow, scikit-learn, and gdown. The notebook does not pin exact versions.

## Files and Artifacts

| File | Purpose |
| --- | --- |
| `project_deep_learninig_Brain_Tumor_MRI.ipynb` | Notebook with original code and saved results |
| `README.md` | Project documentation |
| `brain_tumor.zip` | Downloaded dataset archive, not embedded in the notebook |
| `best_cnn.keras` | Shared checkpoint filename used during both training stages |
| `cnn_model_final.keras` | Final saved custom CNN |
| `vgg16_model_final.keras` | Final saved VGG16-based model |

Model files are produced when the notebook runs; they are not included in this documentation delivery. The reused checkpoint callback can overwrite `best_cnn.keras` during VGG16 training. The two final export filenames remain distinct.

## Interpretation and Reproducibility

- The split is performed at image level. Patient-level separation, duplicate detection, and external validation are not shown.
- Both models use RGB input scaled to `[0, 1]`; the code does not apply VGG16's `preprocess_input`.
- The split seed is fixed, but initialization and augmentation are not fully seeded, and file iteration is unsorted. Exact results can vary across runs.
- Saved models do not bundle the image-loading pipeline or separately serialize the label encoder. Inference requires matching preprocessing and class mapping.
- This educational experiment does not establish clinical diagnostic performance.

## Presentation Update

Markdown sections explain the dataset, architectures, training setup, results, and generated files. The complete original code cell, execution metadata, saved charts, logs, and warnings are preserved without rerunning or changing the models.
