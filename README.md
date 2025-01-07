# TextSniffer - OCR
OCR with Donut-based VisionEncoderDecoderModel (PyTorch Lightning)
This repository contains code to train and evaluate an Optical Character Recognition (OCR) model using NAVER Clova's Donut model. The code leverages PyTorch Lightning for streamlined training loops and experiment management. It also utilizes Albumentations for image augmentations (both basic and advanced), enabling a gradual augmentation strategy that switches augmentation levels after a specified epoch.

Below is an overview of the code, the steps to run it, the libraries and technologies used, and some evaluation results from a sample dataset.

Table of Contents
  - Overview
  - Technologies and Libraries Used
  - Project Structure
  - How to Run
  - Detailed Code Explanation
      1. Imports and Installs
      2. Paths and Hyperparameters
      3. Dataset and Collation
      4. DataModule
      5. Lightning Module
      6. Training, Testing, and Saving
      7. Testing and Evaluation
      8. Final Model and Processor Saving
  - Results
  - Why These Libraries?

# Overview
Objective: Train an OCR system capable of converting images of text to transcriptions.
Model: This uses a VisionEncoderDecoderModel based on the "naver-clova-ix/donut-base" architecture.
Workflow:
Data: Split into train, validation, and test sets with images and corresponding JSON annotations.
Augmentations:
Basic augmentations in the early epochs.
Advanced augmentations after a certain epoch to improve the model's robustness.
Training: PyTorch Lightning handles model training, callbacks, checkpoints, early stopping, etc.
Evaluation: Character-level and word-level accuracy metrics are computed on the test set.

# Technologies and Libraries Used
Python 3.7+
PyTorch: Core deep learning library.
PyTorch Lightning: Simplifies the training loop, handles logging, checkpointing, etc.
Transformers (HuggingFace): Provides the VisionEncoderDecoderModel for the Donut architecture and its tokenizer/processor.
Albumentations: Image augmentation library used for both basic and advanced transformations.
TQDM: For progress bars during training and evaluation.
Numpy: For general numeric computations.
Pillow: For image loading and manipulation.
SentencePiece: Tokenization backend for the Donut model.

# Project Structure
Although the code here is presented in a single notebook/script format, conceptually it can be broken into:
.
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   ├── val/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       ├── images/
│       └── annotations/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── donut_trained_model/  # Checkpoints and best model are saved here
├── final_donut2/         # Final model and processor
└── ocr.ipynb         # The main script containing all the code blocks
Note: Adjust paths as necessary for your environment.

# How to Run
1. Clone this repository
2. Install dependencies (Recommended: use a virtual environment)
  - pip install torch torchvision torchaudio transformers pytorch-lightning sentencepiece Pillow albumentations[imgaug]
3. Prepare your dataset following the directory structure above
  - Place images in train/images, val/images, test/images.
  - Place matching JSON annotations in train/annotations, val/annotations, test/annotations.
  - Ensure train.txt, val.txt, and test.txt in splits/ each list the filenames of the images (one per line).
4. Update paths in the code if needed
  - data_root = "/teamspace/studios/this_studio"  # Or your data root path
5. Run the training script (ocr.ipynb) run all cells in order.
6. Monitor Training
  - PyTorch Lightning logs progress.
  - The best model checkpoint will be saved under donut_trained_model/.
7. Evaluation:
  - The script also runs inference on the test set and prints accuracy metrics and sample predictions vs. references.
8. Final Model
  - The final model is saved in final_donut2/, containing both model weights and the processor.

# Detailed Code Explanation
1. Imports and Installs
Installs are done first to ensure required packages are available:

!pip install torch torchvision torchaudio transformers pytorch-lightning sentencepiece Pillow --quiet
!pip install albumentations[imgaug] --quiet
Then we import necessary libraries (os, json, PIL, random, etc.) to handle file paths, reading JSON, and image manipulations.

2. Paths and Hyperparameters
Paths are defined to point to your dataset splits (train, val, test), along with a directory to store checkpoints.
Example hyperparameters:
learning_rate = 3e-5
weight_decay = 0.001
max_epochs = 30
warmup_ratio = 0.05
batch_size = 2
num_workers = 4
max_length = 512
precision = 16
These can be tweaked based on your hardware or dataset size.

3. Dataset and Collation
Albumentations augmentations are defined in get_basic_augmentations() and get_advanced_augmentations().
DocumentDataset loads each image-annotation pair, processes them with the DonutProcessor, and handles the text tokens and labels.
collate_fn stacks batch items into tensors for the DataLoader.
Gradual Augmentation is achieved via a custom callback GradualAugmentationCallback. It updates the dataset to use get_advanced_augmentations() after a certain epoch.

4. DataModule
PyTorch Lightning’s LightningDataModule organizes training, validation, and test datasets. It uses:
Basic augmentations for training by default.
No augmentations for validation/test sets.
class DonutDataModule(pl.LightningDataModule):
    ...
5. Lightning Module
DonutModelModule wraps VisionEncoderDecoderModel with a Donut architecture from Hugging Face Transformers. Key points:
Tokenizer special tokens: <s_ocr> and </s> tokens are added to handle the OCR prompt and end-of-sequence.
Gradient checkpointing is enabled for memory efficiency.
Training and validation steps are defined, logging loss values.
The optimizer and linear scheduler are configured via configure_optimizers.

6. Training, Testing, and Saving
Load the DonutProcessor from the naver-clova-ix/donut-base checkpoint.
Initialize DonutDataModule with the train/val/test paths.
Calculate steps_per_epoch and total steps to configure warmup ratio for scheduling.
Instantiate the DonutModelModule.
Set up callbacks:
ModelCheckpoint (saves the best model based on val_loss).
EarlyStopping (stops if val_loss does not improve for several epochs).
GradualAugmentationCallback (switch to advanced augmentations at a chosen epoch).
Run trainer.fit(model_module, datamodule=dm).

7. Testing and Evaluation
After training:
Load the best checkpoint:
model_module = DonutModelModule.load_from_checkpoint(best_model_path, processor=processor)
Compute character-level and word-level accuracy using a custom function compute_accuracies().
Print sample references vs. predictions.

8. Final Model and Processor Saving
Save the final model and processor to disk:
model_module.model.save_pretrained(final_save_dir)
processor.save_pretrained(final_save_dir)
They can be loaded later for inference:
from transformers import VisionEncoderDecoderModel, DonutProcessor
model = VisionEncoderDecoderModel.from_pretrained(final_save_dir)
processor = DonutProcessor.from_pretrained(final_save_dir)

# Results
From a sample run, some reported metrics were:

Character-level Accuracy: ~51.43%
Word-level Accuracy: ~74.24%
These metrics depend heavily on the dataset (image quality, text complexity, domain, etc.). With further hyperparameter tuning, more data, or advanced fine-tuning, you may achieve better results.

Sample Outputs
A snippet from the test set results:

Reference: Since 1958 , 13 Labour life Peers and Peeresses have been created ...
Prediction: Since 1958 , 13 Labour life Peers and Peeresses have been created ...

# Why These Libraries?
PyTorch: Industry-standard library for deep learning.
PyTorch Lightning: Removes boilerplate from training loops and provides automatic GPU/TPU support, checkpointing, logging, etc.
Hugging Face Transformers: Offers pre-trained models and a consistent API. The Donut-based VisionEncoderDecoderModel is state-of-the-art for OCR-like tasks.
Albumentations: Efficient image augmentations, giving you both simple and advanced transforms.
TQDM: For tracking progress in training or inference loops.

Happy OCR-ing! If you have any questions or suggestions, please open an issue or reach out.
