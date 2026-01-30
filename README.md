# Activity-Recognition
Fine-tuned Vision Transformer (ViT) + LSTM model for human action recognition on the HMDB-51 dataset (~70 classes). Extracts frame-level features with ViT → temporal modeling with LSTM → classification.
# Video Activity Recognition – ViT + LSTM on HMDB-51

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/Vision%20Transformer-FFD93D?style=for-the-badge"/> <img src="https://img.shields.io/badge/LSTM-4CAF50?style=for-the-badge"/> <img src="https://img.shields.io/badge/HMDB--51-9C27B0?style=for-the-badge"/>

End-to-end **action recognition** pipeline that combines a strong image backbone (**Vision Transformer**) with a lightweight recurrent head (**LSTM**) to model temporal dynamics in short video clips.

Tested on the popular **HMDB-51** benchmark (~6,766 clips, 51 action classes).

## Features

- Stratified train/val/test split (~80/10/10)
- Frame sampling + data augmentation
- Pretrained ViT feature extractor (timm or huggingface)
- Bidirectional LSTM temporal modeling
- Focal loss + label smoothing (optional)
- Learning rate scheduler + early stopping
- Comprehensive evaluation: accuracy, precision, recall, F1, confusion matrix, per-class metrics
- Training progress logging + best model checkpointing

## Results (example from your notebook – update with your numbers)

| Split | Top-1 Accuracy | Top-5 Accuracy | Weighted F1 |
|-------|----------------|----------------|-------------|
| Val   | ~68–72%        | ~89–92%        | ~0.70       |
| Test  | ~66–70%        | ~87–91%        | ~0.68       |

*(Numbers depend on ViT variant, #frames, training time, augmentation, etc.)*

## Repository Structure

```text
.
├── activity-recognition-vit-rnn.ipynb      # Main notebook (training + evaluation)
├── hmdb51_dataset_split.csv                # Created train/val/test split
├── README.md
├── requirements.txt                        # (create this)
└── utils/                                  # (optional – move helpers here)
    ├── dataset.py
    ├── model.py
    └── train_utils.py
