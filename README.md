# Voicemail / Answering Machine Detector

A binary audio classifier that detects whether a call recording is an answering-machine
(voicemail) greeting or a live call, using a convolutional neural network (CNN) trained
on mel spectrograms.

**Author:** Zyan Bezzant (zbezzant)

Built to improve on a text-only voicemail detector used in call-handling workflows: instead
of relying only on transcribed words, the model learns acoustic patterns — continuous greeting
speech, pacing, and tones — directly from audio.

## How It Works

Each 10-second mono recording (16 kHz) is converted into a log-scaled mel spectrogram of
shape `(1, 64, 313)` — 64 frequency bins × ~313 time frames. The spectrogram is treated like
an image and passed through four stacked convolutional blocks, followed by a small classifier
head that outputs a single logit. A sigmoid maps it to a probability P(voicemail); scores
≥ 0.5 are classified as answering machine.

Spectrograms are computed in batches on the GPU via `torchaudio.transforms.MelSpectrogram`
and cached to disk, so subsequent runs skip feature extraction entirely.

### Audio settings

| Parameter | Value | Description |
|---|---|---|
| `SR` | 16000 | Sample rate (standard for speech) |
| `DURATION` | 10 s | Fixed clip length |
| `N_MELS` | 64 | Spectrogram height (frequency bins) |
| `N_FFT` | 1024 | FFT window size |
| `HOP_LENGTH` | 512 | Window step (50% overlap) |
| `FMIN` / `FMAX` | 50 / 8000 Hz | Frequency range |

Each spectrogram is min–max normalized to `[0, 1]`.

### Model architecture

```
Mel spectrogram (1, 64, 313)
│
├─ CNNBlock ×4   [ 2× (Conv2D 3×3 → BatchNorm → ReLU) → MaxPool → Dropout2d(0.2) ]
│     channels: 32 → 64 → 128 → 256
│
├─ AdaptiveAvgPool2d(4, 4)
│
├─ Flatten → Linear(4096, 256) → ReLU → Dropout(0.4)
├─ Linear(256, 128) → ReLU → Dropout(0.3)
│
└─ Linear(128, 1) → logit   (sigmoid applied outside the model)
```

Total parameters: **2,254,497** (all trainable). `BCEWithLogitsLoss` is used for numerical
stability, so the sigmoid lives in the loss / inference step rather than the model.

### Training setup

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` (patience=6, factor=0.5) |
| Gradient clipping | `clip_grad_norm_` at max_norm=1.0 |
| Batch size | 32 |
| Max epochs / early stopping | 60 / patience of 20 epochs on val loss |

In the recorded run, training early-stopped at epoch 48 with a best validation loss of
**0.2061**; the best checkpoint is saved to `cache/best_model.pt`.

## Dataset

12,436 call recordings, hand-labeled (with manual review/correction of initial labels):

| Class | Count |
|---|---|
| Answering machine | 6,249 |
| Non-answering machine | 6,187 |

Stratified splits: **train 9,301 / validation 1,269 / held-out test 1,866** (≈ 73/12/15).

Source recordings arrive as `.ogg` with labels in metadata and are preprocessed into
`.wav` clips: converted format, fixed to 10 seconds, volume-normalized, and trimmed of
long silences.

## Results (held-out test set, n = 1,866)

| Metric | Value |
|---|---|
| Accuracy | **93%** |
| ROC-AUC | **0.9762** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-answering machine | 0.95 | 0.90 | 0.93 | 928 |
| Answering machine | 0.91 | 0.95 | 0.93 | 938 |

## Error Analysis

The notebook includes a Grad-CAM visualization (hooks on the last CNN block) of the five
most confident false positives and false negatives:

- **False positives** — non-voicemail calls with continuous audio, e.g. a caller speaking
  nonstop or loud background noise, which resemble an unbroken voicemail greeting.
- **False negatives** — voicemail recordings containing mostly just the "start recording"
  beep, without extended greeting speech.

## Repository Layout

```
.
├── voicemail_detector.ipynb      # Main notebook
├── answering_machine/            # Voicemail .wav recordings
├── non_answering_machine/        # Non-voicemail .wav recordings
└── cache/                        # Created automatically
    ├── mel_specs.npy             # Cached mel spectrograms
    ├── labels.npy                # Cached labels
    ├── valid_paths.pkl           # File paths (used by the error-analysis table)
    ├── best_model.pt             # Best checkpoint by validation loss
    ├── training_history.png
    ├── confusion_matrix.png
    ├── mel_regular_visualization.png
    └── error_analysis_top_confident_mistakes.png
```

## Getting Started

Requirements: Python 3.10+ and (optionally, but recommended) a CUDA GPU. Tested with
PyTorch 2.5.1+cu121 and librosa 0.11.0 on an NVIDIA RTX 3060 (12 GB).

```bash
pip install torch torchaudio librosa numpy scikit-learn matplotlib seaborn tqdm jupyter
```

Place cleaned 10-second `.wav` files in the two class folders, then run the notebook
top to bottom. The first run extracts and caches spectrograms; later runs load from
`cache/`. On Windows, DataLoader workers are automatically set to 0.

## Usage

**Single-file prediction**

```python
label, prob, source = predict("path/to/call.wav", threshold=0.5)
# label: 'answering_machine' or 'non_answering_machine'
# prob:  model confidence in [0, 1]
```

**Batch sorting** — the final cell walks an input folder of `.wav` files and sorts them
into `AIFoundVoicemails/` and `AINotVoicemails/`, plus ten confidence-bucket folders
(`confidence_00-10` … `confidence_90-100`). Configure `INPUT_DIR`, `OUTPUT_ROOT`,
`THRESHOLD`, and the `MOVE_FILES` copy/move toggle at the top of the cell. In its recorded
run it processed 2,007 files in ~25 s (387 flagged as voicemail, 1,620 as non-voicemail,
0 errors).

## Future Work

- Explicit voicemail-beep detection as an engineered feature to catch beep-only greetings
- Continued data collection and retraining to push accuracy further
- Additional handcrafted audio features (speech rhythm, pitch patterns)

## References

- [librosa.feature.melspectrogram](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html)
- [torchaudio.transforms.MelSpectrogram](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.MelSpectrogram) / [AmplitudeToDB](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.AmplitudeToDB)
- [Environmental Sound Classification with CNN (arXiv:1609.09430)](https://arxiv.org/abs/1609.09430)
- [Very Deep Convolutional Networks (VGG, arXiv:1409.1556)](https://arxiv.org/abs/1409.1556)
- [Attention-Based Models for Speech Recognition (arXiv:1610.02391)](https://arxiv.org/abs/1610.02391)
- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) · [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) · [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) · [clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) · [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) · [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) · [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [MFCCs for Speech Recognition (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/)
- [Audio Classification Using CNNs and Transformers (dev.to)](https://dev.to/aditi_baheti_f4a40487a091/an-in-depth-look-at-audio-classification-using-cnns-and-transformers-1981)