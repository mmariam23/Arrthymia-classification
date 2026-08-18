# PPG Arrhythmia Detection

A deep learning system that classifies cardiac arrhythmias from raw PPG (photoplethysmography) signals — the kind of optical heart-rate sensor found in smartwatches and fitness trackers. The goal is to take a raw wrist-worn sensor signal and flag whether it looks like normal sinus rhythm or one of five arrhythmia types, without needing an ECG.

## What it does

The system takes a stream of raw PPG samples (from a smartwatch or similar wearable), runs it through a signal-processing pipeline, and classifies each 10-second segment into one of six classes:

- **SR** — Sinus Rhythm (normal)
- **PVC** — Premature Ventricular Contraction
- **PAC** — Premature Atrial Contraction
- **VT** — Ventricular Tachycardia
- **SVT** — Supraventricular Tachycardia
- **AF** — Atrial Fibrillation

## Pipeline

Raw PPG samples go through several stages before reaching the model:

1. **Resampling** to a uniform 100 Hz grid — wearable sensors don't sample at perfectly even intervals, so timestamps are used to interpolate onto a fixed grid.
2. **Bandpass filtering** (0.5–10 Hz, 4th-order Butterworth) to remove baseline wander and high-frequency noise.
3. **Moving average smoothing** (window = 5) to reduce residual noise.
4. **Segmentation** into 10-second (1000-sample) windows.
5. **Signal quality checks** — segments are rejected if they're flat, clipped/saturated, or have an implausible or irregular heart rate (a signal quality index, or SQI).
6. **Min-max normalization** to [0, 1].
7. **Heart rate estimation** via peak detection, which also flags bradycardia/tachycardia.
8. **Classification** — segments flagged as arrhythmic by heart rate are passed to the model, a 1D VGG-style convolutional network, for arrhythmia-type classification.

The model itself is a VGG-16-style architecture adapted for 1D time series: stacked Conv1D + BatchNorm + ReLU blocks with max pooling, dropout in the deeper layers, and a small fully-connected classifier head.

## Training results

The model was trained for 50 epochs. Training and validation loss both decrease steadily and validation accuracy converges around 86–87%.

![Training and validation loss/accuracy curves](docs/training_curves.png)

## Evaluation — confusion matrix

On the held-out test set, the model performs very well on the two largest and most distinct classes (SR and AF), and reasonably on PVC, VT, and SVT. The most confusion happens between PAC and the other classes (PVC, AF), which makes sense — PAC morphology overlaps significantly with these on a PPG waveform, and even ECG-based classifiers commonly struggle with this distinction.

![Confusion matrix](docs/confusion_matrix.png)

## Real-world test: where it struggles

Clean, curated training data is not the same as a live signal off a wrist-worn sensor. When I ran the trained model against real PPG streams captured from actual hardware — with motion artifacts, poor contact, and ambient noise — confidence dropped sharply on several segments, and a few normal-rhythm segments were misclassified as AF (visible below as low-confidence AF predictions in the 28–32% range). Only the cleanest segments (SR, ~65% confidence) were classified with any real certainty.

<!-- TODO: add the real-data segment plot here, e.g. docs/real_data_test.png -->
<!-- ![Real-world signal test](docs/real_data_test.png) -->

This is the most important finding of the project, honestly: a model that scores well on curated benchmark data can still be unreliable on noisy real-world sensor data, and confidence scores dropped enough on unclean segments that this pipeline is not yet suitable for stand-alone clinical use. Improving on this — better noise filtering, a denoising front-end, or training with more real-world (not just benchmark) noise — is the clearest next step.

## Project structure

```
server vgg/
├── main.py           entry point (uvicorn)
├── app.py            FastAPI server — /predict, /health
├── model.py           VGG-1D architecture + inference wrapper
├── preprocess.py       bandpass filter, moving average, normalization
├── resampler.py        uniform resampling to 100 Hz
├── sqi.py               signal quality index / segment rejection
├── hr.py                 heart rate estimation + classification
├── requirements.txt
└── Dockerfile
```

## Running it

```bash
pip install -r requirements.txt
python main.py
# server starts at http://localhost:7860
# interactive API docs at http://localhost:7860/docs
```

### API

`GET /health` — server and model status.

`POST /predict` — send raw PPG samples as JSON:

```json
{ "ppg": [91706, 91712, 91724, 91760, ...], "time_ms": null, "fs": 100 }
```

Returns per-segment results (accepted/rejected, signal quality score, BPM, heart-rate status, arrhythmia type if applicable) and an overall diagnosis across all segments.

## Known limitations

- Real-world signal noise (motion artifact, poor sensor contact) reduces classification confidence significantly compared to curated test data — see the real-world test section above.
- PAC is frequently confused with PVC and AF in the confusion matrix.
- The `/predict` endpoint currently expects raw JSON arrays; file upload support (CSV/MAT) is planned but not yet implemented.

## Dataset

Trained on labeled PPG segments spanning the six arrhythmia classes above (see the training notebook for full preprocessing and split details).
