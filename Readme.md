# Sign Language To Text Translator

This project uses a webcam plus a CNN model trained on the Sign MNIST dataset to predict static hand signs and build text on screen.

## Features

- Live webcam prediction with automatic hand detection
- Prediction smoothing to reduce flicker
- Auto-append letters after a stable hand sign
- Save translated text to `translated_text.txt`
- Retrain the model from the included CSV datasets

## Supported Signs

The bundled model follows the Sign MNIST dataset, so it supports static alphabet signs and excludes `J` and `Z` because those letters require motion.

## Project Files

- `translator.py` - run the webcam translator
- `train_model.py` - retrain and save `sign_model.h5`
- `camera.py` - quick webcam test
- `requirements.txt` - Python dependencies

## Setup

```powershell
cd D:\sign-language-translator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

If you already have `sign_model.h5`, start the translator:

```powershell
python translator.py
```

If you want to retrain the model first:

```powershell
python train_model.py
python translator.py
```

`train_model.py` will also look for `C:\Users\Admin\Downloads\archive (1)\asl_alphabet_test` and use those real-image hand samples for a second fine-tuning pass when available.

## Controls

- Hold a clear sign steady to append it automatically
- Move your hand out of the box to reset for the next letter
- `Space` add a blank space
- `B` delete the last character
- `C` clear the full sentence
- `S` save the current sentence to `translated_text.txt`
- `Q` quit

## Notes

- Raise one clear hand in front of the camera. The app now crops the detected hand automatically.
- Try a plain background and bright lighting for better webcam accuracy.
- The included model is best for static ASL-style signs, not full continuous sentence translation.
- If predictions still feel weak, retrain with `python train_model.py` to generate a fresh `sign_model.h5` and `label_map.json`.
