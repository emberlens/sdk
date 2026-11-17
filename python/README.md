# Emberlens SDK

Open your Emberlens exports (COCO, YOLO, Pascal VOC) in Python and use them to
train a model. No dependencies for the core library.

## Install

```
python3 -m venv ~/.emberlens
~/.emberlens/bin/pip install emberlens-sdk
```

## Commands

Run each one as `~/.emberlens/bin/emberlens ...`

- `info EXPORT` shows what is in an export (images, classes, boxes).
- `install train` adds the training libraries (Pillow, PyTorch, torchvision).
- `train classify EXPORT` trains an image classifier on your labels.
- `train detect EXPORT` trains an object detector on your labels.
- `install onnx` then `export-model MODEL` writes a model as a `.onnx` file.

EXPORT is the `.zip` from the Emberlens app, or its folder.

## Train in Colab

[Open the notebook](https://colab.research.google.com/github/emberlens/sdk/blob/main/notebooks/train.ipynb)
to train without installing anything.

## Privacy

Everything runs on your computer. The SDK sends nothing anywhere.

## More

[Full guide](https://github.com/emberlens/sdk#readme). BSD 3-Clause licence.
