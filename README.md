# Emberlens SDK

Use the photos and labels from the Emberlens app to train your own AI models, on your laptop.

The SDK opens the `.zip` you export from the app (COCO, YOLO or Pascal VOC). It lets you:

- look at your photos and boxes,
- pick the ones you want,
- convert them to another format,
- train a model.

You do not need to know Python. One command does each job.

---

## 1. Before you start

1. **Python 3.9 or newer.** Check in a terminal:

   ```bash
   python3 --version
   ```

   No Python, or an older one? Get it from [python.org/downloads](https://www.python.org/downloads/).
2. **An export from the app.** In Emberlens: open the project, tap **Export**, and save the zip to your laptop (AirDrop, Files, email, or a cloud destination).

That is all. There is nothing to configure.

---

## 2. First steps

Open a terminal in this `sdk` folder, then run:

```bash
./emberlens info ~/Downloads/fruit-stand-coco-20261008-1030.zip
```

On Windows, type `emberlens` instead of `./emberlens`.

The first run takes a few seconds. It makes a private Python environment in `python/.venv`. Nothing is installed on the rest of your computer.

You see something like:

```
Fruit stand  (COCO, 4 images, 4 boxes)
  Splits:  train 2, val 1, test 1
  Classes:
    Mango                    2
    Strawberry               2
  Condition:
    Ripe                     2
    Unripe                   1
    (none)                   1
```

Try it now with the sample export that ships with the SDK:

```bash
./emberlens info fixtures/coco_joined
```

---

## 3. What you can do

| I want to… | Command |
|---|---|
| See what is in an export | `./emberlens info EXPORT` |
| Look at every photo with its boxes | `./emberlens preview EXPORT` |
| Convert to YOLO (or `coco`, `voc`, `json`) | `./emberlens convert EXPORT --to yolo` |
| Save each box as its own small image | `./emberlens crops EXPORT` |
| Train a model that names objects | `./emberlens train classify EXPORT` |
| Train a model that finds objects | `./emberlens train detect EXPORT` |
| Use a trained model on new photos | `./emberlens predict MODEL-FOLDER photo.jpg` |
| Check that the SDK works | `./emberlens selftest` |
| See all commands | `./emberlens help` |

`EXPORT` is the zip from the app, or the folder it unzips to.

**Polygons** (outlines drawn in the app) are kept by `convert --to coco` and `--to yolo`, and drawn by `preview`. Pascal VOC has no place for them, so `--to voc` keeps the box around each one. The SDK unzips a zip once, into a folder next to it with the same name.

Results go into the folder you are in, with clear names. Examples: `fruit-stand-coco-yolo/`, `fruit-stand-coco-preview.html`. The SDK never overwrites: if a name exists, it adds `-2`, `-3`, and so on.

### Pick only some boxes: `--where`

```bash
./emberlens preview EXPORT --where class=Mango
./emberlens convert EXPORT --to yolo --where class=Mango --where condition=Unripe
./emberlens info    EXPORT --where condition=none      # boxes with no Condition
./emberlens info    EXPORT --where split=train
```

Letter case does not matter: `mango` and `Mango` match the same boxes.

### Change what the labels are: `--labels`

Your boxes can have a class (Mango) and sub-category values (Condition = Ripe). Choose what the model learns:

| `--labels` | A box with Mango, Ripe, Alphonso becomes |
|---|---|
| `label` (default) | the name in the export, e.g. `Mango_Ripe_Alphonso` |
| `class` | `Mango` |
| `joined` | `Mango_Ripe_Alphonso` |
| `condition` (any sub-category name) | `Ripe` (boxes without a Condition are left out) |

Example: train a model that tells ripe from unripe:

```bash
./emberlens train classify EXPORT --labels condition
```

### New train / val / test split: `--split`

```bash
./emberlens convert EXPORT --to coco --split 80/10/10
```

The split is random but repeatable: the same command always gives the same split.

---

## 4. Training

Training needs PyTorch, a free library of about 1 GB. Install it once:

```bash
./emberlens install train
```

If you forget, the SDK asks you when you first need it.

**Two kinds of model:**

- `./emberlens train classify EXPORT` answers **"what is this object?"**. It learns from the inside of each box. It is fast, minutes on a laptop.
- `./emberlens train detect EXPORT` answers **"where are the objects, and what are they?"**. It is slower; plan longer on a laptop without a graphics card.

Useful options:

- `--epochs 10`: more training rounds. More rounds take longer and usually help.
- `--no-pretrained`: start from zero, with no download. This needs much more data.
- `--device cpu`: force the processor. The default is the fastest found: NVIDIA (`cuda`), Apple (`mps`), or `cpu`.

**What you get:** a folder such as `fruit-stand-coco-classify-model/`. It holds:

- `model.pt`: the trained model.
- `report.json`: how well it did on the val and test photos.

**Use it:**

```bash
./emberlens predict fruit-stand-coco-classify-model my-photo.jpg
```

You get `predictions.json`. For a detector you also get `predictions.html`, a page that shows the found boxes.

**Downloads:** by default, training starts from a model that already knows everyday objects. This works much better with few photos. That starting model is downloaded once from pytorch.org: about 45 MB for `classify`, 75 MB for `detect`. Use `--no-pretrained` to avoid the download.

---

## 5. Use it from your own Python code

You (or your AI coding assistant) can use the SDK in your own scripts:

```python
import emberlens

ds = emberlens.load("fruit-stand-coco.zip")       # zip or folder
print(ds.summary())

for photo in ds:                              # every photo
    for box in photo.boxes:                   # every box
        print(photo.path, box.cls, box.attributes, box.x, box.y, box.w, box.h)

unripe = ds.filter(cls="Mango", condition="Unripe")
unripe.to_yolo("unripe-mangoes-yolo")
train_set = ds.relabel("condition").torch_classification("train")   # PyTorch Dataset
```

Run your script with the SDK's Python, so `import emberlens` works:

```bash
./emberlens run my_script.py
```

Ready-made examples are in [`examples/`](examples/):

- `01_explore.py`: look at the data and filter it.
- `02_my_own_training.py`: your own PyTorch training loop.

**Cheat sheet**

| | |
|---|---|
| `emberlens.load(path)` | open a zip or folder; also works for plain COCO / YOLO / VOC datasets from other tools |
| `ds.images`, `for photo in ds` | photos: `.id`, `.path`, `.width`, `.height`, `.split`, `.boxes` |
| `box.cls`, `box.label`, `box.attributes`, `box.get("condition")` | what the box is |
| `box.x, box.y, box.w, box.h` | position, 0 to 1, top-left corner (`box.pixels(w, h)` for pixels) |
| `box.polygon`, `box.is_polygon`, `box.corners()` | polygon corners `[(x, y), …]` (0 to 1), or `None` for a box |
| `ds.filter(cls=…, split=…, **values)` | keep matching boxes (`condition=None` = no value) |
| `ds.split("train")`, `ds.resplit(0.8, 0.1, 0.1)` | splits |
| `ds.relabel("class" / "joined" / "condition")` | change the labels |
| `ds.counts("class")`, `ds.summary()` | numbers |
| `ds.to_coco(dir)`, `to_yolo`, `to_voc`, `to_json(file)` | write |
| `ds.preview("page.html")`, `ds.crops(dir)` | look, crop |
| `ds.torch_detection(split)`, `ds.torch_classification(split)` | PyTorch datasets |
| `emberlens.train.train_classifier(ds, dir)`, `train_detector`, `predict` | train and predict |

**Tip for AI assistants:** give it this README and [`SPEC.md`](SPEC.md). Then it knows the data.

Prefer your own Python environment? Run:

```bash
pip install -e "sdk/python[train]"
```

Then the `emberlens` command and `import emberlens` work everywhere.

---

## 6. Settings

There are none to set. To start fresh, delete the `python/.venv` folder; the next run makes it again.

| You want | Do this |
|---|---|
| Only reading, filtering, converting, preview | Nothing. It uses only built-in Python and works offline. |
| Crops and image tools | `./emberlens install images` (Pillow) |
| Training | `./emberlens install train` (Pillow, PyTorch, torchvision) |

---

## 7. Privacy

Everything runs on your laptop. The SDK never sends your photos or labels anywhere. It uses the internet only for:

- `./emberlens install …`, which downloads the libraries.
- The first `train` without `--no-pretrained`, which downloads the starting model from pytorch.org.

---

## 8. Problems?

| Message | Fix |
|---|---|
| `needs Python 3.9 or newer` | Install Python from python.org, then run again. |
| `Permission denied: ./emberlens` | Run `chmod +x emberlens` once. |
| `Can't find …` | Check the path. Tip: drag the zip into the terminal to paste its path. |
| `… is not empty. Choose a new folder.` | Use `--out new-folder-name`. |
| `Training needs at least 2 different labels` | Choose other `--labels` or `--where`, or label more photos. |
| `There are no boxes in the train split` | Add `--split 80/10/10`. |
| Anything else | Run `./emberlens selftest` and share its output. |

---

## 9. What is in this folder

```
emberlens, emberlens.cmd        the launcher (macOS/Linux, Windows)
python/emberlens/          the SDK (Python, no required libraries)
python/tests/         its tests (./emberlens selftest)
examples/             small scripts to copy
schema/               JSON Schemas of emberlens.json and dataset.json
fixtures/             sample exports made by the app's own export engine,
                      and expected/ results for checking other-language SDKs
SPEC.md               the exact file formats, for porting the SDK
```

**Licenses:** the SDK is part of Emberlens. The optional libraries are permissive: Pillow (MIT-CMU), PyTorch and torchvision (BSD-3).
