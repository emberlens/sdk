# Nhan export format and SDK specification

Version 1, 2026-10-08 (polygons added the same day). This file is for people who write a Nhan reader in another language, or check one. The Python SDK in `python/nhan/` is the reference implementation. The files in `fixtures/` are the test data.

## 1. Words

| Word | Meaning |
|---|---|
| export | what the app writes: a `.zip` with one dataset in one format |
| class | the main label of a box, e.g. `Mango` |
| sub-category | an extra label, e.g. `Condition`, with values such as `Ripe` and `Unripe` |
| value | one value of a sub-category on one box. A box may have no value. |
| label | the name written in the format's files. It is the class, or the class joined with values (`Mango_Ripe`). |
| card | `nhan.json`, the file that says what each label means (§3) |

## 2. Zip layout

The zip holds the dataset at its root. Readers should also accept one extra top folder. Image file names are `<photo id>.jpg`. Photo ids are opaque strings (UUIDs from the app), safe for file names.

### 2.1 COCO

```
nhan.json
images/{train,val,test}/<id>.jpg
annotations/instances_{train,val,test}.json
```

- One JSON per split: `images[] {id, file_name, width, height}`, `annotations[] {id, image_id, category_id, bbox, area, iscrowd}`, `categories[] {id, name[, supercategory]}`.
- `image_id` and `category_id` are 1-based, and only valid inside one file.
- `bbox = [x, y, w, h]` in pixels of the exported image, top-left origin, 2 decimals.
- A polygon label adds `segmentation: [[x1, y1, x2, y2, …]]`: one ring of pixel corners, 2 decimals. Its `area` is the polygon's area (shoelace formula). `bbox` is the box around the polygon. A box has no `segmentation` key.
- In the separate-fields style, an annotation may have `attributes: {sub-category: value}`. A missing key means no value. A category may have `supercategory`, which is the project's root category name.

### 2.2 YOLO

```
nhan.json
data.yaml
images/{train,val,test}/<id>.jpg
labels/{train,val,test}/<id>.txt
```

- `data.yaml`: `path: .`, `train: images/train`, `val: images/val`, `test: images/test`, `nc: N`, `names: [ … ]`. Names are a YAML flow list. A name is single-quoted when it is not a plain YAML word (`'it''s'` doubles the quote).
- Each `.txt` line is `<index> <cx> <cy> <w> <h>`: normalised 0..1, centre-based, 6 decimals. A photo with no boxes has an empty file.
- **When the export has any polygon**, every line is a YOLO segmentation line instead: `<index> x1 y1 x2 y2 …` (normalised corners, 6 decimals). A box is written as its 4 corners, clockwise from the top-left: `x y  x+w y  x+w y+h  x y+h`. Readers turn exactly that upright rectangle back into a box.
- YOLO has no place for values, so labels are always joined (§3.2).

### 2.3 Pascal VOC

```
nhan.json
JPEGImages/<id>.jpg
Annotations/<id>.xml
ImageSets/Main/{train,val,test}.txt     one photo id per line
```

- XML: `<annotation><filename/><size><width/><height/><depth/></size><object><name/><bndbox><xmin/><ymin/><xmax/><ymax/></bndbox>[<attributes>]</object>…</annotation>`.
- Box corners are integer pixels, **1-based and inclusive**: `xmin = round(x·W) + 1`, `xmax = round((x + w)·W)`. `round` rounds a half away from zero (7.5 → 8, 22.5 → 23), like C++ `std::lround`; not to even.
- VOC XML has no polygon field. A polygon is written as the box around it (`<bndbox>`). Use COCO or YOLO to keep outlines.
- Separate-fields values (CVAT style): `<attributes><attribute><name>Condition</name><value>Ripe</value></attribute>…</attributes>`.

### 2.4 Splits

The app assigns train / val / test with a seeded shuffle (`nhan.json` `split`, `seed`). With fewer than 3 photos, every photo is in train. Every photo is in exactly one split.

## 3. The card: `nhan.json`

Schema: [`schema/nhan-card.schema.json`](schema/nhan-card.schema.json). The app writes it into every export (decision D-022). Older exports and other tools may not have one (§5.4).

```json
{
  "nhan_card_version": 1,
  "format": "yolo",
  "label_style": "joined",
  "separator": ".",
  "lowercase": true,
  "root_category": "fruit",
  "classes": ["mango", "strawberry"],
  "sub_categories": [{"name": "condition", "values": ["ripe", "unripe"]},
                     {"name": "type", "values": ["alphonso", "kesar"]}],
  "labels": [{"name": "mango.ripe.alphonso", "class": "mango",
              "attributes": {"condition": "ripe", "type": "alphonso"}},
             {"name": "strawberry", "class": "strawberry", "attributes": {}}]
}
```

### 3.1 Rules

- `labels` lists **every label name in the files, in export order**. In YOLO, the position in this list is the class index.
- Every name in the card is written exactly as in the files. When `lowercase` is true, all names are lower case.
- A sub-category with no value is **absent** from `attributes`. The text "none" or "null" never appears as a value.
- Readers must refuse a `nhan_card_version` they do not know.
- `shapes` is `boxes`, `polygons` or `mixed`.

### 3.2 Label styles

| `label_style` | Label name in the files | Values come from |
|---|---|---|
| `single` | the class | (no sub-categories) |
| `joined` | class + values in `sub_categories` order, joined by `separator`; missing values are left out (`Mango_Ripe`, `Strawberry`) | the card's `labels[].attributes` |
| `separate_fields` | the class | COCO `attributes` / VOC `<attributes>` per box; card `attributes` are `{}` |

Never split a joined name on the separator: a class or a value can contain the separator. Always use the card.

## 4. The common model, and `dataset.json`

Every reader produces the same model, whatever the source format:

```
Dataset  name, source_format, classes[], sub_categories[{name, values[]}], labels[{name, class, attributes}], images[]
Image    id, file, width, height, split ("train" | "val" | "test"), boxes[]
Box      label, class, attributes{sub-category: value}, x, y, w, h, polygon? [[x, y], …]
```

`x, y` is the top-left corner. `x, y, w, h` are normalised 0..1 of the image size. A polygon label also has `polygon` (at least 3 corners, normalised). `x, y, w, h` is then the box around it. A box has no `polygon` key. `dataset.json` is this model as JSON. Its schema is [`schema/nhan-dataset.schema.json`](schema/nhan-dataset.schema.json). `file` is relative to the JSON file, with `/` separators, and numbers have 6 decimals.

### 4.1 Coordinates per format

| Format | to normalised x, y, w, h |
|---|---|
| COCO | `x = bx/W, y = by/H, w = bw/W, h = bh/H` |
| YOLO | `x = cx − w/2, y = cy − h/2` (w, h as written) |
| VOC | `x = (xmin − 1)/W, y = (ymin − 1)/H, w = (xmax − xmin + 1)/W, h = (ymax − ymin + 1)/H` |
| COCO polygon | corner `i` = `(seg[0][2i]/W, seg[0][2i+1]/H)`; box = bounds of the corners. Only the first ring is read; RLE masks stay boxes. |
| YOLO segmentation | corners as written; an upright 4-corner rectangle (§2.2) is a box |

YOLO files do not store the image size. Read it from the image header (JPEG SOF marker, PNG IHDR). This needs no image library.

## 5. Reading, step by step

1. **Open.** A zip: unzip it. Refuse any member path that leaves the target folder ("zip slip"). Find the dataset root: the folder that holds `nhan.json`, `data.yaml`, `annotations/` or `Annotations/`, going down one level if the zip has a single top folder.
2. **Card.** Read `nhan.json` if present.
3. **Format.** Take `card.format`. Without a card: `Annotations/*.xml` means VOC, `data.yaml` or `labels/**/*.txt` means YOLO, `annotations/*.json` means COCO. **Check for the files, not only the folder name**: macOS and Windows ignore letter case, so `annotations` and `Annotations` are the same folder there.
4. **Read** the format files into the model, with `label` and `class` both set to the name in the file, and any per-box attributes.
5. **Resolve labels.** For every box whose label is in `card.labels`: set `class` from the card, and set `attributes` = the card's attributes, overlaid with the attributes from the file.
6. **Dataset fields.** With a card: `classes`, `sub_categories`, and label order come from it; `name` comes from `card.project.name`. Without a card: collect them from the boxes, and `name` is the folder name.

### 5.1 XML

VOC files are plain UTF-8 XML with the 5 standard entities and numeric character references. A full XML parser is fine. The Python SDK has a small fallback reader for systems where the XML parser library is missing (some Homebrew Python builds).

### 5.2 data.yaml

Only a small YAML subset is used: `key: value` lines, a flow list (`[a, 'b c']`), possibly over several lines, or a block list (`- a`), or an index map (`0: a`). `#` starts a comment. A YAML library is not needed.

### 4.2 Photo orientation

Nhan exports store every photo **upright** (any EXIF Orientation is applied, and the file has no orientation tag), and labels refer to that upright photo. Readers of other datasets should apply EXIF Orientation before using pixels, as the Python SDK's `Image.open()` does.

### 4.3 Resized exports

The export preset can shrink photos (`images.longest_side` in the card). The app stores labels as shares of the photo (0..1), and every pixel value in the files is computed from the **exported** photo size, so labels shrink with the photo:

- COCO `bbox`, `segmentation`, `area`, and VOC corners use the exported width and height, which are also written in the files (`images[].width/height`, `<size>`).
- YOLO and the common model are normalised, so they are the same at every size.
- A photo smaller than `longest_side` is never enlarged.

The fixtures `coco_resized` and `voc_resized` (longest side 40 px) must give the same normalised boxes and polygons as `coco_joined` and `voc_joined` (VOC: within 1 pixel, because it stores whole pixels).

## 6. Writing

Writers produce the layouts of §2 and a new card (with `"written_by"`). Rules:

- Do not write into a folder that is not empty.
- Round pixel values half away from zero (§2.3), so a port writes the same numbers as the app.
- **YOLO:** if a box has values but its label is its class (the separate-fields style), write the joined name (`class` + values, card `separator`, default `_`). No value is lost.
- **COCO / VOC:** write per-box attributes only when the label equals the class (the separate-fields style).
- **Polygons:** COCO writes `segmentation` and the polygon `area`. YOLO switches every line to segmentation lines when any box has a polygon. VOC writes the box.
- **Card labels:** a label's entry carries `class` and `attributes` only when every box with that label agrees on them. Otherwise write `{"name": L, "class": L, "attributes": {}}`. This happens after relabelling by a value, for example. A reader must never get values that are not true for every box.

## 7. Checking a port

`fixtures/` holds real exports made by the app's code: the C++ export engine plus the Dart manifest and card code, through `packages/nhan_data/tool/make_sdk_fixtures.dart`.

| Fixture | Format | Style | Notes |
|---|---|---|---|
| `coco_single` | COCO | single | no sub-categories |
| `coco_joined` | COCO | joined, `_` | |
| `coco_fields` | COCO | separate fields | `attributes`, `supercategory` |
| `yolo_joined` | YOLO | joined, `.`, lowercase | |
| `voc_fields` | VOC | separate fields | `<attributes>` |
| `voc_joined` | VOC | joined, `_` | |
| `coco_resized` | COCO | joined, `_` | photos shrunk to longest side 40 px |
| `voc_resized` | VOC | joined, `_` | photos shrunk to longest side 40 px |

Data: 4 photos (80×60, 64×48, 60×80, 40×40), split 2/1/1. One photo has no boxes. Values: `Type` only fits Mango, and some boxes have no value. The unripe mango on `photo-2` is a **polygon** (a diamond inside its box), so `shapes` is `mixed`. COCO and YOLO keep the polygon; VOC gives its box.

**A port is correct when**, for every fixture, its model written as `dataset.json` (images sorted by `id`, paths relative to the fixture folder) equals `fixtures/expected/<fixture>.json`. Compare numbers after rounding to 6 decimals.

## 8. Map of the Python implementation

| File | Job |
|---|---|
| `load.py` | §5 steps 1–3, 5–6 |
| `readers/coco.py`, `yolo.py`, `voc.py` | §5 step 4 |
| `card.py` | §3 |
| `model.py` | §4 model; filter, relabel, resplit, counts |
| `writers.py` | §6 and `dataset.json` |
| `imagesize.py`, `minixml.py` | §4.1 image header, §5.1 fallback XML |
| `preview.py`, `crops.py`, `torch_data.py`, `train.py`, `cli.py` | tools on top; not needed for a port |
