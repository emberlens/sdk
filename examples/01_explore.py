# Look inside an export and pick the photos you want.
# Run:  ./nhan run examples/01_explore.py path/to/your-export.zip
import sys
from pathlib import Path

import nhan

# No path given: use the sample export that ships with the SDK.
sample = Path(__file__).resolve().parent.parent / "fixtures" / "coco_joined"
ds = nhan.load(sys.argv[1] if len(sys.argv) > 1 else sample)
print(ds.summary())

# Every photo and box
for photo in ds:
    for box in photo.boxes:
        print(photo.id, photo.split, box.cls, box.attributes, f"x={box.x:.2f} y={box.y:.2f}")

# Only unripe mangoes (letter case does not matter)
unripe = ds.filter(cls="mango", condition="unripe")
print(f"\nUnripe mangoes: {unripe.box_count} box(es) in {len(unripe)} photo(s)")

# Make an HTML page to look at them
print("Preview:", unripe.preview("unripe-preview.html"))
