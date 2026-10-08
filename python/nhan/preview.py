"""preview.html: every photo with its boxes, in any web browser. No extra libraries."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990", "#800000"]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fafaf7; --fg:#1c1c1a; --muted:#6b6b66; --card:#ffffff; --line:#e3e3dc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecece6; --muted:#9a9a92; --card:#22221f; --line:#34342f; }} }}
body {{ margin:0; font:15px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }}
header {{ position:sticky; top:0; background:var(--bg); padding:16px; border-bottom:1px solid var(--line); z-index:1; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.meta {{ color:var(--muted); margin-bottom:10px; }}
select {{ font:inherit; padding:4px 8px; margin-right:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:12px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.pic {{ position:relative; }}
.pic img {{ display:block; width:100%; height:auto; }}
.pic svg {{ position:absolute; inset:0; width:100%; height:100%; }}
.pic svg * {{ vector-effect:non-scaling-stroke; }}
.cap {{ padding:8px 10px; font-size:13px; color:var(--muted); }}
.tag {{ display:inline-block; padding:1px 6px; border-radius:6px; color:#fff; margin:2px 2px 0 0; font-size:12px; }}
</style></head><body>
<header><h1>{title}</h1><div class="meta">{meta}</div>
<label>Label <select id="label"><option value="">All</option>{label_options}</select></label>
<label>Split <select id="split"><option value="">All</option><option>train</option><option>val</option><option>test</option></select></label>
<span id="count"></span></header>
<main class="grid" id="grid">{cards}</main>
<script>
const label = document.getElementById('label'), split = document.getElementById('split');
function apply() {{
  let n = 0;
  for (const c of document.querySelectorAll('.card')) {{
    const ok = (!label.value || c.dataset.labels.split('|').includes(label.value)) && (!split.value || c.dataset.split === split.value);
    c.style.display = ok ? '' : 'none'; if (ok) n++;
  }}
  document.getElementById('count').textContent = n + ' photos';
}}
label.onchange = split.onchange = apply; apply();
</script></body></html>
"""


def write_preview(ds, path="preview.html", limit: int = 500) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    colour = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(ds.labels)}
    cards = []
    for img in ds.images[:limit]:
        src = os.path.relpath(img.path.resolve(), out.parent).replace(os.sep, "/")
        rects, tags = [], []
        for b in img.boxes:
            c = colour.get(b.label, "#888")
            text = html.escape(b.label + "".join(f" · {v}" for v in b.attributes.values() if b.label == b.cls))
            if b.polygon is not None:
                pts = " ".join(f"{px:.5f},{py:.5f}" for px, py in b.polygon)
                rects.append(
                    f'<polygon points="{pts}" fill="{c}" fill-opacity="0.25" stroke="{c}" '
                    f'stroke-width="2"><title>{text}</title></polygon>'
                )
            else:
                rects.append(
                    f'<rect x="{b.x:.5f}" y="{b.y:.5f}" width="{b.w:.5f}" height="{b.h:.5f}" '
                    f'fill="none" stroke="{c}" stroke-width="2"><title>{text}</title></rect>'
                )
            tags.append(f'<span class="tag" style="background:{c}">{text}</span>')
        cards.append(
            f'<div class="card" data-split="{img.split}" data-labels="{html.escape("|".join(sorted({b.label for b in img.boxes})))}">'
            f'<div class="pic"><img loading="lazy" src="{html.escape(src)}" alt="{html.escape(img.id)}">'
            f'<svg viewBox="0 0 1 1" preserveAspectRatio="none">{"".join(rects)}</svg></div>'
            f'<div class="cap">{html.escape(img.id)} · {img.split}<br>{"".join(tags) or "no boxes"}</div></div>'
        )
    meta = f"{ds.format.upper()} · {len(ds.images)} photos · {ds.box_count} boxes"
    if len(ds.images) > limit:
        meta += f" · showing the first {limit}"
    page = PAGE.format(
        title=html.escape(ds.name),
        meta=html.escape(meta),
        label_options="".join(f"<option>{html.escape(n)}</option>" for n in ds.labels),
        cards="\n".join(cards),
    )
    out.write_text(page, encoding="utf-8")
    return out
