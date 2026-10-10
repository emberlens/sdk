"""One reader per export format. Each returns (images, label_order).

Readers fill Box.label and the attributes found in the files. The loader
then fills Box.cls and joined-name attributes from emberlens.json.
"""
