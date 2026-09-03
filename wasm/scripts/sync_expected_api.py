#!/usr/bin/env python3
"""Sync expected_api.json from parity_data.go (w2a-only)."""
import json, re, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
text = (root / "parity_data.go").read_text()
items = re.findall(r'"([^"]+)"', text.split("var totalParityItems")[1].split("}")[0])
for dest in [root / "expected_api.json", root.parent / "typescript" / "assets" / "expected_api.json"]:
    dest.write_text(json.dumps(items, indent=2) + "\n")
    print(dest, len(items))
