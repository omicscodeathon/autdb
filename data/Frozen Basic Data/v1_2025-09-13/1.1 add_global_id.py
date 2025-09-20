import pandas as pd
import re
from pathlib import Path
import yaml
import json

# Use current folder
frozen_dir = Path(".")

# 1. Update metadata_ml_ready_splits.xlsx
excel_file = frozen_dir / "metadata_ml_ready_splits.xlsx"
df = pd.read_excel(excel_file)

def make_global(row):
    if row["dataset"] == "MMASD":
        m = re.search(r'(\d+)', str(row["sample_id"]))
        return f"MM_{m.group(1)}" if m else None
    elif row["dataset"] == "Engagnition":
        m = re.search(r'P(\d+)', str(row["sample_id"]))
        return f"EN_P{m.group(1)}" if m else None
    return None

df["participant_id_global"] = df.apply(make_global, axis=1)

out_excel = frozen_dir / "metadata_ml_ready_splits_withGlobalID.xlsx"
df.to_excel(out_excel, index=False)
print(f"Updated Excel saved to: {out_excel}")

# 2. Update schema.yaml
yaml_file = frozen_dir / "schema.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    schema = yaml.safe_load(f)

if "fields" in schema and "participant_id_global" not in [fld["name"] for fld in schema["fields"]]:
    schema["fields"].append({
        "name": "participant_id_global",
        "type": "string",
        "description": "Global participant ID (MM_xxx or EN_Pxx)"})

out_yaml = frozen_dir / "schema_withGlobalID.yaml"
with open(out_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(schema, f, allow_unicode=True)
print(f"Updated YAML saved to: {out_yaml}")

# 3. Update splits_manifest.json
json_file = frozen_dir / "splits_manifest.json"
with open(json_file, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# If schema-like fields exist in JSON, add new field
if "fields" in manifest and "participant_id_global" not in [fld["name"] for fld in manifest["fields"]]:
    manifest["fields"].append({
        "name": "participant_id_global",
        "type": "string",
        "description": "Global participant ID (MM_xxx or EN_Pxx)"})

out_json = frozen_dir / "splits_manifest_withGlobalID.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"Updated JSON saved to: {out_json}")
