"""Quick Hugging Face dataset configuration and schema inspector."""
import json
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset_builder

DATASETS = {
    "BiScope": "HanxiGuo/BiScope_Data",
    "PADBen": "JonathanZha/PADBen",
}

out = Path(__file__).resolve().parents[1] / "outputs"
out.mkdir(parents=True, exist_ok=True)
report = {}

for name, dataset_id in DATASETS.items():
    print(f"\n### {name}: {dataset_id}")
    try:
        configs = get_dataset_config_names(dataset_id)
        report[name] = {}
        print("Configs:", configs)
        for config in configs:
            builder = load_dataset_builder(dataset_id, name=config)
            info = builder.info
            features = {}
            for col, feature in (info.features or {}).items():
                features[col] = {
                    "type": feature.__class__.__name__,
                    "dtype": getattr(feature, "dtype", None),
                    "names": list(feature.names) if hasattr(feature, "names") else None,
                }
            report[name][config] = {
                "splits": list(info.splits.keys()) if info.splits else [],
                "features": features,
                "description": (info.description or "")[:2000],
                "license": info.license,
            }
            print(f"\n[{config}]")
            print("Splits:", report[name][config]["splits"])
            for col, feature in features.items():
                print(f"  {col}: {feature}")
    except Exception as exc:
        report[name] = {"error": str(exc)}
        print("ERROR:", exc)

target = out / "quick_schema_inspection.json"
target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {target}")
