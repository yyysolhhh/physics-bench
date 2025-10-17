import json
from pathlib import Path
from typing import Optional, Any, Dict, List

from datasets import load_dataset, load_dataset_builder


def _ensure_dataset_dir() -> Path:
    base_dir = Path("dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _default_output_name(dataset_name: str, split: str) -> str:
    dataset_suffix = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    return f"{dataset_suffix}_{split}.json"


def _collect_items(dataset_name: str, split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        raise ValueError(f"데이터셋 '{dataset_name}' '{split}' 로드 실패: {e}") from e

    items: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        items.append(dict(row))
    return items


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_metadata_index(index_path: Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _build_metadata_entry(dataset_name: str, split: str, num_samples: int) -> Dict[str, Any]:
    builder = load_dataset_builder(dataset_name)
    info = builder.info
    return {
        "dataset": dataset_name,
        "split": split,
        "num_samples": num_samples,
        "license": str(getattr(info, "license", None)) if getattr(info, "license", None) else None,
        "homepage": info.homepage,
        "citation": info.citation,
        "description": info.description,
        "features": info.features.to_dict() if info.features is not None else None,
    }


def download_huggingface_dataset(
        dataset_name: str,
        split: str,
        output_file: Optional[str] = None,
        limit: Optional[int] = None
) -> None:
    base_dir = _ensure_dataset_dir()
    final_name = output_file or _default_output_name(dataset_name, split)
    output_path = Path(final_name)
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    print(f"Hugging Face에서 '{dataset_name} ({split})' 다운로드 중... ")
    items = _collect_items(dataset_name=dataset_name, split=split, limit=limit)

    _write_json(output_path, items)

    meta_index_path = base_dir / "metadata.json"
    meta_index = _load_metadata_index(meta_index_path)

    entry = _build_metadata_entry(dataset_name=dataset_name, split=split, num_samples=len(items))

    replaced = False
    for i, e in enumerate(meta_index):
        if e.get("dataset") == entry["dataset"] and e.get("split") == entry["split"]:
            meta_index[i] = entry
            replaced = True
            break
    if not replaced:
        meta_index.append(entry)

    _write_json(meta_index_path, meta_index)

    print(f"{len(items)}개 항목 다운로드 완료: {output_path}")
    print(f"메타데이터 누적 저장: {meta_index_path}")
