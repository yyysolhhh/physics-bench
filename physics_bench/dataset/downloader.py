import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def download_huggingface_dataset(
        dataset_name: str,
        split: str,
        output_file: Optional[str] = None,
        limit: Optional[int] = None
) -> None:
    base_dir = Path("dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    if output_file is None:
        dataset_suffix = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        output_file = f"{dataset_suffix}_{split}.json"
    print(f"Hugging Face에서 '{dataset_name}' 다운로드 중...")

    try:
        # load_dataset 사용
        dataset = load_dataset(dataset_name, split=split)
        print(f"데이터셋 로드 완료: {len(dataset)}개 행")
        
        # 데이터를 dict 리스트로 변환
        items = []
        for i, row in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            items.append(dict(row))
        
    except Exception as e:
        print(f"데이터셋 다운로드 실패: {e}")
        raise ValueError(f"데이터셋 '{dataset_name}'을 다운로드할 수 없습니다: {e}")

    else:
        if limit is not None and len(items) > limit:
            items = items[:limit]
    
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    
        print(f"{len(items)}개 항목 다운로드 완료: {output_path}")
