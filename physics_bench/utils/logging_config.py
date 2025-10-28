import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_benchmark_logger(log_file: Optional[str] = None) -> logging.Logger:
    """벤치마크용 로거 설정"""
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택사항)
    if log_file:
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def generate_log_filename(model_name: str = "", test_name: str = "", prefix: str = "benchmark") -> str:
    """로그 파일명 생성 (모델명/datetime 폴더 구조)"""
    # datetime 기반 폴더명 생성 (YYYYMMDD_HHMMSS 형식)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # outputs 폴더 아래에 모델명/datetime 구조 생성
    log_dir = Path("outputs") / model_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일명 생성
    if test_name:
        log_filename = f"{prefix}_{test_name}.log"
    else:
        log_filename = f"{prefix}.log"

    # 전체 경로 반환
    return str(log_dir / log_filename)
