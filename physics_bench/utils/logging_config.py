import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_console_logging(log_file: Optional[str] = None):
    """모든 로거가 콘솔에 출력되도록 기본 설정 (선택적으로 파일에도 저장)
    
    Args:
        log_file: 로그 파일 경로. None이면 콘솔에만 출력, 제공되면 파일에도 저장
    """
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러 추가
    console_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def setup_benchmark_logger(log_file: Optional[str] = None) -> logging.Logger:
    """벤치마크용 로거 설정
    
    Args:
        log_file: 로그 파일 경로. None이면 콘솔에만 출력, 제공되면 파일에도 저장
    """
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 콘솔 핸들러 추가
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택사항)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def generate_log_filename(model_name: str = "", test_name: str = "", prefix: str = "benchmark") -> str:
    """로그 파일명 생성 (outputs/모델명/datetime 폴더 구조)"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    log_dir = Path("outputs") / model_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    if test_name:
        log_filename = f"{prefix}_{test_name}.log"
    else:
        log_filename = f"{prefix}.log"

    return str(log_dir / log_filename)
