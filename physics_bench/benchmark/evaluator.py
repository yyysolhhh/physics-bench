import re
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class PhysicsEvaluator:
    """물리학 문제에 특화된 평가기"""
    
    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """텍스트에서 수치 추출"""
        # LaTeX 수식 제거
        text = re.sub(r'\$.*?\$', '', text)
        
        # 숫자 패턴 찾기 (음수, 소수, 과학적 표기법 포함)
        patterns = [
            r'-?\d+\.?\d*',  # 기본 숫자
            r'-?\d+\.\d+e[+-]?\d+',  # 과학적 표기법
            r'-?\d+\.\d+',  # 소수
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        return None
    
    @staticmethod
    def _extract_unit(text: str) -> str:
        """텍스트에서 단위 추출"""
        # LaTeX 단위 패턴
        latex_units = re.findall(r'\\mathrm\{[^}]*\}', text)
        if latex_units:
            return latex_units[0]
        
        # 일반 단위 패턴
        units = re.findall(r'\b(m|s|kg|N|J|W|V|A|K|°C|°F|Hz|Pa|bar|atm|mol|m/s|m/s²|kg/m³|J/K|W/m²)\b', text)
        if units:
            return units[0]
        
        return ""
    
    @staticmethod
    def _normalize_unit(unit: str) -> str:
        """단위 정규화"""
        unit = unit.strip().lower()
        
        # LaTeX 단위 정규화
        unit = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', unit)
        unit = re.sub(r'\\~', '', unit)
        unit = re.sub(r'\\cdot', '·', unit)
        
        # 공통 단위 매핑
        unit_mapping = {
            'm': 'm',
            'meter': 'm',
            'meters': 'm',
            's': 's',
            'second': 's',
            'seconds': 's',
            'kg': 'kg',
            'kilogram': 'kg',
            'kilograms': 'kg',
            'k': 'k',
            'kelvin': 'k',
            'celsius': '°c',
            'fahrenheit': '°f',
        }
        
        return unit_mapping.get(unit, unit)
    
    @staticmethod
    def _is_numerically_close(pred_num: float, true_num: float, tolerance: float = 1e-6) -> bool:
        """수치적 근사치 비교"""
        if true_num == 0:
            return abs(pred_num) < tolerance
        
        relative_error = abs(pred_num - true_num) / abs(true_num)
        return relative_error < tolerance
    
    def evaluate_single(self, predicted: str, ground_truth: str) -> bool:
        """단일 답변 평가"""
        # 1. 수치 추출
        pred_num = self._extract_number(predicted)
        true_num = self._extract_number(ground_truth)
        
        if pred_num is None or true_num is None:
            # 수치 추출 실패 시 문자열 비교
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        # 2. 단위 추출
        pred_unit = self._extract_unit(predicted)
        true_unit = self._extract_unit(ground_truth)
        
        # 3. 단위 정규화
        pred_unit_norm = self._normalize_unit(pred_unit)
        true_unit_norm = self._normalize_unit(true_unit)
        
        # 4. 수치 비교
        numerical_match = self._is_numerically_close(pred_num, true_num)
        
        # 5. 단위 비교 (단위가 있는 경우에만)
        if true_unit_norm and pred_unit_norm:
            unit_match = pred_unit_norm == true_unit_norm
            return numerical_match and unit_match
        else:
            # 단위가 없으면 수치만 비교
            return numerical_match
    
    def evaluate(self, y_true: list[str], y_pred: list[str]) -> EvaluationResult:
        """배치 평가"""
        assert len(y_true) == len(y_pred)
        total = len(y_true)
        correct = 0
        
        for true_answer, pred_answer in zip(y_true, y_pred):
            if self.evaluate_single(pred_answer, true_answer):
                correct += 1
        
        return EvaluationResult(total=total, correct=correct)


# 기존 호환성을 위한 별칭
ExactMatchEvaluator = PhysicsEvaluator