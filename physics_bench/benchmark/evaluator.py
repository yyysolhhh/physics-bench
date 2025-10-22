import re
import math
import sympy as sp
from dataclasses import dataclass
from typing import Optional, Tuple, Union


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
    """물리학 문제에 특화된 평가기 - 수학 기호와 소수점 모두 지원"""
    
    @staticmethod
    def _evaluate_latex_expression(latex_expr: str) -> Optional[float]:
        """LaTeX 수학 표현식을 계산하여 소수점 값으로 변환"""
        try:
            # LaTeX 수식 정리
            expr = latex_expr.strip()
            
            # LaTeX 특수 문자 변환
            expr = expr.replace('\\sqrt{', 'sqrt(')
            expr = expr.replace('\\pi', 'pi')
            expr = expr.replace('\\sin', 'sin')
            expr = expr.replace('\\cos', 'cos')
            expr = expr.replace('\\tan', 'tan')
            expr = expr.replace('\\frac{', '(')
            expr = expr.replace('}{', '/')
            expr = expr.replace('}', ')')
            expr = expr.replace('$', '')
            
            # SymPy로 계산
            sympy_expr = sp.sympify(expr)
            result = float(sympy_expr.evalf())
            return result
        except:
            return None
    
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
        """단위 정규화 - 기본 SI 단위로 변환"""
        unit = unit.strip()
        
        # LaTeX 단위에서 실제 단위 추출
        unit = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', unit)
        unit = re.sub(r'\\~', '', unit)
        unit = re.sub(r'\\cdot', '·', unit)
        unit = re.sub(r'\$', '', unit)
        unit = unit.strip()
        
        # 단위 변환 매핑 (기본 SI 단위로)
        unit_mapping = {
            # 길이
            'mm': 'm', 'cm': 'm', 'km': 'm', 'm': 'm',
            '미터': 'm', '밀리미터': 'm', '센티미터': 'm', '킬로미터': 'm',
            # 시간  
            's': 's', 'sec': 's', 'min': 's', 'hour': 's', 'h': 's',
            '초': 's', '분': 's', '시간': 's',
            # 질량
            'g': 'kg', 'kg': 'kg', 'mg': 'kg',
            '그램': 'kg', '킬로그램': 'kg', '밀리그램': 'kg',
            # 온도
            'K': 'K', '°C': 'K', '°F': 'K', 'C': 'K',
            '켈빈': 'K', '섭씨': 'K', '화씨': 'K',
            # 에너지
            'J': 'J', 'kJ': 'J', 'MJ': 'J', 'cal': 'J',
            '줄': 'J', '킬로줄': 'J', '메가줄': 'J', '칼로리': 'J',
            # 힘
            'N': 'N', '뉴턴': 'N',
            # 전하
            'C': 'C', 'μC': 'C', 'nC': 'C', 'pC': 'C',
            '쿨롱': 'C', '마이크로쿨롱': 'C',
            # 압력
            'Pa': 'Pa', 'bar': 'Pa', 'atm': 'Pa',
            '파스칼': 'Pa', '바': 'Pa', '기압': 'Pa',
            # 속도
            'm/s': 'm/s', 'm/s²': 'm/s²',
            # 각도
            'rad': 'rad', '°': 'rad', '도': 'rad',
        }
        
        return unit_mapping.get(unit.lower(), unit)
    
    @staticmethod
    def _convert_to_base_unit(value: float, unit: str) -> float:
        """값을 기본 SI 단위로 변환"""
        unit = unit.strip().lower()
        
        # 길이 변환 (mm 기준)
        if unit in ['mm', '밀리미터']:
            return value * 1e-3  # mm -> m
        elif unit in ['cm', '센티미터']:
            return value * 1e-2  # cm -> m  
        elif unit in ['m', '미터']:
            return value * 1e-3  # m -> mm
        elif unit in ['km', '킬로미터']:
            return value * 1e3   # km -> m
        
        # 시간 변환 (s 기준)
        elif unit in ['s', 'sec', '초']:
            return value
        elif unit in ['min', '분']:
            return value * 60    # min -> s
        elif unit in ['hour', 'h', '시간']:
            return value * 3600  # hour -> s
        
        # 질량 변환 (kg 기준)
        elif unit in ['g', '그램']:
            return value * 1e-3  # g -> kg
        elif unit in ['kg', '킬로그램']:
            return value
        elif unit in ['mg', '밀리그램']:
            return value * 1e-6  # mg -> kg
        
        # 온도 변환 (K 기준)
        elif unit in ['k', '켈빈']:
            return value
        elif unit in ['°c', 'c', '섭씨']:
            return value + 273.15  # °C -> K
        elif unit in ['°f', 'f', '화씨']:
            return (value - 32) * 5/9 + 273.15  # °F -> K
        
        # 에너지 변환 (J 기준)
        elif unit in ['j', '줄']:
            return value
        elif unit in ['kj', '킬로줄']:
            return value * 1e3   # kJ -> J
        elif unit in ['mj', '메가줄']:
            return value * 1e6   # MJ -> J
        elif unit in ['cal', '칼로리']:
            return value * 4.184  # cal -> J
        
        # 전하 변환 (C 기준)
        elif unit in ['c', '쿨롱']:
            return value
        elif unit in ['μc', '마이크로쿨롱']:
            return value * 1e-6  # μC -> C
        elif unit in ['nc', '나노쿨롱']:
            return value * 1e-9  # nC -> C
        elif unit in ['pc', '피코쿨롱']:
            return value * 1e-12  # pC -> C
        
        # 압력 변환 (Pa 기준)
        elif unit in ['pa', '파스칼']:
            return value
        elif unit in ['bar', '바']:
            return value * 1e5   # bar -> Pa
        elif unit in ['atm', '기압']:
            return value * 101325  # atm -> Pa
        
        # 기본값 (변환 없음)
        return value
    
    @staticmethod
    def _is_numerically_close(pred_num: float, true_num: float, tolerance: float = 0.01) -> bool:
        """수치적 근사치 비교 - 소수점 2자리까지 허용"""
        if true_num == 0:
            return abs(pred_num) < tolerance
        
        # 소수점 2자리까지 비교 (0.01의 오차 허용)
        pred_rounded = round(pred_num, 2)
        true_rounded = round(true_num, 2)
        
        return abs(pred_rounded - true_rounded) < 0.01
    
    def evaluate_single(self, predicted: str, ground_truth: str) -> bool:
        """단일 답변 평가 - 다단계 평가 시스템"""
        
        # 1단계: 직접 문자열 비교 (정확한 일치)
        if predicted.strip().lower() == ground_truth.strip().lower():
            return True
        
        # 2단계: LaTeX 수식 평가
        pred_latex = self._evaluate_latex_expression(predicted)
        true_latex = self._evaluate_latex_expression(ground_truth)
        
        if pred_latex is not None and true_latex is not None:
            if self._is_numerically_close(pred_latex, true_latex):
                return True
        
        # 3단계: 일반 수치 추출 및 비교
        pred_num = self._extract_number(predicted)
        true_num = self._extract_number(ground_truth)
        
        if pred_num is not None and true_num is not None:
            # 수치 비교
            numerical_match = self._is_numerically_close(pred_num, true_num)
            
            # 단위 비교 (LaTeX 문법 그대로)
            pred_unit = self._extract_unit(predicted)
            true_unit = self._extract_unit(ground_truth)
            
            # 단위 변환을 고려한 비교
            if true_unit and pred_unit:
                # 1. 단위 정규화
                pred_unit_norm = self._normalize_unit(pred_unit)
                true_unit_norm = self._normalize_unit(true_unit)
                
                # 2. 단위가 같으면 수치만 비교
                if pred_unit_norm == true_unit_norm:
                    return numerical_match
                
                # 3. 단위가 다르면 변환 후 비교
                try:
                    pred_converted = self._convert_to_base_unit(pred_num, pred_unit)
                    true_converted = self._convert_to_base_unit(true_num, true_unit)
                    
                    # 변환된 값으로 비교
                    converted_match = self._is_numerically_close(pred_converted, true_converted)
                    return converted_match
                except:
                    # 변환 실패 시 원래 방식으로 비교
                    unit_match = pred_unit.strip() == true_unit.strip()
                    return numerical_match and unit_match
            else:
                return numerical_match
        
        # 4단계: 마지막 수단 - 문자열 유사도 비교
        return predicted.strip().lower() == ground_truth.strip().lower()
    
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