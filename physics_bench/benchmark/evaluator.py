from dataclasses import dataclass
from typing import Optional, Tuple, List

from physics_bench.llm import LLMRegistry
from physics_bench.prompts import PHYSICS_MODEL_JUDGE_PROMPT


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
    # \boxed{ 패턴 (다양한 형식 지원)
    # 패턴 설명: r'(?:\\\\)*(?:\\\$|\$)?\\+boxed\{'
    # - (?:\\\\)* : 백슬래시 0개 이상 (이스케이프된 백슬래시)
    # - (?:\\\$|\$)? : 이스케이프된 달러 또는 일반 달러 (선택적)
    # - \\+ : 백슬래시 1개 이상 (boxed 앞의 실제 백슬래시)
    # - boxed\{ : 리터럴 "boxed{"
    _BOXED_PATTERN = r'(?:\\\\)*(?:\\\$|\$)?\\+boxed\{'
    
    def __init__(self) -> None:
        self._llm = None
        self.last_extracted_pred: Optional[str] = None
        self.last_normalized_gt: Optional[str] = None
        self.last_model_judge_msg: Optional[str] = None
        self.last_student_solution: Optional[str] = None  # 2차 평가에 사용된 student_solution (풀이만)
        self.last_student_answers: Optional[str] = None  # 2차 평가에 사용된 student_answers (정답만)

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        provider = "ollama"
        eval_model = "llama3.1:8b"
        temperature = 0.0
        max_tokens = 512

        self._llm = LLMRegistry.create_client(
            provider=provider,
            model_name=eval_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._llm

    # ---------- 1) 정답 추출 -----------
    @staticmethod
    def _find_matching_brace(text: str, open_brace_pos: int) -> Tuple[Optional[str], Optional[int]]:
        """중괄호 깊이를 추적하여 매칭되는 닫는 중괄호를 찾습니다.
        
        Args:
            text: 검색할 텍스트
            open_brace_pos: 열린 중괄호 '{'의 위치
            
        Returns:
            (content, end_pos): 중괄호 안의 내용과 닫는 중괄호 다음 위치, 매칭 실패 시 (None, None)
        """
        depth = 0
        i = open_brace_pos + 1
        content_start = i
        
        while i < len(text):
            char = text[i]
            
            # 백슬래시 다음 문자는 이스케이프되므로 건너뛰기
            if char == '\\' and i + 1 < len(text):
                i += 2
                continue
            
            if char == '{':
                depth += 1
            elif char == '}':
                if depth == 0:
                    # 매칭되는 닫는 중괄호를 찾음
                    content = text[content_start:i].strip()
                    end_pos = i + 1
                    return content, end_pos
                else:
                    depth -= 1
            
            i += 1
        
        return None, None

    @staticmethod
    def _extract_boxed_all(text: str) -> List[str]:
        """다양한 형식의 \\boxed를 추출: $\\boxed, $\boxed, \\boxed, \boxed (이스케이프 문자 고려)
        LaTeX 중첩 중괄호를 올바르게 처리합니다.
        마지막 boxed를 찾기 위해 텍스트 끝에서부터 역순으로 검색합니다.
        """
        import re
        boxes = []
        
        # \boxed{ 패턴 찾기 (다양한 형식 지원)
        # 매칭 예시:
        # - $\\boxed{...}  (달러 + 백슬래시 2개)
        # - $\boxed{...}   (달러 + 백슬래시 1개)
        # - \\boxed{...}   (백슬래시 2개만)
        # - \boxed{...}    (백슬래시 1개만)
        # - \$\\boxed{...} (이스케이프된 달러 + 백슬래시 2개)
        # - \\$\\boxed{...} (백슬래시 + 달러 + 백슬래시 2개)
        matches = list(re.finditer(PhysicsEvaluator._BOXED_PATTERN, text))
        
        if not matches:
            return boxes
        
        # 마지막 매칭부터 처리 (보통 답변 끝에 boxed가 있음)
        for match in reversed(matches):
            open_brace_pos = match.end() - 1  # '{'의 위치
            
            content, _ = PhysicsEvaluator._find_matching_brace(text, open_brace_pos)
            if content:
                boxes.append(content)
                # 마지막 boxed를 찾았으므로 바로 반환
                return boxes
        
        return boxes

    @staticmethod
    def _split_answers(ans: str) -> List[str]:
        parts = [p.strip() for p in ans.split(',')]
        return [p for p in parts if p]

    @staticmethod
    def _normalize_tex(s: str) -> str:
        """LaTeX 텍스트 정규화 (비교용)"""
        import re
        t = s
        t = t.replace('\u200b', ' ')
        t = t.replace('$', ' ')
        t = re.sub(r"\\left|\\right", "", t)
        t = re.sub(r"\\,|\\;|\\:|~", "", t)
        t = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _normalize_to_double_backslash_boxed(text: str) -> str:
        """다양한 형식의 boxed를 \\boxed{...} 형식으로 정규화"""
        import re
        # 이미 \\boxed 형식이면 그대로 반환
        if '\\boxed{' in text:
            # $ 제거
            text = text.replace('$', '')
            # \boxed를 \\boxed로 통일 (이미 \\boxed가 아닌 경우만)
            text = re.sub(r'(?<!\\)\\boxed\{', r'\\boxed{', text)
            return text.strip()

        # boxed가 없으면 원본 반환
        return text.strip()

    def _extract_final_answers(self, text: str) -> List[str]:
        """boxed에서 답을 추출하여 \\boxed{...} 형식으로 반환"""
        boxes = self._extract_boxed_all(text)
        if boxes:
            last = boxes[-1]
            # 콤마로 분리된 여러 답 처리
            answers = self._split_answers(last)
            # 각 답을 \\boxed{...} 형식으로 변환
            normalized_answers = []
            for ans in answers:
                # \\boxed{답} 형식으로 정규화 (JSON 저장 시 \\boxed로 저장됨)
                normalized = f"\\boxed{{{ans}}}"
                normalized_answers.append(normalized)
            return normalized_answers
        # 박스가 없으면 빈 결과 (엄격 모드)
        return []

    # ---------- 2) 타입별 등가성 판단(간소화) -----------
    @staticmethod
    def _try_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _sympy_equal(a: str, b: str) -> Optional[bool]:
        try:
            import sympy as sp
            a_s = a.replace('^', '**')
            b_s = b.replace('^', '**')
            expr_a = sp.sympify(a_s)
            expr_b = sp.sympify(b_s)
            return sp.simplify(expr_a - expr_b) == 0
        except Exception:
            return None

    @staticmethod
    def _num_close(x: float, y: float, tol: float = 1e-8) -> bool:
        if y == 0:
            return abs(x) <= tol
        return abs((x - y)) <= max(tol, abs(y) * tol)

    def _pair_equal(self, a: str, b: str, tol: float = 1e-8) -> bool:
        """\\boxed{...} 형식의 두 답을 비교"""
        # \\boxed{...} 형식에서 내용만 추출 (Python 문자열에서 \boxed는 실제로 \boxed이므로 정규식에서 \\boxed로 매칭)
        import re
        a_content = re.sub(r'\\boxed\{([\s\S]*)\}', r'\1', a)
        b_content = re.sub(r'\\boxed\{([\s\S]*)\}', r'\1', b)

        # 내용을 정규화
        a_norm = self._normalize_tex(a_content)
        b_norm = self._normalize_tex(b_content)

        # 수치 비교
        fa = self._try_float(a_norm)
        fb = self._try_float(b_norm)
        if fa is not None and fb is not None:
            return self._num_close(fa, fb, tol)

        # SymPy 등가성 비교
        sym = self._sympy_equal(a_norm, b_norm)
        if sym is not None:
            return bool(sym)

        # 마지막 수단: 정규화된 내용 문자열 비교
        return a_norm == b_norm

    def _auto_judge(self, pred_text: str, gt_text: str, tol: float = 1e-8) -> Tuple[bool, List[str], List[str]]:
        pred_list = self._extract_final_answers(pred_text)
        gt_list = self._extract_final_answers(gt_text)
        self.last_extracted_pred = ', '.join(pred_list) if pred_list else ''
        self.last_normalized_gt = ', '.join(gt_list) if gt_list else ''

        if not pred_list or not gt_list:
            return False, pred_list, gt_list
        if len(pred_list) != len(gt_list):
            return False, pred_list, gt_list

        remaining_gt = gt_list.copy()
        remaining_pred = pred_list.copy()
        while remaining_gt:
            g = remaining_gt.pop(0)
            for i, p in enumerate(remaining_pred):
                if self._pair_equal(g, p, tol):
                    remaining_pred.pop(i)
                    break
            else:
                return False, pred_list, gt_list
        return True, pred_list, gt_list

    # ---------- 3) 2차 모델 심판 -----------
    def _judge_with_model(
            self,
            question: str,
            student_ans_list: List[str],
            ref_ans_list: List[str],
            *,
            student_solution: str = "",
            reference_solution: str = "",
            student_boxed_answer: str = "",
            reference_boxed_answer: str = "",
    ) -> Tuple[Optional[bool], Optional[str]]:
        llm = self._get_llm()
        # boxed 형식이 있으면 우선 사용, 없으면 리스트를 문자열로 변환
        student_answers = student_boxed_answer if student_boxed_answer else (
            '; '.join(student_ans_list) if student_ans_list else '')
        reference_answers = reference_boxed_answer if reference_boxed_answer else (
            '; '.join(ref_ans_list) if ref_ans_list else '')

        user_prompt = PHYSICS_MODEL_JUDGE_PROMPT.format(
            question=question,
            reference_solution=reference_solution or '',
            reference_answers=reference_answers,
            student_solution=student_solution or '',
            student_answers=student_answers,
        )
        raw = llm.generate_answer(system_prompt="", user_prompt=user_prompt)
        msg = raw.strip()
        try:
            # "## Equivalence Judgement" 섹션에서 TRUE/FALSE 추출
            up = msg.upper()
            if "## EQUIVALENCE JUDGEMENT" in up:
                seg = up.split("## EQUIVALENCE JUDGEMENT", 1)[1]
                val = seg.splitlines()[1].strip() if len(seg.splitlines()) > 1 else seg.strip()
                if "TRUE" in val:
                    return True, msg
                if "FALSE" in val:
                    return False, msg
        except Exception:
            pass
        return None, msg

    # ---------- 1차 평가 (규칙 기반 자동 판정) -----------
    def evaluate_first_stage(self, predicted_raw: str, ground_truth_raw: str) -> Tuple[bool, List[str], List[str]]:
        """1차 규칙 기반 자동 판정
        
        Returns:
            (is_correct, pred_list, gt_list): 판정 결과, 추출된 예측 답 리스트, 추출된 정답 리스트
        """
        ok, pred_list, gt_list = self._auto_judge(predicted_raw, ground_truth_raw)
        return ok, pred_list, gt_list

    # ---------- 2차 평가 (모델 판정) -----------
    def _extract_boxed_answer_and_solution(self, text: str) -> Tuple[str, str]:
        """텍스트에서 \\boxed{...} 형식의 정답과 나머지 풀이를 분리
        
        Returns:
            (boxed_answer, solution_text): \\boxed{...} 형식의 정답, 나머지 풀이 텍스트
        """
        import re
        boxes = self._extract_boxed_all(text)

        if not boxes:
            return "", text.strip()

        # 마지막 boxed를 \\boxed 형식으로 정규화
        last_boxed_content = boxes[-1]
        normalized_boxed = f"\\boxed{{{last_boxed_content}}}"

        # 원본 텍스트에서 마지막 boxed 부분을 제거하여 풀이만 추출
        solution_text = text
        matches = list(re.finditer(self._BOXED_PATTERN, solution_text))
        if matches:
            # 마지막 매칭부터 시작
            last_match = matches[-1]
            start_pos = last_match.start()
            open_brace_pos = last_match.end() - 1
            
            _, end_pos = self._find_matching_brace(solution_text, open_brace_pos)
            if end_pos is not None:
                solution_text = (solution_text[:start_pos] +
                               solution_text[end_pos:]).strip()

        return normalized_boxed, solution_text

    def evaluate_second_stage(
            self,
            question: str,
            predicted_raw: str,
            ground_truth_raw: str,
            pred_list: List[str],
            gt_list: List[str],
            *,
            student_solution: Optional[str] = None,
            reference_solution: Optional[str] = None,
    ) -> bool:
        """2차 모델 심판
        
        Args:
            question: 문제 텍스트
            predicted_raw: 원본 예측 답변
            ground_truth_raw: 원본 정답
            pred_list: 추출된 예측 답 리스트
            gt_list: 추출된 정답 리스트
            student_solution: 학생 풀이 (선택)
            reference_solution: 참고 해설 (선택)
            
        Returns:
            판정 결과 (True/False)
        """
        self.last_model_judge_msg = None

        # student_solution이 있으면 정답과 풀이를 분리
        student_boxed_answer = ""
        student_solution_text = ""

        if student_solution:
            student_boxed_answer, student_solution_text = self._extract_boxed_answer_and_solution(student_solution)
            # boxed에서 답 리스트 재추출 (원본 그대로 사용)
            if student_boxed_answer:
                pred_list = self._extract_final_answers(student_boxed_answer)
        elif not pred_list:
            # student_solution이 없고 pred_list도 비어있으면 predicted_raw에서 추출
            pred_list = self._extract_final_answers(predicted_raw)

        # 정답을 원본 그대로 사용
        student_answers_str = student_boxed_answer if student_boxed_answer else '; '.join(
            pred_list) if pred_list else ''
        ref_answers_str = ground_truth_raw if ground_truth_raw else '; '.join(gt_list) if gt_list else ''

        # 2차 평가에 사용된 student_solution과 student_answers 저장
        self.last_student_solution = student_solution_text if student_solution_text else (
            predicted_raw if not student_solution else "")
        self.last_student_answers = student_answers_str

        verdict, msg = self._judge_with_model(
            question=question,
            student_ans_list=pred_list,
            ref_ans_list=gt_list,
            student_solution=self.last_student_solution,
            reference_solution=reference_solution or "",
            student_boxed_answer=self.last_student_answers,
            reference_boxed_answer=ref_answers_str,
        )
        self.last_model_judge_msg = msg

        if isinstance(verdict, bool):
            return verdict

        # 최후의 수단: 아주 엄격한 문자열 비교
        return predicted_raw.strip() == ground_truth_raw.strip()
