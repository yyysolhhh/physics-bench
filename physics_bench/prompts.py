PHYSICS_TUTOR_SYSTEM_PROMPT = ""

PHYSICS_USER_PROMPT = "Question: {question}"

# 간단한 물리학 프롬프트 (수치 계산 강조)
PHYSICS_NUMERICAL_PROMPT = """물리학 문제를 해결해주세요.

답변 형식:
- 수학 기호는 최대한 계산하여 숫자로 변환
- 복잡한 수식은 단계별로 계산하여 최종 숫자 도출
- 최종 답변은 "답: {수치} {단위}" 형태로 제시
- 숫자와 단위를 $로 묶지 말고 공백으로 구분
- 단위는 latex 문법으로 표시
- 답변에는 오직 최종 답만 포함하고 풀이 과정은 생략
"""

PHYSICS_EVALUATION_PROMPT = """
당신은 물리학 문제 채점 전문가입니다. 아래 입력을 보고 예측 답변이 정답과 의미적으로 동일한지 판단하세요.

요구사항:
- 수학적 등가성, 단위 변환, 소수점 반올림은 허용 범위 내에서 동일로 간주
- 단, 문제에서 특정 형식(예: 단위, 유효자리수)이 강제된 경우 이를 우선 적용
- 복잡한 정규화는 하지 말고, 필요한 최소한의 비교로 판정

출력 형식(반드시 다음의 한 줄 JSON만 출력):
{"is_correct": true|false, "reason": "간단 근거"}

입력:
- Question: {question}
- GroundTruth: {ground_truth}
- Predicted: {predicted}
"""

# 물리학 벤치마크용 프롬프트
PHYSICS_BENCHMARK_PROMPT = """
"""

# 간단한 물리학 프롬프트 (빠른 평가용)
PHYSICS_SIMPLE_PROMPT = """
"""

PHYSICS_DETAILED_PROMPT = """
"""
