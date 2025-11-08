PHYSICS_USER_PROMPT = "{question}"

PHYSICS_SYSTEM_PROMPT_KO = """
다음 물리학 문제를 풀고 정답만 LaTeX의 \\boxed{...} 안에 넣어 출력하세요. 정답만 출력하고 풀이 과정은 포함하지 마세요.
"""

PHYSICS_SYSTEM_PROMPT_EN = """
Solve the following physics problem. Make sure to put the answer (and only answer) inside \\boxed{}.
"""

PHYSICS_MODEL_JUDGE_PROMPT = """
# CONTEXT #
I am a teacher, and I have some undergraduate-level physics problems. I am tasked with evaluating the correctness of a student's answer. Below, I am provided with a problem, a reference solution, and the reference final answer(s). Additionally, a student's solution together with their final answer(s) is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
A. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.
B. Consider Physical Equivalence: Pay special attention to transferring the units of both answers and equivalent variables given in the problem description. Feel free to ignore some physical constants appropriately.
C. Provide a Justification: Conclude with a brief explanation as to why you believe the student's output is correct or incorrect, highlighting any key differences in meaning or content.

# STYLE #
Teaching report.

# TONE #
Professional, scientific.

# AUDIENCE #
Students. Enable them to better understand whether the answer they produce is correct.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]
## Justification
[Conclude with a brief explanation as to why you believe the student's answer is correct or incorrect.]

# ATTENTION #
- The reference solution is ALWAYS correct. The reference final answer is extracted from the reference solution by certain rules, and may sometimes not capture all the meaning of the reference solution. You should carefully judge whether the student gives the same final answer as the reference answer based on corresponding solutions.
- The Equivalence Judgement is only TRUE or FALSE. The answer is TRUE whenever the student's final answer is physically equivalent to the reference one.
- Do not hesitate to refer to the corresponding solutions to determine physical equivalence of the final answers if appropriately.
- Minor formatting differences, rounding within reasonable tolerance, equivalent units, and algebraically equivalent expressions should be considered correct.
- Add "=== report over ===" at the end of the report.

<physics solution>
**Question**:
{question}

**Reference Solution**:
{reference_solution}

**Reference Answer(s)**:
{reference_answers}

**Student Solution**:
{student_solution}

**Student Answer(s)**:
{student_answers}

</physics solution>
"""

PHYSICS_SOLUTION_PROMPT_KO = """
다음은 대학 수준 물리학 문제입니다. 주어진 요구사항과 정보에 따라 답을 계산하세요. 풀이 과정에서 사용된 변수와 공식, 결과는 LaTeX 형식을 사용하여 표현하세요. 풀이의 마지막에 "So the final answer is \\boxed{answer}." 형식으로 결과를 명시적으로 제시하세요.
"""

PHYSICS_SOLUTION_PROMPT_EN = """
The following is an undergraduate-level physics problem. Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is \\boxed{answer}." and give the result explicitly.
"""
