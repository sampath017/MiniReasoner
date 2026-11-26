"""Leakage-free arithmetic dataset generation, inspection, disk/Drive persistence, and SFT masking."""

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np
import torch
from tokenizers import Tokenizer

from ..core.tokenizer import sha256


GENERATOR_VERSION = "arith_v2"
SCRATCHPAD_GENERATOR_VERSION = "arith_v3"
SUPPORTED_OPS = ("add", "sub", "mul", "div", "mod")
DEFAULT_OPS = ("add", "sub", "mul", "div")
ANSWER_FORMATS = ("direct", "scratchpad")
ANSWER_FORMAT_CHOICES = ("direct", "cot", "scratchpad")
_STRICT_INT_RE = re.compile(r"-?\d+")
_SCRATCHPAD_RE = re.compile(r"^Work:\s*([^\r\n]+)\r?\nAnswer:\s*(-?\d+)$")

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
)


def int_to_words(n: int) -> str:
    """Convert integers up to 999,999 (positive or negative) cleanly into English words."""
    if n < 0:
        return f"minus {int_to_words(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        ten, rem = divmod(n, 10)
        return _TENS[ten] if rem == 0 else f"{_TENS[ten]}-{_ONES[rem]}"
    if n < 1000:
        hundred, rem = divmod(n, 100)
        return f"{_ONES[hundred]} hundred" if rem == 0 else f"{_ONES[hundred]} hundred {int_to_words(rem)}"
    if n < 1_000_000:
        thousand, rem = divmod(n, 1000)
        return f"{int_to_words(thousand)} thousand" if rem == 0 else f"{int_to_words(thousand)} thousand {int_to_words(rem)}"
    return str(n)


@dataclass(frozen=True)
class CanonicalProblem:
    op: str        # "add" | "sub" | "mul" | "div" | "mod" | "multi"
    a: int         # canonical left operand (or primary answer seed for multi-step)
    b: int         # canonical right operand
    answer: int
    category: str
    kind: str = "binary"          # "binary" | "multi_seq" | "multi_combine" | "multi_pemdas"
    steps_spec: tuple | None = None
    canonical_id_override: str | None = None

    @property
    def canonical_id(self) -> str:
        if self.canonical_id_override is not None:
            return self.canonical_id_override
        return f"{self.op}:{self.a}:{self.b}"


@dataclass(frozen=True)
class ArithmeticExample:
    canonical_id: str
    op: str
    category: str
    prompt: str
    answer_text: str
    answer: int
    variant: str


def normalize_answer_format(answer_format: str) -> str:
    """Validate arithmetic target formatting mode (`direct` vs `cot` / `scratchpad`)."""
    fmt = answer_format.strip().lower()
    if fmt == "cot":
        return "scratchpad"
    if fmt not in ANSWER_FORMATS:
        raise ValueError(f"Unsupported answer_format {answer_format!r}. Choose from {ANSWER_FORMAT_CHOICES}.")
    return fmt


def normalize_ops(ops: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    """Validate and canonicalize an operation list while preserving unique order."""
    raw = ops.split(",") if isinstance(ops, str) else list(ops)
    cleaned: list[str] = []
    for item in raw:
        op = item.strip().lower()
        if not op:
            continue
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported arithmetic op {op!r}. Choose from {SUPPORTED_OPS}.")
        if op not in cleaned:
            cleaned.append(op)
    if not cleaned:
        raise ValueError("Specify at least one arithmetic operation.")
    return tuple(cleaned)


def _has_addition_carry(a: int, b: int) -> bool:
    a, b = abs(a), abs(b)
    carry = 0
    while a > 0 or b > 0 or carry > 0:
        da, db = a % 10, b % 10
        if da + db + carry >= 10:
            return True
        carry = (da + db + carry) // 10
        a //= 10
        b //= 10
    return False


def _has_subtraction_borrow(a: int, b: int) -> bool:
    if a < b:
        return True
    x, y = a, b
    while x > 0 or y > 0:
        if (x % 10) < (y % 10):
            return True
        x //= 10
        y //= 10
    return False


def _digit_bucket(a: int, b: int) -> str:
    d = len(str(max(abs(a), abs(b))))
    return "1d" if d == 1 else ("2d" if d == 2 else f"{d}d")


def classify_problem(op: str, a: int, b: int) -> str:
    """Assign a descriptive category reflecting operation, sign, digit scale, and carry/borrow."""
    if op == "add":
        if a == 0 or b == 0:
            return "add_zero"
        if a < 0 or b < 0:
            return "add_signed"
        if a == b:
            return "add_equal"
        tag = "carry" if _has_addition_carry(a, b) else "no_carry"
        return f"add_{_digit_bucket(a, b)}_{tag}"

    if op == "sub":
        if b == 0 or a == 0:
            return "sub_zero"
        if a == b:
            return "sub_equal"
        if a < 0 or b < 0:
            return "sub_signed_inputs"
        if a < b:
            return "sub_negative_result"
        tag = "borrow" if _has_subtraction_borrow(a, b) else "no_borrow"
        return f"sub_{_digit_bucket(a, b)}_{tag}"

    if op == "mul":
        if a == 0 or b == 0:
            return "mul_zero"
        if abs(a) == 1 or abs(b) == 1:
            return "mul_unit"
        if a < 0 or b < 0:
            return "mul_signed"
        return f"mul_{_digit_bucket(a, b)}"

    if op == "div":
        if a == 0:
            return "div_zero"
        if abs(b) == 1:
            return "div_by_unit"
        if a < 0 or b < 0:
            return "div_signed"
        return f"div_{_digit_bucket(a, b)}"

    if op == "mod":
        if a == 0 or a % b == 0:
            return "mod_exact"
        if a < 0 or b < 0:
            return "mod_signed"
        return f"mod_{_digit_bucket(a, b)}"

    raise ValueError(f"Unsupported arithmetic op {op!r}")


def _eval_binary(op: str, a: int, b: int) -> int:
    """Exact Python integer evaluation for all supported arithmetic operations."""
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ZeroDivisionError("Division by zero cannot form an integer arithmetic problem.")
        if a % b != 0:
            raise ValueError(f"Exact integer division requires clean divisibility ({a} / {b}).")
        sign = -1 if (a < 0) ^ (b < 0) else 1
        return sign * (abs(a) // abs(b))
    if op == "mod":
        if b == 0:
            raise ZeroDivisionError("Modulo by zero is undefined.")
        return a % b
    raise ValueError(f"Unsupported arithmetic op {op!r}")


def canonical_problem(op: str, a: int, b: int) -> CanonicalProblem:
    """Normalize commutative problems (`add`, `mul`) so swapped order maps to one canonical key."""
    if op in ("add", "mul"):
        ca, cb = (a, b) if a <= b else (b, a)
    elif op in ("sub", "div", "mod"):
        ca, cb = a, b
    else:
        raise ValueError(f"Unsupported arithmetic op {op!r}")
    answer = _eval_binary(op, ca, cb)

    return CanonicalProblem(
        op=op,
        a=ca,
        b=cb,
        answer=answer,
        category=classify_problem(op, ca, cb),
    )


def _msb_place_terms(n: int) -> list[int]:
    """Decompose positive integer `n` into non-zero place-value terms from MSB to LSB."""
    digits = str(n)
    power = len(digits) - 1
    terms: list[int] = []
    for ch in digits:
        d = int(ch)
        if d:
            terms.append(d * (10**power))
        power -= 1
    return terms or [0]


def _positive_add_steps(a: int, b: int) -> list[str]:
    big, small = (a, b) if a >= b else (b, a)
    if small == 0:
        return [f"{big}+0={big}"]
    terms = _msb_place_terms(small)
    if len(terms) == 1:
        return [f"{big}+{small}={big + small}"]
    steps: list[str] = []
    acc = big
    for term in terms:
        nxt = acc + term
        steps.append(f"{acc}+{term}={nxt}")
        acc = nxt
    return steps


def _positive_sub_steps(a: int, b: int) -> list[str]:
    if b == 0:
        return [f"{a}-0={a}"]
    if a == b:
        return [f"{a}-{b}=0"]
    terms = _msb_place_terms(b)
    if len(terms) == 1:
        return [f"{a}-{b}={a - b}"]
    steps: list[str] = []
    acc = a
    for term in terms:
        nxt = acc - term
        steps.append(f"{acc}-{term}={nxt}")
        acc = nxt
    return steps


def _work_steps_for_op(op: str, a: int, b: int) -> list[str]:
    """Build deterministic numeric place-value steps solving `a op b` cleanly."""
    if op == "add":
        if a >= 0 and b >= 0:
            return _positive_add_steps(a, b)
        if a < 0 and b < 0:
            return _positive_add_steps(-a, -b) + [f"neg={a + b}"]
        pos, neg_mag = (a, -b) if b < 0 else (b, -a)
        if pos >= neg_mag:
            return _positive_sub_steps(pos, neg_mag)
        return _positive_sub_steps(neg_mag, pos) + [f"neg={a + b}"]

    if op == "sub":
        if b == 0:
            return [f"{a}-0={a}"]
        if a == b:
            rhs = f"({b})" if b < 0 else str(b)
            return [f"{a}-{rhs}=0"]
        if a >= 0 and b >= 0:
            if a >= b:
                return _positive_sub_steps(a, b)
            return _positive_sub_steps(b, a) + [f"neg={a - b}"]
        return _work_steps_for_op("add", a, -b)

    if op == "mul":
        if a == 0 or b == 0:
            rhs = f"({b})" if b < 0 else str(b)
            return [f"{a}*{rhs}=0"]
        pa, pb = abs(a), abs(b)
        sign_neg = (a < 0) ^ (b < 0)
        if pa == 1 or pb == 1 or (pa < 10 and pb < 10):
            steps = [f"{pa}*{pb}={pa * pb}"]
        else:
            big, small = (pa, pb) if pa >= pb else (pb, pa)
            terms = _msb_place_terms(small)
            if len(terms) == 1:
                other = small
                terms = _msb_place_terms(big)
            else:
                other = big
            if len(terms) == 1:
                steps = [f"{other}*{terms[0]}={pa * pb}"]
            else:
                partials = [other * t for t in terms]
                steps = [f"{other}*{t}={p}" for t, p in zip(terms, partials)]
                acc = partials[0]
                for p in partials[1:]:
                    steps.append(f"{acc}+{p}={acc + p}")
                    acc += p
        return steps + ([f"neg={-(pa * pb)}"] if sign_neg else [])

    if op == "div":
        if a == 0:
            rhs = f"({b})" if b < 0 else str(b)
            return [f"0/{rhs}=0"]
        pa, pb = abs(a), abs(b)
        pq = pa // pb
        sign_neg = (a < 0) ^ (b < 0)
        q_terms = _msb_place_terms(pq)
        if pb == 1 or len(q_terms) == 1:
            steps = [f"{pa}/{pb}={pq}"]
        else:
            rem = pa
            steps = []
            for qt in q_terms:
                chunk = pb * qt
                nxt = rem - chunk
                steps.append(f"{pb}*{qt}={chunk}")
                if nxt > 0:
                    steps.append(f"{rem}-{chunk}={nxt}")
                rem = nxt
            steps.append("+".join(str(qt) for qt in q_terms) + f"={pq}")
        return steps + ([f"neg=-{pq}"] if sign_neg else [])

    if op == "mod":
        q, r = divmod(a, b)
        prod = q * b
        rhs = f"({prod})" if prod < 0 else str(prod)
        return [f"{q}*{b}={prod}", f"{a}-{rhs}={r}"]

    raise ValueError(f"Unsupported arithmetic op {op!r}")


def format_work_trace(problem: CanonicalProblem) -> str:
    """Build deterministic `Work: ...` line breaking multi-digit or multi-step math into simple steps."""
    if problem.kind == "binary":
        steps = _work_steps_for_op(problem.op, problem.a, problem.b)
    elif problem.kind == "multi_seq":
        (op1, a, b), (op2, c) = problem.steps_spec
        r1 = _eval_binary(op1, a, b)
        steps = _work_steps_for_op(op1, a, b) + _work_steps_for_op(op2, r1, c)
    elif problem.kind == "multi_combine":
        (op1, a, b), (op2, c, d), op_join = problem.steps_spec
        r1 = _eval_binary(op1, a, b)
        r2 = _eval_binary(op2, c, d)
        steps = (
            _work_steps_for_op(op1, a, b)
            + _work_steps_for_op(op2, c, d)
            + _work_steps_for_op(op_join, r1, r2)
        )
    elif problem.kind == "multi_pemdas":
        op1, a, b, op2, c = problem.steps_spec
        if op2 == "mul":
            r1 = _eval_binary("mul", b, c)
            steps = _work_steps_for_op("mul", b, c) + _work_steps_for_op(op1, a, r1)
        else:
            r1 = _eval_binary("mul", a, b)
            steps = _work_steps_for_op("mul", a, b) + _work_steps_for_op(op2, r1, c)
    else:
        raise ValueError(f"Unsupported problem kind {problem.kind!r}")
    return "Work: " + "; ".join(steps)


def _sample_magnitude(rng: np.random.Generator, max_digits: int) -> int:
    """Sample uniformly by digit length (`1..max_digits`) so 1d/2d/3d+ scales all stay balanced."""
    digits = int(rng.integers(1, max_digits + 1))
    low = 0 if digits == 1 else 10 ** (digits - 1)
    high = 10**digits
    return int(rng.integers(low, high))


def _maybe_negate(rng: np.random.Generator, value: int, allow_negative: bool) -> int:
    if value != 0 and allow_negative and rng.random() < 0.35:
        return -value
    return value


def _generate_op_problems(
    op: str,
    *,
    max_digits: int,
    allow_negative: bool,
    target_count: int,
    rng: np.random.Generator,
    seed_full_1d: bool = False,
) -> list[CanonicalProblem]:
    """Deterministically generate canonical problems balanced across 1..`max_digits` and edge cases."""
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    limit = 10**max_digits

    # For 1-digit problems (`0..9`), full enumeration is small (~100-400) and instant.
    if max_digits == 1 and target_count >= 400:
        lo = -(limit - 1) if allow_negative else 0
        problems: list[CanonicalProblem] = []
        seen: set[tuple[str, int, int]] = set()
        if op == "add":
            for x in range(lo, limit):
                for y in range(x, limit):
                    p = canonical_problem("add", x, y)
                    problems.append(p)
            return problems
        if op == "sub":
            for x in range(lo, limit):
                for y in range(lo, limit if allow_negative else x + 1):
                    problems.append(canonical_problem("sub", x, y))
            return problems
        if op == "mul":
            for x in range(lo, limit):
                for y in range(x, limit):
                    problems.append(canonical_problem("mul", x, y))
            return problems
        if op == "div":
            for divisor in range(lo, limit):
                if divisor == 0:
                    continue
                max_q = (limit - 1) // abs(divisor)
                q_lo = -max_q if allow_negative else 0
                for q in range(q_lo, max_q + 1):
                    p = canonical_problem("div", divisor * q, divisor)
                    key = (p.op, p.a, p.b)
                    if key not in seen:
                        seen.add(key)
                        problems.append(p)
            return problems
        if op == "mod":
            for x in range(0, limit):
                for y in range(1, limit):
                    problems.append(canonical_problem("mod", x, y))
            return problems

    seen = set()
    problems = []

    def try_add(x: int, y: int) -> None:
        if len(problems) >= target_count:
            return
        prob = canonical_problem(op, x, y)
        key = (prob.op, prob.a, prob.b)
        if key not in seen:
            seen.add(key)
            problems.append(prob)

    # Seed fundamental 1-digit edge cases first so zeros, units, signs, and tiny ops never miss coverage.
    if seed_full_1d and target_count >= 400:
        base_1d = list(range(min(10, limit)))
        edge_vals = base_1d + ([-1, -5, -9] if allow_negative else [])
    else:
        edge_vals = [0, 1, 2, 5, 9] + ([-1, -5, -9] if allow_negative else [])
    for x in edge_vals:
        for y in edge_vals:
            if op == "sub" and not allow_negative and x < y:
                continue
            if op == "div":
                if y != 0:
                    try_add(x * y, y)
                continue
            if op == "mod":
                if y > 0 and x >= 0:
                    try_add(x, y)
                continue
            try_add(x, y)

    max_attempts = max(10_000, target_count * 12)
    attempts = 0
    while len(problems) < target_count and attempts < max_attempts:
        attempts += 1
        if op in ("add", "mul"):
            x = _maybe_negate(rng, _sample_magnitude(rng, max_digits), allow_negative)
            y = _maybe_negate(rng, _sample_magnitude(rng, max_digits), allow_negative)
            try_add(x, y)
        elif op == "sub":
            x = _maybe_negate(rng, _sample_magnitude(rng, max_digits), allow_negative)
            y = _maybe_negate(rng, _sample_magnitude(rng, max_digits), allow_negative)
            if not allow_negative and x < y:
                x, y = y, x
            try_add(x, y)
        elif op == "div":
            # Construct dividend = divisor * quotient with |dividend| < 10**max_digits
            div_digits = int(rng.integers(1, max(2, max_digits) + 1))
            d_hi = min(limit, 10**div_digits)
            divisor_mag = int(rng.integers(1, max(2, d_hi)))
            max_q = max(0, (limit - 1) // divisor_mag)
            q_mag = int(rng.integers(0, max_q + 1))
            divisor = _maybe_negate(rng, divisor_mag, allow_negative)
            quotient = _maybe_negate(rng, q_mag, allow_negative)
            try_add(divisor * quotient, divisor)
        elif op == "mod":
            divisor = int(rng.integers(1, limit))
            dividend = _sample_magnitude(rng, max_digits)
            if allow_negative and rng.random() < 0.25 and dividend != 0:
                dividend = -dividend
            try_add(dividend, divisor)

    return sorted(problems, key=lambda p: (p.op, p.a, p.b))


def _sample_multistep_operand(
    rng: np.random.Generator,
    max_digits: int,
    allow_negative: bool,
    *,
    cap_digits: int | None = None,
    neg_prob: float = 0.2,
) -> int:
    d_lim = min(max_digits, cap_digits) if cap_digits is not None else max_digits
    val = _sample_magnitude(rng, max(1, d_lim))
    if val != 0 and allow_negative and rng.random() < neg_prob:
        return -val
    return val


def _canonicalize_pair(op: str, a: int, b: int) -> tuple[str, int, int]:
    if op in ("add", "mul") and a > b:
        return op, b, a
    return op, a, b


def _generate_multistep_problems(
    *,
    ops: tuple[str, ...],
    max_digits: int,
    allow_negative: bool,
    target_count: int,
    rng: np.random.Generator,
) -> list[CanonicalProblem]:
    """Generate leakage-free multi-step chained, combined-pair, and PEMDAS problems."""
    chain_ops = [op for op in ops if op in ("add", "sub", "mul")] or ["add"]
    join_ops = [op for op in ops if op in ("add", "sub", "mul")] or ["add"]
    has_mul = "mul" in ops
    add_sub_ops = [op for op in ops if op in ("add", "sub")]

    problems: list[CanonicalProblem] = []
    seen: set[str] = set()
    max_attempts = max(10_000, target_count * 12)
    attempts = 0
    while len(problems) < target_count and attempts < max_attempts:
        attempts += 1
        pick = rng.random()
        if pick < 0.4:
            # Sequential 2-step chain: (a op1 b) op2 c
            op1 = chain_ops[int(rng.integers(0, len(chain_ops)))]
            op2 = chain_ops[int(rng.integers(0, len(chain_ops)))]
            cap1 = 2 if op1 == "mul" else min(max_digits, 3)
            a = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=cap1)
            b = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=1 if op1 == "mul" and abs(a) >= 10 else cap1)
            op1, a, b = _canonicalize_pair(op1, a, b)
            r1 = _eval_binary(op1, a, b)
            c_cap = 1 if op2 == "mul" else min(max_digits, 2 if abs(r1) >= 100 else 3)
            c = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=c_cap)
            if not allow_negative and (r1 < 0 or (op2 == "sub" and r1 < c)):
                continue
            ans = _eval_binary(op2, r1, c)
            if not allow_negative and ans < 0:
                continue
            cid = f"multi_seq:{op1}:{a}:{b}:{op2}:{c}"
            if cid in seen:
                continue
            seen.add(cid)
            style = "word" if len(problems) % 2 == 1 else "sym"
            problems.append(
                CanonicalProblem(
                    op="multi",
                    a=ans,
                    b=r1,
                    answer=ans,
                    category=f"multi_seq_{style}",
                    kind="multi_seq",
                    steps_spec=((op1, a, b), (op2, c)),
                    canonical_id_override=cid,
                )
            )
        elif pick < 0.8 or not (has_mul and add_sub_ops):
            # Combined pair: (a op1 b) op_join (c op2 d) e.g. (10 + 10) + (2 + 3)
            op1 = chain_ops[int(rng.integers(0, len(chain_ops)))]
            op2 = chain_ops[int(rng.integers(0, len(chain_ops)))]
            op_join = join_ops[int(rng.integers(0, len(join_ops)))]
            cap1 = 2 if op1 == "mul" else min(max_digits, 2 if op_join == "mul" else 3)
            cap2 = 2 if op2 == "mul" else min(max_digits, 2 if op_join == "mul" else 3)
            a = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=cap1)
            b = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=1 if op1 == "mul" and abs(a) >= 10 else cap1)
            c = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=cap2)
            d = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=1 if op2 == "mul" and abs(c) >= 10 else cap2)
            op1, a, b = _canonicalize_pair(op1, a, b)
            op2, c, d = _canonicalize_pair(op2, c, d)
            if not allow_negative and ((op1 == "sub" and a < b) or (op2 == "sub" and c < d)):
                continue
            r1 = _eval_binary(op1, a, b)
            r2 = _eval_binary(op2, c, d)
            if op_join in ("add", "mul") and (op1, a, b) > (op2, c, d):
                (op1, a, b), (op2, c, d) = (op2, c, d), (op1, a, b)
                r1, r2 = r2, r1
            if op_join == "mul" and (abs(r1) >= 100 and abs(r2) >= 100):
                continue
            ans = _eval_binary(op_join, r1, r2)
            if not allow_negative and ans < 0:
                continue
            cid = f"multi_combine:{op1}:{a}:{b}:{op_join}:{op2}:{c}:{d}"
            if cid in seen:
                continue
            seen.add(cid)
            style = "word" if len(problems) % 2 == 1 else "sym"
            problems.append(
                CanonicalProblem(
                    op="multi",
                    a=ans,
                    b=r1,
                    answer=ans,
                    category=f"multi_combine_{style}",
                    kind="multi_combine",
                    steps_spec=((op1, a, b), (op2, c, d), op_join),
                    canonical_id_override=cid,
                )
            )
        else:
            # PEMDAS/BODMAS order-of-operations expression without parentheses
            other_op = add_sub_ops[int(rng.integers(0, len(add_sub_ops)))]
            mul_on_right = bool(rng.random() < 0.7)
            if mul_on_right:
                op1, op2 = other_op, "mul"
                a = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=min(max_digits, 3))
                b = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=min(max_digits, 2))
                c = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=1 if abs(b) >= 10 else 2)
                prod = b * c
                if not allow_negative and op1 == "sub" and a < prod:
                    continue
                ans = _eval_binary(op1, a, prod)
            else:
                op1, op2 = "mul", other_op
                a = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=min(max_digits, 2))
                b = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=1 if abs(a) >= 10 else 2)
                c = _sample_multistep_operand(rng, max_digits, allow_negative, cap_digits=min(max_digits, 3))
                prod = a * b
                if not allow_negative and op2 == "sub" and prod < c:
                    continue
                ans = _eval_binary(op2, prod, c)
            if not allow_negative and ans < 0:
                continue
            cid = f"multi_pemdas:{op1}:{a}:{b}:{op2}:{c}"
            if cid in seen:
                continue
            seen.add(cid)
            problems.append(
                CanonicalProblem(
                    op="multi",
                    a=ans,
                    b=prod,
                    answer=ans,
                    category="multi_pemdas",
                    kind="multi_pemdas",
                    steps_spec=(op1, a, b, op2, c),
                    canonical_id_override=cid,
                )
            )

    return sorted(problems, key=lambda p: p.canonical_id)


def enumerate_canonical_problems(
    *,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    seed: int = 42,
    include_multistep: bool = False,
    seed_full_1d: bool = False,
) -> list[CanonicalProblem]:
    """Generate stratified canonical problems across operations and `1..max_digits` scales."""
    op_list = normalize_ops(ops)
    rng = np.random.default_rng(seed)
    all_problems: list[CanonicalProblem] = []
    for op in op_list:
        all_problems.extend(
            _generate_op_problems(
                op,
                max_digits=max_digits,
                allow_negative=allow_negative,
                target_count=samples_per_op,
                rng=rng,
                seed_full_1d=seed_full_1d,
            )
        )
    if include_multistep:
        all_problems.extend(
            _generate_multistep_problems(
                ops=op_list,
                max_digits=max_digits,
                allow_negative=allow_negative,
                target_count=samples_per_op,
                rng=rng,
            )
        )
    return all_problems


def split_canonical_problems(
    problems: list[CanonicalProblem],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[CanonicalProblem]]:
    """Split canonical problems stratified by category BEFORE rendering wording variants."""
    if not (0.0 < val_fraction < 0.5 and 0.0 < test_fraction < 0.5 and val_fraction + test_fraction < 0.9):
        raise ValueError("val_fraction and test_fraction must leave room for training problems.")

    rng = np.random.default_rng(seed)
    by_category: dict[str, list[CanonicalProblem]] = {}
    for problem in problems:
        by_category.setdefault(problem.category, []).append(problem)

    splits: dict[str, list[CanonicalProblem]] = {"train": [], "val": [], "test": []}
    for category in sorted(by_category):
        items = list(by_category[category])
        order = rng.permutation(len(items))
        shuffled = [items[int(i)] for i in order]
        n = len(shuffled)
        if n >= 3:
            n_val = max(1, round(n * val_fraction))
            n_test = max(1, round(n * test_fraction))
            while n_val + n_test >= n and n_val > 1:
                n_val -= 1
            while n_val + n_test >= n and n_test > 1:
                n_test -= 1
        elif n == 2:
            n_val, n_test = 1, 0
        else:
            n_val, n_test = 0, 0

        splits["val"].extend(shuffled[:n_val])
        splits["test"].extend(shuffled[n_val:n_val + n_test])
        splits["train"].extend(shuffled[n_val + n_test:])

    for name in splits:
        splits[name].sort(key=lambda p: (p.op, p.a, p.b, p.canonical_id))
    return splits


def _fmt_rhs(y: int) -> str:
    return f"({y})" if y < 0 else str(y)


def _op_strings(op: str) -> tuple[str, str]:
    return {
        "add": ("+", "plus"),
        "sub": ("-", "minus"),
        "mul": ("*", "times"),
        "div": ("/", "divided by"),
        "mod": ("%", "modulo"),
    }[op]


def _action_phrase(op: str, val: int) -> str:
    verb = {"add": "add", "sub": "subtract", "mul": "multiply by", "div": "divide by"}[op]
    return f"{verb} {_fmt_rhs(val)}"


def _render_multistep_question_variants(
    problem: CanonicalProblem,
    *,
    include_variants: bool,
) -> list[tuple[str, str]]:
    """Return `(variant_name, question_text)` pairs for multi-step problems."""
    if problem.kind == "multi_seq":
        (op1, a, b), (op2, c) = problem.steps_spec
        s1, w1 = _op_strings(op1)
        s2, _ = _op_strings(op2)
        expr1 = f"{a} {s1} {_fmt_rhs(b)}"
        sym_q = f"What is ({expr1}) {s2} {_fmt_rhs(c)}?"
        word_q = f"Start with {a}, {_action_phrase(op1, b)}, and then {_action_phrase(op2, c)}. What is the final result?"
        canonical_q = word_q if problem.category.endswith("_word") else sym_q
        variants = [("canonical", canonical_q)]
        if not include_variants:
            return variants
        alt_tag, alt_q = ("multi_seq_sym", sym_q) if problem.category.endswith("_word") else ("multi_seq_word", word_q)
        variants.extend([
            (alt_tag, alt_q),
            ("calculate", f"Calculate ({expr1}) {s2} {_fmt_rhs(c)}."),
            ("equation", f"({expr1}) {s2} {_fmt_rhs(c)} ="),
            ("conversational_take", f"What if I take {a}, {_action_phrase(op1, b)}, and next {_action_phrase(op2, c)}?"),
            ("conversational_first", f"First compute {a} {w1} {_fmt_rhs(b)}, then {_action_phrase(op2, c)}. What do you get?"),
        ])
        return variants

    if problem.kind == "multi_combine":
        (op1, a, b), (op2, c, d), op_join = problem.steps_spec
        s1, _ = _op_strings(op1)
        s2, _ = _op_strings(op2)
        sj, _ = _op_strings(op_join)
        e1 = f"{a} {s1} {_fmt_rhs(b)}"
        e2 = f"{c} {s2} {_fmt_rhs(d)}"
        sym_q = f"What is ({e1}) {sj} ({e2})?"
        join_templates = {
            "add": (
                f"If you compute {e1} and also take {e2}, what is the sum of both results?",
                f"What if I do {e1} and take {e2}, then add them together for the final result?",
                f"Calculate {e1} and {e2}, and add the two numbers together.",
            ),
            "sub": (
                f"Compute {e1} and {e2}, then subtract the second result from the first.",
                f"If you start with {e1} and take away {e2}, what is the final result?",
                f"What if I calculate {e1} and subtract {e2} from it?",
            ),
            "mul": (
                f"Find {e1} and {e2}, then multiply those two results together.",
                f"What if I compute {e1} and {e2}, then multiply the two numbers together?",
                f"Calculate {e1} and {e2}, and find the product of both results.",
            ),
        }[op_join]
        canonical_q = join_templates[0] if problem.category.endswith("_word") else sym_q
        variants = [("canonical", canonical_q)]
        if not include_variants:
            return variants
        alt_tag, alt_q = ("multi_combine_sym", sym_q) if problem.category.endswith("_word") else ("multi_combine_word", join_templates[0])
        variants.extend([
            (alt_tag, alt_q),
            ("calculate", f"Calculate ({e1}) {sj} ({e2})."),
            ("equation", f"({e1}) {sj} ({e2}) ="),
            ("conversational_take", join_templates[1]),
            ("conversational_calc", join_templates[2]),
        ])
        return variants

    if problem.kind == "multi_pemdas":
        op1, a, b, op2, c = problem.steps_spec
        s1, _ = _op_strings(op1)
        s2, _ = _op_strings(op2)
        expr = f"{a} {s1} {_fmt_rhs(b)} {s2} {_fmt_rhs(c)}"
        variants = [("canonical", f"What is {expr}?")]
        if include_variants:
            variants.extend([
                ("calculate", f"Calculate {expr}."),
                ("equation", f"{expr} ="),
                ("pemdas_word", f"Using order of operations, what is {expr}?"),
            ])
        return variants

    raise ValueError(f"Unsupported multi-step kind {problem.kind!r}")


def render_examples(
    problem: CanonicalProblem,
    *,
    include_variants: bool,
    answer_format: str = "direct",
    include_conversational: bool = False,
) -> list[ArithmeticExample]:
    """Render canonical and optional wording/commuted variants in `direct` or `scratchpad` format."""
    fmt = normalize_answer_format(answer_format)
    suffix = "\nAnswer:" if fmt == "direct" else "\n"
    answer_text = (
        str(problem.answer)
        if fmt == "direct"
        else f"{format_work_trace(problem)}\nAnswer: {problem.answer}"
    )

    if problem.kind != "binary":
        return [
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"{question}{suffix}",
                answer_text=answer_text,
                answer=problem.answer,
                variant=variant_name,
            )
            for variant_name, question in _render_multistep_question_variants(
                problem, include_variants=include_variants,
            )
        ]

    sym, word_op = _op_strings(problem.op)
    examples = [
        ArithmeticExample(
            canonical_id=problem.canonical_id,
            op=problem.op,
            category=problem.category,
            prompt=f"What is {problem.a} {sym} {_fmt_rhs(problem.b)}?{suffix}",
            answer_text=answer_text,
            answer=problem.answer,
            variant="canonical",
        )
    ]
    if not include_variants:
        return examples

    pairs = [(problem.a, problem.b, False)]
    if problem.op in ("add", "mul") and problem.a != problem.b:
        pairs.append((problem.b, problem.a, True))

    for x, y, commuted in pairs:
        var_suffix = "_commuted" if commuted else ""
        rhs = _fmt_rhs(y)
        if commuted:
            examples.append(
                ArithmeticExample(
                    canonical_id=problem.canonical_id,
                    op=problem.op,
                    category=problem.category,
                    prompt=f"What is {x} {sym} {rhs}?{suffix}",
                    answer_text=answer_text,
                    answer=problem.answer,
                    variant="canonical_commuted",
                )
            )
        examples.extend([
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"Calculate {x} {sym} {rhs}.{suffix}",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"calculate{var_suffix}",
            ),
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"{x} {sym} {rhs} ={suffix}",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"equation{var_suffix}",
            ),
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"What is {x} {word_op} {rhs}?{suffix}",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"word_op{var_suffix}",
            ),
        ])
        if include_conversational and not commuted:
            examples.extend([
                ArithmeticExample(
                    canonical_id=problem.canonical_id,
                    op=problem.op,
                    category=problem.category,
                    prompt=f"Can you do the math for {x} {sym} {rhs}?{suffix}",
                    answer_text=answer_text,
                    answer=problem.answer,
                    variant="conversational_math",
                ),
                ArithmeticExample(
                    canonical_id=problem.canonical_id,
                    op=problem.op,
                    category=problem.category,
                    prompt=f"If I compute {x} {word_op} {rhs}, what is the final result?{suffix}",
                    answer_text=answer_text,
                    answer=problem.answer,
                    variant="conversational_result",
                ),
            ])
        if max(abs(x), abs(y)) < 1000 and not commuted:
            if problem.op == "mod":
                word_prompt = f"What is the remainder when {int_to_words(x)} is divided by {int_to_words(y)}?{suffix}"
            else:
                word_prompt = f"What is {int_to_words(x)} {word_op} {int_to_words(y)}?{suffix}"
            examples.append(
                ArithmeticExample(
                    canonical_id=problem.canonical_id,
                    op=problem.op,
                    category=problem.category,
                    prompt=word_prompt,
                    answer_text=answer_text,
                    answer=problem.answer,
                    variant="word_numbers",
                )
            )
    return examples


def dataset_slug(
    *,
    ops: tuple[str, ...] = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    seed: int = 42,
    answer_format: str = "direct",
    include_multistep: bool = False,
) -> str:
    """Stable human-readable directory name for caching an arithmetic SFT dataset."""
    ops_tag = "-".join(normalize_ops(ops))
    sign_tag = "signed" if allow_negative else "nonneg"
    fmt = normalize_answer_format(answer_format)
    if fmt == "direct" and not include_multistep:
        return f"{GENERATOR_VERSION}_{ops_tag}_d{max_digits}_{sign_tag}_n{samples_per_op}_s{seed}"
    mode_tag = fmt + ("_multistep" if include_multistep else "")
    return f"{SCRATCHPAD_GENERATOR_VERSION}_{mode_tag}_{ops_tag}_d{max_digits}_{sign_tag}_n{samples_per_op}_s{seed}"


def build_arithmetic_dataset(
    *,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    answer_format: str = "direct",
    include_multistep: bool = False,
) -> tuple[dict[str, list[ArithmeticExample]], dict]:
    """Generate leakage-free `train`, `val`, and `test` arithmetic splits with manifest metadata."""
    op_list = normalize_ops(ops)
    fmt = normalize_answer_format(answer_format)
    use_v3 = fmt != "direct" or include_multistep
    gen_version = SCRATCHPAD_GENERATOR_VERSION if use_v3 else GENERATOR_VERSION
    problems = enumerate_canonical_problems(
        ops=op_list,
        max_digits=max_digits,
        allow_negative=allow_negative,
        samples_per_op=samples_per_op,
        seed=seed,
        include_multistep=include_multistep,
        seed_full_1d=use_v3,
    )
    canonical_splits = split_canonical_problems(
        problems,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    examples_by_split: dict[str, list[ArithmeticExample]] = {}
    category_counts: dict[str, dict[str, int]] = {}
    op_counts: dict[str, dict[str, int]] = {}
    canonical_counts: dict[str, int] = {}

    for split_name, split_problems in canonical_splits.items():
        include_variants = split_name == "train"
        rendered: list[ArithmeticExample] = []
        cats: dict[str, int] = {}
        ops_cnt: dict[str, int] = {}
        for prob in split_problems:
            cats[prob.category] = cats.get(prob.category, 0) + 1
            ops_cnt[prob.op] = ops_cnt.get(prob.op, 0) + 1
            rendered.extend(
                render_examples(
                    prob,
                    include_variants=include_variants,
                    answer_format=fmt,
                    include_conversational=use_v3,
                )
            )
        examples_by_split[split_name] = rendered
        category_counts[split_name] = cats
        op_counts[split_name] = ops_cnt
        canonical_counts[split_name] = len(split_problems)

    manifest = {
        "generator_version": gen_version,
        "answer_format": fmt,
        "include_multistep": include_multistep,
        "ops": list(op_list),
        "max_digits": max_digits,
        "allow_negative": allow_negative,
        "samples_per_op": samples_per_op,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "canonical_counts": canonical_counts,
        "example_counts": {k: len(v) for k, v in examples_by_split.items()},
        "op_counts": op_counts,
        "category_counts": category_counts,
    }
    return examples_by_split, manifest


def save_arithmetic_dataset(
    directory: Path,
    splits: dict[str, list[ArithmeticExample]],
    manifest: dict,
) -> Path:
    """Write `train.jsonl`, `val.jsonl`, `test.jsonl`, and SHA-256 verified `manifest.json` to disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_meta = {}
    for split_name in ("train", "val", "test"):
        out_path = directory / f"{split_name}.jsonl"
        rows = splits[split_name]
        with out_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in rows:
                stream.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        file_meta[out_path.name] = {
            "rows": len(rows),
            "bytes": out_path.stat().st_size,
            "sha256": sha256(out_path),
        }

    full_manifest = {**manifest, "files": file_meta}
    (directory / "manifest.json").write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    return directory


def load_arithmetic_dataset(directory: Path) -> tuple[dict[str, list[ArithmeticExample]], dict]:
    """Load and SHA-256 verify a prepared arithmetic SFT dataset from disk."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_info = manifest.get("files", {})
    splits: dict[str, list[ArithmeticExample]] = {}
    for split_name in ("train", "val", "test"):
        fname = f"{split_name}.jsonl"
        fpath = directory / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing prepared split {fpath}")
        expected_sha = files_info.get(fname, {}).get("sha256")
        if expected_sha and sha256(fpath) != expected_sha:
            raise ValueError(f"Checksum mismatch for {fpath}")
        items: list[ArithmeticExample] = []
        with fpath.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    items.append(ArithmeticExample(**json.loads(line)))
        splits[split_name] = items
    return splits, manifest


def summarize_arithmetic_dataset(
    splits: dict[str, list[ArithmeticExample]],
    manifest: dict,
    *,
    samples_per_op: int = 2,
) -> str:
    """Format a readable inspection summary of split counts, operations, categories, and examples."""
    fmt = manifest.get("answer_format", "direct")
    multistep = bool(manifest.get("include_multistep", False))
    lines = [
        "=" * 74,
        f"ARITHMETIC SFT DATASET ({manifest['generator_version']})",
        "=" * 74,
        f"Operations:     {', '.join(manifest['ops'])} | Max Digits: {manifest['max_digits']} | Negatives: {manifest['allow_negative']} | Format: {fmt} | Multistep: {multistep} | Seed: {manifest['seed']}",
        f"Canonical Pool: train={manifest['canonical_counts']['train']:,} | val={manifest['canonical_counts']['val']:,} | sealed_test={manifest['canonical_counts']['test']:,}",
        f"Rendered Rows:  train={manifest['example_counts']['train']:,} | val={manifest['example_counts']['val']:,} | sealed_test={manifest['example_counts']['test']:,}",
        "-" * 74,
        "Per-Operation Canonical Split Breakdown:",
    ]
    display_ops = list(manifest["ops"]) + (
        ["multi"] if "multi" in manifest["op_counts"].get("train", {}) and "multi" not in manifest["ops"] else []
    )
    for op in display_ops:
        tr_c = manifest["op_counts"]["train"].get(op, 0)
        va_c = manifest["op_counts"]["val"].get(op, 0)
        te_c = manifest["op_counts"]["test"].get(op, 0)
        lines.append(f"  {op:<6} | train={tr_c:<7,} val={va_c:<6,} sealed_test={te_c:<6,}")

    lines.append("-" * 74)
    lines.append("Sample Training Prompts (with wording variants):")
    shown_per_op: dict[str, int] = {}
    for ex in splits["train"]:
        cnt = shown_per_op.get(ex.op, 0)
        if cnt < samples_per_op and (cnt == 0 or ex.variant != "canonical"):
            shown_per_op[ex.op] = cnt + 1
            one_prompt = ex.prompt.replace("\n", "\\n")
            one_ans = ex.answer_text.replace("\n", "\\n")
            lines.append(f"  [{ex.op} | {ex.category} | {ex.variant}] {one_prompt!r} -> {one_ans!r}")

    lines.append("-" * 74)
    lines.append("Sample Held-Out Val Prompts (canonical evaluation format):")
    seen_val_ops: set[str] = set()
    for ex in splits["val"]:
        key = ex.category if ex.op == "multi" else ex.op
        if key not in seen_val_ops:
            seen_val_ops.add(key)
            one_prompt = ex.prompt.replace("\n", "\\n")
            one_ans = ex.answer_text.replace("\n", "\\n")
            lines.append(f"  [{ex.op} | {ex.category}] {one_prompt!r} -> {one_ans!r}")
    lines.append("=" * 74)
    return "\n".join(lines)


def prepare_arithmetic_dataset(
    *,
    output_dir: Path | None = None,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    answer_format: str = "direct",
    include_multistep: bool = False,
    sync_drive: bool = True,
    drive_data_root: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, list[ArithmeticExample]], dict]:
    """Generate, save to disk, print preview, and optionally cache to Google Drive."""
    from ..settings import DRIVE_DATA_ROOT, SFT_DATA_DIR

    op_list = normalize_ops(ops)
    fmt = normalize_answer_format(answer_format)
    target_dir = Path(output_dir) if output_dir is not None else (
        SFT_DATA_DIR / dataset_slug(
            ops=op_list, max_digits=max_digits, allow_negative=allow_negative,
            samples_per_op=samples_per_op, seed=seed,
            answer_format=fmt, include_multistep=include_multistep,
        )
    )
    if (target_dir / "manifest.json").is_file() and not overwrite:
        splits, manifest = load_arithmetic_dataset(target_dir)
        print(f"Loaded existing verified arithmetic dataset at {target_dir}", flush=True)
    else:
        splits, manifest = build_arithmetic_dataset(
            ops=op_list,
            max_digits=max_digits,
            allow_negative=allow_negative,
            samples_per_op=samples_per_op,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            answer_format=fmt,
            include_multistep=include_multistep,
        )
        save_arithmetic_dataset(target_dir, splits, manifest)
        manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        print(f"Saved prepared arithmetic dataset to {target_dir}", flush=True)

    print(summarize_arithmetic_dataset(splits, manifest), flush=True)

    if sync_drive:
        from .drive_corpus import publish_sft_dataset

        remote = publish_sft_dataset(drive_data_root or DRIVE_DATA_ROOT, target_dir, manifest)
        print(f"Cached & SHA-256 verified arithmetic dataset on Google Drive: {remote}", flush=True)

    return target_dir, splits, manifest


def load_or_restore_arithmetic_dataset(
    *,
    dataset_dir: Path | None = None,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    seed: int = 42,
    answer_format: str = "direct",
    include_multistep: bool = False,
    use_drive: bool = True,
    drive_data_root: str | None = None,
) -> tuple[Path, dict[str, list[ArithmeticExample]], dict]:
    """Load prepared SFT data from local disk or Google Drive before training starts."""
    from ..settings import DRIVE_DATA_ROOT, SFT_DATA_DIR

    op_list = normalize_ops(ops)
    fmt = normalize_answer_format(answer_format)
    use_v3 = fmt != "direct" or include_multistep
    target_dir = Path(dataset_dir) if dataset_dir is not None else (
        SFT_DATA_DIR / dataset_slug(
            ops=op_list, max_digits=max_digits, allow_negative=allow_negative,
            samples_per_op=samples_per_op, seed=seed,
            answer_format=fmt, include_multistep=include_multistep,
        )
    )
    if (target_dir / "manifest.json").is_file():
        splits, manifest = load_arithmetic_dataset(target_dir)
        return target_dir, splits, manifest

    if use_drive:
        from .drive_corpus import reuse_sft_dataset

        expected_identity = {
            "generator_version": SCRATCHPAD_GENERATOR_VERSION if use_v3 else GENERATOR_VERSION,
            "ops": list(op_list),
            "max_digits": max_digits,
            "allow_negative": allow_negative,
            "samples_per_op": samples_per_op,
            "seed": seed,
        }
        if use_v3:
            expected_identity["answer_format"] = fmt
            expected_identity["include_multistep"] = include_multistep
        restored = reuse_sft_dataset(drive_data_root or DRIVE_DATA_ROOT, target_dir, expected_identity)
        if restored is not None:
            splits, manifest = load_arithmetic_dataset(target_dir)
            return target_dir, splits, manifest

    fmt_flag = " --format cot" if fmt == "scratchpad" else ""
    ms_flag = " --multistep" if include_multistep else ""
    raise FileNotFoundError(
        f"Prepared SFT arithmetic dataset not found at {target_dir}. "
        f"Run `bash scripts/prepare_data.sh{fmt_flag}{ms_flag}` first to generate, inspect, and cache the dataset."
    )


def encode_sft_example(
    tokenizer: Tokenizer,
    prompt: str,
    answer_text: str,
    *,
    context_length: int,
    eos_id: int,
) -> tuple[list[int], list[int]]:
    """Encode `(prompt, answer + EOS)` with `-100` mask over prompt tokens."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False).ids + [eos_id]
    if not prompt_ids:
        prompt_ids = [eos_id]
    max_full = context_length + 1
    if len(prompt_ids) + len(answer_ids) > max_full:
        keep_prompt = max(1, max_full - len(answer_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        if len(prompt_ids) + len(answer_ids) > max_full:
            raise ValueError("Answer alone exceeds context_length; cannot train masked SFT sample.")
    full = prompt_ids + answer_ids
    x = full[:-1]
    y = ([-100] * (len(prompt_ids) - 1)) + answer_ids
    return x, y


def encode_sft_split(
    examples: list[ArithmeticExample],
    tokenizer: Tokenizer,
    *,
    context_length: int,
    eos_id: int,
) -> list[tuple[list[int], list[int]]]:
    return [
        encode_sft_example(
            tokenizer, ex.prompt, ex.answer_text, context_length=context_length, eos_id=eos_id,
        )
        for ex in examples
    ]


def collate_sft_batch(
    encoded: list[tuple[list[int], list[int]]],
    indices: np.ndarray | list[int],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length SFT sequences to `(B, max_len)` with `-100` target masking."""
    batch = [encoded[int(i)] for i in indices]
    max_len = max(len(x) for x, _ in batch)
    x_tensor = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    y_tensor = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row_i, (x_seq, y_seq) in enumerate(batch):
        length = len(x_seq)
        x_tensor[row_i, :length] = torch.tensor(x_seq, dtype=torch.long)
        y_tensor[row_i, :length] = torch.tensor(y_seq, dtype=torch.long)
    return x_tensor.to(device), y_tensor.to(device)


def extract_strict_integer(raw_output: str) -> tuple[bool, int | None]:
    """Parse strictly one integer (optional leading `-` sign) after stripping whitespace."""
    cleaned = raw_output.strip()
    match = _STRICT_INT_RE.fullmatch(cleaned)
    if match is None:
        return False, None
    return True, int(match.group(0))


def extract_scratchpad_integer(raw_output: str) -> tuple[bool, int | None]:
    """Parse strict `Work: <steps>\\nAnswer: <int>` completion with no trailing junk."""
    cleaned = raw_output.strip()
    match = _SCRATCHPAD_RE.fullmatch(cleaned)
    if match is None or not match.group(1).strip():
        return False, None
    return True, int(match.group(2))


def score_arithmetic_completion(
    raw_output: str,
    expected_answer: int,
    *,
    answer_format: str = "direct",
) -> dict:
    """Deterministic verifier matching Phase 4 SFT evaluation and Phase 5 GRPO spec."""
    fmt = normalize_answer_format(answer_format)
    extractor = extract_scratchpad_integer if fmt == "scratchpad" else extract_strict_integer
    valid_format, extracted = extractor(raw_output)
    exact_match = valid_format and (extracted == expected_answer)
    return {
        "valid_format": valid_format,
        "extracted": extracted,
        "exact_match": exact_match,
        "reward": 1.0 if exact_match else 0.0,
    }


def select_stratified_eval_problems(
    examples: list[ArithmeticExample],
    limit: int,
    *,
    seed: int = 42,
) -> list[ArithmeticExample]:
    """Sample up to `limit` canonical evaluation problems evenly across categories."""
    if limit <= 0 or len(examples) <= limit:
        return list(examples)
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[ArithmeticExample]] = {}
    op_to_cats: dict[str, list[str]] = {}
    for ex in examples:
        by_cat.setdefault(ex.category, []).append(ex)
        cats_for_op = op_to_cats.setdefault(ex.op, [])
        if ex.category not in cats_for_op:
            cats_for_op.append(ex.category)

    shuffled_cats = {}
    for cat, items in by_cat.items():
        perm = rng.permutation(len(items))
        shuffled_cats[cat] = [items[int(i)] for i in perm]

    # Interleave categories across operations so small limits still cover every op evenly.
    for op_name in op_to_cats:
        op_to_cats[op_name].sort()
    categories: list[str] = []
    cat_idx = 0
    op_names = sorted(op_to_cats)
    while len(categories) < len(shuffled_cats):
        for op_name in op_names:
            if cat_idx < len(op_to_cats[op_name]):
                categories.append(op_to_cats[op_name][cat_idx])
        cat_idx += 1

    selected: list[ArithmeticExample] = []
    idx = 0
    while len(selected) < limit:
        added_this_round = False
        for cat in categories:
            if idx < len(shuffled_cats[cat]):
                selected.append(shuffled_cats[cat][idx])
                added_this_round = True
                if len(selected) >= limit:
                    break
        if not added_this_round:
            break
        idx += 1
    return selected


def safe_eval_arithmetic_expr(expr: str) -> tuple[int, bool]:
    """Safely evaluate integer arithmetic (`+`, `-`, `*`, exact `/`) via AST, flagging operator presence."""
    cleaned = expr.strip()
    if not cleaned or len(cleaned) > 256:
        raise ValueError("Empty or oversized arithmetic expression.")
    tree = ast.parse(cleaned, mode="eval")

    def _eval_node(node: ast.AST) -> tuple[int, bool]:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError("Only integer literals are permitted.")
            return int(node.value), False
        if isinstance(node, ast.UnaryOp):
            val, has_op = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return val, has_op
            if isinstance(node.op, ast.USub):
                return -val, has_op
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            lval, _ = _eval_node(node.left)
            rval, _ = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return lval + rval, True
            if isinstance(node.op, ast.Sub):
                return lval - rval, True
            if isinstance(node.op, ast.Mult):
                return lval * rval, True
            if isinstance(node.op, ast.Div):
                if rval == 0 or lval % rval != 0:
                    raise ValueError("Exact integer division required.")
                sign = -1 if (lval < 0) ^ (rval < 0) else 1
                return sign * (abs(lval) // abs(rval)), True
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported syntax in trace expression: {type(node).__name__}")

    return _eval_node(tree)


def verify_work_trace(work_body: str) -> tuple[bool, int | None]:
    """Verify every `;`-separated `Work:` equation holds (`LHS == RHS`, no no-op tautologies) and return final RHS."""
    raw_clauses = [part.strip() for part in work_body.split(";")]
    if not raw_clauses or any(not clause for clause in raw_clauses):
        return False, None

    last_val: int | None = None
    for clause in raw_clauses:
        if "=" not in clause:
            return False, None
        lhs, rhs = clause.split("=", 1)
        rhs_match = _STRICT_INT_RE.fullmatch(rhs.strip())
        if rhs_match is None:
            return False, None
        rhs_val = int(rhs_match.group(0))
        lhs_clean = lhs.strip()

        if lhs_clean == "neg":
            if last_val is None or rhs_val != -last_val:
                return False, None
        else:
            try:
                lhs_val, has_binop = safe_eval_arithmetic_expr(lhs_clean)
            except (SyntaxError, ValueError, RecursionError):
                return False, None
            if not has_binop or lhs_val != rhs_val:
                return False, None
        last_val = rhs_val

    return True, last_val


def score_strict_scratchpad_rollout(
    raw_output: str,
    expected_answer: int,
    *,
    ended_with_eos: bool = True,
) -> dict:
    """Strictly binary `{0.0, 1.0}` GRPO verifier: EOS + strict format + valid `Work:` trace + trace ends at exact `Answer:`."""
    cleaned = raw_output.strip()
    match = _SCRATCHPAD_RE.fullmatch(cleaned)
    valid_format = match is not None and bool(match.group(1).strip())
    extracted = int(match.group(2)) if valid_format else None

    valid_work = False
    last_work_val: int | None = None
    if valid_format:
        valid_work, last_work_val = verify_work_trace(match.group(1).strip())

    trace_ends_at_answer = valid_format and valid_work and (last_work_val == extracted)
    answer_match = valid_format and (extracted == expected_answer)
    strict_pass = bool(ended_with_eos) and valid_format and valid_work and trace_ends_at_answer and answer_match

    return {
        "ended_with_eos": bool(ended_with_eos),
        "valid_format": valid_format,
        "valid_work": valid_work,
        "trace_ends_at_answer": trace_ends_at_answer,
        "last_work_val": last_work_val,
        "extracted": extracted,
        "exact_match": answer_match,
        "strict_pass": strict_pass,
        "reward": 1.0 if strict_pass else 0.0,
    }


_EASY_RL_CATEGORIES = frozenset({
    "add_1d_no_carry",
    "add_1d_carry",
    "add_2d_no_carry",
    "add_2d_carry",
    "add_3d_no_carry",
    "add_3d_carry",
    "add_equal",
    "add_zero",
    "add_signed",
    "sub_1d_no_borrow",
    "sub_equal",
    "sub_zero",
    "mul_1d",
    "mul_unit",
    "mul_zero",
    "div_1d",
    "div_by_unit",
    "div_zero",
})


def select_hard_rl_candidates(
    train_examples: list[ArithmeticExample],
    *,
    exclude_easy_categories: bool = True,
    canonical_only_first: bool = True,
) -> tuple[list[ArithmeticExample], dict[str, list[ArithmeticExample]]]:
    """Filter `train` examples to non-trivial candidates and index all wording variants by `canonical_id`."""
    by_canonical: dict[str, list[ArithmeticExample]] = {}
    pool: list[ArithmeticExample] = []
    for ex in train_examples:
        if exclude_easy_categories and ex.category in _EASY_RL_CATEGORIES:
            continue
        by_canonical.setdefault(ex.canonical_id, []).append(ex)

    for cid in sorted(by_canonical):
        items = by_canonical[cid]
        if canonical_only_first:
            primary = next((it for it in items if it.variant == "canonical"), items[0])
            pool.append(primary)
        else:
            pool.extend(items)
    return pool, by_canonical


def save_rl_hard_dataset(directory: Path, examples: list[ArithmeticExample], manifest: dict) -> Path:
    """Persist SHA-256 verified hard RL training examples and manifest cleanly to disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fpath = directory / "hard_train.jsonl"
    op_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    with fpath.open("w", encoding="utf-8", newline="\n") as stream:
        for ex in examples:
            op_counts[ex.op] = op_counts.get(ex.op, 0) + 1
            cat_counts[ex.category] = cat_counts.get(ex.category, 0) + 1
            stream.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    full_manifest = {
        **manifest,
        "count": len(examples),
        "op_counts": dict(sorted(op_counts.items())),
        "category_counts": dict(sorted(cat_counts.items())),
        "files": {"hard_train.jsonl": {"count": len(examples), "sha256": sha256(fpath)}},
    }
    (directory / "manifest.json").write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    return directory


def load_rl_hard_dataset(directory: Path) -> tuple[list[ArithmeticExample], dict]:
    """Load and SHA-256 verify a previously mined hard RL training dataset."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hard RL manifest not found in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fpath = directory / "hard_train.jsonl"
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing hard RL split {fpath}")
    expected_sha = manifest.get("files", {}).get("hard_train.jsonl", {}).get("sha256")
    if expected_sha and sha256(fpath) != expected_sha:
        raise ValueError(f"Checksum mismatch for {fpath}")
    items: list[ArithmeticExample] = []
    with fpath.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                items.append(ArithmeticExample(**json.loads(line)))
    return items, manifest
