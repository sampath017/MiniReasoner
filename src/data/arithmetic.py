"""Leakage-free arithmetic dataset generation, inspection, disk/Drive persistence, and SFT masking."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np
import torch
from tokenizers import Tokenizer

from ..core.tokenizer import sha256


GENERATOR_VERSION = "arith_v2"
SUPPORTED_OPS = ("add", "sub", "mul", "div", "mod")
DEFAULT_OPS = ("add", "sub", "mul", "div")
_STRICT_INT_RE = re.compile(r"-?\d+")

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
    op: str        # "add" | "sub" | "mul" | "div" | "mod"
    a: int         # canonical left operand
    b: int         # canonical right operand
    answer: int
    category: str

    @property
    def canonical_id(self) -> str:
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


def canonical_problem(op: str, a: int, b: int) -> CanonicalProblem:
    """Normalize commutative problems (`add`, `mul`) so swapped order maps to one canonical key."""
    if op == "add":
        ca, cb = (a, b) if a <= b else (b, a)
        answer = ca + cb
    elif op == "sub":
        ca, cb = a, b
        answer = ca - cb
    elif op == "mul":
        ca, cb = (a, b) if a <= b else (b, a)
        answer = ca * cb
    elif op == "div":
        if b == 0:
            raise ZeroDivisionError("Division by zero cannot form an integer arithmetic problem.")
        if a % b != 0:
            raise ValueError(f"Exact integer division requires clean divisibility ({a} / {b}).")
        ca, cb = a, b
        # Sign-exact integer quotient matching clean multiplication inverse (q * b == a)
        sign = -1 if (a < 0) ^ (b < 0) else 1
        answer = sign * (abs(a) // abs(b))
    elif op == "mod":
        if b == 0:
            raise ZeroDivisionError("Modulo by zero is undefined.")
        ca, cb = a, b
        answer = ca % cb
    else:
        raise ValueError(f"Unsupported arithmetic op {op!r}")

    return CanonicalProblem(
        op=op,
        a=ca,
        b=cb,
        answer=answer,
        category=classify_problem(op, ca, cb),
    )


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


def enumerate_canonical_problems(
    *,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    seed: int = 42,
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
        splits[name].sort(key=lambda p: (p.op, p.a, p.b))
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


def render_examples(problem: CanonicalProblem, *, include_variants: bool) -> list[ArithmeticExample]:
    """Render canonical `"What is a op b?\\nAnswer:"` and optional phrasing/commuted variants."""
    sym, word_op = _op_strings(problem.op)
    answer_text = str(problem.answer)

    examples = [
        ArithmeticExample(
            canonical_id=problem.canonical_id,
            op=problem.op,
            category=problem.category,
            prompt=f"What is {problem.a} {sym} {_fmt_rhs(problem.b)}?\nAnswer:",
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
        suffix = "_commuted" if commuted else ""
        rhs = _fmt_rhs(y)
        if commuted:
            examples.append(
                ArithmeticExample(
                    canonical_id=problem.canonical_id,
                    op=problem.op,
                    category=problem.category,
                    prompt=f"What is {x} {sym} {rhs}?\nAnswer:",
                    answer_text=answer_text,
                    answer=problem.answer,
                    variant="canonical_commuted",
                )
            )
        examples.append(
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"Calculate {x} {sym} {rhs}.\nAnswer:",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"calculate{suffix}",
            )
        )
        examples.append(
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"{x} {sym} {rhs} =\nAnswer:",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"equation{suffix}",
            )
        )
        examples.append(
            ArithmeticExample(
                canonical_id=problem.canonical_id,
                op=problem.op,
                category=problem.category,
                prompt=f"What is {x} {word_op} {rhs}?\nAnswer:",
                answer_text=answer_text,
                answer=problem.answer,
                variant=f"word_op{suffix}",
            )
        )
        if max(abs(x), abs(y)) < 1000 and not commuted:
            if problem.op == "mod":
                word_prompt = f"What is the remainder when {int_to_words(x)} is divided by {int_to_words(y)}?\nAnswer:"
            else:
                word_prompt = f"What is {int_to_words(x)} {word_op} {int_to_words(y)}?\nAnswer:"
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
) -> str:
    """Stable human-readable directory name for caching an arithmetic SFT dataset."""
    ops_tag = "-".join(normalize_ops(ops))
    sign_tag = "signed" if allow_negative else "nonneg"
    return f"{GENERATOR_VERSION}_{ops_tag}_d{max_digits}_{sign_tag}_n{samples_per_op}_s{seed}"


def build_arithmetic_dataset(
    *,
    ops: tuple[str, ...] | list[str] | str = DEFAULT_OPS,
    max_digits: int = 3,
    allow_negative: bool = True,
    samples_per_op: int = 12_000,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, list[ArithmeticExample]], dict]:
    """Generate leakage-free `train`, `val`, and `test` arithmetic splits with manifest metadata."""
    op_list = normalize_ops(ops)
    problems = enumerate_canonical_problems(
        ops=op_list,
        max_digits=max_digits,
        allow_negative=allow_negative,
        samples_per_op=samples_per_op,
        seed=seed,
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
            rendered.extend(render_examples(prob, include_variants=include_variants))
        examples_by_split[split_name] = rendered
        category_counts[split_name] = cats
        op_counts[split_name] = ops_cnt
        canonical_counts[split_name] = len(split_problems)

    manifest = {
        "generator_version": GENERATOR_VERSION,
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
    lines = [
        "=" * 74,
        f"ARITHMETIC SFT DATASET ({manifest['generator_version']})",
        "=" * 74,
        f"Operations:     {', '.join(manifest['ops'])} | Max Digits: {manifest['max_digits']} | Negatives: {manifest['allow_negative']} | Seed: {manifest['seed']}",
        f"Canonical Pool: train={manifest['canonical_counts']['train']:,} | val={manifest['canonical_counts']['val']:,} | sealed_test={manifest['canonical_counts']['test']:,}",
        f"Rendered Rows:  train={manifest['example_counts']['train']:,} | val={manifest['example_counts']['val']:,} | sealed_test={manifest['example_counts']['test']:,}",
        "-" * 74,
        "Per-Operation Canonical Split Breakdown:",
    ]
    for op in manifest["ops"]:
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
            one_line = ex.prompt.replace("\n", "\\n")
            lines.append(f"  [{ex.op} | {ex.category} | {ex.variant}] {one_line!r} -> {ex.answer_text!r}")

    lines.append("-" * 74)
    lines.append("Sample Held-Out Val Prompts (canonical evaluation format):")
    seen_val_ops: set[str] = set()
    for ex in splits["val"]:
        if ex.op not in seen_val_ops:
            seen_val_ops.add(ex.op)
            one_line = ex.prompt.replace("\n", "\\n")
            lines.append(f"  [{ex.op} | {ex.category}] {one_line!r} -> {ex.answer_text!r}")
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
    sync_drive: bool = True,
    drive_data_root: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, list[ArithmeticExample]], dict]:
    """Generate, save to disk, print preview, and optionally cache to Google Drive."""
    from ..settings import DRIVE_DATA_ROOT, SFT_DATA_DIR

    op_list = normalize_ops(ops)
    target_dir = Path(output_dir) if output_dir is not None else (
        SFT_DATA_DIR / dataset_slug(
            ops=op_list, max_digits=max_digits, allow_negative=allow_negative,
            samples_per_op=samples_per_op, seed=seed,
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
    use_drive: bool = True,
    drive_data_root: str | None = None,
) -> tuple[Path, dict[str, list[ArithmeticExample]], dict]:
    """Load prepared SFT data from local disk or Google Drive before training starts."""
    from ..settings import DRIVE_DATA_ROOT, SFT_DATA_DIR

    op_list = normalize_ops(ops)
    target_dir = Path(dataset_dir) if dataset_dir is not None else (
        SFT_DATA_DIR / dataset_slug(
            ops=op_list, max_digits=max_digits, allow_negative=allow_negative,
            samples_per_op=samples_per_op, seed=seed,
        )
    )
    if (target_dir / "manifest.json").is_file():
        splits, manifest = load_arithmetic_dataset(target_dir)
        return target_dir, splits, manifest

    if use_drive:
        from .drive_corpus import reuse_sft_dataset

        expected_identity = {
            "generator_version": GENERATOR_VERSION,
            "ops": list(op_list),
            "max_digits": max_digits,
            "allow_negative": allow_negative,
            "samples_per_op": samples_per_op,
            "seed": seed,
        }
        restored = reuse_sft_dataset(drive_data_root or DRIVE_DATA_ROOT, target_dir, expected_identity)
        if restored is not None:
            splits, manifest = load_arithmetic_dataset(target_dir)
            return target_dir, splits, manifest

    raise FileNotFoundError(
        f"Prepared SFT arithmetic dataset not found at {target_dir}. "
        "Run `bash scripts/prepare_sft_data.sh` first to generate, inspect, and cache the dataset."
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


def score_arithmetic_completion(raw_output: str, expected_answer: int) -> dict:
    """Deterministic verifier matching Phase 4 SFT evaluation and Phase 5 GRPO spec."""
    valid_format, extracted = extract_strict_integer(raw_output)
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
    for ex in examples:
        by_cat.setdefault(ex.category, []).append(ex)

    shuffled_cats = {}
    for cat, items in by_cat.items():
        perm = rng.permutation(len(items))
        shuffled_cats[cat] = [items[int(i)] for i in perm]

    selected: list[ArithmeticExample] = []
    idx = 0
    categories = sorted(shuffled_cats)
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
