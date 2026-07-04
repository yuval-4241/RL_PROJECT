"""
Reuses the SRT repo's real \\boxed{} extraction and math-equivalence scoring
(verl.utils.reward_score) without importing the top-level `verl` package.

`import verl` executes verl/__init__.py, which pulls in torch, tensordict and
ray (see verl/protocol.py, verl/utils/tokenizer.py) -- multi-GB heavyweight
deps that this Groq-only, inference-only project has no other use for.

Instead we load the exact repo source files (reward_score/__init__.py,
math.py, math_verify.py) directly via importlib, registering lightweight
namespace stand-ins for `verl` and `verl.utils` so the repo files' internal
`from . import ...` / `from verl.utils.reward_score.math import ...`
statements resolve normally. This is the same code the repo runs during
training -- nothing here is reimplemented.
"""
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent / "srt_repo"
VERL_ROOT = REPO_ROOT / "verl"

MATH_DAPO_SOURCE = "math_dapo"


def _ensure_namespace_package(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_reward_score_package() -> types.ModuleType:
    key = "verl.utils.reward_score"
    if key in sys.modules:
        return sys.modules[key]

    if not VERL_ROOT.exists():
        raise FileNotFoundError(
            f"SRT repo not found at {REPO_ROOT}. Expected a checkout/symlink of "
            "https://github.com/tajwarfahim/srt at RL_Project/srt_repo."
        )

    _ensure_namespace_package("verl", VERL_ROOT)
    _ensure_namespace_package("verl.utils", VERL_ROOT / "utils")

    init_path = VERL_ROOT / "utils" / "reward_score" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        key, init_path, submodule_search_locations=[str(init_path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


_reward_score = _load_reward_score_package()


def extract_boxed_answer(solution_str: str):
    """verl.utils.reward_score._extract_verifiable_part_of_solution for data_source='math_dapo'."""
    return _reward_score._extract_verifiable_part_of_solution(
        data_source=MATH_DAPO_SOURCE,
        solution_str=solution_str,
    )


def score_against_ground_truth(solution_str: str, ground_truth: str) -> float:
    """verl.utils.reward_score._default_compute_score for data_source='math_dapo'."""
    return _reward_score._default_compute_score(
        data_source=MATH_DAPO_SOURCE,
        solution_str=solution_str,
        ground_truth=ground_truth,
    )
