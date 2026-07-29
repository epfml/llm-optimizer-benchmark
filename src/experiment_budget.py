from dataclasses import asdict, dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class TokenBudgetPlan:
    train_token_budget: int
    tokens_per_iteration: int
    iterations: int
    eval_at_tokens: tuple
    eval_at_steps: tuple
    data_unique_tokens: int
    target_data_exposure: float

    def to_dict(self):
        return asdict(self)


def _positive_int(value, name):
    if value is None or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return int(value)


def _tokens_to_steps(tokens, tokens_per_iteration, name):
    if tokens % tokens_per_iteration != 0:
        raise ValueError(
            f"{name}={tokens} must be divisible by tokens_per_iteration="
            f"{tokens_per_iteration}. Choose an exact optimizer-step boundary."
        )
    return tokens // tokens_per_iteration


def make_token_budget_plan(
    *,
    train_token_budget: int,
    tokens_per_iteration: int,
    data_unique_tokens: int,
    eval_at_tokens: Optional[Iterable[int]] = None,
    strict_sub_one_pass: bool = False,
) -> TokenBudgetPlan:
    """Convert token budgets to exact optimizer-step boundaries."""

    train_token_budget = _positive_int(train_token_budget, "train_token_budget")
    tokens_per_iteration = _positive_int(tokens_per_iteration, "tokens_per_iteration")
    data_unique_tokens = _positive_int(data_unique_tokens, "data_unique_tokens")

    if strict_sub_one_pass and train_token_budget > data_unique_tokens:
        raise ValueError(
            "strict_sub_one_pass requires train_token_budget <= "
            f"data_unique_tokens, got {train_token_budget} > {data_unique_tokens}."
        )

    iterations = _tokens_to_steps(
        train_token_budget, tokens_per_iteration, "train_token_budget"
    )

    requested_eval_tokens = (
        tuple(eval_at_tokens) if eval_at_tokens is not None else (train_token_budget,)
    )
    if not requested_eval_tokens:
        requested_eval_tokens = (train_token_budget,)

    normalized_eval_tokens = []
    for token_count in requested_eval_tokens:
        token_count = _positive_int(token_count, "eval_at_tokens")
        if token_count > train_token_budget:
            raise ValueError(
                f"eval_at_tokens contains {token_count}, which exceeds "
                f"train_token_budget={train_token_budget}."
            )
        normalized_eval_tokens.append(token_count)

    normalized_eval_tokens.append(train_token_budget)
    normalized_eval_tokens = tuple(sorted(set(normalized_eval_tokens)))
    eval_at_steps = tuple(
        _tokens_to_steps(token_count, tokens_per_iteration, "eval_at_tokens")
        for token_count in normalized_eval_tokens
    )

    return TokenBudgetPlan(
        train_token_budget=train_token_budget,
        tokens_per_iteration=tokens_per_iteration,
        iterations=iterations,
        eval_at_tokens=normalized_eval_tokens,
        eval_at_steps=eval_at_steps,
        data_unique_tokens=data_unique_tokens,
        target_data_exposure=train_token_budget / data_unique_tokens,
    )
