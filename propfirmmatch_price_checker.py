import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

TRPC_BASE = "https://propfirmmatch.com/api/trpc"
PROPFIRM_COMPARE_BASE = "https://propfirm.compare/api"
PAGE_SIZE = 50
TIMEOUT = 60.0

OUTPUT_DIR = Path("data/parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFD_JSON_PATH = Path("tpt/cfd_lowest_challenge_fees.json")
FUTURES_JSON_PATH = Path("tpt/futures_lowest_challenge_fees.json")


@dataclass(frozen=True)
class FirmConfig:
    market: str
    firm_key: str
    display_name: str
    firm_name_pattern: str


FIRM_CONFIGS: tuple[FirmConfig, ...] = (
    FirmConfig("cfd", "fundednext", "FundedNext", r"(?i)^Funded\s*Next$"),
    FirmConfig("cfd", "hantec_trader", "Hantec Trader", r"(?i)^Hantec\s+Trader$"),
    FirmConfig("cfd", "ftmo", "FTMO", r"(?i)^FTMO$"),
    FirmConfig("cfd", "city_traders_imperium", "City Traders Imperium", r"(?i)^City\s+Traders\s+Imperium$"),
    FirmConfig("cfd", "funding_pips", "Funding Pips", r"(?i)^Funding\s*Pips$"),
    FirmConfig("cfd", "maven_trading", "Maven Trading", r"(?i)^Maven$"),
    FirmConfig("cfd", "brightfunded", "BrightFunded", r"(?i)^Bright\s?Funded$"),
    FirmConfig("cfd", "the_5ers", "The 5%ers", r"(?i)^The\s*5ers$"),
    FirmConfig("cfd", "e8_markets", "E8 Markets", r"(?i)^E8\s+Markets$"),
    FirmConfig("cfd", "alpha_capital_group", "Alpha Capital Group", r"(?i)^Alpha\s+Capital$"),
    FirmConfig("futures", "lucid_trading", "Lucid Trading", r"(?i)^Lucid\s+Trading$"),
    FirmConfig("futures", "tradeify", "Tradeify", r"(?i)^Tradeify$"),
    FirmConfig("futures", "fundednext_futures", "FundedNext Futures", r"(?i)^FundedNext\s+Futures$"),
    FirmConfig("futures", "my_funded_futures", "My Funded Futures", r"(?i)^My\s+Funded\s+Futures$"),
    FirmConfig("futures", "alpha_futures", "Alpha Futures", r"(?i)^Alpha\s+Futures$"),
    FirmConfig("futures", "take_profit_trader", "Take Profit Trader", r"(?i)^Take\s+Profit\s+Trader$"),
    FirmConfig("futures", "apex_trader_funding", "Apex Trader Funding", r"(?i)^Apex\s+Trader\s+Funding$"),
    FirmConfig("futures", "tradeday", "TradeDay", r"(?i)^TradeDay$"),
    FirmConfig("futures", "topstep", "Topstep", r"(?i)^Topstep$"),
    FirmConfig("futures", "top_one_futures", "Top One Futures", r"(?i)^Top\s*One\s*Futures$"),
)

TARGET_SIZES_BY_MARKET: dict[str, tuple[str, ...]] = {
    "cfd": ("25K", "50K", "100K"),
    "futures": ("50K", "100K", "150K"),
}

CHALLENGE_NAME_INCLUDE_BY_KEY: dict[str, str] = {
    "fundednext": r"(?i)^Funded\s*Next\s+-",
    "hantec_trader": r"(?i)^Hantec\s+Trader\s+-",
    "ftmo": r"(?i)^FTMO\s+-",
    "city_traders_imperium": r"(?i)^City\s+Traders\s+Imperium\s+-",
    "funding_pips": r"(?i)^Funding\s*Pips\s+-",
    "maven_trading": r"(?i)^Maven\s+-",
    "brightfunded": r"(?i)^Bright\s?Funded\s+-",
    "the_5ers": r"(?i)^The\s*5%?ers\s+-\s+High\s+Stakes",
    "e8_markets": r"(?i)^E8\s+Markets\s+-",
    "alpha_capital_group": r"(?i)^Alpha\s+Capital\s+-",
}

PROPFIRM_COMPARE_SLUGS: dict[str, str] = {
    "lucid_trading": "lucid-trading",
    "tradeify": "tradeify",
    "fundednext_futures": "fundednext-futures",
    "my_funded_futures": "my-funded-futures",
    "alpha_futures": "alpha-futures",
    "take_profit_trader": "take-profit-trader",
    "apex_trader_funding": "apex-trader-funding",
    "tradeday": "tradeday",
    "topstep": "topstep",
    "top_one_futures": "top-one-futures",
}

PROPFIRM_COMPARE_DISCOUNT_CODE_OVERRIDES: dict[str, str] = {
    "lucid_trading": "VAULT",
}

FIRM_LABEL_BY_KEY: dict[str, str] = {
    "fundednext": "FundedNext",
    "hantec_trader": "Hantec Trader",
    "ftmo": "FTMO",
    "city_traders_imperium": "City Traders Imperium",
    "funding_pips": "FundingPips",
    "maven_trading": "Maven Trading",
    "brightfunded": "BrightFunded",
    "the_5ers": "The 5%ers",
    "e8_markets": "E8 Markets",
    "alpha_capital_group": "Alpha Capital Group",
    "lucid_trading": "Lucid Trading",
    "tradeify": "Tradeify",
    "fundednext_futures": "FundedNext Futures",
    "my_funded_futures": "MyFundedFutures",
    "alpha_futures": "Alpha Futures",
    "take_profit_trader": "Take Profit Trader",
    "apex_trader_funding": "Apex Trader Funding",
    "tradeday": "TradeDay",
    "topstep": "Topstep",
    "top_one_futures": "Top One Futures",
}

SOURCE_URL_BY_KEY: dict[str, str] = {
    "fundednext": "https://propfirmmatch.com/prop-firms/fundednext/challenges",
    "hantec_trader": "https://propfirmmatch.com/prop-firms/hantec-trader/challenges",
    "ftmo": "https://propfirmmatch.com/prop-firms/ftmo/challenges",
    "city_traders_imperium": "https://propfirmmatch.com/prop-firms/city-traders-imperium/challenges",
    "funding_pips": "https://propfirmmatch.com/prop-firms/funding-pips/challenges",
    "maven_trading": "https://propfirmmatch.com/prop-firms/maven-trading/challenges",
    "brightfunded": "https://propfirmmatch.com/prop-firms/brightfunded/challenges",
    "the_5ers": "https://propfirmmatch.com/prop-firms/the-5-ers/challenges",
    "e8_markets": "https://propfirmmatch.com/prop-firms/e8-markets/challenges",
    "alpha_capital_group": "https://propfirmmatch.com/prop-firms/alpha-capital-group/challenges",
    "lucid_trading": "https://propfirm.compare/propfirm/lucid-trading",
    "tradeify": "https://propfirmmatch.com/futures/prop-firms/tradeify/challenges",
    "fundednext_futures": "https://propfirmmatch.com/futures/prop-firms/fundednext-futures/challenges",
    "my_funded_futures": "https://propfirmmatch.com/futures/prop-firms/my-funded-futures/challenges",
    "alpha_futures": "https://propfirmmatch.com/futures/prop-firms/alpha-futures/challenges",
    "take_profit_trader": "https://propfirmmatch.com/futures/prop-firms/take-profit-trader/challenges",
    "apex_trader_funding": "https://propfirmmatch.com/futures/prop-firms/apex-trader-funding/challenges",
    "tradeday": "https://propfirmmatch.com/futures/prop-firms/tradeday/challenges",
    "topstep": "https://propfirmmatch.com/futures/prop-firms/topstep/challenges",
    "top_one_futures": "https://propfirmmatch.com/futures/prop-firms/top-one-futures/challenges",
}

SIZE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b5\s*k\b|\b5,?000\b"), "5K"),
    (re.compile(r"(?i)\b10\s*k\b|\b10,?000\b"), "10K"),
    (re.compile(r"(?i)\b25\s*k\b|\b25,?000\b"), "25K"),
    (re.compile(r"(?i)\b50\s*k\b|\b50,?000\b"), "50K"),
    (re.compile(r"(?i)\b100\s*k\b|\b100,?000\b"), "100K"),
    (re.compile(r"(?i)\b150\s*k\b|\b150,?000\b"), "150K"),
    (re.compile(r"(?i)\b200\s*k\b|\b200,?000\b"), "200K"),
)

STEP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b1\s*[-_ ]?step\b|\bone\s*[-_ ]?step\b|\b1\s*phase\b|\bone\s*phase\b"), "one_step"),
    (re.compile(r"(?i)\b2\s*[-_ ]?step\b|\btwo\s*[-_ ]?step\b|\b2\s*phase\b|\btwo\s*phase\b"), "two_step"),
)


def to_float_money(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d.]", "", text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def first_discount_code(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        code = value.strip()
        if not code:
            return None
        return code.split(",")[0].strip() or None

    if isinstance(value, list):
        for item in value:
            code = first_discount_code(item)
            if code:
                return code
        return None

    if isinstance(value, dict):
        for key in ("name", "code", "promoCode", "promo_code"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        for key in ("promo", "discount", "discountCode", "discountCodes", "challengePromos", "promos"):
            nested = value.get(key)
            code = first_discount_code(nested)
            if code:
                return code

        for nested in value.values():
            if isinstance(nested, (dict, list)):
                code = first_discount_code(nested)
                if code:
                    return code

    return None


def parse_discount_percent(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\d.]", "", text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_size(text: str | None) -> str | None:
    if not text:
        return None
    for pattern, normalized in SIZE_PATTERNS:
        if pattern.search(text):
            return normalized
    return None


def normalize_step(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for pattern, normalized in STEP_PATTERNS:
        if pattern.search(text):
            return normalized
    return None


def extract_challenge_step(row: dict[str, Any], challenge_name: str) -> str | None:
    for key in ("steps", "step", "challengeSteps", "challengeType", "accountType"):
        normalized = normalize_step(row.get(key))
        if normalized:
            return normalized

    normalized_name = normalize_step(challenge_name)
    if normalized_name:
        return normalized_name

    phase2_target = to_float_money(row.get("phase2ProfitTarget"))
    if phase2_target is not None and phase2_target > 0:
        return "two_step"

    return None


def canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def size_key(size: str) -> str:
    return size.lower()


def iso_utc_now_z() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def challenge_name_allowed(firm_key: str, challenge_name: str) -> bool:
    pattern = CHALLENGE_NAME_INCLUDE_BY_KEY.get(firm_key)
    if not pattern:
        return True
    return bool(re.search(pattern, challenge_name))


def trpc_input(payload: dict[str, Any], include_meta: bool = False) -> str:
    wrapper: dict[str, Any] = {"0": {"json": payload}}
    if include_meta:
        wrapper["0"]["meta"] = {"values": {"search": ["undefined"]}, "v": 1}
    return json.dumps(wrapper, separators=(",", ":"))


def trpc_get(client: httpx.Client, method: str, payload: dict[str, Any], include_meta: bool = False) -> Any:
    response = client.get(
        f"{TRPC_BASE}/{method}",
        params={"batch": "1", "input": trpc_input(payload=payload, include_meta=include_meta)},
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected tRPC response for {method}")
    return data[0]["result"]["data"]["json"]


def fetch_all_firms(client: httpx.Client) -> list[dict[str, Any]]:
    response = client.get(
        f"{TRPC_BASE}/challenge.getListFilteringOptions",
        params={
            "batch": "1",
            "input": json.dumps(
                {"0": {"json": None, "meta": {"values": ["undefined"], "v": 1}}},
                separators=(",", ":"),
            ),
        },
    )
    response.raise_for_status()
    payload = response.json()[0]["result"]["data"]["json"]
    firms = payload.get("firms") or []
    return [firm for firm in firms if isinstance(firm, dict) and firm.get("id") and firm.get("name")]


def resolve_firm_ids(all_firms: list[dict[str, Any]], configs: Iterable[FirmConfig]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for config in configs:
        for firm in all_firms:
            name = str(firm.get("name") or "")
            if re.search(config.firm_name_pattern, name):
                resolved[config.firm_key] = str(firm["id"])
                break
    return resolved


def compute_discount_percent(original_price: float | None, discounted_price: float | None) -> float | None:
    if original_price is None or discounted_price is None:
        return None
    if original_price <= 0:
        return None
    discount = (1.0 - (discounted_price / original_price)) * 100.0
    return round(discount, 2) if discount >= 0 else None


def apply_activation_fee(price: float | None, activation_fee: float | None) -> float | None:
    if price is None:
        return None
    if activation_fee is None:
        return price
    return round(price + activation_fee, 2)


def format_usd_compact(value: float) -> str:
    rounded = int(round(value))
    return f"${rounded:,.0f}"


def build_max_loss_label(row: dict[str, Any]) -> str | None:
    max_amount = to_float_money(row.get("phase1MaxDrawdownAmount"))
    daily_amount = to_float_money(row.get("phase1MaxDailyLossAmount"))
    if max_amount is None:
        max_amount = to_float_money(row.get("maxDrawdownAmount"))
    if daily_amount is None:
        daily_amount = to_float_money(row.get("maxDailyLossAmount"))

    if max_amount is not None:
        label = format_usd_compact(max_amount)
        if daily_amount is not None:
            label = f"{label} / {format_usd_compact(daily_amount)} daily"
        return label

    max_pct = to_float_money(row.get("maxDrawdown"))
    daily_pct = to_float_money(row.get("maxDailyLoss"))
    if max_pct is None:
        max_pct = to_float_money(row.get("phase1MaxDrawdown"))
    if daily_pct is None:
        daily_pct = to_float_money(row.get("phase1MaxDailyLoss"))

    if max_pct is not None:
        max_text = f"{max_pct:g}% max"
        if daily_pct is not None:
            return f"{max_text} / {daily_pct:g}% daily"
        return max_text

    return None


def build_max_loss_label_from_compare_item(item: dict[str, Any]) -> str | None:
    drawdown_amount = to_float_money(item.get("drawdownAmount"))
    daily_loss_amount = to_float_money(item.get("dailyLossLimitAmount"))

    if drawdown_amount is None:
        return None

    label = format_usd_compact(drawdown_amount)
    if daily_loss_amount is not None and daily_loss_amount > 0:
        label = f"{label} / {format_usd_compact(daily_loss_amount)} daily"
    return label


def extract_profit_target_phase_values(row: dict[str, Any]) -> tuple[float | None, float | None]:
    phase1 = to_float_money(row.get("phase1ProfitTarget"))
    phase2 = to_float_money(row.get("phase2ProfitTarget"))
    if phase1 is None:
        phase1 = to_float_money(row.get("profitTargetSum"))
    return phase1, phase2


def build_profit_target_label(row: dict[str, Any]) -> str | None:
    phase1_pct = to_float_money(row.get("phase1ProfitTarget"))
    phase2_pct = to_float_money(row.get("phase2ProfitTarget"))
    phase1_amount = to_float_money(row.get("phase1ProfitTargetAmount"))
    phase2_amount = to_float_money(row.get("phase2ProfitTargetAmount"))

    if phase1_pct is not None and phase2_pct is not None:
        return f"{phase1_pct:g}% / {phase2_pct:g}%"
    if phase1_pct is not None:
        return f"{phase1_pct:g}%"

    if phase1_amount is not None and phase2_amount is not None:
        return f"{format_usd_compact(phase1_amount)} / {format_usd_compact(phase2_amount)}"
    if phase1_amount is not None:
        return format_usd_compact(phase1_amount)

    total_pct = to_float_money(row.get("profitTargetSum"))
    if total_pct is not None:
        return f"{total_pct:g}%"

    return None


def extract_ptdd_ratio(row: dict[str, Any]) -> float | None:
    ratio = to_float_money(row.get("ptddRatio"))
    if ratio is not None:
        return round(ratio, 4)

    phase1_target = to_float_money(row.get("phase1ProfitTarget"))
    phase1_drawdown = to_float_money(row.get("phase1MaxDrawdown"))
    if phase1_drawdown is None:
        phase1_drawdown = to_float_money(row.get("maxDrawdown"))
    if phase1_target is not None and phase1_drawdown is not None and phase1_drawdown > 0:
        return round(phase1_target / phase1_drawdown, 4)

    phase1_target_amount = to_float_money(row.get("phase1ProfitTargetAmount"))
    phase1_drawdown_amount = to_float_money(row.get("phase1MaxDrawdownAmount"))
    if phase1_drawdown_amount is None:
        phase1_drawdown_amount = to_float_money(row.get("maxDrawdownAmount"))
    if phase1_target_amount is not None and phase1_drawdown_amount is not None and phase1_drawdown_amount > 0:
        return round(phase1_target_amount / phase1_drawdown_amount, 4)

    return None


def fetch_firm_challenges(client: httpx.Client, firm_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip = 0
    while True:
        payload = {
            "skip": skip,
            "limit": PAGE_SIZE,
            "search": None,
            "filter": {
                "bookmarksOnly": False,
                "firmIds": [firm_id],
                "ignoreListingStatus": True,
                "applyDiscount": True,
            },
        }
        result = trpc_get(client=client, method="challenge.listFiltered", payload=payload, include_meta=True)
        data = result.get("data") or []
        has_more = bool(result.get("hasMore"))
        if not data:
            break
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
        if not has_more:
            break
        skip += PAGE_SIZE
    return rows


def fetch_propfirm_compare_firm(client: httpx.Client, firm_slug: str) -> dict[str, Any] | None:
    response = client.get(f"{PROPFIRM_COMPARE_BASE}/prop-firms/by-slug/{firm_slug}")
    if response.status_code != 200:
        return None
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def parse_account_size_pricing(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    return []


def build_propfirm_compare_fallback_rows(config: FirmConfig, payload: dict[str, Any]) -> list[dict[str, Any]]:
    discount_code = first_discount_code(payload.get("discountCodes"))
    if not discount_code:
        discount_code = PROPFIRM_COMPARE_DISCOUNT_CODE_OVERRIDES.get(config.firm_key)
    top_level_discount_pct = parse_discount_percent(payload.get("discount"))
    account_rows = parse_account_size_pricing(payload.get("accountSizePricing"))
    firm_name = str(payload.get("name") or config.display_name)
    fallback_profit_split_pct = to_float_money(payload.get("profitSplit"))

    rows: list[dict[str, Any]] = []
    for item in account_rows:
        original_price = to_float_money(item.get("price"))
        if original_price is None:
            continue

        activation_fee = to_float_money(item.get("activationFee")) if config.market == "futures" else None
        base_original_price = original_price

        account_size_value = item.get("accountSize")
        account_size = str(account_size_value).strip() if account_size_value is not None else None
        normalized_size = normalize_size(account_size)
        if normalized_size is None:
            normalized_size = normalize_size(str(item.get("accountName") or ""))

        discount_override_pct = parse_discount_percent(item.get("discountOverride"))
        applied_discount_pct = discount_override_pct if discount_override_pct is not None else top_level_discount_pct
        discounted_price = base_original_price
        if applied_discount_pct is not None:
            discounted_price = round(base_original_price * (1.0 - (applied_discount_pct / 100.0)), 2)

        original_price = apply_activation_fee(base_original_price, activation_fee)
        discounted_price = apply_activation_fee(discounted_price, activation_fee)

        account_type = str(item.get("accountType") or "").strip()
        account_name = str(item.get("accountName") or "").strip()
        challenge_label_parts = [firm_name]
        if account_name:
            challenge_label_parts.append(account_name)
        if account_type:
            challenge_label_parts.append(account_type)

        challenge_name = " - ".join(challenge_label_parts)

        profit_target_amount = to_float_money(item.get("profitTargetAmount"))
        max_drawdown_amount = to_float_money(item.get("drawdownAmount"))
        ptdd_ratio = None
        if profit_target_amount is not None and max_drawdown_amount is not None and profit_target_amount > 0:
            ptdd_ratio = round(max_drawdown_amount / profit_target_amount, 4)

        rows.append(
            {
                "market": config.market,
                "firm_key": config.firm_key,
                "firm_name": firm_name,
                "challenge_name": challenge_name,
                "challenge_step": extract_challenge_step(item, challenge_name),
                "account_size": account_size,
                "normalized_size": normalized_size,
                "original_price": original_price,
                "discounted_price": discounted_price,
                "discount_pct": compute_discount_percent(
                    original_price=original_price,
                    discounted_price=discounted_price,
                ),
                "discount_code": discount_code,
                "currency": "USD",
                "source": "propfirm_compare_api",
                "activation_fee": activation_fee,
                "max_loss_label": build_max_loss_label_from_compare_item(item),
                "max_drawdown_amount": max_drawdown_amount,
                "max_daily_loss_amount": to_float_money(item.get("dailyLossLimitAmount")),
                "profit_target_phase1": None,
                "profit_target_phase2": None,
                "profit_target_label": format_usd_compact(profit_target_amount)
                if profit_target_amount is not None
                else None,
                "profit_split_pct": fallback_profit_split_pct,
                "ptdd_ratio": ptdd_ratio,
            }
        )

    return rows


def build_price_rows(config: FirmConfig, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    price_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        challenge_name = str(row.get("name") or "").strip()
        if not challenge_name or re.search(r"(?i)crypto", challenge_name):
            continue
        if not challenge_name_allowed(config.firm_key, challenge_name):
            continue

        account_size = str(row.get("accountSize") or "").strip() or None
        normalized_size = normalize_size(account_size) or normalize_size(challenge_name)
        challenge_step = extract_challenge_step(row, challenge_name)

        original_price = to_float_money(row.get("price"))
        discounted_price = to_float_money(row.get("discountedPrice"))
        effective_price = discounted_price if discounted_price is not None else original_price

        activation_fee = None
        if config.market == "futures":
            activation_fee = to_float_money(row.get("activationFee"))
            if activation_fee is None:
                activation_fee = to_float_money(row.get("activationFeeAmount"))
        original_price = apply_activation_fee(original_price, activation_fee)
        effective_price = apply_activation_fee(effective_price, activation_fee)

        if effective_price is None:
            continue

        discount_code = first_discount_code(row.get("challengePromos") or row.get("promos"))
        discount_pct = compute_discount_percent(original_price=original_price, discounted_price=effective_price)
        profit_target_phase1, profit_target_phase2 = extract_profit_target_phase_values(row)

        price_rows.append(
            {
                "market": config.market,
                "firm_key": config.firm_key,
                "firm_name": config.display_name,
                "challenge_name": challenge_name,
                "challenge_step": challenge_step,
                "account_size": account_size,
                "normalized_size": normalized_size,
                "original_price": original_price,
                "discounted_price": effective_price,
                "discount_pct": discount_pct,
                "discount_code": discount_code,
                "currency": "USD",
                "source": "propfirmmatch_api",
                "activation_fee": activation_fee,
                "max_loss_label": build_max_loss_label(row),
                "max_drawdown_amount": to_float_money(row.get("phase1MaxDrawdownAmount"))
                or to_float_money(row.get("maxDrawdownAmount")),
                "max_daily_loss_amount": to_float_money(row.get("phase1MaxDailyLossAmount"))
                or to_float_money(row.get("maxDailyLossAmount")),
                "profit_target_phase1": profit_target_phase1,
                "profit_target_phase2": profit_target_phase2,
                "profit_target_label": build_profit_target_label(row),
                "profit_split_pct": to_float_money(row.get("profitSplit")),
                "ptdd_ratio": extract_ptdd_ratio(row),
            }
        )
    return price_rows


def pick_cheapest_targets(df: pl.DataFrame) -> pl.DataFrame:
    target_df = df.filter(
        ((pl.col("market") == "cfd") & pl.col("normalized_size").is_in(list(TARGET_SIZES_BY_MARKET["cfd"])))
        | ((pl.col("market") == "futures") & pl.col("normalized_size").is_in(list(TARGET_SIZES_BY_MARKET["futures"])))
    )

    if target_df.height == 0:
        return target_df

    return (
        target_df.sort(["firm_key", "normalized_size", "discounted_price", "challenge_name"])
        .unique(subset=["firm_key", "normalized_size"], keep="first")
        .sort(["market", "firm_key", "normalized_size"])
    )


def build_firm_price_payload(rows: list[dict[str, Any]], target_sizes: tuple[str, ...]) -> dict[str, Any]:
    by_size: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        normalized_size = row.get("normalized_size")
        if not isinstance(normalized_size, str):
            continue
        by_size.setdefault(normalized_size, []).append(row)

    challenge_fees_by_size_usd: dict[str, dict[str, float]] = {}
    lowest_challenge_fee_usd: dict[str, dict[str, float]] = {}
    lowest_challenge_fee_one_step_usd: dict[str, dict[str, float]] = {}
    lowest_challenge_fee_two_step_usd: dict[str, dict[str, float]] = {}
    observed_prices_usd: dict[str, list[float]] = {}
    max_loss_by_size_usd: dict[str, str] = {}
    activation_fee_by_size_usd: dict[str, float] = {}
    profit_target_by_size: dict[str, str] = {}
    ptdd_ratio_by_size: dict[str, float] = {}
    profit_split_pct_by_size: dict[str, float] = {}

    for normalized_size, size_rows in sorted(by_size.items()):
        candidates = [r for r in size_rows if isinstance(r.get("discounted_price"), int | float)]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: (float(r["discounted_price"]), str(r.get("challenge_name") or "")))

        discounted = round(float(best["discounted_price"]), 2)
        original_value = best.get("original_price")
        original = round(float(original_value), 2) if isinstance(original_value, int | float) else discounted

        pair = {"discounted": discounted, "original": original}
        key = size_key(normalized_size)
        challenge_fees_by_size_usd[key] = pair

        max_loss_label = best.get("max_loss_label")
        if isinstance(max_loss_label, str) and max_loss_label.strip():
            max_loss_by_size_usd[key] = max_loss_label.strip()

        activation_fee = best.get("activation_fee")
        if isinstance(activation_fee, int | float) and float(activation_fee) > 0:
            activation_fee_by_size_usd[key] = round(float(activation_fee), 2)

        profit_target_label = best.get("profit_target_label")
        if isinstance(profit_target_label, str) and profit_target_label.strip():
            profit_target_by_size[key] = profit_target_label.strip()

        ptdd_ratio = best.get("ptdd_ratio")
        if isinstance(ptdd_ratio, int | float):
            ptdd_ratio_by_size[key] = round(float(ptdd_ratio), 4)

        profit_split_pct = best.get("profit_split_pct")
        if isinstance(profit_split_pct, int | float):
            profit_split_pct_by_size[key] = round(float(profit_split_pct), 2)

        observed = sorted({discounted, original})
        observed_prices_usd[key] = observed

        if normalized_size in target_sizes:
            lowest_challenge_fee_usd[key] = pair

    def step_pair_for_size(size_rows: list[dict[str, Any]], step_key: str) -> dict[str, float] | None:
        step_candidates = [
            r
            for r in size_rows
            if r.get("challenge_step") == step_key and isinstance(r.get("discounted_price"), int | float)
        ]
        if not step_candidates:
            return None
        step_best = min(
            step_candidates, key=lambda r: (float(r["discounted_price"]), str(r.get("challenge_name") or ""))
        )
        step_discounted = round(float(step_best["discounted_price"]), 2)
        step_original_value = step_best.get("original_price")
        step_original = (
            round(float(step_original_value), 2) if isinstance(step_original_value, int | float) else step_discounted
        )
        return {"discounted": step_discounted, "original": step_original}

    for normalized_size, size_rows in sorted(by_size.items()):
        key = size_key(normalized_size)
        one_step_pair = step_pair_for_size(size_rows=size_rows, step_key="one_step")
        if one_step_pair is not None:
            lowest_challenge_fee_one_step_usd[key] = one_step_pair
        two_step_pair = step_pair_for_size(size_rows=size_rows, step_key="two_step")
        if two_step_pair is not None:
            lowest_challenge_fee_two_step_usd[key] = two_step_pair

    discount_code = None
    for row in sorted(
        rows, key=lambda r: (float(r.get("discounted_price") or 0.0), str(r.get("challenge_name") or ""))
    ):
        code = row.get("discount_code")
        if isinstance(code, str) and code.strip():
            discount_code = code.strip().split(",")[0].strip()
            break

    discount_percentage = None
    target_rows_for_fields = [
        row for row in rows if isinstance(row.get("normalized_size"), str) and row["normalized_size"] in target_sizes
    ]
    target_rows_sorted = sorted(
        target_rows_for_fields,
        key=lambda r: (float(r.get("discounted_price") or 0.0), str(r.get("challenge_name") or "")),
    )
    discount_rows_sorted = [row for row in target_rows_sorted if isinstance(row.get("discount_pct"), int | float)]
    if discount_rows_sorted:
        discount_percentage = round(float(discount_rows_sorted[0]["discount_pct"]), 2)

    profit_target_pct: dict[str, float] | float | None = None
    profit_split_pct: float | None = None
    ptdd_ratio: float | None = None
    max_loss: str | None = None

    for row in target_rows_sorted:
        phase1_target = row.get("profit_target_phase1")
        phase2_target = row.get("profit_target_phase2")
        if isinstance(phase1_target, int | float):
            if isinstance(phase2_target, int | float):
                profit_target_pct = {"phase1": round(float(phase1_target), 2), "phase2": round(float(phase2_target), 2)}
            else:
                profit_target_pct = round(float(phase1_target), 2)
            break

    for row in target_rows_sorted:
        split = row.get("profit_split_pct")
        if isinstance(split, int | float):
            profit_split_pct = round(float(split), 2)
            break

    for row in target_rows_sorted:
        ratio = row.get("ptdd_ratio")
        if isinstance(ratio, int | float):
            ptdd_ratio = round(float(ratio), 4)
            break

    for row in target_rows_sorted:
        label = row.get("max_loss_label")
        if isinstance(label, str) and label.strip():
            max_loss = label.strip()
            break

    return {
        "lowest_challenge_fee_usd": lowest_challenge_fee_usd,
        "lowest_challenge_fee_one_step_usd": lowest_challenge_fee_one_step_usd,
        "lowest_challenge_fee_two_step_usd": lowest_challenge_fee_two_step_usd,
        "observed_prices_usd": observed_prices_usd,
        "challenge_fees_by_size_usd": challenge_fees_by_size_usd,
        "max_loss_by_size_usd": max_loss_by_size_usd,
        "activation_fee_by_size_usd": activation_fee_by_size_usd,
        "profit_target_by_size": profit_target_by_size,
        "ptdd_ratio_by_size": ptdd_ratio_by_size,
        "profit_split_pct_by_size": profit_split_pct_by_size,
        "max_loss": max_loss,
        "profit_target_pct": profit_target_pct,
        "profit_split_pct": profit_split_pct,
        "ptdd_ratio": ptdd_ratio,
        "discount_code": discount_code,
        "discount_percentage": discount_percentage,
    }


def flatten_price_map(value: Any) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if not isinstance(value, dict):
        return flattened
    for size_key_name, size_value in value.items():
        if not isinstance(size_key_name, str) or not isinstance(size_value, dict):
            continue
        for side in ("discounted", "original"):
            side_value = size_value.get(side)
            if isinstance(side_value, int | float):
                flattened[f"{size_key_name}.{side}"] = round(float(side_value), 2)
    return flattened


def collect_price_changes(
    market: str,
    firm_name: str,
    old_lowest: Any,
    new_lowest: Any,
    old_all_sizes: Any,
    new_all_sizes: Any,
    old_one_step: Any,
    new_one_step: Any,
    old_two_step: Any,
    new_two_step: Any,
) -> list[str]:
    changes: list[str] = []
    old_lowest_map = flatten_price_map(old_lowest)
    new_lowest_map = flatten_price_map(new_lowest)
    old_all_map = flatten_price_map(old_all_sizes)
    new_all_map = flatten_price_map(new_all_sizes)
    old_one_step_map = flatten_price_map(old_one_step)
    new_one_step_map = flatten_price_map(new_one_step)
    old_two_step_map = flatten_price_map(old_two_step)
    new_two_step_map = flatten_price_map(new_two_step)

    for key in sorted(set(old_lowest_map) | set(new_lowest_map)):
        old_value = old_lowest_map.get(key)
        new_value = new_lowest_map.get(key)
        if old_value != new_value:
            changes.append(f"{market}:{firm_name}:lowest:{key} {old_value} -> {new_value}")

    for key in sorted(set(old_all_map) | set(new_all_map)):
        old_value = old_all_map.get(key)
        new_value = new_all_map.get(key)
        if old_value != new_value:
            changes.append(f"{market}:{firm_name}:all_sizes:{key} {old_value} -> {new_value}")

    for key in sorted(set(old_one_step_map) | set(new_one_step_map)):
        old_value = old_one_step_map.get(key)
        new_value = new_one_step_map.get(key)
        if old_value != new_value:
            changes.append(f"{market}:{firm_name}:one_step:{key} {old_value} -> {new_value}")

    for key in sorted(set(old_two_step_map) | set(new_two_step_map)):
        old_value = old_two_step_map.get(key)
        new_value = new_two_step_map.get(key)
        if old_value != new_value:
            changes.append(f"{market}:{firm_name}:two_step:{key} {old_value} -> {new_value}")

    return changes


def update_market_json(json_path: Path, market_rows: list[dict[str, Any]], market: str, now_z: str) -> list[str]:
    changes: list[str] = []
    if not json_path.exists():
        return changes

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    firms = payload.get("firms")
    if not isinstance(firms, list):
        return changes

    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in market_rows:
        firm_key = row.get("firm_key")
        if isinstance(firm_key, str):
            by_key.setdefault(firm_key, []).append(row)

    target_sizes = TARGET_SIZES_BY_MARKET[market]
    for firm in firms:
        if not isinstance(firm, dict):
            continue
        firm_name = str(firm.get("firm") or "")
        firm_name_canon = canonical_name(firm_name)

        matched_key = None
        for key, label in FIRM_LABEL_BY_KEY.items():
            if canonical_name(label) == firm_name_canon:
                matched_key = key
                break
        if not matched_key:
            continue

        rows = by_key.get(matched_key) or []
        if not rows:
            continue

        updates = build_firm_price_payload(rows=rows, target_sizes=target_sizes)
        old_lowest = firm.get("lowest_challenge_fee_usd")
        old_all_sizes = firm.get("challenge_fees_by_size_usd")
        old_one_step = firm.get("lowest_challenge_fee_one_step_usd")
        old_two_step = firm.get("lowest_challenge_fee_two_step_usd")

        firm["lowest_challenge_fee_usd"] = updates["lowest_challenge_fee_usd"]
        firm["lowest_challenge_fee_one_step_usd"] = updates["lowest_challenge_fee_one_step_usd"]
        firm["lowest_challenge_fee_two_step_usd"] = updates["lowest_challenge_fee_two_step_usd"]
        firm["observed_prices_usd"] = updates["observed_prices_usd"]
        firm["challenge_fees_by_size_usd"] = updates["challenge_fees_by_size_usd"]
        firm["max_loss_by_size_usd"] = updates["max_loss_by_size_usd"]
        firm["activation_fee_by_size_usd"] = updates["activation_fee_by_size_usd"]
        firm["profit_target_by_size"] = updates["profit_target_by_size"]
        firm["ptdd_ratio_by_size"] = updates["ptdd_ratio_by_size"]
        firm["profit_split_pct_by_size"] = updates["profit_split_pct_by_size"]
        firm["discount_code"] = updates["discount_code"]

        if updates["max_loss"] is not None:
            firm["max_loss"] = updates["max_loss"]
        if updates["profit_target_pct"] is not None:
            firm["profit_target_pct"] = updates["profit_target_pct"]
        if updates["profit_split_pct"] is not None:
            firm["profit_split_pct"] = updates["profit_split_pct"]
        if updates["ptdd_ratio"] is not None:
            firm["ptdd_ratio"] = updates["ptdd_ratio"]

        changes.extend(
            collect_price_changes(
                market=market,
                firm_name=firm_name,
                old_lowest=old_lowest,
                new_lowest=updates["lowest_challenge_fee_usd"],
                old_all_sizes=old_all_sizes,
                new_all_sizes=updates["challenge_fees_by_size_usd"],
                old_one_step=old_one_step,
                new_one_step=updates["lowest_challenge_fee_one_step_usd"],
                old_two_step=old_two_step,
                new_two_step=updates["lowest_challenge_fee_two_step_usd"],
            )
        )

        if "discount_percentage" in firm and updates["discount_percentage"] is not None:
            firm["discount_percentage"] = updates["discount_percentage"]

        source_url = SOURCE_URL_BY_KEY.get(matched_key)
        if source_url:
            firm["pricing_source_urls"] = [source_url]

    payload["generated_at"] = now_z
    notes = payload.get("notes")
    if isinstance(notes, dict):
        notes["date_updated"] = now_z[:10]

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check current prop firm challenge prices from propfirmmatch API")
    parser.add_argument(
        "--firm-key",
        action="append",
        default=None,
        help="Optional firm key(s) to limit checks (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_firm_keys = set(args.firm_key or [])
    selected_configs = (
        [cfg for cfg in FIRM_CONFIGS if cfg.firm_key in requested_firm_keys]
        if requested_firm_keys
        else list(FIRM_CONFIGS)
    )

    if not selected_configs:
        print("No matching firm keys found in configuration.")
        return

    now_utc = iso_utc_now_z()

    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}) as client:
        all_firms = fetch_all_firms(client)
        firm_ids = resolve_firm_ids(all_firms=all_firms, configs=selected_configs)

        all_price_rows: list[dict[str, Any]] = []
        for config in selected_configs:
            firm_id = firm_ids.get(config.firm_key)
            if not firm_id:
                print(f"[{config.firm_key}] firm ID not found")
                raw_rows = []
            else:
                raw_rows = fetch_firm_challenges(client=client, firm_id=firm_id)

            price_rows = build_price_rows(config=config, raw_rows=raw_rows)

            if not price_rows and config.market == "futures":
                firm_slug = PROPFIRM_COMPARE_SLUGS.get(config.firm_key)
                if firm_slug:
                    fallback_payload = fetch_propfirm_compare_firm(client=client, firm_slug=firm_slug)
                    if fallback_payload:
                        fallback_rows = build_propfirm_compare_fallback_rows(config=config, payload=fallback_payload)
                        if fallback_rows:
                            price_rows = fallback_rows
                            print(f"[{config.firm_key}] fallback rows from propfirm.compare: {len(fallback_rows)}")

            all_price_rows.extend(price_rows)
            print(f"[{config.firm_key}] challenges fetched: {len(price_rows)}")

    if not all_price_rows:
        print("No challenge rows found.")
        return

    all_df = pl.DataFrame(all_price_rows, infer_schema_length=None).with_columns(
        pl.col("market").cast(pl.Utf8),
        pl.col("firm_key").cast(pl.Utf8),
        pl.col("firm_name").cast(pl.Utf8),
        pl.col("challenge_name").cast(pl.Utf8),
        pl.col("challenge_step").cast(pl.Utf8),
        pl.col("account_size").cast(pl.Utf8),
        pl.col("normalized_size").cast(pl.Utf8),
        pl.col("original_price").cast(pl.Float64),
        pl.col("discounted_price").cast(pl.Float64),
        pl.col("discount_pct").cast(pl.Float64),
        pl.col("discount_code").cast(pl.Utf8),
        pl.col("currency").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
        pl.col("activation_fee").cast(pl.Float64),
        pl.col("max_loss_label").cast(pl.Utf8),
        pl.col("max_drawdown_amount").cast(pl.Float64),
        pl.col("max_daily_loss_amount").cast(pl.Float64),
        pl.col("profit_target_phase1").cast(pl.Float64),
        pl.col("profit_target_phase2").cast(pl.Float64),
        pl.col("profit_target_label").cast(pl.Utf8),
        pl.col("profit_split_pct").cast(pl.Float64),
        pl.col("ptdd_ratio").cast(pl.Float64),
    )

    all_df = all_df.unique(
        subset=["market", "firm_key", "challenge_name", "account_size", "discounted_price"],
        keep="first",
    )

    cheapest_df = pick_cheapest_targets(all_df)

    all_csv = OUTPUT_DIR / "propfirmmatch_20_firms_all_challenges.csv"
    all_parquet = OUTPUT_DIR / "propfirmmatch_20_firms_all_challenges.parquet"
    cheapest_csv = OUTPUT_DIR / "propfirmmatch_20_firms_cheapest_targets.csv"
    cheapest_parquet = OUTPUT_DIR / "propfirmmatch_20_firms_cheapest_targets.parquet"
    summary_json = OUTPUT_DIR / "propfirmmatch_20_firms_cheapest_targets.json"

    all_df.write_csv(all_csv)
    all_df.write_parquet(all_parquet)
    cheapest_df.write_csv(cheapest_csv)
    cheapest_df.write_parquet(cheapest_parquet)

    summary_payload = {
        "generated_at": now_utc,
        "firm_count_requested": len(selected_configs),
        "rows_total": all_df.height,
        "rows_cheapest": cheapest_df.height,
        "target_sizes": TARGET_SIZES_BY_MARKET,
        "data": cheapest_df.to_dicts(),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    all_rows = all_df.to_dicts()
    cfd_changes = update_market_json(
        json_path=CFD_JSON_PATH,
        market_rows=[row for row in all_rows if row.get("market") == "cfd"],
        market="cfd",
        now_z=now_utc,
    )
    futures_changes = update_market_json(
        json_path=FUTURES_JSON_PATH,
        market_rows=[row for row in all_rows if row.get("market") == "futures"],
        market="futures",
        now_z=now_utc,
    )

    print(f"Generated at: {now_utc}")
    print(f"Saved all rows CSV: {all_csv}")
    print(f"Saved all rows Parquet: {all_parquet}")
    print(f"Saved cheapest CSV: {cheapest_csv}")
    print(f"Saved cheapest Parquet: {cheapest_parquet}")
    print(f"Saved cheapest JSON: {summary_json}")
    print(f"Updated website JSON: {CFD_JSON_PATH}")
    print(f"Updated website JSON: {FUTURES_JSON_PATH}")
    all_changes = cfd_changes + futures_changes
    if all_changes:
        print("\nPrice changes since previous run:")
        for change in all_changes:
            print(f"- {change}")
    else:
        print("\nPrice changes since previous run: none")
    print("\nCheapest target prices:")
    print(
        cheapest_df.select(
            [
                "market",
                "firm_key",
                "normalized_size",
                "original_price",
                "discounted_price",
                "discount_pct",
                "discount_code",
            ]
        )
    )


if __name__ == "__main__":
    main()
