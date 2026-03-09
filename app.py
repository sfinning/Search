from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from propfirmmatch_price_checker import (
    FIRM_CONFIGS,
    PROPFIRM_COMPARE_SLUGS,
    TIMEOUT as PROPFIRM_TIMEOUT,
    build_price_rows,
    build_propfirm_compare_fallback_rows,
    fetch_all_firms,
    fetch_firm_challenges,
    fetch_propfirm_compare_firm,
    resolve_firm_ids,
)

PROPFIRM_TABLE_CACHE_TTL_SECONDS = int(os.getenv("PROPFIRM_TABLE_CACHE_TTL_SECONDS", "300"))
PROPFIRM_TABLE_COLUMNS = (
    "market",
    "firm_key",
    "firm_name",
    "challenge_name",
    "challenge_step",
    "account_size",
    "original_price",
    "discounted_price",
    "discount_pct",
    "discount_code",
    "activation_fee",
    "max_loss_label",
    "profit_target_label",
    "profit_split_pct",
    "ptdd_ratio",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "web"

_propfirm_table_cache: dict[str, Any] = {}
_propfirm_table_cache_lock = asyncio.Lock()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _propfirm_table_rows_sync() -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=PROPFIRM_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}) as client:
        all_firms = fetch_all_firms(client)
        firm_ids = resolve_firm_ids(all_firms=all_firms, configs=FIRM_CONFIGS)

        for config in FIRM_CONFIGS:
            firm_id = firm_ids.get(config.firm_key)
            raw_rows = fetch_firm_challenges(client=client, firm_id=firm_id) if firm_id else []
            price_rows = build_price_rows(config=config, raw_rows=raw_rows)

            if not price_rows and config.market == "futures":
                firm_slug = PROPFIRM_COMPARE_SLUGS.get(config.firm_key)
                if firm_slug:
                    fallback_payload = fetch_propfirm_compare_firm(client=client, firm_slug=firm_slug)
                    if fallback_payload:
                        fallback_rows = build_propfirm_compare_fallback_rows(config=config, payload=fallback_payload)
                        if fallback_rows:
                            price_rows = fallback_rows

            all_rows.extend(price_rows)

    deduped_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for row in all_rows:
        dedupe_key = (
            row.get("market"),
            row.get("firm_key"),
            row.get("challenge_name"),
            row.get("account_size"),
            row.get("discounted_price"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        deduped_rows.append(
            {
                "market": row.get("market"),
                "firm_key": row.get("firm_key"),
                "firm_name": row.get("firm_name"),
                "challenge_name": row.get("challenge_name"),
                "challenge_step": row.get("challenge_step"),
                "account_size": row.get("account_size"),
                "original_price": _float_or_none(row.get("original_price")),
                "discounted_price": _float_or_none(row.get("discounted_price")),
                "discount_pct": _float_or_none(row.get("discount_pct")),
                "discount_code": row.get("discount_code"),
                "activation_fee": _float_or_none(row.get("activation_fee")),
                "max_loss_label": row.get("max_loss_label"),
                "profit_target_label": row.get("profit_target_label"),
                "profit_split_pct": _float_or_none(row.get("profit_split_pct")),
                "ptdd_ratio": _float_or_none(row.get("ptdd_ratio")),
            }
        )

    deduped_rows.sort(
        key=lambda row: (
            str(row.get("market") or ""),
            str(row.get("firm_name") or ""),
            str(row.get("account_size") or ""),
            _float_or_none(row.get("discounted_price")) or float("inf"),
            str(row.get("challenge_name") or ""),
        )
    )
    return deduped_rows


@app.get("/api/propfirmmatch/challenges")
async def propfirmmatch_challenges(refresh: bool = False):
    now = datetime.now(timezone.utc)

    async with _propfirm_table_cache_lock:
        cached_payload = _propfirm_table_cache.get("payload")
        cached_at = _propfirm_table_cache.get("fetched_at")
        if (
            not refresh
            and isinstance(cached_at, datetime)
            and isinstance(cached_payload, dict)
            and (now - cached_at).total_seconds() < PROPFIRM_TABLE_CACHE_TTL_SECONDS
        ):
            return cached_payload

    try:
        rows = await asyncio.to_thread(_propfirm_table_rows_sync)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Failed to fetch PropFirmMatch data: {e}"},
            status_code=502,
        )

    payload = {
        "generated_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "row_count": len(rows),
        "columns": PROPFIRM_TABLE_COLUMNS,
        "rows": rows,
    }

    async with _propfirm_table_cache_lock:
        _propfirm_table_cache["fetched_at"] = now
        _propfirm_table_cache["payload"] = payload

    return payload


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "propfirmsearch",
        "time": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


@app.get("/")
async def root():
    return RedirectResponse(url="/propfirmsearch.html", status_code=307)


@app.get("/propfirmsearch.html")
async def propfirmsearch_page():
    page = static_dir / "propfirmsearch.html"
    if not page.exists():
        return JSONResponse(
            {"status": "error", "message": "Missing web/propfirmsearch.html in deployment bundle"},
            status_code=500,
        )
    return FileResponse(page)


if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8081")),
        reload=os.getenv("RELOAD", "1") == "1",
    )



