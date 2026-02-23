"""Kraken Futures order execution for BTC signal trading.

Uses Kraken Futures perpetual BTC/USD inverse (symbol: pi_xbtusd).
Supports all three model signals: LONG (+1), SHORT (-1), FLAT (0).

Paper → Live is a single environment variable change — no code changes:
    KRAKEN_FUTURES_ENV=demo  (default)  →  demo-futures.kraken.com  (paper money)
    KRAKEN_FUTURES_ENV=live             →  futures.kraken.com        (real money)

Get demo API keys:  https://demo-futures.kraken.com  → Settings → API Keys
Get live API keys:  https://futures.kraken.com       → Settings → API Keys

Credentials (same format for demo and live):
    Local:  export KRAKEN_API_KEY=... KRAKEN_API_SECRET=...
    Lambda: store in AWS Secrets Manager as {"api_key": "...", "api_secret": "..."}
            set KRAKEN_SECRET_NAME env var

Usage:
    uv run python src/btc_trade.py --signal 1 --dry-run    # show what would happen
    uv run python src/btc_trade.py --signal 1              # execute on demo (default)
    KRAKEN_FUTURES_ENV=live uv run python src/btc_trade.py --signal 1  # live
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEMO_URL = "https://demo-futures.kraken.com"
_LIVE_URL = "https://futures.kraken.com"
_API_PREFIX = "/derivatives/api/v3"

SYMBOL = "pi_xbtusd"   # BTC/USD perpetual inverse — 1 contract = $1 USD notional

TRADE_LOG = Path(__file__).resolve().parent.parent / "outputs" / "btc_trades.jsonl"
LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def _base_url() -> str:
    env = os.environ.get("KRAKEN_FUTURES_ENV", "demo").lower()
    return _LIVE_URL if env == "live" else _DEMO_URL


def _is_demo() -> bool:
    return os.environ.get("KRAKEN_FUTURES_ENV", "demo").lower() != "live"


# ---------------------------------------------------------------------------
# Kraken Futures authentication
# ---------------------------------------------------------------------------
def _sign(endpoint_path: str, nonce: str, api_secret: str, post_data: str = "") -> str:
    """Kraken Futures HMAC-SHA512 signature.

    message  = postData + nonce + endpoint_path
    digest   = SHA-256(message)
    authent  = Base64(HMAC-SHA512(Base64Decode(secret), digest))
    """
    message = post_data + nonce + endpoint_path
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    mac = hmac.new(base64.b64decode(api_secret), digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("utf-8")


def _get(path: str, api_key: str, api_secret: str) -> dict:
    """Authenticated GET to Kraken Futures API."""
    nonce = str(int(time.time() * 1000))
    endpoint = f"{_API_PREFIX}{path}"
    headers = {
        "APIKey": api_key,
        "Authent": _sign(endpoint, nonce, api_secret),
        "Nonce": nonce,
    }
    resp = requests.get(f"{_base_url()}{endpoint}", headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("result") != "success":
        raise RuntimeError(f"Kraken Futures error: {result}")
    return result


def _post(path: str, data: dict, api_key: str, api_secret: str) -> dict:
    """Authenticated POST to Kraken Futures API."""
    nonce = str(int(time.time() * 1000))
    endpoint = f"{_API_PREFIX}{path}"
    post_data = urllib.parse.urlencode(data)
    headers = {
        "APIKey": api_key,
        "Authent": _sign(endpoint, nonce, api_secret, post_data),
        "Nonce": nonce,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    resp = requests.post(
        f"{_base_url()}{endpoint}",
        headers=headers,
        data=post_data,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("result") != "success":
        raise RuntimeError(f"Kraken Futures error: {result}")
    return result


# ---------------------------------------------------------------------------
# Account / market queries
# ---------------------------------------------------------------------------
def get_position(api_key: str, api_secret: str) -> dict:
    """Get current open position for pi_xbtusd.

    Returns:
        {"side": "long"/"short"/"flat", "size": <contracts>, "entry_price": <float>}
    """
    result = _get("/openpositions", api_key, api_secret)
    for pos in result.get("openPositions", []):
        if pos.get("symbol") == SYMBOL:
            return {
                "side": pos["side"].lower(),          # "long" or "short"
                "size": abs(float(pos["size"])),
                "entry_price": float(pos.get("price", 0)),
            }
    return {"side": "flat", "size": 0.0, "entry_price": 0.0}


def get_btc_price() -> float:
    """BTC/USD mark price from public Futures ticker (no auth needed)."""
    resp = requests.get(f"{_base_url()}{_API_PREFIX}/tickers", timeout=10)
    resp.raise_for_status()
    for ticker in resp.json().get("tickers", []):
        if ticker.get("symbol") == SYMBOL:
            return float(ticker.get("markPrice") or ticker.get("last", 0))
    raise RuntimeError(f"Ticker not found for {SYMBOL}")


def get_available_margin(api_key: str, api_secret: str) -> float:
    """Available BTC margin in the flex account."""
    result = _get("/accounts", api_key, api_secret)
    xbt = (result.get("accounts", {})
                 .get("flex", {})
                 .get("currencies", {})
                 .get("xbt", {}))
    return float(xbt.get("available", 0.0))


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------
def execute_trade(
    signal: int,
    api_key: str,
    api_secret: str,
    position_size_usd: float = 1000.0,
    dry_run: bool = False,
) -> dict:
    """Execute trade on Kraken Futures based on model signal.

    Args:
        signal:            +1 LONG, 0 FLAT, -1 SHORT
        api_key:           Kraken Futures API key (demo or live)
        api_secret:        Kraken Futures API secret
        position_size_usd: Target position size in USD contracts.
                           For pi_xbtusd: 1 contract = $1 notional.
        dry_run:           Print intent but don't submit orders.

    Returns:
        Dict with action, orders placed, timestamps, environment.
    """
    env_label = "DEMO" if _is_demo() else "LIVE"
    current = get_position(api_key, api_secret)
    btc_price = get_btc_price()
    current_side = current["side"]   # "long", "short", or "flat"
    target_side = {1: "long", 0: "flat", -1: "short"}[signal]

    print(f"\n  Environment: {env_label}")
    print(f"  BTC price:   ${btc_price:,.2f}")
    print(f"  Position:    {current_side.upper()}"
          + (f"  {current['size']:.0f} contracts" if current_side != "flat" else ""))
    print(f"  Signal:      {LABEL[signal]} ({signal:+d})  →  target: {target_side.upper()}")

    # No change needed
    if current_side == target_side:
        print(f"\n  → HOLD — already {current_side.upper()}, no order needed.")
        return {
            "action": "hold",
            "signal": signal,
            "signal_label": LABEL[signal],
            "current_position": current_side,
            "btc_price": btc_price,
            "environment": env_label,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Build the order list:
    #   1. Close existing position (if any)
    #   2. Open new position (if target is not flat)
    # Kraken Futures market orders reduce/flip positions automatically,
    # but being explicit about close + open avoids size ambiguity.
    orders_to_place = []

    if current_side == "long":
        orders_to_place.append(("sell", current["size"]))   # close long
    elif current_side == "short":
        orders_to_place.append(("buy", current["size"]))    # close short

    if target_side == "long":
        orders_to_place.append(("buy", position_size_usd))
    elif target_side == "short":
        orders_to_place.append(("sell", position_size_usd))

    transition = f"{current_side} → {target_side}"
    print(f"\n  → {transition.upper()}")
    for side, size in orders_to_place:
        print(f"     {side.upper():4s}  {size:>6.0f} contracts  (~${size:,.0f})")

    if dry_run:
        print(f"\n  DRY RUN — no orders submitted.")
        return {
            "action": f"{transition} [dry_run]",
            "signal": signal,
            "signal_label": LABEL[signal],
            "orders": [{"side": s, "size": sz} for s, sz in orders_to_place],
            "btc_price": btc_price,
            "dry_run": True,
            "environment": env_label,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Submit orders
    placed = []
    for side, size in orders_to_place:
        result = _post(
            "/sendorder",
            {"orderType": "mkt", "symbol": SYMBOL, "side": side, "size": int(size)},
            api_key,
            api_secret,
        )
        order_id = result.get("sendStatus", {}).get("order_id", "unknown")
        placed.append({"side": side, "size": int(size), "order_id": order_id})
        print(f"  Placed: {side.upper()} {size:.0f} → order_id={order_id}")

    return {
        "action": transition,
        "signal": signal,
        "signal_label": LABEL[signal],
        "orders": placed,
        "position_before": current_side,
        "position_after": target_side,
        "btc_price": btc_price,
        "position_size_usd": position_size_usd,
        "environment": env_label,
        "executed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Credential loading (same for demo and live)
# ---------------------------------------------------------------------------
def load_credentials(secret_name: str = None) -> tuple[str, str]:
    """Load Kraken API credentials.

    Priority:
        1. KRAKEN_API_KEY + KRAKEN_API_SECRET environment variables
        2. AWS Secrets Manager (for Lambda) — secret_name or KRAKEN_SECRET_NAME env var
           Secret JSON format: {"api_key": "...", "api_secret": "..."}
    """
    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    if api_key and api_secret:
        return api_key, api_secret

    secret_name = secret_name or os.environ.get("KRAKEN_SECRET_NAME")
    if not secret_name:
        raise RuntimeError(
            "No Kraken credentials found.\n"
            "Set KRAKEN_API_KEY + KRAKEN_API_SECRET, "
            "or set KRAKEN_SECRET_NAME for AWS Secrets Manager."
        )

    import boto3
    sm = boto3.client("secretsmanager")
    resp = sm.get_secret_value(SecretId=secret_name)
    secret = json.loads(resp["SecretString"])
    return secret["api_key"], secret["api_secret"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    signal = None
    size = float(os.environ.get("POSITION_SIZE_USD", 1000))

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--signal" and i + 1 < len(args):
            signal = int(args[i + 1])
        elif arg == "--size" and i + 1 < len(args):
            size = float(args[i + 1])

    if signal is None:
        print("Usage: uv run python src/btc_trade.py --signal <1|0|-1> [--dry-run] [--size <usd>]")
        sys.exit(1)

    env = os.environ.get("KRAKEN_FUTURES_ENV", "demo").upper()
    print(f"\nBTC Trade Executor  [{env}]")
    print(f"{'='*50}")
    print(f"Signal:   {LABEL[signal]} ({signal:+d})")
    print(f"Size:     ${size:,.0f} USD contracts")
    if dry_run:
        print(f"Mode:     DRY RUN (queries Kraken, no orders placed)")
    print(f"{'='*50}")

    api_key, api_secret = load_credentials()
    result = execute_trade(signal, api_key, api_secret, position_size_usd=size, dry_run=dry_run)

    print(f"\nResult:\n{json.dumps(result, indent=2)}")

    if not dry_run and result.get("action") not in ("hold",):
        TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nLogged → {TRADE_LOG}")


if __name__ == "__main__":
    main()
