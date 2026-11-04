"""Streamlit dashboard. Run with `token-eval dashboard` or
`streamlit run token_eval/dashboard.py`."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from token_eval.budget import Budget, Level
from token_eval.storage import DEFAULT_DB, TokenStore, utcnow

st.set_page_config(page_title="token-eval", layout="wide")
st.title("token-eval dashboard")

with st.sidebar:
    db_path = st.text_input("Database", os.environ.get("TOKEN_EVAL_DB", DEFAULT_DB))
    budget_path = st.text_input("Budget config", os.environ.get("TOKEN_EVAL_BUDGET", "budget.json"))
    days = st.slider("Days of history", 1, 90, 14)
    task_id = st.text_input("Task id (optional)") or None
    session_id = st.text_input("Session id (optional)") or None

if not Path(db_path).exists():
    st.info(f"No database at {db_path} yet. Run `python examples/demo.py` to create one.")
    st.stop()

store = TokenStore(db_path)
since = utcnow() - timedelta(days=days)
calls = pd.DataFrame(store.calls(since=since))
savings = pd.DataFrame(store.savings(since=since))

if calls.empty:
    st.info("No calls recorded in this window.")
else:
    calls["ts"] = pd.to_datetime(calls["ts"], format="ISO8601")
    calls["day"] = calls["ts"].dt.date
    calls["tokens"] = (calls["prompt_tokens"] + calls["completion_tokens"]
                       + calls["cache_write_tokens"] + calls["cache_read_tokens"])
    calls["tag"] = calls["tag"].fillna("(none)")

saved_usd = float(savings["usd_saved"].sum()) if not savings.empty else 0.0
saved_tok = int((savings["tokens_before"] - savings["tokens_after"]).sum()) if not savings.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calls", 0 if calls.empty else len(calls))
c2.metric("Tokens", f"{0 if calls.empty else int(calls['tokens'].sum()):,}")
c3.metric("Cost", f"${0 if calls.empty else calls['cost_usd'].sum():.4f}")
c4.metric("Saved by optimize()", f"{saved_tok:,} tok", f"${saved_usd:.4f} per call")

st.subheader("Budget")
if Path(budget_path).exists():
    status = Budget.from_file(budget_path, store=store, task_id=task_id,
                              session_id=session_id).status()
    badge = {Level.CONTINUE: "🟢", Level.WARN: "🟡", Level.STOP: "🔴"}[status.level]
    st.write(f"{badge} **{status.level.value}** — {status.recommendation}")
    for name, u in status.scopes.items():
        if (name == "task" and not task_id) or (name == "session" and not session_id):
            continue
        if not (u.limit.tokens or u.limit.usd):
            continue
        st.progress(min(u.fraction, 1.0),
                    text=f"{name}: {u.used_tokens:,} tokens / ${u.used_usd:.4f} "
                         f"({u.fraction:.0%} of limit)")
else:
    st.caption(f"No budget config at {budget_path}.")

if not calls.empty:
    left, right = st.columns(2)
    with left:
        st.subheader("Daily usage")
        daily = calls.groupby("day")[["tokens", "cost_usd"]].sum()
        metric = st.radio("Metric", ["tokens", "cost_usd"], horizontal=True)
        st.bar_chart(daily[metric])
    with right:
        st.subheader("Cost by tag")
        st.bar_chart(calls.groupby("tag")["cost_usd"].sum().sort_values(ascending=False))

    st.subheader("Top expensive prompts")
    top = calls.sort_values("cost_usd", ascending=False).head(15)
    st.dataframe(top[["ts", "model", "tag", "prompt_tokens", "completion_tokens", "cost_usd",
                      "prompt_preview"]], hide_index=True, width="stretch")

st.subheader("Savings achieved")
if savings.empty:
    st.caption("No optimize() runs recorded yet (`token-eval optimize prompt.json`).")
else:
    savings["saved_tokens"] = savings["tokens_before"] - savings["tokens_after"]
    st.dataframe(savings[["ts", "model", "tag", "tokens_before", "tokens_after", "saved_tokens",
                          "usd_saved"]], hide_index=True, width="stretch")
