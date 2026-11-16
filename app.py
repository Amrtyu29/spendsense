"""SpendSense dashboard.

Run:  python -m streamlit run app.py     (or double-click run_app.bat on Windows)

Load the demo month or your own statement, move the day slider to watch the
agent work through the month, and answer its questions: it learns from you.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from expense_agent.app_engine import Feedback, default_budgets, demo_month, load_statement, replay, statement_months
from expense_agent.data_gen import CATEGORIES

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "statement_unlabelled.csv"
DEMO = ROOT / "data" / "demo"
COLORS = {"WARN": "#F5A524", "INFO": "#2F80FF", "REALLOCATE": "#8B6CFF", "VERIFY": "#F5A524",
          "BLOCK": "#FF5C7A", "ASK_CATEGORY": "#19C6B5"}
LABELS = {"WARN": "Warning", "INFO": "Heads-up", "REALLOCATE": "Re-plan budget", "VERIFY": "Please confirm",
          "BLOCK": "Blocked", "ASK_CATEGORY": "Question"}

st.set_page_config(page_title="SpendSense", layout="wide")
st.markdown("""
<style>
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; margin-right:8px; }
.dayhead { color:#A9B8D3; font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; margin:18px 0 4px; }
.muted { color:#A9B8D3; font-size:0.86rem; }
</style>""", unsafe_allow_html=True)

if "fb" not in st.session_state:
    st.session_state.fb = Feedback()
fb: Feedback = st.session_state.fb


def badge(kind: str) -> str:
    c = COLORS.get(kind, "#A9B8D3")
    return f'<span class="badge" style="color:{c}; border:1px solid {c}; background:{c}22">{LABELS.get(kind, kind)}</span>'


def rs(x: float) -> str:
    return f"Rs {x:,.0f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("SpendSense")
    st.caption("An intelligent AI financial assistant")
    source_label = st.radio("Data", ["Demo month", "My statement (CSV)"],
                            help="The demo month is simulated. A statement is any bank or UPI export with the columns date, description, amount (and optionally time).")
    source = "demo" if source_label == "Demo month" else "statement"
    statement, ym, simulate = None, None, False
    if source == "demo":
        simulate = st.toggle("Simulated user reacts to alerts", value=True,
                             help="On: a simulated person responds to warnings, so you can compare with and without the agent. Off: you are the user and answer the agent yourself.")
        year, month = demo_month()
    else:
        bundled = {"Demo: an ordinary month": DEMO / "1_normal_month.csv",
                   "Demo: an overspending month": DEMO / "2_overspending_month.csv",
                   "Demo: suspicious payments": DEMO / "3_suspicious_payments.csv",
                   "Sample: twelve months": SAMPLE}
        bundled = {k: v for k, v in bundled.items() if v.exists()}
        choice = st.selectbox("Statement", list(bundled) + ["Upload my own file"],
                              help="The demo files are in data/demo/. Each shows a different situation. Use my_statement_template.csv there as a starting point for your own.")
        up = st.file_uploader("Statement CSV", type=["csv"]) if choice == "Upload my own file" else None
        try:
            if up is not None:
                statement = load_statement(up)
            elif choice in bundled:
                statement = load_statement(bundled[choice])
        except Exception as e:  # show a readable message instead of a stack trace
            st.error(str(e))
        if statement is not None:
            months = statement_months(statement)
            ym = st.selectbox("Month", months, index=len(months) - 1, format_func=lambda t: f"{pd.Timestamp(t[0], t[1], 1):%B %Y}")
            year, month = ym
    if source == "statement" and statement is None:
        st.info("Choose a CSV file to upload, or pick one of the demo statements above.")
        st.stop()

    days = pd.Timestamp(year, month, 1).days_in_month
    day = st.slider("Day of the month", 1, days, days, help="Scrub through the month to watch the agent decide day by day.")
    tolerance = st.select_slider("How much do alerts bother you?", options=[30, 60, 120, 200, 320], value=120,
                                 format_func=lambda v: {30: "Not at all", 60: "A little", 120: "Normal", 200: "A lot", 320: "Very much"}[v],
                                 help="This is the fatigue cost in the utility formula. Higher means fewer, more certain alerts.")
    with st.expander("Monthly budgets"):
        st.caption("Proposed at 110% of your historical average. Edit any of them.")
        base_budgets = default_budgets(7 if source == "demo" else 0)
        edited = st.data_editor(pd.DataFrame({"category": list(base_budgets), "budget": list(base_budgets.values())}),
                                hide_index=True, disabled=["category"], key="budget_editor",
                                column_config={"budget": st.column_config.NumberColumn(min_value=0, step=50, format="Rs %d")})
        budgets = dict(zip(edited["category"], edited["budget"]))
    if fb.count():
        st.caption(f"You have given the agent {fb.count()} piece(s) of feedback.")
        if st.button("Forget my feedback"):
            st.session_state.fb = Feedback()
            st.rerun()

# ------------------------------------------------------------------ run the agent
res = replay(source=source, upto_day=day, feedback=fb, simulate_user=simulate, fatigue_cost=float(tolerance),
             budgets=budgets, statement=statement, ym=ym)
cats: pd.DataFrame = res["categories"]
tx: pd.DataFrame = res["transactions"]
feed = res["feed"]
manual = not simulate   # the person answers the agent themselves

st.header(f"{res['label']} · day {res['day']} of {res['days']}")
st.caption("Simulated demo month. The agent was trained on six earlier labelled months." if source == "demo"
           else "Your statement. Categories for merchants the agent has never seen may be uncertain: correct them in the Transactions tab and it learns.")

k = st.columns(5)
total_budget, total_spent = cats["budget"].sum(), cats["spent"].sum()
k[0].metric("Spent so far", rs(total_spent), f"{total_spent / total_budget:.0%} of {rs(total_budget)}", delta_color="off")
k[1].metric("Projected month-end", rs(cats["projected"].sum()), f"{cats['projected'].sum() - total_budget:+,.0f} vs budget", delta_color="inverse")
k[2].metric("Categories at risk", int((cats["risk of overspending"] >= 0.5).sum()), "50% or more chance of overspending", delta_color="off")
k[3].metric("Budget alerts sent", sum(f["kind"] in ("WARN", "INFO", "REALLOCATE") for f in feed))
held_now = int((tx["status"] == "held for review").sum()) if len(tx) else 0
k[4].metric("Payments held", held_now, "waiting for your answer" if held_now else "none waiting", delta_color="off")

t_budget, t_feed, t_pay, t_tx, t_learn = st.tabs(["Budgets", "Agent decisions", "Payments to review", "Transactions", "What the agent learned"])

# ------------------------------------------------------------------ budgets
with t_budget:
    show = cats[["category", "budget", "adjusted budget", "spent", "used", "projected", "plus/minus", "risk of overspending"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "budget": st.column_config.NumberColumn(format="Rs %d"),
        "adjusted budget": st.column_config.NumberColumn(format="Rs %d", help="After any re-plan the agent proposed this month"),
        "spent": st.column_config.NumberColumn(format="Rs %d"),
        "used": st.column_config.ProgressColumn("budget used", min_value=0, max_value=1.5, format="percent"),
        "projected": st.column_config.NumberColumn("projected month-end", format="Rs %d"),
        "plus/minus": st.column_config.NumberColumn("± uncertainty", format="Rs %d"),
        "risk of overspending": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
    })
    c1, c2 = st.columns([1, 3])
    risky = cats.sort_values("risk of overspending", ascending=False)["category"].tolist()
    cat = c1.selectbox("Category to plot", risky)
    row = cats[cats["category"] == cat].iloc[0]
    c1.metric("Budget", rs(row["budget"]))
    c1.metric("Projected", rs(row["projected"]), f"± {row['plus/minus']:,.0f}", delta_color="off")
    c1.metric("Chance of overspending", f"{row['risk of overspending']:.0%}")
    line = pd.DataFrame({"day": range(res["day"] + 1), "Rs": res["series"][cat], "series": "with the agent"})
    if res["counterfactual"] is not None and simulate:
        cf = res["counterfactual"][cat]
        line = pd.concat([line, pd.DataFrame({"day": range(len(cf)), "Rs": cf, "series": "without the agent"})])
    chart = alt.Chart(line).mark_line(strokeWidth=2.5).encode(
        x=alt.X("day:Q", title="day of the month", scale=alt.Scale(domain=[0, res["days"]])),
        y=alt.Y("Rs:Q", title="cumulative spend (Rs)"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["with the agent", "without the agent"], range=["#19C6B5", "#A9B8D3"])))
    rule = alt.Chart(pd.DataFrame({"y": [row["budget"]]})).mark_rule(color="#FF5C7A", strokeDash=[4, 4]).encode(y="y:Q")
    marks = pd.DataFrame([{"day": f["day"], "kind": LABELS[f["kind"]]} for f in feed if f["category"] == cat and f["kind"] in ("WARN", "INFO", "REALLOCATE")])
    layers = [chart, rule]
    if len(marks):
        layers.append(alt.Chart(marks).mark_rule(color="#F5A524", opacity=0.7).encode(x="day:Q", tooltip=["day", "kind"]))
    c2.altair_chart(alt.layer(*layers).properties(height=330), width="stretch")
    c2.caption("Dashed red line: budget. Orange lines: days the agent alerted you about this category.")

# ------------------------------------------------------------------ decisions feed
with t_feed:
    shown = [f for f in feed if f["kind"] != "ASK_CATEGORY"]
    if not shown:
        st.info("No alerts yet. The agent stays quiet until an action is worth the interruption (and never before day 5 for budgets).")
    last_day = None
    for i, f in enumerate(shown):
        if f["day"] != last_day:
            st.markdown(f'<div class="dayhead">Day {f["day"]}</div>', unsafe_allow_html=True)
            last_day = f["day"]
        st.markdown(badge(f["kind"]) + f["message"], unsafe_allow_html=True)
        with st.expander("Why the agent chose this"):
            if f["utilities"]:
                u = pd.DataFrame([{"action": a, "expected utility": v} for a, v in f["utilities"].items() if v is not None])
                st.dataframe(u.sort_values("expected utility", ascending=False), hide_index=True)
                st.caption("The action with the highest expected utility wins. For budgets: money a warning is expected to save, minus the annoyance cost."
                           + (f" The agent currently believes you act on {f['p_act']:.0%} of warnings." if f["p_act"] else ""))
            elif f["score"] is not None:
                st.caption(f"Anomaly score {f['score']:.0%}.")
        if manual and f["kind"] in ("WARN", "INFO", "REALLOCATE"):
            key = (f["day"], f["category"])
            given = fb.alerts.get(key)
            if given:
                st.caption({"helpful": "You marked this helpful.", "dismiss": "You dismissed this.", "reject_realloc": "You rejected this re-plan."}[given])
            else:
                b = st.columns([1, 1, 6])
                if f["kind"] == "REALLOCATE":
                    if b[0].button("Reject", key=f"rej{i}"):
                        fb.alerts[key] = "reject_realloc"; st.rerun()
                else:
                    if b[0].button("Helpful", key=f"help{i}"):
                        fb.alerts[key] = "helpful"; st.rerun()
                    if b[1].button("Dismiss", key=f"dis{i}"):
                        fb.alerts[key] = "dismiss"; st.rerun()

# ------------------------------------------------------------------ payments to review
with t_pay:
    flagged = [f for f in feed if f["kind"] in ("VERIFY", "BLOCK")]
    if not flagged:
        st.info("No payment has looked suspicious so far.")
    for i, f in enumerate(flagged):
        r = tx[tx["txn_id"] == f["txn_id"]].iloc[0]
        with st.container(border=True):
            a, b = st.columns([4, 1])
            a.markdown(badge(f["kind"]) + f"**{r['description']}**", unsafe_allow_html=True)
            a.markdown(f"<span class='muted'>{r['when']:%d %b %H:%M} · guessed {r['category']} · risk {r['risk']:.0%}</span>", unsafe_allow_html=True)
            a.write(f"Why: {r['evidence']}.")
            b.metric("Amount", rs(r["amount"]))
            if r["status"] == "held for review" and manual:
                c = st.columns([2, 2.6, 5])
                if c[0].button("It was me", key=f"ok{i}"):
                    fb.payments[f["txn_id"]] = "legit"; st.rerun()
                if c[1].button("Not me (fraud)", key=f"bad{i}"):
                    fb.payments[f["txn_id"]] = "confirm_fraud"; st.rerun()
                st.caption("Held out of your budget totals until you answer.")
            else:
                st.caption({"counted": "Confirmed genuine: counted in the budget, and the merchant is now trusted.",
                            "fraud (excluded)": "Confirmed fraud: excluded from the budget.",
                            "held for review": "Held: waiting for an answer."}[r["status"]])

# ------------------------------------------------------------------ transactions
with t_tx:
    if tx.empty:
        st.info("No transactions up to this day.")
    else:
        asked = {f["txn_id"] for f in feed if f["kind"] == "ASK_CATEGORY"}
        view = tx.copy()
        view["agent is unsure"] = view["txn_id"].isin(asked)
        cols = ["when", "description", "amount", "category", "confidence", "risk", "status", "agent is unsure"]
        if "true category" in view and simulate:
            cols.insert(4, "true category")
        config = {"when": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                  "amount": st.column_config.NumberColumn(format="Rs %d"),
                  "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                  "risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                  "category": st.column_config.SelectboxColumn(options=CATEGORIES, required=True)}
        if manual:
            st.caption("Change a category in the table to correct the agent. It updates its model immediately and uses your label from then on.")
            ed = st.data_editor(view[["txn_id"] + cols], hide_index=True, width="stretch", column_config={**config, "txn_id": None},
                                disabled=[c for c in cols if c != "category"], key=f"tx_{res['day']}_{len(fb.categories)}_{source}_{res['label']}_{len(tx)}")
            changed = ed[ed["category"].values != view["category"].values]
            if len(changed):
                for _, r in changed.iterrows():
                    fb.categories[int(r["txn_id"])] = r["category"]
                st.rerun()
        else:
            st.caption("Read-only while the simulated user is on. Turn the toggle off in the sidebar to correct categories yourself.")
            st.dataframe(view[cols], hide_index=True, width="stretch", column_config=config)
        acc = None
        if "true category" in tx:
            acc = (tx["category"] == tx["true category"]).mean()
        st.caption(f"{len(tx)} transactions, {int(view['agent is unsure'].sum())} where the agent asked for help"
                   + (f", {acc:.0%} categorised correctly so far." if acc is not None else "."))

# ------------------------------------------------------------------ learning
with t_learn:
    L = res["learned"]
    m = st.columns(4)
    m[0].metric("Believed chance you act on a warning", f"{L['p_act']:.0%}", "starts at 60%", delta_color="off")
    m[1].metric("Alert fatigue scale", f"{L['fatigue_scale']:.2f}", "1.00 at the start", delta_color="off")
    m[2].metric("Categoriser updates", f"{L['categoriser_updates']:,}", f"{res['history_size']} from history", delta_color="off")
    m[3].metric("Words and features known", f"{L['vocabulary']:,}")
    st.markdown("""
**How the agent learns from you**

- **Responsiveness.** Seven days after a warning it compares your spending with a reference rate. If you cut back, its belief that you act on warnings goes up, and warnings become more worthwhile.
- **Tolerance.** Each *Dismiss* raises the cost it assigns to interrupting you; each *Helpful* lowers it.
- **Categories.** Every correction is an immediate update of its Naive Bayes counts, weighted three times a normal example.
- **Merchants.** When you confirm a held payment was yours, that merchant stops looking new.
""")
    if res["user_stats"]:
        s = res["user_stats"]
        st.caption(f"Simulated user this month: acted on {s['responded']} alert(s), ignored {s['ignored']}, dismissed {s['dismissed']}, "
                   f"confirmed {s['fraud_confirmed']} fraud(s), cleared {s['false_flags']} false flag(s), answered {s['category_answers']} category question(s).")
