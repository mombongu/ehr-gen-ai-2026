"""
Mock EHR "Pre-Op Readiness" board: patient chart and OR schedule.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...        (or OPENAI_API_KEY + PREOP_LLM_PROVIDER=openai + PREOP_LLM_MODEL=...)
    streamlit run app.py

Without an API key the app still runs using the rules-only engine.
Fully synthetic patients. Teaching rules, not clinical guidelines.
"""
from __future__ import annotations

import calendar
import csv
import os
from datetime import datetime
from html import escape as esc

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from preop import criteria as C
from preop import data as D
from preop import schedule as S
from preop.llm import LLM, default_provider
from preop.pipeline import action_text, run_case

st.set_page_config(page_title="Pre-Op Readiness", page_icon="🩺", layout="wide")

# Streamlit Community Cloud exposes configured credentials through st.secrets,
# while the LLM client reads environment variables.
try:
    app_secrets = st.secrets
except StreamlitSecretNotFoundError:
    app_secrets = {}
for setting in ("PREOP_LLM_PROVIDER", "PREOP_LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(setting) and setting in app_secrets:
        os.environ[setting] = str(app_secrets[setting])

ROLE_LABEL = {"nurse": "Pre-op nurse", "anesthesia": "Anesthesiologist"}
STATUS_CLS = {"Ready": "ready", "Outstanding items": "later", "Needs clinical review": "resched", "Recommend hold": "hold"}
KEY_TESTS = {
    "nurse": ["Hemoglobin", "Potassium", "Glucose", "POC glucose", "INR", "Serum hCG", "UA nitrite", "Urine culture"],
    "anesthesia": ["Hemoglobin", "Hematocrit", "Platelets", "PT", "INR", "PTT", "Sodium", "Potassium", "Creatinine",
                   "eGFR", "Glucose", "POC glucose", "HbA1c", "Albumin", "AST", "ALT", "Total bilirubin"],
}
ICON = {"ready": "🟢", "open": "⚪", "later": "🟠", "resched": "🔴", "hold": "🟣"}
CK = {"ok": ("✓", "Complete"), "missing": ("✗", "Missing"), "outdated": ("!", "Expired"), "pending": ("…", "Pending")}
DECLINE_REASONS = ["Already done elsewhere", "Not indicated for this patient", "Patient declined", "Covered by another order", "Other"]
DECISIONS = os.path.join(D.ROOT, "results", "decisions.csv")

# Colors read on both Streamlit's light and dark themes: mid-tone ink, translucent fills.
st.markdown("""<style>
:root{--ok:#2e9e64;--warn:#c8820f;--bad:#d0473b;--hold:#9a4fc4;--muted:rgba(128,128,128,.95);--line:rgba(128,128,128,.3);--soft:rgba(128,128,128,.1)}
.s-ready{--c:var(--ok);--cb:rgba(46,158,100,.14)} .s-open{--c:var(--muted);--cb:var(--soft)} .s-later{--c:var(--warn);--cb:rgba(200,130,15,.15)}
.s-resched{--c:var(--bad);--cb:rgba(208,71,59,.14)} .s-hold{--c:var(--hold);--cb:rgba(154,79,196,.15)}
.banner{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:8px;line-height:1.6}
.banner .nm{font-size:1.3em;font-weight:700;margin-right:10px} .banner .mt{color:var(--muted);font-size:.92em}
.allergy{color:var(--bad);font-weight:600}
.sugg{margin-top:6px;padding:6px 10px;border-radius:6px;background:var(--cb);border-left:4px solid var(--c)}
.sugg .lb{color:var(--c);font-size:.75em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-right:8px}
.card{padding:16px 20px;border-radius:10px;background:var(--cb);border:1px solid var(--c);margin-bottom:10px}
.card .ti{font-size:1.6em;font-weight:700;color:var(--c);line-height:1.2}
.card .sub{margin-top:2px} .card .fx{margin-top:6px;font-size:.9em;color:var(--muted)}
.card ul{margin:6px 0;padding-left:0;list-style:none} .card li{margin:2px 0}
.lead{font-size:.85em;color:var(--muted);margin-left:6px} .lead.days{color:var(--bad);font-weight:600}
.bar{padding:8px 14px;border-radius:6px;font-weight:600;background:var(--cb);color:var(--c);margin-bottom:10px}
.bar span{font-weight:400;color:inherit;opacity:.9}
.ck-ok{color:var(--ok)} .ck-missing{color:var(--bad)} .ck-outdated,.ck-pending{color:var(--warn)}
table.ck{border-collapse:collapse;width:100%;font-size:.9em} table.ck td,table.ck th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--soft);vertical-align:top}
table.ck th{font-size:.75em;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.legend span{margin-right:14px;font-size:.88em;white-space:nowrap} .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;background:var(--c)}
.og-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.og{display:grid;min-width:max-content;--q:13px}
.og .hd{font-size:.72em;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:8px;background:var(--soft);grid-row:1}
.og .tm{font-family:monospace;font-size:.7em;color:var(--muted);padding:2px 6px 0;border-top:1px solid var(--line);grid-column:1}
.og .lane{border-left:1px solid var(--soft);background:repeating-linear-gradient(to bottom,var(--line) 0 1px,transparent 1px calc(var(--q)*4))}
.blk{display:flex;flex-direction:column;margin:1px 3px;overflow:hidden;border-left:4px solid var(--c);background:var(--cb);border-radius:6px;padding:4px 6px;font-size:.72em;line-height:1.25;z-index:1}
.blk .st{font-size:.85em;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--c)}
.blk .tt{font-family:monospace} .blk .pl{color:var(--muted)} .blk .mv{font-style:italic}
.cal{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;text-align:center;font-size:.85em}
.cal .w{font-size:.75em;font-weight:600;color:var(--muted)}
.cal .d{position:relative;border-radius:6px;padding:4px 0;line-height:1.1}
.cal .d.has{border:1px solid var(--line);font-weight:600} .cal .d.sel{background:#0d6b73;color:#fff;border-color:#0d6b73}
.cal .d small{display:block;font-size:.72em;font-weight:400;opacity:.75} .cal .d.off{opacity:.45}
.cal .d i{position:absolute;top:3px;right:3px;width:6px;height:6px;border-radius:50%;background:var(--c)}
.sg{border-left:4px solid var(--c);padding:2px 0 2px 10px;margin:8px 0;font-size:.92em}
.src{color:var(--muted);font-size:.85em}
</style>""", unsafe_allow_html=True)

BOARD = S.by_id()
DAYS = sorted({x["case"]["surgery_date"] for x in BOARD.values()})
ss = st.session_state
ss.setdefault("sid", "S082")
ss.setdefault("view", "Patient chart")
ss.setdefault("day", BOARD[ss.sid]["case"]["surgery_date"])
ss.setdefault("results", {})
ss.setdefault("dec_time", {})


def open_chart(sid):
    """Callback: jump from the schedule to a patient's chart."""
    ss.sid, ss.view = sid, "Patient chart"


def open_day(day):
    ss.day = day


def html(s):
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Pre-Op Readiness")
view = st.sidebar.radio("View", ["Patient chart", "OR schedule"], key="view", horizontal=True)
role = st.sidebar.radio("I am the…", list(ROLE_LABEL), format_func=ROLE_LABEL.get, key="role")
provider = default_provider()
engines = ["AI (LLM)", "Rules only"] if provider != "mock" else ["Rules only", "AI (LLM)"]
engine = "llm" if st.sidebar.selectbox("Engine", engines).startswith("AI") else "rules"
if engine == "llm" and provider == "mock":
    st.sidebar.warning("No API key found. AI mode will show the rules-only output.")

if view == "Patient chart":
    st.sidebar.divider()
    sf = st.sidebar.selectbox("Readiness", [""] + list(S.SCHED), format_func=lambda s: S.SCHED.get(s, "Any status"))
    pf = st.sidebar.multiselect("Procedure", C.PROCEDURES)
    split = st.sidebar.selectbox("Cases", ["all", "dev", "test"])
    rows = [x for x in BOARD.values() if (not sf or x["state"] == sf) and (not pf or x["case"]["procedure"] in pf)
            and (split == "all" or x["split"] == split)]
    ids = [x["id"] for x in sorted(rows, key=lambda x: (x["case"]["surgery_date"], x["case"]["surgery_time"]))]
    if ss.sid not in ids:
        ids.insert(0, ss.sid)

    def label(sid):
        x = BOARD[sid]
        tag = f"{S.open_count(x)} open" if x["state"] == "open" else S.SCHED[x["state"]]
        return f"{ICON[x['state']]} {x['case']['surgery_date'][5:]} {x['case']['surgery_time']} · {x['patient']['name']} · {tag}"

    st.sidebar.selectbox(f"Patient ({len(rows)} cases)", ids, key="sid", format_func=label)
    st.sidebar.caption("🟢 Ready · ⚪ items outstanding · 🟠 move later · 🔴 reschedule · 🟣 clinical hold")
st.sidebar.caption("Fictional patients · teaching rules, not clinical guidelines · not for patient care")


# ---------------------------------------------------------------------------
# OR schedule
# ---------------------------------------------------------------------------
def month_calendar(day):
    y, m = int(day[:4]), int(day[5:7])
    by_day = {}
    for x in BOARD.values():
        by_day.setdefault(x["case"]["surgery_date"], []).append(x["state"])
    cells = [f"<span class='w'>{w}</span>" for w in "MTWTFSS"]
    cells += ["<span></span>"] * calendar.monthrange(y, m)[0]
    for i in range(1, calendar.monthrange(y, m)[1] + 1):
        d = f"{y:04d}-{m:02d}-{i:02d}"
        sts = by_day.get(d)
        if not sts:
            cells.append(f"<span class='d off'>{i}</span>")
            continue
        flag = "s-resched" if {"resched", "hold"} & set(sts) else "s-later" if "later" in sts else ""
        cells.append(f"<span class='d has {'sel' if d == day else ''}'>{i}<small>{len(sts)}</small>"
                     f"{f'<i class={flag}></i>' if flag else ''}</span>")
    return (f"<b>{calendar.month_name[m]} {y}</b><div class='cal'>{''.join(cells)}</div>"
            "<div class='src' style='margin-top:6px'><span class='s-resched'>●</span> reschedule or hold "
            "<span class='s-later'>●</span> move later · number = cases</div>")


def timeline(cases, sugg):
    """Room-by-room grid, one row per 15 minutes from 07:00 to 18:00."""
    rooms = sorted({x["case"]["surgery_location"] for x in cases})
    g0, g1, step = S.to_min("07:00"), S.to_min("18:00"), 15
    nrows = (g1 - g0) // step
    row = lambda m: 2 + (m - g0) // step
    overlaps = lambda x: any(y is not x and y["case"]["surgery_location"] == x["case"]["surgery_location"]
                             and S.start(y) < S.end(x) and S.start(x) < S.end(y) for y in cases)
    parts = ["<div class='hd'></div>"] + [f"<div class='hd' style='grid-column:{i + 2}'>{esc(S.short_room(r))}</div>" for i, r in enumerate(rooms)]
    parts += [f"<div class='tm' style='grid-row:{row(m)} / span 4'>{S.hhmm(m)}</div>" for m in range(g0, g1, 60)]
    parts += [f"<div class='lane' style='grid-column:{i + 2};grid-row:2 / span {nrows}'></div>" for i in range(len(rooms))]
    for x in cases:
        stt, g = x["state"], sugg.get(x["id"])
        tip = f"{x['patient']['name']} · {D.procedure_label(x['case'])} · {S.span(S.start(x), S.dur(x))} ({S.dur_label(S.dur(x))}) · {S.SCHED[stt]}"
        parts.append(
            f"<div class='blk s-{stt}' title='{esc(tip, quote=True)}' style='grid-column:{rooms.index(x['case']['surgery_location']) + 2};"
            f"grid-row:{row(S.start(x))} / span {S.dur(x) // step}'><span class='tt'>{S.span(S.start(x), S.dur(x))}</span>"
            f"<span class='st'>{S.SCHED[stt]}{' · overlap' if overlaps(x) else ''}</span><b>{esc(x['patient']['name'])}</b>"
            f"<span class='pl'>{esc(D.procedure_label(x['case']))}</span>"
            + (f"<span class='mv'>Suggested: {esc(g['tag'].lower())}</span>" if g else "") + "</div>")
    return (f"<div class='og-wrap'><div class='og' style='grid-template-columns:52px repeat({len(rooms)}, minmax(118px, 1fr));"
            f"grid-template-rows:auto repeat({nrows}, var(--q))'>{''.join(parts)}</div></div>")


def schedule_view():
    if ss.day not in DAYS:
        ss.day = DAYS[0]
    di = DAYS.index(ss.day)
    cases = [x for x in BOARD.values() if x["case"]["surgery_date"] == ss.day]
    items = S.day_suggestions(ss.day)
    sugg = {g["x"]["id"]: g for g in items}

    nav = st.columns([0.6, 5, 0.6, 3], vertical_alignment="center")
    nav[0].button("‹", disabled=di == 0, on_click=open_day, args=(DAYS[di - 1],), help="Previous surgery day")
    nav[1].subheader(f"{S.fmt_day(ss.day, long=True)} · {len(cases)} cases")
    nav[2].button("›", disabled=di == len(DAYS) - 1, on_click=open_day, args=(DAYS[min(di + 1, len(DAYS) - 1)],), help="Next surgery day")

    def day_label(d):
        sts = [x["state"] for x in BOARD.values() if x["case"]["surgery_date"] == d]
        flag = " 🔴" if {"resched", "hold"} & set(sts) else " 🟠" if "later" in sts else ""
        return f"{S.fmt_day(d)} · {len(sts)} cases{flag}"
    nav[3].selectbox("Go to day", DAYS, key="day", format_func=day_label, label_visibility="collapsed")

    counts = [(s, sum(x["state"] == s for x in cases)) for s in S.SCHED]
    html("<div class='legend'>" + "".join(f"<span class='s-{s}'><i></i>{S.SCHED[s]} {n}</span>" for s, n in counts if n) + "</div>")

    main, side = st.columns([3, 1.1])
    with main:
        html(timeline(cases, sugg))
        st.caption("Hover a block for details. Open any patient below.")
        order = sorted(cases, key=lambda x: (x["case"]["surgery_location"], S.start(x)))
        for i, x in enumerate(order):
            if i % 4 == 0:
                cols = st.columns(4)
            cols[i % 4].button(f"{ICON[x['state']]} {x['case']['surgery_time']} {x['patient']['name']}", key=f"go_{x['id']}",
                               on_click=open_chart, args=(x["id"],), width="stretch")
    with side:
        with st.container(border=True):
            html(month_calendar(ss.day))
        with st.container(border=True):
            st.markdown("**Suggested adjustments**")
            st.caption(f"{S.fmt_day(ss.day)} · based on readiness")
            if not items:
                st.write("No adjustments needed. Every case can go as scheduled.")
            for g in items:
                x = g["x"]
                html(f"<div class='sg s-{x['state']}'><b>{esc(x['patient']['name'])}</b> "
                     f"<span class='src'>{S.span(S.start(x), S.dur(x))} · {esc(S.short_room(x['case']['surgery_location']))}</span>"
                     f"<div><b>{esc(g['text'])}</b></div><div class='src'>{esc(g['reason'])}</div></div>")
                st.button("Open chart", key=f"sg_{x['id']}", on_click=open_chart, args=(x["id"],))
            st.caption("Suggestions only. Confirm with the surgeons, anesthesia and the patients before changing the schedule.")


if view == "OR schedule":
    schedule_view()
    st.stop()

# ---------------------------------------------------------------------------
# Patient chart
# ---------------------------------------------------------------------------
sid = ss.sid
x = BOARD[sid]
chart = D.build_chart(sid)
c, p = chart["case"], chart["patient"]
proc = D.procedure_label(c)
when = f"{c['surgery_date']} {S.span(S.start(x), S.dur(x))}"

sg = S.suggested_time(x)
html(f"""<div class="banner"><span class="nm">{esc(p['name'])}</span>{esc(p['age'])} y/o {esc(p['sex'])} · MRN {esc(p['mrn'])} · BMI {esc(p['bmi'])} ·
  <span class="{'' if p['allergies'].startswith('No known') else 'allergy'}">Allergies: {esc(p['allergies'])}</span><br>
  <span class="mt"><b>{esc(proc)}</b> · {when} ({S.dur_label(S.dur(x))}) · {esc(c['surgery_location'])} · {esc(c['surgeon'])} ·
  Anesthesia: {esc(c['planned_anesthesia'])} · Post-op: {esc(c['postoperative_location'])}</span>
  {f'''<div class="sugg s-{sg['cls']}"><span class="lb">Suggested new time</span><b>{S.fmt_day(sg['date'])} · {sg['time']} · {esc(sg['room'])}</b>
  <span class="src"> {esc(sg['why'])}</span></div>''' if sg else ''}</div>""")

tabs = st.tabs(["Pre-Op Readiness", "Labs", "Medications", "Tests & Imaging", "Consents & Blood", "H&P"])


def top_card(status, n, headline):
    """Ready card, reschedule card, or a plain status bar, as on the demo page."""
    open_items = S.open_reqs(x)
    done = len(x["reqs"]) - len(open_items)
    if not n and status == "Ready":
        return (f"<div class='card s-ready'><div class='ti'>✓ Ready for surgery</div>"
                f"<div class='sub'>Nothing outstanding for the {ROLE_LABEL[role].lower()}</div>"
                f"<div class='fx'><b>{esc(proc)}</b> · {when} · {esc(c['surgery_location'])} · Checklist {done} of {len(x['reqs'])} complete</div></div>")
    if len(open_items) >= 2:
        slow = [r for r in open_items if S.takes_days(r)]
        hold = status == "Recommend hold"
        hard = hold or x["state"] == "resched"
        note = ("A clinical finding recommends holding this case. Review with the surgeon and anesthesia before choosing a new date."
                if hold and not slow else
                f"{' and '.join(r['item'] for r in slow)} {'is' if len(slow) == 1 else 'are'} unlikely to be ready in time. Talk with the surgeon about a new date."
                if hard else "Everything can be done on the day. A later slot gives time to finish before the patient goes to the OR.")
        items = "".join(f"<li><span class='ck-{r['status']}'>{CK[r['status']][1]}</span> <b>{esc(r['item'])}</b>"
                        f"<span class='lead {'days' if S.takes_days(r) else ''}'>{'Usually needs days' if S.takes_days(r) else 'Can be done today'}</span></li>"
                        for r in open_items)
        return (f"<div class='card s-{'resched' if hard else 'later'}'><div class='ti'>{'Consider rescheduling' if hard else 'Consider moving later today'}</div>"
                f"<div class='sub'>{len(open_items)} pre-op items are not ready for {esc(proc)} at {c['surgery_date']} {c['surgery_time']} · {esc(status)}</div>"
                f"<ul>{items}</ul><div class='src'>{esc(note)}</div></div>")
    return f"<div class='bar s-{STATUS_CLS.get(status, 'open')}'>{esc(status)} <span>· {esc(headline)}</span></div>"


def checklist_html():
    done = sum(r["status"] == "ok" for r in x["reqs"])
    rows = "".join(f"<tr><td class='ck-{r['status']}'><b>{CK[r['status']][0]} {CK[r['status']][1]}</b></td><td><b>{esc(r['item'])}</b></td>"
                   f"<td>{esc(r['note'])}</td><td class='src'>{esc(r['why'])}</td></tr>" for r in x["reqs"])
    return (f"<b>Pre-op checklist</b> <span class='src'>· {done} of {len(x['reqs'])} complete</span>"
            f"<table class='ck'><tr><th>Status</th><th>Item</th><th>On file</th><th>Needed because</th></tr>{rows}</table>")


def stamp(dk):
    ss.dec_time[dk] = datetime.now().strftime("%b %d %H:%M")


with tabs[0]:
    key = (sid, role, engine)
    if key not in ss.results and not (engine == "llm" and provider != "mock"):
        ss.results[key] = run_case(sid, role=role, engine=engine)
    res = ss.results.get(key)
    if engine == "llm" and provider != "mock":
        if st.button("Run the AI review again" if res else f"Run AI readiness review for the {ROLE_LABEL[role].lower()}", type="primary"):
            with st.spinner("Reading the chart (Modules 2-4)…"):
                ss.results[key] = res = run_case(sid, role=role, engine="llm", llm=LLM(), use_cache=res is None)
    if res is None:
        res = run_case(sid, role=role, engine="rules")
        st.caption("Showing the rules-only check until the AI review is run.")
    steps, summ = res["next_steps"], res["summary"]
    n = len(steps["actions"])
    html(top_card(steps["status"], n, summ.get("headline") or f"{n} item{'s' * (n != 1)} to address before surgery."))
    st.caption(f"{ROLE_LABEL[role]} view · engine: {res['engine']} · {steps.get('status_reason', '')}")

    dkey = lambda code: f"dec|{sid}|{role}|{code}"
    left, right = st.columns([1, 1.15])
    with left:
        accepted = [a for a in steps["actions"] if ss.get(dkey(a["code"])) == "Accept"]
        if n:
            with st.container(border=True):
                tasks = [(a, i, s) for a in accepted for i, s in enumerate(C.ACTION_FOLLOW_UP.get(a["code"], ([], ""))[0])]
                done = sum(bool(ss.get(f"todo|{sid}|{role}|{a['code']}|{i}")) for a, i, _ in tasks)
                st.markdown(f"**To-do · {ROLE_LABEL[role]}**" + (f" <span class='src'>· {done} of {len(tasks)} done</span>" if tasks else ""),
                            unsafe_allow_html=True)
                if not accepted:
                    st.caption("Accept a next step to add its tasks here.")
                for a in accepted:
                    todo_steps, closes = C.ACTION_FOLLOW_UP.get(a["code"], ([], ""))
                    st.markdown(f"<span class='src'><b>{esc(action_text(a['code']))}</b></span>", unsafe_allow_html=True)
                    for i, s in enumerate(todo_steps):
                        st.checkbox(s, key=f"todo|{sid}|{role}|{a['code']}|{i}")
                    if all(ss.get(f"todo|{sid}|{role}|{a['code']}|{i}") for i in range(len(todo_steps))):
                        st.caption(f"Tasks done · waiting for the chart: {closes}")
        with st.container(border=True):
            for sec in summ.get("sections", []):
                st.markdown(f"**{sec['title']}**")
                for b in sec.get("bullets", []):
                    st.markdown(f"- {b}")
            st.markdown("**Key values**")
            rows = []
            for t in KEY_TESTS[role]:
                lab = C.latest_lab(chart, t)
                if lab:
                    rows.append(dict(Test=t, Value=f"{lab['result_value']} {'' if lab['unit'] == 'text' else lab['unit']}",
                                     Flag=lab["flag"], Date=lab["collection_date"], ID=lab["lab_id"]))
            for d in chart["diagnostics"]:
                if role == "anesthesia" or d["test_type"] == "EKG":
                    rows.append(dict(Test=d["test_type"], Value=(f"{d['result_value']} {d['result_unit']}".strip() or d["interpretation"]),
                                     Flag="New" if d["compared_to_prior"] == "New finding" else "", Date=d["test_date"], ID=d["test_id"]))
            if rows:
                st.dataframe(pd.DataFrame(rows).style.map(lambda v: "color:#d0473b;font-weight:600" if v in ("H", "L", "Critical", "Abnormal", "New") else "",
                                                          subset=["Flag"]), hide_index=True, width="stretch")
            else:
                st.caption("No values in chart.")

    with right:
        st.markdown(f"**Next steps · {ROLE_LABEL[role]}**")
        if not n:
            st.success("Nothing outstanding.")
        for a in steps["actions"]:
            dk = dkey(a["code"])
            badge = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(a.get("priority"), "⚪")
            with st.container(border=True):
                st.markdown(f"{badge} **{action_text(a['code'])}**")
                st.caption(f"{a.get('reason', '')} · Evidence: {', '.join(a.get('evidence_ids', [])) or '-'}")
                dec = st.radio("Decision", ["Pending", "Accept", "Decline"], horizontal=True, key=dk,
                               label_visibility="collapsed", on_change=stamp, args=(dk,))
                t = f" · {ss.dec_time[dk]}" if dk in ss.dec_time else ""
                if dec == "Accept":
                    st.markdown(f"<span class='s-later' style='color:var(--c);font-weight:600'>Accepted{t} · still open · tasks added to your to-do list</span>"
                                f"<br><span class='src'><b>Closes when:</b> {esc(C.ACTION_FOLLOW_UP.get(a['code'], ([], ''))[1])}</span>",
                                unsafe_allow_html=True)
                elif dec == "Decline":
                    st.caption(f"Declined{t}")
                    st.selectbox("Reason", [""] + DECLINE_REASONS, key=f"why|{dk}", format_func=lambda r: r or "Choose a reason…")
        if steps.get("other_considerations"):
            st.markdown("**Other considerations**")
            for o in steps["other_considerations"]:
                st.markdown(f"- {o}")
        if n and st.button("Save decisions"):
            os.makedirs(os.path.dirname(DECISIONS), exist_ok=True)
            new = not os.path.exists(DECISIONS)
            with open(DECISIONS, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "surgery_id", "role", "engine", "action_code", "decision", "decline_reason"])
                for a in steps["actions"]:
                    dk = dkey(a["code"])
                    w.writerow([datetime.now().isoformat(timespec="seconds"), sid, role, res["engine"], a["code"],
                                ss.get(dk, "Pending"), ss.get(f"why|{dk}", "")])
            st.success(f"Decisions saved to {os.path.relpath(DECISIONS, D.ROOT)}.")

    with st.container(border=True):
        html(checklist_html())
    if res.get("qa"):
        with st.expander("Module 2 answers (question answering, with evidence)"):
            for a in res["qa"].get("answers", []):
                st.markdown(f"**{a.get('question_id')}** {a.get('answer')}  \n`{', '.join(a.get('evidence_ids', []))}`")
    st.caption("Every suggestion needs human review.")

# ---------------------------------------------------------------------------
# Chart tabs
# ---------------------------------------------------------------------------
def show(rows, cols, hl=None, title=None):
    if title:
        st.markdown(f"**{title}**")
    if not rows:
        st.caption("None in chart.")
        return
    df = pd.DataFrame(rows)[cols]
    if hl:
        bad = {"H", "L", "Critical", "Abnormal", "Missing", "Refused", "Pending", "Positive", "New finding"}
        df = df.style.map(lambda v: "color:#d0473b;font-weight:600" if v in bad else "", subset=hl)
    st.dataframe(df, hide_index=True, width="stretch")


with tabs[1]:
    show(sorted(chart["labs"], key=lambda r: (r["panel"], r["test_name"])),
         ["panel", "test_name", "result_value", "unit", "reference_low", "reference_high", "flag", "collection_date", "lab_id"], ["flag"])
with tabs[2]:
    show([dict(m, preop_instruction=m["preop_instruction"] or "⚠ none documented") for m in chart["meds"]],
         ["medication_name", "dose", "route", "frequency", "preop_instruction", "medication_id"])
with tabs[3]:
    show(chart["diagnostics"], ["test_type", "test_date", "result_value", "result_unit", "interpretation", "compared_to_prior", "test_id"],
         ["compared_to_prior"], title="Diagnostic tests")
    show(chart["imaging"], ["imaging_type", "imaging_date", "report", "imaging_id"], title="Imaging")
    show(chart["clearances"], ["clearance_type", "status", "requested_date", "completed_date", "assessment_note", "clearance_id"], ["status"],
         title="Clearances")
with tabs[4]:
    show(chart["consents"], ["consent_type", "status", "signed_date", "consent_id"], ["status"], title="Consents")
    if chart["blood"]:
        show([chart["blood"]], ["abo_rh", "antibody_screen", "type_screen_date", "type_screen_expires", "prbc_crossmatched_units",
                                "ffp_units_available", "platelet_units_available"], ["antibody_screen"], title="Blood bank")
    else:
        st.caption("No blood bank record (not required for this procedure).")
with tabs[5]:
    if chart["notes"]:
        for note in chart["notes"]:
            st.text(note["note_text"])
    else:
        st.warning("No H&P in chart.")
    st.caption(f"Anesthesia history: {p['anesthesia_complication_history']}")
