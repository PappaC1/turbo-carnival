import io
import re
import numpy as np
import pdfplumber
import pandas as pd
import streamlit as st
from racing_model import build_features, choose_selections

st.set_page_config(page_title="SA Punters Predictor", page_icon="🏇", layout="centered", initial_sidebar_state="collapsed")
st.title("🏇 SA Punters Predictor")
st.caption("Hollywoodbets Punters Challenge • Computaform AI")

def first_val(series):
    return next(iter(series))

def parse_computaform_pdf(uploaded_file) -> pd.DataFrame:
    pages_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    full_text = "\n".join(pages_text)

    course = "FAIRVIEW"
    cm = re.search(r"TAB\s+(FAIRVIEW|TURFFONTEIN|GREYVILLE|KENILWORTH|VAAL|SCOTTSVILLE|DURBANVILLE)", full_text, re.I)
    if cm:
        course = cm.group(1).upper()
    going = "Polytrack" if "POLYTRACK" in "\n".join(pages_text[:4]).upper() else "Good"

    runners = []
    for txt in pages_text:
        if "Race Ref:" not in txt or "WIN - PLACE - SWINGER" not in txt:
            continue
        rm = re.search(r"\b\d{2}:\d{2}\s+(\d{1,2})\b", txt)
        dm = re.search(r"\b(\d{3,4})m\b", txt)
        if not rm:
            continue
        r_num = int(rm.group(1))
        dist = int(dm.group(1)) if dm else 1200
        r_type = "Maiden" if "MAIDEN" in txt.upper() else ("Plate" if "PLATE" in txt.upper() else "Handicap")

        # Extract CFORM & SPEED ratings
        ratings = {}
        fb = re.search(r"CFORM RATINGS(.*?)TIPSTER", txt, re.S | re.I)
        if fb:
            for line in fb.group(1).split("\n"):
                if "TOP %" in line or "BEST WEIGHTED" in line:
                    continue
                for bm in re.finditer(r"\b(\d{1,2})\s+[A-Za-z][A-Za-z\s\'\(\)\-]+?\s+([+-]\d+\.\d+)\b", line):
                    ratings.setdefault(int(bm.group(1)), {})["bva"] = float(bm.group(2))
                for im in re.finditer(r"\b(\d{1,2})\s+[A-Za-z][A-Za-z\s\'\(\)\-]+?\s+(\d{1,3})\b", line):
                    no, val = int(im.group(1)), float(im.group(2))
                    ent = ratings.setdefault(no, {})
                    if "cf" not in ent and val <= 75:
                        ent["cf"] = val
                    elif "sp" not in ent:
                        ent["sp"] = val

        for line in txt.split("\n"):
            m = re.search(
                r"^\s*(\d{1,2})\s+(?:([\d▲▼]+)\s+)?([A-Z][A-Z\s\'\(\)\-]+?)\s+(\d+)-(\d+)-(\d+)-(\d+)\s+\d+%\s+R[\d,\.]+\s+R[\d,\.]+\s+(\d)([A-Za-z]{1,3})\s+([\d\.]+)(?:-[\d\.]+)?\s+(\d{1,2})\s+([A-Za-z\s/\.\-]+?)\s+(?:[\d\.]+|--)\s*%\s+\*?([A-Za-z\s/\.\-]+?)\s+(?:[\d\.]+|--)\s*%\s*(?:(\d{2,3})\s+(\d{2,3})\s+)?.*?(\d+[-/]\d+)\s*$",
                line,
            )
            if not m:
                continue
            no = int(m.group(1))
            l3 = ",".join(["99" if d == "0" else d for d in re.findall(r"\d", m.group(2) or "")[:3]])
            starts, wins, s2, s3 = int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
            num, den = map(float, m.group(16).replace("-", "/").split("/"))
            rmeta = ratings.get(no, {})
            runners.append({
                "race_id": f"{course[:3]}_R{r_num}", "course": course, "race_number": r_num,
                "distance_m": dist, "going": going, "race_type": r_type,
                "saddle_number": no, "horse_name": m.group(3).strip().title(),
                "draw": int(m.group(11)), "weight_kg": float(m.group(10)), "age": int(m.group(8)),
                "trainer_name": m.group(12).strip(), "jockey_name": m.group(13).strip(),
                "wins_last_12_months": wins, "places_last_12_months": wins + s2 + s3,
                "starts_last_12_months": starts, "last_3_positions": l3,
                "official_rating": float(m.group(15)) if m.group(15) else np.nan,
                "market_odds": round((num / den) + 1.0, 2),
                "cform_rating": rmeta.get("cf", np.nan), "cf_speed_rating": rmeta.get("sp", np.nan),
                "best_vs_ave": rmeta.get("bva", np.nan),
            })
    df = pd.DataFrame(runners)
    return df.drop_duplicates(subset=["race_id", "saddle_number"]).sort_values(["race_number", "saddle_number"]) if not df.empty else df

def score_runners(today_df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(today_df)
    probs = []
    for _, grp in df.groupby("race_id", sort=False):
        mkt = grp["implied_probability"].values / grp["implied_probability"].sum()
        z = np.zeros(len(grp))
        for col, w in [("cform_rating", 0.30), ("cf_speed_rating", 0.25), ("recent_form_score", 0.15), ("best_vs_ave", 0.15), ("official_rating", 0.10), ("win_rate_12m", 0.05)]:
            arr = grp[col].fillna(grp[col].median()).fillna(0).values
            if np.std(arr) > 0:
                z += w * ((arr - np.mean(arr)) / np.std(arr))
        rp = np.exp(z * 0.55)
        blend = (0.55 * mkt) + (0.45 * (rp / rp.sum()))
        probs.extend(blend / blend.sum())
    df["model_win_probability"] = probs
    df["model_edge"] = df["model_win_probability"] - df["implied_probability"]
    return df

uploaded_pdf = st.file_uploader("Upload Computaform PDF", type=["pdf"])
if not uploaded_pdf:
    st.info("Tap **Browse files** above to upload today's **Computaform PDF**.")
    st.stop()

with st.spinner("Analyzing Computaform card..."):
    today_df = parse_computaform_pdf(uploaded_pdf)
if today_df.empty:
    st.error("No runners detected. Ensure you uploaded an official Computaform PDF.")
    st.stop()

predictions = score_runners(today_df)
selection_df = pd.DataFrame([s.__dict__ for s in choose_selections(predictions)])
course_title = str(first_val(today_df["course"]))

tab1, tab2 = st.tabs(["🏆 Punters Challenge Card", "🔍 Race-by-Race"])
with tab1:
    st.subheader(f"{course_title} — Punters Card")
    best_picks = selection_df[selection_df["is_best_selection"]].sort_values("race_number")
    for _, pick in best_picks.iterrows():
        rm = predictions[(predictions["race_number"] == pick["race_number"]) & (predictions["horse_name"] == pick["horse_name"])]
        saddle = int(first_val(rm["saddle_number"])) if not rm.empty else "-"
        jockey = str(first_val(rm["jockey_name"])) if not rm.empty else ""
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**R{int(pick['race_number'])}: #{saddle} {pick['horse_name']}**")
            c1.caption(f"{pick['selection_type']} • {jockey} • Fcst: {pick['market_odds']:.2f}")
            c2.metric("Win %", f"{pick['model_win_probability']:.0%}")
    st.download_button("📥 Download Punters Card (CSV)", best_picks.to_csv(index=False), f"{course_title}_punters.csv", "text/csv", use_container_width=True)

with tab2:
    sel_race = st.selectbox("Select Race", selection_df["race_id"].unique().tolist())
    r_sub = predictions[predictions["race_id"] == sel_race]
    st.subheader(f"Race {int(first_val(r_sub['race_number']))} — {first_val(r_sub['course'])}")
    for _, row in selection_df[selection_df["race_id"] == sel_race].iterrows():
        with st.container(border=True):
            st.markdown(f"### {'🥇' if row['is_best_selection'] else '•'} {row['selection_type']}: **{row['horse_name']}**\nWin: **{row['model_win_probability']:.1%}** | Odds: **{row['market_odds']:.2f}** | Edge: **{row['model_edge']:+.1%}**")
    cols = ["saddle_number", "horse_name", "jockey_name", "draw", "cform_rating", "cf_speed_rating", "market_odds", "model_win_probability", "model_edge"]
    st.dataframe(r_sub[cols].sort_values("model_win_probability", ascending=False), use_container_width=True, hide_index=True)
