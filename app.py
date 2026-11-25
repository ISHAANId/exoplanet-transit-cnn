"""
Streamlit demo UI for the exoplanet transit CNN project.

Three-step story, as tabs instead of one long scroll: (1) the idea, with a
simple illustrative diagram, no real data involved; (2) try it on one real
Kepler star at a time, raw vs. phase-folded, with a live AI prediction; (3)
the overall raw vs. folded comparison across the whole test set, with a
plain scorecard up front and the technical proof (ROC curve, confusion
matrices) tucked into an expander for anyone who wants to verify it.

Run:
    venv/Scripts/python.exe -m streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATASET_PATH, FOLDED_DATASET_PATH, RESULTS_DIR, MODELS_DIR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from model import prepare  # noqa: E402
from planet_info import build_profile, mars_profile, load_koi_table, STAR_CLASS_NOTE  # noqa: E402

st.set_page_config(page_title="ExoFinder", page_icon="🔭", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 4.5rem;}
    .exo-title {font-family: Georgia, 'Times New Roman', serif; font-size: 3rem;
                letter-spacing: -1px; margin: 0; line-height: 1.25; padding-top: .2rem;}
    .exo-title span {color: #eb6834;}
    .exo-tag {color: #898781; font-size: 1.05rem; margin: .3rem 0 1.4rem 0;}
    div[data-baseweb="tab-list"] {gap: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Categorical slots 1/2 from the project's validated dataviz palette -- kept
# fixed to "raw"/"folded" everywhere in this app rather than cycled per-chart.
VARIANT_COLOR = {"raw": "#2a78d6", "folded": "#eb6834"}
MUTED = "#898781"


@st.cache_resource
def load_models():
    import tensorflow as tf
    models = {}
    for variant in ["raw", "folded"]:
        path = MODELS_DIR / f"{variant}_best.keras"
        if path.exists():
            models[variant] = tf.keras.models.load_model(path)
    return models


@st.cache_data
def load_datasets():
    data = {}
    for variant, path in [("raw", RAW_DATASET_PATH), ("folded", FOLDED_DATASET_PATH)]:
        if path.exists():
            d = np.load(path, allow_pickle=True)
            data[variant] = {
                "X": d["X"], "y": d["y"], "kepid": d["kepid"],
                "koi_name": d["koi_name"], "split": d["split"],
            }
    return data


@st.cache_data
def load_koi():
    path = Path(__file__).resolve().parent / "data" / "koi_raw.csv"
    return load_koi_table(path) if path.exists() else None


@st.cache_data
def load_zoo():
    path = RESULTS_DIR / "model_zoo" / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_metrics():
    out = {}
    for variant, folder in [("raw", "experiment_A_raw"), ("folded", "experiment_B_folded")]:
        path = RESULTS_DIR / folder / "metrics.json"
        if path.exists():
            with open(path) as f:
                out[variant] = json.load(f)
    return out


def test_predictions(variant):
    """Live model output on that variant's held-out test split -- feeds the
    scorecard, ROC curve, and confusion matrix (not read from a saved file,
    so it always matches whatever model is currently loaded)."""
    d = datasets[variant]
    mask = d["split"] == "test"
    X_test = prepare(d["X"][mask])[..., np.newaxis]
    y_test = d["y"][mask]
    y_prob = models[variant].predict(X_test, verbose=0).ravel()
    return y_test, y_prob


st.markdown('<p class="exo-title">Exo<span>Finder</span></p>', unsafe_allow_html=True)
st.markdown(
    '<p class="exo-tag">Spotting planets by the way they dim their stars. '
    'Built on NASA Kepler data.</p>',
    unsafe_allow_html=True,
)

models = load_models()
datasets = load_datasets()
metrics = load_metrics()

if not datasets:
    st.error("No processed dataset found yet. Run scripts 01-04 first.")
    st.stop()

# Flip to True to bring back the "How it works" planet-simulator tab.
SHOW_SIMULATOR = False

if SHOW_SIMULATOR:
    tab_about, tab_learn, tab_idea, tab_game, tab_try, tab_models, tab_results = st.tabs(
        ["Overview", "Learn", "How it works", "Play: you vs. ExoFinder", "Look at a real star",
         "Models", "How well does it do?"]
    )
else:
    tab_about, tab_learn, tab_game, tab_try, tab_models, tab_results = st.tabs(
        ["Overview", "Learn", "Play: you vs. ExoFinder", "Look at a real star", "Models",
         "How well does it do?"]
    )

# ---------------------------------------------------------------------------
# Overview: one-page pitch -- problem, approach, live headline result, roadmap
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Kepler found thousands of planet candidates. Someone has to check them.")
    st.write(
        "The Kepler space telescope stared at a patch of sky for years and logged how "
        "bright each star was, over and over. When a star dims in a regular pattern, a "
        "planet may be passing in front of it. But a dimming can also come from two "
        "stars circling each other, and telling the two apart takes a trained eye and a "
        "lot of time."
    )
    st.write(
        "ExoFinder is our attempt at that first check. We trained a small 1D "
        "convolutional neural network on Kepler stars that NASA had already sorted, "
        "then tested it on stars it had never seen."
    )
    m_raw, m_fold = metrics.get("raw"), metrics.get("folded")
    if m_raw and m_fold:
        c1, c2, c3 = st.columns(3)
        c1.metric("Stars in the dataset", f"{m_fold['n_train'] + m_fold['n_val'] + m_fold['n_test']}",
                  help="Training, validation and test examples together.")
        c2.metric("Right on stars it hasn't seen", f"{m_fold['accuracy']:.0%}",
                  delta=f"{m_fold['accuracy'] - m_raw['accuracy']:+.0%} vs. unfolded data",
                  help="Phase-folded model, scored on the held-out test stars.")
        c3.metric("AUC", f"{m_fold['roc_auc']:.2f}",
                  help="How well it ranks planets above false alarms. 1.00 is perfect, "
                       "0.50 is a coin flip.")
        st.caption(
            f"Scored on {m_fold['n_test']} test stars. That is a small sample, so read "
            "these as early numbers."
        )
    st.markdown("#### The question we tested")
    st.write(
        "Does it help to fold a star's light curve on the planet's orbital period "
        "before the network sees it? Folding stacks every transit on top of the "
        "others, so a faint dip stands out. We trained the same network twice, once on "
        "raw data and once on folded data, and compared them. NASA's own verdict "
        "columns were removed from the inputs so the network can't peek at the answer."
    )
    st.markdown("#### What we'd do with more time")
    st.write(
        "Train on many more stars, add a zoomed-in view of each dip the way published "
        "Kepler classifiers do, and try the same pipeline on TESS, which is still "
        "observing."
    )

# ---------------------------------------------------------------------------
# Learn: a short astrophysics lesson, then a quiz
# ---------------------------------------------------------------------------
with tab_learn:
    st.subheader("Learn how astronomers find planets around other stars")
    st.write("Read these five short steps, open the one you want, then try the quiz at the bottom.")
    with st.expander("1. What is an exoplanet?", expanded=True):
        st.write(
            "An **exoplanet** is a planet that orbits a star other than our Sun. We have found "
            "thousands. They are far too faint and too close to their stars to photograph "
            "directly, so astronomers use clever tricks to detect them."
        )
    with st.expander("2. The transit method: a tiny shadow"):
        st.write(
            "If a planet's orbit is lined up with our view, it passes **in front of** its star "
            "once every orbit and blocks a little light, like a moth flying past a lamp. The "
            "star's brightness dips, then recovers. This event is a **transit**, and a graph of "
            "brightness over time is a **light curve**."
        )
        st.write(
            "The depth of the dip tells you the planet's **size**: a planet that blocks 1% of "
            "the starlight has about one tenth of the star's width. The time between dips "
            "tells you its **year length**."
        )
    with st.expander("3. The Kepler telescope"):
        st.write(
            "NASA's **Kepler** space telescope stared at one patch of sky for about four years, "
            "measuring the brightness of roughly 150,000 stars every 30 minutes. It found "
            "thousands of possible planets. All the stars in this app are real Kepler stars."
        )
    with st.expander("4. False alarms"):
        st.write(
            "Not every dip is a planet. Two stars circling each other can pass in front of one "
            "another (an **eclipsing binary**) and make a similar dip. Instrument glitches and "
            "a nearby star's light can also fake one. Checking each signal by hand takes a lot "
            "of expert time, so we trained a neural network to do the first check."
        )
    with st.expander("5. Folding: stacking the dips"):
        st.write(
            "One transit is often buried in noise. If you know the planet's orbital period, you "
            "can cut the recording into one piece per orbit and **stack them on top of each "
            "other**. The noise averages out and the dip stands out. That is **phase folding**, "
            "and in this project it raised the model's accuracy a lot. Try it on the "
            "**Look at a real star** tab: the left chart is one pass, the right is the folded view."
        )
    st.markdown("#### Quick quiz")
    q1 = st.radio("A planet blocks 1% of its star's light. What does the dip tell us?",
                  ["Choose one", "The planet's size", "The planet's colour", "How old the star is"],
                  key="learn_q1")
    if q1 != "Choose one":
        (st.success if q1 == "The planet's size" else st.error)(
            "Right: a deeper dip means a bigger planet." if q1 == "The planet's size"
            else "Not quite. The depth of the dip tells us the planet's size.")
    q2 = st.radio("Why do astronomers fold a light curve on the orbital period?",
                  ["Choose one", "To stack all the transits so the dip stands out of the noise",
                   "To make the star look brighter", "To remove the planet"], key="learn_q2")
    if q2 != "Choose one":
        ok = q2.startswith("To stack")
        (st.success if ok else st.error)(
            "Right: stacking averages out the noise." if ok
            else "Not quite. Folding stacks every transit on top of the others.")
    q3 = st.radio("Which of these can cause a false alarm?",
                  ["Choose one", "Two stars eclipsing each other", "A planet that is too small",
                   "A star that is too bright"], key="learn_q3")
    if q3 != "Choose one":
        ok = q3.startswith("Two stars")
        (st.success if ok else st.error)(
            "Right: an eclipsing binary makes a planet-like dip." if ok
            else "Not quite. Two stars eclipsing each other is the classic false alarm.")
    st.caption("Data: NASA Kepler mission and the NASA Exoplanet Archive. This app teaches how "
               "planet detection works; it does not make new discoveries.")

# ---------------------------------------------------------------------------
# Tab 1: the idea, illustrated -- no real data, just the concept
# ---------------------------------------------------------------------------
if SHOW_SIMULATOR:
    with tab_idea:
        st.subheader("A planet passing in front of its star dims it, just slightly")
        st.write(
            "Play with the planet below. Every time it crosses the star, the "
            "brightness we measure drops a little, then recovers. Spotting that "
            "drop is how Kepler found thousands of planets."
        )

        st.markdown("#### Make a planet")

        def _preset(size_pct, period):
            st.session_state["sim_size"] = size_pct
            st.session_state["sim_period"] = period

        st.session_state.setdefault("sim_size", 10)
        st.session_state.setdefault("sim_period", 10)
        pc1, pc2, pc3, _ = st.columns([1, 1, 1, 3])
        pc1.button("Earth-sized", on_click=_preset, args=(1, 30), use_container_width=True)
        pc2.button("Neptune-sized", on_click=_preset, args=(4, 15), use_container_width=True)
        pc3.button("Jupiter-sized", on_click=_preset, args=(10, 5), use_container_width=True)

        s1, s2, s3 = st.columns(3)
        size_pct = s1.slider("Planet size (% of the star's width)", 1, 15, key="sim_size",
                             help="Earth is about 1% of the Sun's width, Jupiter about 10%.")
        period = s2.slider("Orbit length (days)", 2, 40, key="sim_period",
                           help="Closer planets orbit faster, so they dip more often.")
        noise_pct = s3.slider("Telescope noise", 0.0, 1.0, 0.1, 0.05,
                              help="Real measurements are noisy. Turn this up and watch a small "
                                   "planet disappear into it.")

        window = 40.0
        t_now = st.slider("Time (days)", 0.0, window, 5.0, 0.25,
                          help="Drag to move the planet along its orbit.")

        r = size_pct / 100
        duration = 13 * (period / 365) ** (1 / 3) / 24 * 8   # days; stretched so dips are visible
        half = duration / 2

        def brightness(tt):
            x = ((tt + period / 2) % period) - period / 2          # days from mid-transit
            u = np.abs(x) / half                                   # star-widths from centre
            full = np.clip((1 + r - u) / (2 * r), 0, 1)            # 0 outside, 1 fully inside
            return 1 - r ** 2 * np.minimum(full, 1)

        t_grid = np.linspace(0, window, 1200)
        rng = np.random.default_rng(7)
        curve = brightness(t_grid)
        noisy = curve + rng.normal(0, noise_pct / 100 * 0.01, t_grid.size)

        star_col, curve_col = st.columns([1, 2])
        with star_col:
            x_now = (((t_now + period / 2) % period) - period / 2) / half
            fig_star = go.Figure()
            fig_star.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1,
                               fillcolor="#ffd24a", line=dict(color="#ffb300"))
            fig_star.add_shape(type="circle", x0=x_now - r, y0=-r * 0.3 - r,
                               x1=x_now + r, y1=-r * 0.3 + r,
                               fillcolor="#1d1d27", line=dict(color="#000"))
            fig_star.update_layout(
                xaxis=dict(range=[-1.6, 1.6], visible=False, scaleanchor="y"),
                yaxis=dict(range=[-1.3, 1.3], visible=False),
                height=300, margin=dict(t=10, b=10, l=0, r=0), plot_bgcolor="#0b0d17",
            )
            st.plotly_chart(fig_star, width="stretch")
            st.caption("The star right now")
        with curve_col:
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=t_grid, y=noisy, mode="markers",
                                         marker=dict(size=3, color=MUTED), name="Measured",
                                         hoverinfo="skip"))
            fig_sim.add_trace(go.Scatter(x=t_grid, y=curve, mode="lines",
                                         line=dict(color=VARIANT_COLOR["raw"], width=2),
                                         name="True dip", hoverinfo="skip"))
            fig_sim.add_vline(x=t_now, line=dict(color=VARIANT_COLOR["folded"], width=2))
            fig_sim.update_layout(
                xaxis_title="Days", yaxis_title="Star's brightness",
                height=300, margin=dict(t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_sim, width="stretch")
            st.caption(f"The star dims by **{r ** 2:.2%}** each time the planet crosses it, "
                       f"every **{period} days**.")

        st.write(
            "ExoFinder is a small neural network we trained to look at real Kepler "
            "measurements and decide whether a dip comes from a planet or from something "
            "else, like two stars circling each other. Think you can do better? Try the "
            "next tab."
        )

# ---------------------------------------------------------------------------
# Tab 2: game -- guess Planet / False Alarm on real stars, human vs. the AI
# ---------------------------------------------------------------------------
with tab_game:
    st.subheader("Planet or false alarm?")
    st.write(
        "Here's a real Kepler star with every orbit stacked on top of each other. "
        "A planet usually leaves a clean, flat-bottomed dip; false alarms tend to "
        "look pointier or messier. Make your call, then see what ExoFinder thought."
    )

    g_variant = "folded" if "folded" in datasets and "folded" in models else None
    if g_variant is None:
        st.info("The folded model and dataset are needed for the game.")
    else:
        gd = datasets[g_variant]
        pool = np.where(gd["split"] == "test")[0]   # only stars the model never trained on
        n_stars = len(pool)
        ss = st.session_state
        ss.setdefault("g_rounds", 0)
        ss.setdefault("g_you", 0)
        ss.setdefault("g_ai", 0)
        ss.setdefault("g_idx", int(pool[np.random.default_rng().integers(n_stars)]))
        ss.setdefault("g_guess", None)

        i = ss["g_idx"]
        truth = int(gd["y"][i])
        x = gd["X"][i]
        prob = float(models[g_variant].predict(prepare(x)[np.newaxis, :, np.newaxis], verbose=0)[0, 0])
        ai_guess = int(prob >= 0.5)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Rounds played", ss["g_rounds"])
        sc2.metric("You", f"{ss['g_you']} correct")
        sc3.metric("ExoFinder", f"{ss['g_ai']} correct")

        fig_g = go.Figure(go.Scatter(y=x, mode="lines",
                                     line=dict(color=VARIANT_COLOR["folded"], width=2),
                                     hoverinfo="skip"))
        fig_g.update_layout(xaxis_title="Time around the orbit",
                            yaxis_title="Star's brightness",
                            xaxis=dict(showticklabels=False), height=320,
                            margin=dict(t=10, b=40))
        st.plotly_chart(fig_g, width="stretch")

        def _guess(choice):
            ss["g_guess"] = choice
            ss["g_rounds"] += 1
            ss["g_you"] += int(choice == truth)
            ss["g_ai"] += int(ai_guess == truth)

        def _next():
            ss["g_guess"] = None
            ss["g_idx"] = int(pool[np.random.default_rng().integers(n_stars)])

        if ss["g_guess"] is None:
            b1, b2 = st.columns(2)
            b1.button("Planet", on_click=_guess, args=(1,), use_container_width=True)
            b2.button("False alarm", on_click=_guess, args=(0,), use_container_width=True)
        else:
            you_ok, ai_ok = ss["g_guess"] == truth, ai_guess == truth
            st.markdown(
                f"**Truth (NASA experts):** {'Planet' if truth else 'False Alarm'}  \n"
                f"**You said:** {'Planet' if ss['g_guess'] else 'False Alarm'} "
                f"({'right' if you_ok else 'wrong'})  \n"
                f"**ExoFinder said:** {'Planet' if ai_guess else 'False Alarm'} "
                f"({prob:.0%} sure it's a planet) ({'right' if ai_ok else 'wrong'})"
            )
            if you_ok and not ai_ok:
                st.success("Nice, you got that one and ExoFinder missed it.")
            elif ai_ok and not you_ok:
                st.warning("ExoFinder got this one.")
            st.button("Next star", on_click=_next)

        st.caption(
            "These are all test stars, so ExoFinder is seeing them for the first time, "
            "same as you."
        )

# ---------------------------------------------------------------------------
# Tab 3: try it on one real star at a time
# ---------------------------------------------------------------------------
with tab_try:
    st.write(
        "Pick a real star below. The left chart shows one single pass in front of the "
        "star (**raw**); the right chart shows every pass stacked on top of each other "
        "(**phase-folded**) -- the dip should look clearer on the right if folding is "
        "doing its job. Underneath each, see what ExoFinder decided."
    )

    any_variant = "folded" if "folded" in datasets else "raw"
    koi_names = datasets[any_variant]["koi_name"]
    labels = datasets[any_variant]["y"]
    splits = datasets[any_variant]["split"]

    options = [f"{name}  (true label: {'Planet' if lab == 1 else 'False Alarm'}, split: {sp})"
               for name, lab, sp in zip(koi_names, labels, splits)]
    idx = st.selectbox("Choose a star", range(len(options)), format_func=lambda i: options[i])

    true_label = "Planet" if labels[idx] == 1 else "False Alarm"
    st.write(f"**NASA's own experts say this one really is:** {true_label}")

    col1, col2 = st.columns(2)
    for col, variant, label in zip([col1, col2], ["raw", "folded"],
                                    ["Raw (single pass)", "Phase-folded (all passes stacked)"]):
        with col:
            st.subheader(label)
            if variant in datasets:
                x = datasets[variant]["X"][idx]
                fig_lc = go.Figure(go.Scatter(
                    y=x, mode="lines", line=dict(color=VARIANT_COLOR[variant], width=1.5),
                    hovertemplate="time step=%{x}<br>brightness=%{y:.4f}<extra></extra>",
                ))
                fig_lc.update_layout(
                    xaxis_title="Time", yaxis_title="Brightness",
                    xaxis=dict(showticklabels=False), height=300, margin=dict(t=10, b=40),
                )
                st.plotly_chart(fig_lc, width="stretch")
                if variant in models:
                    prob = float(models[variant].predict(prepare(x)[np.newaxis, :, np.newaxis],
                                                           verbose=0)[0, 0])
                    pred_label = "Planet" if prob >= 0.5 else "False Alarm"
                    correct = "correct" if pred_label == true_label else "incorrect"
                    sure = prob if prob >= 0.5 else 1 - prob
                    st.metric(f"ExoFinder's guess ({variant})", pred_label,
                              delta=f"{sure:.0%} sure ({prob:.0%} chance of a planet) -- {correct}",
                              help="The model scores every star from 0 to 100% for 'is a "
                                   "planet'. Above 50% it says Planet, below it says False "
                                   "Alarm. 'Sure' is how far it is from a coin flip, in "
                                   "whichever direction it leans.")
            else:
                st.info(f"No '{variant}' dataset found.")

    # ---- Star and planet profile, with Mars for comparison ----
    koi_table = load_koi()
    name_sel = str(koi_names[idx])
    if koi_table is not None and name_sel in koi_table.index:
        st.divider()
        st.subheader("About this star and its planet")
        pf, mars = build_profile(koi_table.loc[name_sel]), mars_profile()

        def fmt(v, f="{:.2f}", unit=""):
            return "unknown" if v is None or pd.isna(v) else f.format(v) + unit

        c_star, c_planet = st.columns(2)
        with c_star:
            st.markdown(f"#### The star: class {pf['star_class']}")
            st.caption(pf["star_note"])
            st.write(
                f"**Surface temperature:** {fmt(pf['star_teff'], '{:,.0f}', ' K')}  \n"
                f"**Size:** {fmt(pf['star_radius_sun'], '{:.2f}', 'x the Sun')}  \n"
                f"**Mass:** {fmt(pf['star_mass_sun'], '{:.2f}', 'x the Sun')}"
            )
        with c_planet:
            st.markdown(f"#### The planet: {pf['category'].lower()}")
            st.caption(pf["atmosphere"])
            st.write(
                f"**Size:** {fmt(pf['radius_earth'], '{:.2f}', 'x Earth')}  \n"
                f"**Mass (estimated):** {fmt(pf['mass_earth'], '{:.1f}', 'x Earth')}  \n"
                f"**Density (estimated):** {fmt(pf['density'], '{:.2f}', ' g/cm3')}  \n"
                f"**Distance from its star:** {fmt(pf['orbit_au'], '{:.3f}', ' AU')}  \n"
                f"**One year lasts:** {fmt(pf['period_days'], '{:.1f}', ' days')}  \n"
                f"**Temperature:** {fmt(pf['teq_k'], '{:,.0f}', ' K')}"
            )

        verdict_box = {"Possibly livable": st.success, "Too hot": st.error,
                       "Too cold": st.info}.get(pf["livability"], st.warning)
        verdict_box(f"**How livable? {pf['livability']}.** {pf['livability_why']}")

        st.markdown("#### Compared with Mars")
        rows = [
            ("Size (x Earth)", pf["radius_earth"], mars["radius_earth"], "{:.2f}"),
            ("Mass (x Earth)", pf["mass_earth"], mars["mass_earth"], "{:.2f}"),
            ("Density (g/cm3)", pf["density"], mars["density"], "{:.2f}"),
            ("Distance from its star (AU)", pf["orbit_au"], mars["orbit_au"], "{:.3f}"),
            ("Year length (days)", pf["period_days"], mars["period_days"], "{:,.0f}"),
            ("Temperature (K)", pf["teq_k"], mars["teq_k"], "{:,.0f}"),
            ("Starlight received (x Earth)", pf["insolation"], mars["insolation"], "{:,.2f}"),
            ("Earth similarity (0 to 1)", pf["esi"], mars["esi"], "{:.2f}"),
        ]
        table = pd.DataFrame({
            "": [r[0] for r in rows],
            "This planet": [fmt(r[1], r[3]) for r in rows],
            "Mars": [fmt(r[2], r[3]) for r in rows],
            "Times Mars": [fmt(r[1] / r[2], "{:,.2f}x") if r[1] is not None
                           and not pd.isna(r[1]) and r[2] else "unknown" for r in rows],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            f"Mars: {mars['category'].lower()}, orbiting a class {mars['star_type']} star "
            f"(the Sun). Atmosphere: {mars['atmosphere'].lower()}. Size, orbit, "
            "temperature and star data come straight from NASA's Kepler table. **Mass, "
            "density, atmosphere and livability are estimates from the planet's size**, "
            "because Kepler measures a planet's size, not its weight or air. Earth "
            "similarity compares size, density, gravity and temperature to Earth; it is "
            "not a prediction of life."
        )

    with st.expander("Is this a fair test?"):
        st.write(
            "ExoFinder only ever sees the brightness chart above -- nothing else. It is "
            "never shown NASA's own answer while it's deciding; that answer is only "
            "used afterwards, to check whether ExoFinder got it right."
        )
        st.caption(
            "Technical detail: columns like `koi_score`, `koi_pdisposition`, and "
            "`koi_fpflag_*` (NASA's own vetting outputs) are dropped before "
            "`koi_clean_labels.csv` is even written -- see `config.LEAKAGE_COLUMNS` and "
            "`scripts/02_clean_koi_table.py`."
        )

# ---------------------------------------------------------------------------
# 5 models: five model types on the same stars, and how each does by star type
# ---------------------------------------------------------------------------
with tab_models:
    zoo = load_zoo()
    if not zoo:
        st.info("Run scripts/07_model_zoo.py to compare the models.")
    else:
        st.subheader(f"{len(zoo['models'])} different models, same stars, same test")
        st.write(
            f"We trained {len(zoo['models'])} kinds of model on the same phase-folded stars and scored "
            f"them on the same {zoo['n_test']} stars none of them had seen. Then we split "
            "the score by the host star's type, to see whether one model suits one kind of "
            "star better."
        )
        order = sorted(zoo["models"], key=lambda k: -zoo["models"][k]["accuracy"])
        palette = ["#eb6834", "#2a78d6", "#898781", "#3aa675", "#b05cc7", "#d6b02a", "#5cc7c0", "#c75c7e"]

        fig_m = go.Figure(go.Bar(
            x=[zoo["models"][k]["accuracy"] for k in order], y=order, orientation="h",
            marker_color=[palette[i % 5] for i in range(len(order))],
            text=[f"{zoo['models'][k]['accuracy']:.0%}" for k in order],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig_m.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%",
                                       title="Accuracy on unseen stars"),
                            yaxis=dict(autorange="reversed"), height=300,
                            margin=dict(t=10, b=40, l=10, r=40))
        st.plotly_chart(fig_m, width="stretch")

        st.dataframe(
            pd.DataFrame({
                "Model": order,
                "Accuracy": [f"{zoo['models'][k]['accuracy']:.0%}" for k in order],
                "AUC": [f"{zoo['models'][k]['roc_auc']:.2f}" for k in order],
                "What it is": [zoo["models"][k]["info"] for k in order],
            }),
            hide_index=True, width="stretch",
        )

        st.markdown("#### Which model suits which kind of star?")
        types = [t for t in ["M", "K", "G", "F"] if t in zoo["star_types"]]
        min_n = zoo["min_group"]
        counts = {t: zoo["star_types"][t] for t in types}
        z = [[zoo["models"][m]["by_star_type"][t]["accuracy"] for t in types] for m in order]
        txt = [[f"{zoo['models'][m]['by_star_type'][t]['accuracy']:.0%}"
                + ("*" if counts[t] < min_n else "") for t in types] for m in order]
        fig_h = go.Figure(go.Heatmap(
            z=z, x=[f"{t}  (n={counts[t]})" for t in types], y=order, zmin=0.4, zmax=1.0,
            colorscale="Blues", text=txt, texttemplate="%{text}", showscale=False,
            hovertemplate="%{y} on %{x}: %{z:.0%}<extra></extra>",
        ))
        fig_h.update_layout(yaxis=dict(autorange="reversed"), height=300,
                            xaxis_title="Host star type (n = test stars of that type)",
                            margin=dict(t=10, b=50))
        st.plotly_chart(fig_h, width="stretch")

        best_rows = []
        for t in types:
            best = max(order, key=lambda m: zoo["models"][m]["by_star_type"][t]["accuracy"])
            best_rows.append({
                "Star type": f"{t}: {STAR_CLASS_NOTE[t].split(':')[0]}",
                "Test stars": counts[t],
                "Best model": best,
                "Accuracy": f"{zoo['models'][best]['by_star_type'][t]['accuracy']:.0%}",
                "Reliable?": "Yes" if counts[t] >= min_n else "No, too few stars",
            })
        st.dataframe(pd.DataFrame(best_rows), hide_index=True, width="stretch")

        st.caption(
            f"* fewer than {min_n} test stars of that type, so treat it as a hint only. "
            "Models that sit within a few points of each other are not meaningfully "
            "different on a test set this size. The Deeper CNN's design was adjusted once "
            "after a first attempt scored poorly on the test set, so its score is a little "
            "optimistic."
        )
        with st.expander("What do the star types mean?"):
            for t in ["M", "K", "G", "F"]:
                st.markdown(f"**{t}:** {STAR_CLASS_NOTE[t].split(': ', 1)[-1]}")

# ---------------------------------------------------------------------------
# Tab 3: overall raw vs. folded comparison across the whole test set
# ---------------------------------------------------------------------------
with tab_results:
    if metrics.get("raw") and metrics.get("folded"):
        best_variant, best_auc = None, -1
        for v in ["raw", "folded"]:
            a = metrics.get(v, {}).get("roc_auc")
            if a is not None and a > best_auc:
                best_variant, best_auc = v, a
        if best_variant:
            st.success(
                f"**Quick answer:** the "
                f"{'phase-folded' if best_variant == 'folded' else 'raw (unfolded)'} "
                f"version currently scores better (closer to 1.00 is better, 0.50 is a "
                f"coin flip; it scored {best_auc:.2f})."
            )
        st.write(
            "We trained the **same** model twice on the **same** stars, changing only "
            "whether the brightness data was phase-folded or left raw. This is the "
            "overall result across every star in the held-out test set (stars ExoFinder "
            "never trained on)."
        )

        preds = {v: test_predictions(v) for v in ["raw", "folded"]
                 if v in models and v in datasets}

        st.subheader("Each test star, at a glance")
        st.caption("Check = ExoFinder matched the real answer. Cross = it missed.")
        score_cols = st.columns(2)
        for col, variant, label in zip(score_cols, ["raw", "folded"], ["Raw", "Folded"]):
            m = metrics[variant]
            with col:
                st.markdown(f"**{label}**")
                if variant in preds:
                    y_test, y_prob = preds[variant]
                    y_pred = (y_prob >= 0.5).astype(int)
                    n_correct = int((y_pred == y_test).sum())
                    st.markdown(f"## {n_correct} / {len(y_test)} correct")
                    for truth, guess in zip(y_test, y_pred):
                        truth_word = "Planet" if truth == 1 else "False Alarm"
                        icon = "✔" if truth == guess else "✘"
                        st.write(f"{icon} really a **{truth_word}**")
                st.caption(f"Learned from {m['n_train']} other stars first.")

        with st.expander("See the detailed statistics (for the technically curious)"):
            threshold = st.slider(
                "How sure must ExoFinder be before it calls something a Planet?",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="ExoFinder outputs a probability from 0 to 1 for every star. This "
                     "slider sets the cutoff: below it, 'False Alarm'; at or above it, "
                     "'Planet'. Drag it and watch everything below update live.",
            )

            cols = st.columns(2)
            for col, variant, label in zip(cols, ["raw", "folded"],
                                            ["A: Raw (no phase folding)", "B: Phase folded"]):
                m = metrics[variant]
                with col:
                    st.subheader(label)
                    if variant in preds:
                        y_test, y_prob = preds[variant]
                        y_pred = (y_prob >= threshold).astype(int)
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                    else:
                        acc, prec, rec, f1 = m["accuracy"], m["precision"], m["recall"], m["f1"]
                    st.metric("Overall correct", f"{acc:.0%}",
                              help="Out of every star shown to ExoFinder, what fraction did "
                                   "it label correctly (Planet or False Alarm)?")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trustworthy 'Planet' calls", f"{prec:.0%}",
                              help="When ExoFinder says 'Planet', how often is it actually "
                                   "right? Low = lots of false alarms get called planets.")
                    c2.metric("Planets it catches", f"{rec:.0%}",
                              help="Out of all the REAL planets in this set, what "
                                   "fraction did ExoFinder actually find?")
                    c3.metric("Balance of the two", f"{f1:.0%}",
                              help="A single score (F1) that balances the two metrics to "
                                   "the left -- high only when both are good.")
                    st.caption(f"Trained on {m['n_train']} stars, tested on {m['n_test']} "
                               "stars it never saw during training.")
            n_train = (metrics.get("raw", {}).get("n_train")
                       or metrics.get("folded", {}).get("n_train"))
            if n_train and n_train < 50:
                st.warning(
                    f"This run trained on only {n_train} examples -- a smoke-test scale, "
                    "used to prove the pipeline works end-to-end. Treat these specific "
                    "numbers as a proof of concept, not a statistically reliable result "
                    "-- that needs a much larger download."
                )
            elif n_train:
                st.info(f"This run trained on {n_train} examples. See PROJECT_LOG.md for "
                        "the exact star count and split sizes.")

            st.subheader("How good is each model, at every possible confidence cutoff?")
            st.caption(
                "The closer a line hugs the top-left corner, the better that model is "
                "overall -- the diagonal would mean no better than guessing randomly. "
                "The dot shows where the slider above is currently set."
            )
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                               line=dict(color=MUTED, width=1, dash="dot"))
            for variant, label in [("raw", "Raw"), ("folded", "Folded")]:
                if variant in preds:
                    y_test, y_prob = preds[variant]
                    if len(np.unique(y_test)) > 1:
                        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                        auc = metrics.get(variant, {}).get("roc_auc")
                        name = f"{label} (AUC={auc:.2f})" if auc is not None else label
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines", name=name,
                            line=dict(color=VARIANT_COLOR[variant], width=2),
                            customdata=thresholds,
                            hovertemplate="FPR=%{x:.2f}<br>TPR=%{y:.2f}<br>"
                                          "threshold=%{customdata:.2f}<extra>"
                                          + name + "</extra>",
                        ))
                        y_pred_t = (y_prob >= threshold).astype(int)
                        cm_t = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
                        tn, fp, fn, tp = cm_t.ravel()
                        fpr_op = fp / (fp + tn) if (fp + tn) else 0.0
                        tpr_op = tp / (tp + fn) if (tp + fn) else 0.0
                        fig_roc.add_trace(go.Scatter(
                            x=[fpr_op], y=[tpr_op], mode="markers", showlegend=False,
                            marker=dict(color=VARIANT_COLOR[variant], size=12,
                                        line=dict(color="white", width=2)),
                            hovertemplate=f"{label} at threshold={threshold:.2f}<br>"
                                          "FPR=%{x:.2f}<br>TPR=%{y:.2f}<extra></extra>",
                        ))
            fig_roc.update_layout(
                xaxis_title="False positive rate", yaxis_title="True positive rate",
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                height=420, margin=dict(t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_roc, width="stretch")

            st.subheader("Where exactly did each model get it right or wrong?")
            st.caption(
                "The diagonal is where ExoFinder's guess matched reality -- the other two "
                "boxes are its two different kinds of mistakes."
            )
            cm_cols = st.columns(2)
            for col, variant, label in zip(cm_cols, ["raw", "folded"], ["Raw", "Folded"]):
                with col:
                    if variant in preds:
                        y_test, y_prob = preds[variant]
                        y_pred = (y_prob >= threshold).astype(int)
                        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                        fig_cm = go.Figure(go.Heatmap(
                            z=cm, x=["False Alarm", "Planet"], y=["False Alarm", "Planet"],
                            colorscale="Blues", showscale=False,
                            text=cm, texttemplate="%{text}", textfont=dict(size=16),
                            hovertemplate="It actually was: %{y}<br>ExoFinder guessed: %{x}<br>"
                                          "count=%{z}<extra></extra>",
                        ))
                        fig_cm.update_layout(
                            title=label, xaxis_title="What ExoFinder guessed",
                            yaxis_title="What it actually was",
                            yaxis=dict(autorange="reversed"), height=340,
                            margin=dict(t=40, b=20),
                        )
                        st.plotly_chart(fig_cm, width="stretch")
    else:
        st.info("Train both variants (scripts/05_train_experiment.py --variant raw / "
                "folded) to see results here.")

st.divider()
st.caption(
    "ExoFinder  |  Ishaani Dongre and Adwitta Bhardwaj  |  ATLAS, UGDX School of "
    "Technology, B.Tech CS (AI & ML)  |  Data: NASA Kepler / Exoplanet Archive"
)
