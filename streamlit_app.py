"""
Star Tracker — Streamlit Visual Debugger
Run: python -m streamlit run streamlit_app.py
"""

import os, sys, io, tempfile, random
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import config

# ── WCS camera auto-detection ─────────────────────────────────────────────────
def camera_from_fits(fits_image):
    """
    Build a StarTrackerCamera whose focal length matches the FITS WCS plate-scale.
    Falls back to the global config if no WCS is present.
    SkyView images are ~1 deg FOV / 512 px  →  f ≈ 29 000 px.
    Default synthetic images are 20 deg FOV / 1024 px → f ≈ 2 900 px.
    Mixing these up makes every angular distance 10× wrong.
    """
    from camera.camera_model import StarTrackerCamera
    h = fits_image.header
    H, W = fits_image.data.shape

    # Try to read plate scale (deg/px) from WCS
    plate_scale = None
    for key in ('CDELT1', 'CD1_1'):
        val = h.get(key, None)
        if val is not None:
            plate_scale = abs(float(val))   # degrees per pixel
            break

    if plate_scale and plate_scale > 0:
        fov_deg = plate_scale * max(W, H)
        focal_px = (W / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        crpix1 = float(h.get('CRPIX1', W / 2.0))
        crpix2 = float(h.get('CRPIX2', H / 2.0))
        # also tighten the triangle-matching pair window to the actual FOV
        config.TRIANGLE_MAX_PAIR_ANGLE_DEG = min(fov_deg * 1.2, 20.0)
        config.TRIANGLE_MIN_PAIR_ANGLE_DEG = plate_scale * 2
        camera = StarTrackerCamera(
            focal_length=focal_px,
            cx=crpix1, cy=crpix2,
            width=W, height=H,
            k1=0.0, k2=0.0,
        )
        return camera, fov_deg, plate_scale
    else:
        return StarTrackerCamera(), config.FOV_DEG, None

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Star Tracker Visual Debugger",
    page_icon="⭐", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #161b22; }
.block-container { padding-top: 1.2rem; }
.panel-title {
    color: #58a6ff; font-size: 13px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 6px 0 4px 0; border-bottom: 1px solid #30363d;
    margin-bottom: 8px;
}
.status-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
}
.terminal-box {
    background: #0d1117; border: 1px solid #3fb950;
    border-radius: 8px; padding: 14px 16px; font-family: 'Courier New', monospace;
    font-size: 12px; color: #3fb950; line-height: 1.7;
    white-space: pre-wrap; margin-top: 8px;
}
.badge-tri  { background:#1f6feb; color:#fff; padding:2px 10px; border-radius:12px; font-size:11px; }
.badge-gnn  { background:#8b5cf6; color:#fff; padding:2px 10px; border-radius:12px; font-size:11px; }
.badge-fail { background:#f85149; color:#fff; padding:2px 10px; border-radius:12px; font-size:11px; }
</style>
""", unsafe_allow_html=True)


# ── resource caching ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading catalogue & triangle DB…")
def load_resources():
    from catalogue.hipparcos import load_catalogue
    from modules.m5_triangle_match import load_triangle_database
    return load_catalogue(config.CATALOGUE_FILE), \
           load_triangle_database(config.TRIANGLE_DB_FILE)


# ── pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(fits_path, mag_lim, tol, occlusion_frac=0.0, force_gnn=False):
    """
    Run full pipeline, optionally removing `occlusion_frac` of detected stars
    to simulate occlusion.  Returns result dict with all intermediate data.
    FOV / focal-length are now auto-detected from the FITS WCS header.
    """
    config.HIPPARCOS_MAG_LIMIT          = mag_lim
    config.TRIANGLE_ANGLE_TOLERANCE_DEG = tol

    from modules.m1_image_input import load_fits_image
    from modules.m2_preprocessing import preprocess_image
    from modules.m3_star_detection import detect_stars
    from modules.m4_pixel_to_vector import convert_pixels_to_vectors
    from modules.m5_triangle_match import match_stars
    from modules.m7_quest import quest_from_matches
    from modules.m9_output import format_output
    from validation.ground_truth import extract_ground_truth

    out = {}
    catalogue, tri_db = load_resources()

    fits = load_fits_image(fits_path)
    camera, detected_fov, plate_scale = camera_from_fits(fits)
    out['raw']         = fits.data.copy()
    out['detected_fov'] = detected_fov
    out['plate_scale']  = plate_scale
    out['camera_repr']  = repr(camera)
    out['ground_truth'] = extract_ground_truth(fits)

    cleaned = preprocess_image(fits.data)
    out['preprocessed'] = cleaned

    stars = detect_stars(cleaned)
    out['all_detected'] = stars[:]   # keep full list for panel 1

    # ── occlusion simulation ──────────────────────────────────────────────────
    if occlusion_frac > 0:
        n_keep = max(config.MIN_STARS_FOR_MATCH,
                     int(len(stars) * (1 - occlusion_frac)))
        # randomly remove stars (but keep brightest few)
        keep = stars[:3] + random.sample(stars[3:], max(0, n_keep - 3))
        stars = sorted(keep, key=lambda s: s.instrumental_mag)
    out['detected_stars'] = stars
    out['n_detected']     = len(stars)
    out['occluded']       = occlusion_frac > 0

    if len(stars) < config.MIN_STARS_FOR_MATCH:
        out['error'] = f"Only {len(stars)} stars after occlusion (need ≥{config.MIN_STARS_FOR_MATCH})"
        return out

    star_vecs = convert_pixels_to_vectors(stars, camera)
    out['star_vectors'] = star_vecs

    if not force_gnn:
        matches, success = match_stars(star_vecs, tri_db, catalogue)
        out['triangle_success'] = success
    else:
        matches, success = [], False
        out['triangle_success'] = False
        
    out['gnn_used']         = False
    out['gnn_success']      = False

    if not success:
        try:
            from modules.m6_gnn import gnn_identify_stars, TORCH_AVAILABLE
            if TORCH_AVAILABLE:
                matches, success = gnn_identify_stars(stars, camera, catalogue, tri_db)
                out['gnn_used']    = True
                out['gnn_success'] = success
        except Exception as e:
            out['gnn_error'] = str(e)

    out['matches']       = matches
    out['match_success'] = success
    out['catalogue']     = catalogue

    if not success:
        out['error'] = "Star identification failed (both Triangle + GNN)"
        return out

    try:
        quest_res = quest_from_matches(star_vecs, matches, catalogue)
        out['quest'] = quest_res
        gt_q = out['ground_truth']['quaternion'] if out['ground_truth'] else None
        out['attitude'] = format_output(
            quest_res,
            method='gnn' if out['gnn_used'] else 'triangle',
            ground_truth_q=gt_q
        )
    except Exception as e:
        out['error'] = f"Attitude estimation failed: {e}"
        out['quest'] = None
        out['attitude'] = None

    return out


# ── figure builders ───────────────────────────────────────────────────────────
def _dark_fig(nrows=1, ncols=1, figsize=(5, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    for ax in (np.ravel(axes) if hasattr(axes, '__len__') else [axes]):
        ax.set_facecolor('#0d1117')
    return fig, axes


def _imshow_ax(ax, data):
    vmin, vmax = np.percentile(data, [1, 99])
    ax.imshow(data, cmap='inferno', vmin=vmin, vmax=vmax, origin='upper')
    ax.axis('off')


def fig_panel1(pre, stars):
    """Panel 1: image + cyan detection circles + magnitude labels."""
    fig, ax = _dark_fig(figsize=(5, 5))
    _imshow_ax(ax, pre)
    for s in stars:
        c = mpatches.Circle((s.u, s.v), radius=8, linewidth=1.2,
                             edgecolor='#00eeff', facecolor='none')
        ax.add_patch(c)
        ax.text(s.u + 10, s.v, f"{s.instrumental_mag:.1f}",
                color='#00eeff', fontsize=4.5, va='center',
                fontfamily='monospace')
    ax.set_title(f"Detection  ({len(stars)} stars)", color='#8b949e',
                 fontsize=9, pad=3)
    fig.tight_layout(pad=0)
    return fig


def fig_panel2(pre, stars, matches):
    """Panel 2: image + coloured match circles + HIP ID labels."""
    matched_idx = {m.star_index for m in matches}
    fig, ax = _dark_fig(figsize=(5, 5))
    _imshow_ax(ax, pre)
    for i, s in enumerate(stars):
        col = '#3fb950' if i in matched_idx else '#f85149'
        c = mpatches.Circle((s.u, s.v), radius=9, linewidth=1.3,
                             edgecolor=col, facecolor='none')
        ax.add_patch(c)
    for m in matches:
        s = stars[m.star_index]
        ax.text(s.u + 10, s.v, f"HIP {m.hip_id}",
                color='#f0a500', fontsize=4.5, va='center',
                fontfamily='monospace')
    ax.set_title(f"Identification  ({len(matches)} matched)",
                 color='#8b949e', fontsize=9, pad=3)
    fig.tight_layout(pad=0)
    return fig


def fig_globe(attitude):
    """3-D boresight globe (Plotly)."""
    ra_r  = np.radians(attitude.ra_boresight)
    dec_r = np.radians(attitude.dec_boresight)
    bx = np.cos(dec_r) * np.cos(ra_r)
    by = np.cos(dec_r) * np.sin(ra_r)
    bz = np.sin(dec_r)

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones(40), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=sx, y=sy, z=sz,
                             colorscale=[[0,'#161b22'],[1,'#21262d']],
                             opacity=0.4, showscale=False))
    for dec_d in [-60,-30,0,30,60]:
        d  = np.radians(dec_d)
        ra = np.linspace(0, 2*np.pi, 80)
        fig.add_trace(go.Scatter3d(
            x=np.cos(d)*np.cos(ra), y=np.cos(d)*np.sin(ra),
            z=np.full(80, np.sin(d)),
            mode='lines', line=dict(color='#30363d', width=1),
            showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[0, bx], y=[0, by], z=[0, bz],
        mode='lines+markers',
        line=dict(color='#58a6ff', width=6),
        marker=dict(size=[2,12], color=['#58a6ff','#f85149']),
        name='Boresight'))
    fig.update_layout(
        paper_bgcolor='#0d1117',
        scene=dict(bgcolor='#0d1117',
                   xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=0, b=0), height=280)
    return fig


def terminal_text(attitude, method_label):
    """Build the green terminal-style attitude string."""
    q = attitude.quaternion
    lines = [
        f"★ STAR TRACKER — ATTITUDE OUTPUT",
        f"{'─'*38}",
        f"  Method   : {method_label}",
        f"  Stars    : {attitude.n_stars_used}",
        f"",
        f"  Quaternion (scalar-first):",
        f"  q0 = {q[0]:+.8f}",
        f"  q1 = {q[1]:+.8f}",
        f"  q2 = {q[2]:+.8f}",
        f"  q3 = {q[3]:+.8f}",
        f"",
        f"  Euler Angles (ZYX):",
        f"  Roll  = {attitude.roll:+.4f} °",
        f"  Pitch = {attitude.pitch:+.4f} °",
        f"  Yaw   = {attitude.yaw:+.4f} °",
        f"",
        f"  Boresight:",
        f"  RA  = {attitude.ra_boresight:.4f} °",
        f"  Dec = {attitude.dec_boresight:+.4f} °",
        f"",
        f"  Residual = {attitude.residual_arcsec:.2f} arcsec RMS",
    ]
    if attitude.angular_error_arcsec is not None:
        lines.append(f"  GT Error = {attitude.angular_error_arcsec:.2f} arcsec")
    lines.append(f"{'─'*38}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⭐ Star Tracker")
    st.markdown("*Visual Pipeline Debugger*")
    st.divider()

    source = st.selectbox("Image Source", [
        "Demo — Orion (synthetic)",
        "Upload FITS (real SkyView etc.)",
    ])
    fits_path = None

    if source == "Upload FITS (real SkyView etc.)":
        up = st.file_uploader("Upload .fits", type=["fits","fit","fts"])
        if up:
            tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
            tmp.write(up.read()); tmp.flush()
            fits_path = tmp.name
    else:
        fits_path = os.path.join(config.DATA_DIR, "demo_image.fits")

    st.divider()
    st.markdown("#### ⚙ Pipeline Settings")
    fov_v  = st.slider("FOV (°)",            5.0, 40.0, float(config.FOV_DEG), 0.5)
    mag_v  = st.slider("Mag limit",          4.0,  8.5, float(config.HIPPARCOS_MAG_LIMIT), 0.1)
    tol_v  = st.slider("Angle tolerance (°)", 0.001, 0.020, 0.010, 0.001, format="%.3f")

    st.divider()
    st.markdown("#### 🎬 Demo Scenarios")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        run_normal   = st.button("🌟 Normal",   use_container_width=True, type="primary")
    with col_b:
        run_occluded = st.button("🌑 Occlusion", use_container_width=True)
    with col_c:
        run_gnn_only = st.button("🧠 GNN Only", use_container_width=True)

    occ_frac = st.slider("Occlusion fraction", 0.0, 0.8, 0.5, 0.05,
                         help="Fraction of detected stars removed before matching")
    st.caption("Normal: triangle matching\nOcclusion/GNN Only: forces GNN fallback")


# ── ensure demo image exists ──────────────────────────────────────────────────
if fits_path == os.path.join(config.DATA_DIR, "demo_image.fits") \
        and not os.path.exists(fits_path):
    with st.spinner("Generating demo image (Orion's Belt)…"):
        from catalogue.hipparcos import load_catalogue
        from synthetic.image_generator import generate_synthetic_image, save_synthetic_image
        cat = load_catalogue(config.CATALOGUE_FILE)
        res = generate_synthetic_image(cat, ra_deg=84.0, dec_deg=-1.0)
        save_synthetic_image(res, fits_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# ⭐ Star Tracker — Visual Debugger")

run_any = run_normal or run_occluded or run_gnn_only
if not run_any:
    st.info("Choose a scenario in the sidebar and click **🌟 Normal**, **🌑 Occlusion**, or **🧠 GNN Only**.")
    st.stop()

if fits_path is None or not os.path.exists(fits_path):
    st.error("No FITS file. Upload one or select Demo.")
    st.stop()

occ = occ_frac if run_occluded else 0.0
scenario_name = "🧠 GNN Only" if run_gnn_only else ("🌑 Occlusion Simulation" if run_occluded else "🌟 Normal Sky")

with st.spinner(f"Running pipeline — {scenario_name}…"):
    result = run_pipeline(fits_path, mag_v, tol_v, occlusion_frac=occ, force_gnn=run_gnn_only)

# ── determine method label ────────────────────────────────────────────────────
if result.get('gnn_used'):
    method_label = "GNN Recovery"
    badge = '<span class="badge-gnn">GNN RECOVERY</span>'
elif result.get('triangle_success'):
    method_label = "Triangle Matching"
    badge = '<span class="badge-tri">TRIANGLE MATCHING</span>'
else:
    method_label = "FAILED"
    badge = '<span class="badge-fail">FAILED</span>'

attitude  = result.get('attitude')
matches   = result.get('matches', [])
stars     = result.get('detected_stars', [])
pre       = result.get('preprocessed')
raw       = result.get('raw')
catalogue = result.get('catalogue')

# ── camera / FOV info banner ─────────────────────────────────────────────────
fov_info = result.get('detected_fov', '?')
ps_info  = result.get('plate_scale')
cam_msg  = (f"📡 Camera auto-configured from WCS: **FOV ≈ {fov_info:.3f}°**, "
            f"plate scale = {ps_info*3600:.2f} arcsec/px"
            if ps_info else
            f"📡 No WCS in FITS — using default 20° FOV synthetic camera")
if run_gnn_only:
    st.info(f"**GNN Only** — Triangle matching bypassed. "
            f"GNN {'succeeded' if result.get('gnn_success') else 'failed'}.")
elif run_occluded:
    st.warning(f"**Occlusion Simulation** — {int(occ*100)}% of stars removed. "
               f"Triangle matching {'succeeded' if result.get('triangle_success') else 'failed'}. "
               f"GNN {'recovered' if result.get('gnn_used') and result.get('gnn_success') else 'attempted'}.")
else:
    st.success(f"**Normal Scenario** — {result.get('n_detected',0)} stars detected.")
st.caption(cam_msg)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# THREE PANELS
# ══════════════════════════════════════════════════════════════════════════════
p1, p2, p3 = st.columns(3)

# ── Panel 1: Detection ────────────────────────────────────────────────────────
with p1:
    st.markdown('<div class="panel-title">📷 Panel 1 — Star Detection</div>',
                unsafe_allow_html=True)
    if pre is not None and stars:
        st.pyplot(fig_panel1(pre, stars), use_container_width=True)
        st.caption(f"**{len(stars)}** stars detected  |  "
                   f"{'⚠ occluded' if result.get('occluded') else '✓ full field'}")
    elif 'error' in result:
        st.error(result['error'])

# ── Panel 2: Identification ───────────────────────────────────────────────────
with p2:
    st.markdown('<div class="panel-title">🔗 Panel 2 — Star Identification</div>',
                unsafe_allow_html=True)
    if pre is not None and matches and stars:
        st.pyplot(fig_panel2(pre, stars, matches), use_container_width=True)

        # confidence from matches
        conf_vals = [m.confidence for m in matches if m.confidence > 0]
        avg_conf  = np.mean(conf_vals) * 100 if conf_vals else 0

        st.markdown(f"""
<div class="status-box">
  <b>Method:</b> {badge}<br>
  <b>Stars matched:</b> {len(matches)} / {len(stars)}<br>
  <b>Avg confidence:</b> {avg_conf:.1f} %<br>
  <b>Triangle OK:</b> {'✅' if result.get('triangle_success') else '❌'} &nbsp;
  <b>GNN used:</b> {'✅' if result.get('gnn_used') else '—'}
</div>""", unsafe_allow_html=True)

        # Star match table (compact)
        if catalogue is not None:
            rows = [{"HIP": m.hip_id,
                     "RA°": round(float(catalogue['ra_deg'][m.catalogue_index]),3),
                     "Dec°": round(float(catalogue['dec_deg'][m.catalogue_index]),3),
                     "Vmag": round(float(catalogue['vmag'][m.catalogue_index]),2),
                     "Conf": round(m.confidence, 3)}
                    for m in matches]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=160)
    elif 'error' in result:
        st.error(result['error'])
        st.markdown(f'<div class="status-box"><b>Method:</b> {badge}</div>',
                    unsafe_allow_html=True)

# ── Panel 3: Attitude Output ──────────────────────────────────────────────────
with p3:
    st.markdown('<div class="panel-title">🛰 Panel 3 — Attitude Output</div>',
                unsafe_allow_html=True)
    if attitude is not None:
        txt = terminal_text(attitude, method_label)
        st.markdown(f'<div class="terminal-box">{txt}</div>', unsafe_allow_html=True)

        st.plotly_chart(fig_globe(attitude), use_container_width=True)
    else:
        st.error("No attitude solution computed.")
        if 'error' in result:
            st.caption(result['error'])

# ── Step-by-step expander (bonus) ────────────────────────────────────────────
with st.expander("🔬 Step-by-step pipeline view", expanded=False):
    t_raw, t_pre, t_det = st.tabs(["Raw Image", "Preprocessed", "Detection Detail"])
    with t_raw:
        if raw is not None:
            fig_r, ax_r = _dark_fig(figsize=(5,5))
            _imshow_ax(ax_r, raw)
            ax_r.set_title("Raw FITS", color='#8b949e', fontsize=9)
            st.pyplot(fig_r, use_container_width=True)
    with t_pre:
        if pre is not None and raw is not None:
            fig_p, axes_p = _dark_fig(1, 2, figsize=(9,4))
            _imshow_ax(axes_p[0], raw); axes_p[0].set_title("Raw", color='#8b949e', fontsize=9)
            _imshow_ax(axes_p[1], pre); axes_p[1].set_title("Background-subtracted",
                                                              color='#8b949e', fontsize=9)
            fig_p.tight_layout(pad=0.5)
            st.pyplot(fig_p, use_container_width=True)
    with t_det:
        if stars:
            mags = [s.instrumental_mag for s in stars]
            snrs = [s.snr for s in stars]
            fig_h, axh = _dark_fig(1, 2, figsize=(9, 3))
            for ax in axh: ax.set_facecolor('#161b22')
            axh[0].hist(mags, bins=20, color='#58a6ff', edgecolor='#0d1117')
            axh[0].set_title("Magnitude dist.", color='#8b949e', fontsize=9)
            axh[1].hist(snrs, bins=20, color='#3fb950', edgecolor='#0d1117')
            axh[1].set_title("SNR dist.", color='#8b949e', fontsize=9)
            for ax in axh:
                ax.tick_params(colors='#8b949e')
                for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
            fig_h.tight_layout(pad=0.4)
            st.pyplot(fig_h, use_container_width=True)
