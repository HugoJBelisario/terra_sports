import zlib
import hashlib
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "terra_sports.svg"


def get_page_icon():
    """
    Use the first favicon-like asset we find so branding can be updated
    by replacing a file in /assets without touching code again.
    """
    candidate_names = (
        "favicon.png",
        "favicon.ico",
        "favicon.jpg",
        "favicon.jpeg",
        "favicon.svg",
    )

    for name in candidate_names:
        icon_path = ASSETS_DIR / name
        if icon_path.exists():
            return str(icon_path)

    return None

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Terra Sports Dashboard",
    page_icon=get_page_icon(),
    layout="wide",
)

# --------------------------------------------------
# Timing constants
# --------------------------------------------------
KINEMATIC_FPS = 250
MS_PER_FRAME = 1000 / KINEMATIC_FPS  # 4 ms per frame
REPORT_METRIC_LOGIC_VERSION = "report_metrics_v2_br_plus4_normalized"

import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from dotenv import load_dotenv
from db.connection import get_connection

def login():
    raw_users = st.secrets["auth"]["users"]
    users = {str(username).strip(): str(password) for username, password in raw_users.items()}

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown(
        """
        <style>
        .login-shell {
            padding-top: 2rem;
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stTextInput"] input,
        div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
            width: 100%;
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: #BD0318;
            border: 1px solid #BD0318;
            color: #FFFFFF;
        }

        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: #A10214;
            border-color: #A10214;
            color: #FFFFFF;
        }

        div[data-testid="stButton"] button[kind="secondary"]:focus:not(:active) {
            color: #FFFFFF;
            border-color: #BD0318;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, center_col, right_col = st.columns([1.2, 1, 1.2])
    with center_col:
        st.markdown('<div class="login-shell">', unsafe_allow_html=True)
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_clicked = st.button("Log in", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if login_clicked:
        submitted_username = username.strip()
        submitted_password = password
        if submitted_username in users and users[submitted_username] == submitted_password:
            st.session_state.authenticated = True
            st.session_state.user = submitted_username
            st.rerun()
        else:
            st.error("Invalid username or password")

    return False


if not login():
    st.stop()

# --- Safe color resolver for named/hex colors ---
def to_rgba(color, alpha=0.35):
    """
    Convert a hex color (#RRGGBB) or basic named color to an rgba() string
    without requiring matplotlib.
    """
    named_colors = {
        "blue": (31, 119, 180),
        "orange": (255, 127, 14),
        "green": (44, 160, 44),
        "red": (214, 39, 40),
        "purple": (148, 103, 189),
        "brown": (140, 86, 75),
        "pink": (227, 119, 194),
        "gray": (127, 127, 127),
        "olive": (188, 189, 34),
        "teal": (23, 190, 207),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "deeppink": (255, 20, 147),
        "dodgerblue": (30, 144, 255),
        "crimson": (220, 20, 60),
        "darkorange": (255, 140, 0),
        "charcoal": (55, 65, 81),
        "darkblue": (0, 0, 139),
        "darkred": (139, 0, 0),
        "darkgreen": (0, 100, 0)
    }

    if isinstance(color, str):
        color = color.lower()

        # Hex color
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"

        # Named color
        if color in named_colors:
            r, g, b = named_colors[color]
            return f"rgba({r},{g},{b},{alpha})"

    # Fallback (neutral gray)
    return f"rgba(150,150,150,{alpha})"


def rel_frame_to_ms(rel_frame):
    return int(round(rel_frame * MS_PER_FRAME))


def ms_to_rel_frame(milliseconds):
    return int(round(milliseconds / MS_PER_FRAME))


SEGMENT_DISPLAY_NAMES = {
    "Pelvis": "Pelvis Rotation",
    "Torso": "Torso Rotation",
    "Elbow": "Elbow Extension",
    "Shoulder": "Shoulder Internal Rotation",
    "Shoulder IR": "Shoulder Internal Rotation",
}


def segment_display_name(label):
    return SEGMENT_DISPLAY_NAMES.get(label, label)


def add_event_iqr_band(fig, event_frames, color, show_band, opacity=0.10):
    if not show_band or not event_frames:
        return

    event_q1_frame = int(np.percentile(event_frames, 25))
    event_q3_frame = int(np.percentile(event_frames, 75))
    event_start_ms = rel_frame_to_ms(event_q1_frame)
    event_end_ms = rel_frame_to_ms(event_q3_frame)

    if event_start_ms == event_end_ms:
        return

    fig.add_vrect(
        x0=event_start_ms,
        x1=event_end_ms,
        fillcolor=color,
        opacity=opacity,
        layer="below",
        line_width=0
    )

load_dotenv()

@st.cache_data(ttl=300)
def get_all_pitchers():
    """
    Returns all athlete names from the athletes table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT athlete_name
                FROM athletes
                WHERE athlete_name IS NOT NULL
                ORDER BY athlete_name
            """)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# --------------------------------------------------
# Velocity Bounds Helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_velocity_bounds(athlete_name, selected_dates):
    """
    Returns (min_velocity, max_velocity) for the selected pitcher and dates.
    Assumes pitch velocity is stored as `pitch_velo` on the takes table.
    """
    if athlete_name is None:
        return None, None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if not selected_dates or "All Dates" in selected_dates:
                cur.execute("""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.pitch_velo IS NOT NULL
                """, (athlete_name,))
            else:
                placeholders = ",".join(["%s"] * len(selected_dates))
                cur.execute(f"""
                    SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.take_date IN ({placeholders})
                      AND t.pitch_velo IS NOT NULL
                """, (athlete_name, *selected_dates))

            row = cur.fetchone()
            return row if row else (None, None)
    finally:
        conn.close()

@st.cache_data(ttl=300, show_spinner=False)
def get_control_group_take_pool(handedness_filter):
    """
    Returns control-group candidates from all takes in the database, optionally
    filtered by pitcher handedness.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            params = []
            handedness_clause = ""
            if handedness_filter in ("R", "L"):
                handedness_clause = "AND a.handedness = %s"
                params.append(handedness_filter)

            cur.execute(f"""
                WITH ids AS (
                    SELECT
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_CGVel') AS cat_kk_cgvel,
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_ProxEndPos') AS cat_kk_prox_pos,
                        (SELECT category_id FROM categories WHERE category_name = 'KINETIC_KINEMATIC_DistEndPos') AS cat_kk_dist_endpos,
                        (SELECT segment_id FROM segments WHERE segment_name = 'LHA') AS seg_hand_l,
                        (SELECT segment_id FROM segments WHERE segment_name = 'RHA') AS seg_hand_r,
                        (SELECT segment_id FROM segments WHERE segment_name = 'LAR') AS seg_arm_l,
                        (SELECT segment_id FROM segments WHERE segment_name = 'RAR') AS seg_arm_r
                ),
                candidate_takes AS (
                    SELECT
                        t.take_id,
                        t.pitch_velo,
                        a.athlete_name,
                        a.handedness,
                        CASE WHEN a.handedness = 'L' THEN i.seg_hand_l ELSE i.seg_hand_r END AS seg_hand_dom,
                        CASE WHEN a.handedness = 'L' THEN i.seg_arm_l ELSE i.seg_arm_r END AS seg_arm_dom,
                        i.seg_hand_l,
                        i.seg_hand_r,
                        i.cat_kk_cgvel,
                        i.cat_kk_prox_pos,
                        i.cat_kk_dist_endpos
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    CROSS JOIN ids i
                    WHERE t.pitch_velo IS NOT NULL
                      AND t.throw_type = 'Mound'
                      AND a.handedness IN ('R', 'L')
                      {handedness_clause}
                      AND EXISTS (
                          SELECT 1
                          FROM time_series_data d
                          WHERE d.take_id = t.take_id
                      )
                ),
                hand_vel AS (
                    SELECT
                        t.take_id,
                        t.pitch_velo,
                        t.handedness,
                        d.frame,
                        d.x_data,
                        LAG(d.x_data) OVER (
                            PARTITION BY t.take_id
                            ORDER BY d.frame
                        ) AS prev_x
                    FROM time_series_data d
                    JOIN candidate_takes t ON t.take_id = d.take_id
                    WHERE d.category_id = t.cat_kk_cgvel
                      AND d.segment_id IN (t.seg_hand_l, t.seg_hand_r)
                      AND d.x_data IS NOT NULL
                ),
                cross_15 AS (
                    SELECT DISTINCT ON (take_id)
                        take_id,
                        frame AS cross_frame
                    FROM hand_vel
                    WHERE x_data >= 15
                      AND (prev_x < 15 OR prev_x IS NULL)
                    ORDER BY take_id, frame
                ),
                positive_phase AS (
                    SELECT
                        h.take_id,
                        h.frame,
                        h.x_data,
                        LAG(h.x_data, 5) OVER (
                            PARTITION BY h.take_id
                            ORDER BY h.frame
                        ) AS prev_5_x,
                        LEAD(h.x_data) OVER (
                            PARTITION BY h.take_id
                            ORDER BY h.frame
                        ) AS next_x
                    FROM hand_vel h
                    JOIN cross_15 c ON c.take_id = h.take_id
                    WHERE h.frame >= c.cross_frame
                      AND h.x_data > 0
                ),
                per_take_br AS (
                    SELECT DISTINCT ON (p.take_id)
                        p.take_id,
                        p.frame AS br_frame
                    FROM positive_phase p
                    WHERE p.next_x IS NOT NULL
                      AND p.prev_5_x IS NOT NULL
                      AND p.x_data > p.next_x
                      AND p.x_data > p.prev_5_x
                    ORDER BY p.take_id, p.frame ASC
                ),
                per_take_arm_points AS (
                    SELECT
                        br.take_id,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.x_data END) AS x_arm,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.y_data END) AS y_arm,
                        MAX(CASE WHEN d_arm.segment_id = t.seg_arm_dom THEN d_arm.z_data END) AS z_arm,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.x_data END) AS x_hand,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.y_data END) AS y_hand,
                        MAX(CASE WHEN d_hand.segment_id = t.seg_hand_dom THEN d_hand.z_data END) AS z_hand
                    FROM per_take_br br
                    JOIN candidate_takes t ON t.take_id = br.take_id
                    LEFT JOIN time_series_data d_arm
                        ON d_arm.take_id = br.take_id
                       AND d_arm.frame = br.br_frame
                       AND d_arm.category_id = t.cat_kk_prox_pos
                       AND d_arm.segment_id = t.seg_arm_dom
                    LEFT JOIN time_series_data d_hand
                        ON d_hand.take_id = br.take_id
                       AND d_hand.frame = br.br_frame
                       AND d_hand.category_id = t.cat_kk_dist_endpos
                       AND d_hand.segment_id = t.seg_hand_dom
                    GROUP BY br.take_id
                ),
                per_take_arm_angle AS (
                    SELECT
                        p.take_id,
                        CASE
                            WHEN p.x_arm IS NULL OR p.x_hand IS NULL THEN NULL
                            ELSE DEGREES(
                                ATAN2(
                                    (p.z_hand - p.z_arm),
                                    NULLIF(SQRT(
                                        POWER(p.x_hand - p.x_arm, 2) +
                                        POWER(p.y_hand - p.y_arm, 2)
                                    ), 0)
                                )
                            )
                        END AS arm_slot_deg
                    FROM per_take_arm_points p
                )
                SELECT
                    t.take_id,
                    t.pitch_velo,
                    t.athlete_name,
                    t.handedness,
                    a.arm_slot_deg
                FROM candidate_takes t
                LEFT JOIN per_take_arm_angle a ON a.take_id = t.take_id
                ORDER BY t.take_id
            """, tuple(params))
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_session_dates_for_pitcher(athlete_name):
    """
    Returns distinct session dates (take_date) for a given pitcher.
    """
    if athlete_name is None:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT t.take_date
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE a.athlete_name = %s
                ORDER BY t.take_date
            """, (athlete_name,))
            return [row[0].strftime("%Y-%m-%d") for row in cur.fetchall()]
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pitcher_handedness(athlete_name):
    """
    Returns handedness ('R' or 'L') for a given pitcher.
    """
    if athlete_name is None:
        return None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT handedness
                FROM athletes
                WHERE athlete_name = %s
                LIMIT 1
            """, (athlete_name,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pelvis_angular_velocity(take_ids):
    """
    Returns pelvis angular velocity (z_data) over frames for given take_ids.
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Torso Angular Velocity (Z) helper
# --------------------------------------------------

@st.cache_data(ttl=300)
def get_torso_angular_velocity(take_ids):
    """
    Returns torso angular velocity (z_data) over frames for given take_ids.
    Category: ORIGINAL
    Segment: TORSO_ANGULAR_VELOCITY
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Torso-Pelvis Angular Velocity (Z) helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_torso_pelvis_angular_velocity(take_ids):
    """
    Returns torso-pelvis angular velocity (z_data) over frames for given take_ids.
    Category: ORIGINAL
    Segment: TORSO_PELVIS_ANGULAR_VELOCITY
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

# --------------------------------------------------
# Elbow Angular Velocity (X) helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_elbow_angular_velocity(take_ids, handedness):
    """
    Returns elbow angular velocity (x_data) over frames for given take_ids.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGULAR_VELOCITY
      LHP → LT_ELBOW_ANGULAR_VELOCITY
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_ELBOW_ANGULAR_VELOCITY"
        if handedness == "R"
        else "LT_ELBOW_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_elbow_flexion_angle(take_ids, handedness):
    """
    Returns elbow flexion angle (x_data) for the throwing elbow.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGLE
      LHP → LT_ELBOW_ANGLE
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_ELBOW_ANGLE"
        if handedness == "R"
        else "LT_ELBOW_ANGLE"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- SHOULDER EXTERNAL ROTATION ANGLE helper ---
@st.cache_data(ttl=300)
def get_shoulder_er_angle(take_ids, handedness):
    """
    Returns shoulder external rotation angle (z_data) for the throwing shoulder.

    Category: ORIGINAL
    Segments:
      RHP → RT_SHOULDER_ANGLE
      LHP → LT_SHOULDER_ANGLE
    """
    if not take_ids:
        return {}

    segment = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                    SELECT
                        ts.take_id,
                        ts.frame,
                        ts.z_data
                    FROM time_series_data ts
                    JOIN categories c ON ts.category_id = c.category_id
                    JOIN segments s ON ts.segment_id = s.segment_id
                    WHERE c.category_name = 'JOINT_ANGLES'
                      AND s.segment_name = %s
                      AND ts.take_id IN ({placeholders})
                      AND ts.z_data IS NOT NULL
                    ORDER BY ts.take_id, ts.frame
                """, (segment, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()

# --- SHOULDER ABDUCTION ANGLE helper ---

@st.cache_data(ttl=300)
def get_shoulder_abduction_angle(take_ids, handedness):
    """
    Returns shoulder abduction angle (y_data) for the throwing shoulder.

    Category: JOINT_ANGLES
    Segments:
      RHP → RT_SHOULDER
      LHP → LT_SHOULDER
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER"
        if handedness == "R"
        else "LT_SHOULDER"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, y in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(y)

            return data
    finally:
        conn.close()

# --- SHOULDER HORIZONTAL ABDUCTION ANGLE helper ---

@st.cache_data(ttl=300)
def get_front_knee_flexion_angle(take_ids, handedness):
    """
    Returns front (lead) knee flexion angle (x_data).

    Category: ORIGINAL
    Segments:
      RHP → LT_KNEE_ANGLE
      LHP → RT_KNEE_ANGLE
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "LT_KNEE_ANGLE"
        if handedness == "R"
        else "RT_KNEE_ANGLE"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- FRONT KNEE EXTENSION VELOCITY helper ---
@st.cache_data(ttl=300)
def get_front_knee_extension_velocity(take_ids, handedness):
    """
    Returns front (lead) knee angular velocity (x_data).

    Category: ORIGINAL
    Segments:
      RHP → LT_KNEE_ANGULAR_VELOCITY
      LHP → RT_KNEE_ANGULAR_VELOCITY
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "LT_KNEE_ANGULAR_VELOCITY"
        if handedness == "R"
        else "RT_KNEE_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_shoulder_horizontal_abduction_angle(take_ids, handedness):
    """
    Returns shoulder horizontal abduction angle (x_data) for the throwing shoulder.

    Category: JOINT_ANGLES
    Segments:
      RHP → RT_SHOULDER
      LHP → LT_SHOULDER
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER"
        if handedness == "R"
        else "LT_SHOULDER"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- TORSO ANGLE COMPONENTS helper ---
@st.cache_data(ttl=300)
def get_torso_angle_components(take_ids):
    """
    Returns torso angle components for each take.

    Category: ORIGINAL
    Segment: TORSO_ANGLE

    Components:
      x_data → Forward Trunk Tilt
      y_data → Lateral Trunk Tilt
      z_data → Trunk Angle
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x, y, z in rows:
                data.setdefault(take_id, {
                    "frame": [],
                    "x": [],
                    "y": [],
                    "z": []
                })
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_pelvis_angle(take_ids):
    """
    Returns pelvis angle (z_data).

    Category: ORIGINAL
    Segment: PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_pelvic_lateral_tilt(take_ids):
    """
    Returns pelvic lateral tilt from pelvis angle (y_data).

    Category: ORIGINAL
    Segment: PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, y in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(y)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_hip_shoulder_separation(take_ids):
    """
    Returns hip–shoulder separation angle (z_data).

    Category: ORIGINAL
    Segment: TORSO_PELVIS_ANGLE
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_shoulder_ir_velocity(take_ids, handedness):
    """
    Returns shoulder internal rotation angular velocity (x_data).

    Category: ORIGINAL
    Segments:
      RHP → RT_SHOULDER_ANGULAR_VELOCITY
      LHP → LT_SHOULDER_ANGULAR_VELOCITY

    L-handed pitchers will be sign-normalized later.
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER_ANGULAR_VELOCITY"
        if handedness == "R"
        else "LT_SHOULDER_ANGULAR_VELOCITY"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_arm_proximal_energy_transfer(take_ids, handedness):
    """
    Arm proximal energy transfer (power flowing into the arm).

    Category: SEGMENT_POWERS
    Segments:
      RHP → RAR_PROX
      LHP → LAR_PROX
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR_PROX" if handedness == "R" else "LAR_PROX"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()

# --- DISTAL ARM SEGMENT POWER loader ---
@st.cache_data(ttl=300)
def get_distal_arm_segment_power(take_ids, handedness):
    """
    Returns distal throwing arm segment power (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP → RTA_DIST_R
      LHP → RTA_DIST_L
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_DIST_R" if handedness == "R" else "RTA_DIST_L"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_glove_side_trunk_shoulder_energy_flow(take_ids, handedness):
    """
    Returns glove-side distal arm/trunk-shoulder energy flow (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP -> RTA_DIST_L
      LHP -> RTA_DIST_R
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_DIST_L" if handedness == "R" else "RTA_DIST_R"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_glove_arm_energy_flow(take_ids, handedness):
    """
    Returns glove-side proximal arm energy flow (Watts).

    Category: SEGMENT_POWERS
    Segments:
      RHP -> LAR_PROX
      LHP -> RAR_PROX
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LAR_PROX" if handedness == "R" else "RAR_PROX"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'SEGMENT_POWERS'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_trunk_shoulder_rot_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder rotational energy flow.

    Category: JCS_STP_ROT
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ROT'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Trunk–Shoulder Elevation/Depression Energy Flow loader ---

@st.cache_data(ttl=300)
def get_trunk_shoulder_elev_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder elevation/depression energy flow.

    Category: JCS_STP_ELEV
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ELEV'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Trunk–Shoulder Horizontal Abduction/Adduction Energy Flow loader ---

@st.cache_data(ttl=300)
def get_trunk_shoulder_horizabd_energy_flow(take_ids, handedness):
    """
    Trunk–Shoulder horizontal abduction/adduction energy flow.

    Category: JCS_STP_HORIZABD
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RTA_RAR" if handedness == "R" else "RTA_LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_HORIZABD'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


# --- Arm Rotational Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_rot_energy_flow(take_ids, handedness):
    """
    Arm rotational energy flow.

    Category: JCS_STP_ROT
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ROT'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()


# --- Arm Elevation/Depression Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_elev_energy_flow(take_ids, handedness):
    """
    Arm elevation/depression energy flow.

    Category: JCS_STP_ELEV
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_ELEV'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()


# --- Arm Horizontal Abduction/Adduction Energy Flow loader ---
@st.cache_data(ttl=300)
def get_arm_horizabd_energy_flow(take_ids, handedness):
    """
    Arm horizontal abduction/adduction energy flow.

    Category: JCS_STP_HORIZABD
    Segments:
      RHP → RTA_RAR
      LHP → RTA_LAR
    Component: x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RAR" if handedness == "R" else "LAR"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'JCS_STP_HORIZABD'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for tid, frame, x in rows:
                data.setdefault(tid, {"frame": [], "value": []})
                data[tid]["frame"].append(frame)
                data[tid]["value"].append(x)
            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_energy_flow_from_segment(take_ids, segment_name, component="x"):
    """
    Generic energy-flow loader by segment name and component.
    """
    if not take_ids or not segment_name:
        return {}
    component_col = {
        "x": "ts.x_data",
        "y": "ts.y_data",
        "z": "ts.z_data",
    }.get(component, "ts.x_data")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    {component_col}
                FROM time_series_data ts
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND {component_col} IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()

NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP = {
    "RPV_DIST_STP_FLEX": ("RPV_DIST", "JCS_STP_FLEX"),
    "RPV_DIST_STP_SIDE": ("RPV_DIST", "JCS_STP_SIDE"),
    "RPV_DIST_STP_ROT": ("RPV_DIST", "JCS_STP_ROT"),
    "RTA_PROX_STP_FLEX": ("RTA_PROX", "JCS_STP_FLEX"),
    "RTA_PROX_STP_SIDE": ("RTA_PROX", "JCS_STP_SIDE"),
    "RTA_PROX_STP_ROT": ("RTA_PROX", "JCS_STP_ROT"),
    "RTA_PROX_STP_X": ("RTA_PROX", "JCS_STP_X"),
    "RTA_PROX_STP_Y": ("RTA_PROX", "JCS_STP_Y"),
    "RTA_PROX_STP_Z": ("RTA_PROX", "JCS_STP_Z"),
}

NEW_TRUNK_PELVIS_ENERGY_METRICS = list(NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP.keys())

NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP = {
    "RPV_DIST_STP_FLEX": "#0F766E",
    "RPV_DIST_STP_SIDE": "#1D4ED8",
    "RPV_DIST_STP_ROT": "#A16207",
    "RTA_PROX_STP_FLEX": "#059669",
    "RTA_PROX_STP_SIDE": "#2563EB",
    "RTA_PROX_STP_ROT": "#CA8A04",
    "RTA_PROX_STP_X": "#BE123C",
    "RTA_PROX_STP_Y": "#6D28D9",
    "RTA_PROX_STP_Z": "#7C3AED",
}

ENERGY_TORQUE_METRICS = {
    "Throwing Shoulder Rotational Torque (Relative to Trunk)",
}


def get_energy_yaxis_title(selected_metrics):
    metric_set = set(selected_metrics or [])
    has_torque = bool(metric_set & ENERGY_TORQUE_METRICS)
    has_energy_flow = bool(metric_set - ENERGY_TORQUE_METRICS)

    if has_torque and has_energy_flow:
        return "Energy Flow (W) / Torque (N-m)"
    if has_torque:
        return "Torque (N-m)"
    return "Energy Flow (W)"


@st.cache_data(ttl=300)
def get_energy_flow_from_category_segment(take_ids, category_name, segment_name, component="x"):
    """
    Generic energy-flow loader by category, segment name, and component.
    """
    if not take_ids or not category_name or not segment_name:
        return {}
    component_col = {
        "x": "ts.x_data",
        "y": "ts.y_data",
        "z": "ts.z_data",
    }.get(component, "ts.x_data")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    {component_col}
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = %s
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND {component_col} IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (category_name, segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_hand_cg_velocity(take_ids, handedness):
    """
    Returns CG velocity (x_data) for the throwing hand based on handedness.
    Category: KINETIC_KINEMATIC_CGVel
    Segments: RHA / LHA
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RHA" if handedness == "R" else "LHA"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "x": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_hand_speed(take_ids, handedness):
    """
    Returns hand speed magnitude from CG velocity components:
      speed = sqrt(x^2 + y^2 + z^2)
    Category: KINETIC_KINEMATIC_CGVel
    Segments: RHA / LHA
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RHA" if handedness == "R" else "LHA"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()
            data = {}
            for take_id, frame, x, y, z in rows:
                if x is None or y is None or z is None:
                    continue

                speed = float(np.sqrt(x**2 + y**2 + z**2))
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(speed)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_center_of_mass_velocity_x(take_ids):
    """
    Returns Center of Mass velocity in the x direction.

    Category: PROCESSED
    Segment: CenterOfMass_VELO
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'PROCESSED'
                  AND s.segment_name = 'CenterOfMass_VELO'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, x in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_shoulder_er_angles(take_ids, handedness):
    """
    Returns shoulder joint angle z_data for MER detection.
    Category: JOINT_ANGLES
    Segments:
      - RT_SHOULDER for R-handed
      - LT_SHOULDER for L-handed
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_SHOULDER" if handedness == "R" else "LT_SHOULDER"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["z"].append(z)

            return data
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_forearm_pron_sup_angle(take_ids, handedness):
    """
    Forearm Pronation / Supination angle.

    Category: ORIGINAL
    Segments:
      RHP → RT_ELBOW_ANGLE
      LHP → LT_ELBOW_ANGLE
    Component: z_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_ELBOW_ANGLE" if handedness == "R" else "LT_ELBOW_ANGLE"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            data = {}
            for take_id, frame, z in rows:
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)

            return data
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_peak_glove_knee_pre_br(take_ids, handedness, br_frames):
    """
    Peak glove-side knee height (Z position) prior to Ball Release.

    Category: KINETIC_KINEMATIC_ProxEndPos
    Segments:
      RHP → LSK
      LHP → RSK
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LSK" if handedness == "R" else "RSK"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_ProxEndPos'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.z_data DESC, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                br_frame = br_frames.get(take_id)
                if br_frame is None:
                    continue
                if frame < br_frame and take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()


# --------------------------------------------------
# Foot Plant event helper
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_foot_plant_frame(
    take_ids,
    handedness,
    knee_peak_frames,
    br_frames
):
    """
    Estimate Foot Plant as the LAST frame where lead ankle Z velocity is negative
    between peak knee height and ball release.

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                knee_frame = knee_peak_frames.get(take_id)
                br_frame   = br_frames.get(take_id)

                if knee_frame is None or br_frame is None:
                    continue

                # constrain search window: knee peak → ball release
                if frame < knee_frame or frame > br_frame:
                    continue

                # last downward ankle velocity frame
                if z < 0:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_peak_ankle_prox_x_velocity(
    take_ids,
    handedness
):
    """
    Peak lead ankle proximal X velocity.

    Category: KINETIC_KINEMATIC_ProxEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_ProxEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.x_data DESC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, x in rows:
                if take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_ankle_min_frame(
    take_ids,
    handedness,
    ankle_prox_x_peak_frames,
    shoulder_er_max_frames
):
    """
    Deepest lead ankle distal Z-velocity dip between ankle prox-X peak and max shoulder ER.

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.z_data ASC, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, _z in rows:
                px_frame = ankle_prox_x_peak_frames.get(take_id)
                er_frame = shoulder_er_max_frames.get(take_id)

                if px_frame is None or er_frame is None:
                    continue

                if frame < px_frame or frame > er_frame:
                    continue

                if take_id not in out:
                    out[take_id] = int(frame)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_foot_plant_frame_zero_cross(
    take_ids,
    handedness,
    ankle_min_frames,
    shoulder_er_max_frames
):
    """
    Refined Foot Plant using zero-cross logic.

    Search window:
      lead ankle distal Z minimum → max shoulder ER

    Rule:
      first frame where ankle Z velocity >= -0.05
      foot plant frame = frame - 1

    Category: KINETIC_KINEMATIC_DistEndVel
    Segments:
      RHP → LFT
      LHP → RFT
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_DistEndVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (segment_name, *take_ids))

            rows = cur.fetchall()

            out = {}
            for take_id, frame, z in rows:
                ankle_min_frame = ankle_min_frames.get(take_id)
                er_frame = shoulder_er_max_frames.get(take_id)

                if ankle_min_frame is None or er_frame is None:
                    continue

                # refined biomechanical bounds
                if frame < ankle_min_frame or frame > er_frame:
                    continue

                # zero-cross detection
                if z >= -0.05 and take_id not in out:
                    out[take_id] = int(frame - 1)

            return out
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_lead_heel_contact_frame(
    take_ids,
    handedness,
    start_frames,
    end_frames,
    anchor_frames,
    contact_ratio=0.15,
    absolute_floor_buffer=0.03,
    min_consecutive_frames=3,
    pre_anchor_frames=4,
    post_anchor_frames=6,
    flattening_tolerance=0.01
):
    """
    Estimate lead-foot contact timing from heel height using a take-specific near-floor threshold.

    Category: LANDMARK_ORIGINAL
    Segments:
      RHP → L_HEEL
      LHP → R_HEEL
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    heel_segment = "L_HEEL" if handedness == "R" else "R_HEEL"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s   ON ts.segment_id  = s.segment_id
                WHERE c.category_name = 'LANDMARK_ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame ASC
            """, (heel_segment, *take_ids))

            rows = cur.fetchall()

            rows_by_take = {}
            for take_id, frame, z in rows:
                rows_by_take.setdefault(take_id, []).append((int(frame), float(z)))

            out = {}
            for take_id, take_rows in rows_by_take.items():
                start_frame = start_frames.get(take_id)
                end_frame = end_frames.get(take_id)
                anchor_frame = anchor_frames.get(take_id)

                if start_frame is None or end_frame is None or start_frame > end_frame:
                    continue
                if anchor_frame is None:
                    continue

                full_window_rows = [
                    (frame, z)
                    for frame, z in take_rows
                    if start_frame <= frame <= end_frame
                ]
                if not full_window_rows:
                    continue

                heel_values = [z for _, z in full_window_rows]
                heel_floor = min(heel_values)
                heel_ceil = max(heel_values)
                heel_range = heel_ceil - heel_floor
                relative_threshold = (
                    heel_floor
                    if heel_range <= 1e-9 else
                    heel_floor + contact_ratio * heel_range
                )
                absolute_threshold = heel_floor + absolute_floor_buffer
                heel_threshold = min(relative_threshold, absolute_threshold)

                search_start = max(start_frame, int(anchor_frame) - pre_anchor_frames)
                search_end = min(end_frame, int(anchor_frame) + post_anchor_frames)
                search_rows = [
                    (frame, z)
                    for frame, z in full_window_rows
                    if search_start <= frame <= search_end
                ]
                if len(search_rows) < min_consecutive_frames:
                    continue

                for i in range(0, len(search_rows) - min_consecutive_frames + 1):
                    block = search_rows[i:i + min_consecutive_frames]
                    block_values = [z for _, z in block]
                    if not all(z <= heel_threshold for z in block_values):
                        continue

                    # Contact should look settled, not like a single-frame downward spike.
                    block_diffs = [
                        block_values[j + 1] - block_values[j]
                        for j in range(len(block_values) - 1)
                    ]
                    if any(diff < -flattening_tolerance for diff in block_diffs):
                        continue

                    out[take_id] = int(block[0][0])
                    break

            return out
    finally:
        conn.close()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
# --- Sidebar Logo ---
st.sidebar.image(
    "assets/terra_sports.svg",
    use_container_width=True
)


st.sidebar.markdown("### Dashboard Controls")
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        font-size: 1.17em;
        font-weight: 600;
        background-color: #C62828;
        color: #FFFFFF;
        border: 1px solid #C62828;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #B71C1C;
        color: #FFFFFF;
        border: 1px solid #B71C1C;
    }
    /* Make selected multiselect tags readable in the sidebar */
    div[data-testid="stSidebar"] div[data-baseweb="tag"] {
        max-width: 100% !important;
        height: auto !important;
        white-space: normal !important;
    }
    div[data-testid="stSidebar"] div[data-baseweb="tag"] > span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
    }
    /* Make selected multiselect tags readable in the main page as well */
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
        max-width: 100% !important;
        height: auto !important;
        white-space: normal !important;
    }
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] > span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Initialize session state for excluded takes
# -------------------------------
if "excluded_take_ids" not in st.session_state:
    st.session_state["excluded_take_ids"] = []
if "create_groups_mode" not in st.session_state:
    st.session_state["create_groups_mode"] = False
if "show_control_group_velocity" not in st.session_state:
    st.session_state["show_control_group_velocity"] = False
if "control_group_take_ids" not in st.session_state:
    st.session_state["control_group_take_ids"] = []
if "excluded_control_group_take_ids" not in st.session_state:
    st.session_state["excluded_control_group_take_ids"] = []
if "control_group_handedness" not in st.session_state:
    st.session_state["control_group_handedness"] = "Both"
if "control_group_arm_slot_ids" not in st.session_state:
    st.session_state["control_group_arm_slot_ids"] = []
if "control_group_pitchers" not in st.session_state:
    st.session_state["control_group_pitchers"] = []
if "control_group_velocity_range" not in st.session_state:
    st.session_state["control_group_velocity_range"] = (50.0, 100.0)
control_group_arm_slot_categories = [
    ("Over The Top", 50, 90),
    ("High 3/4", 30, 49),
    ("Low 3/4", 10, 29),
    ("Sidearm", -5, 9),
    ("Submarine", -90, -6),
]

for category_label, _, _ in control_group_arm_slot_categories:
    category_key = f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}"
    if category_key not in st.session_state:
        st.session_state[category_key] = True
if "control_group_status_message" not in st.session_state:
    st.session_state["control_group_status_message"] = ""

group_mode_enabled = st.session_state.get("create_groups_mode", False)
if "group_count" not in st.session_state:
    st.session_state["group_count"] = 1

pitcher_names = get_all_pitchers()
group_configs = []
selected_pitchers = []
pitcher_filters = {}

def build_pitcher_filters_for_group(selected_group_pitchers, group_index, show_group_prefix):
    group_pitcher_filters = {}
    multi_pitcher_group = len(selected_group_pitchers) > 1

    for i, pitcher in enumerate(selected_group_pitchers):
        label_suffix = f" - {pitcher}" if multi_pitcher_group else ""
        if multi_pitcher_group:
            st.sidebar.markdown(f"**{pitcher} Filters**")

        session_dates = get_session_dates_for_pitcher(pitcher)
        if session_dates:
            session_dates_with_all = ["All Dates"] + session_dates
            session_dates_label = (
                f"Group {group_index} Session Dates{label_suffix}"
                if show_group_prefix else
                f"Session Dates{label_suffix}"
            )
            selected_dates_i = st.sidebar.multiselect(
                session_dates_label,
                options=session_dates_with_all,
                default=["All Dates"],
                key=f"group{group_index}_select_session_dates_{i}"
            )
        else:
            st.sidebar.info(f"No session dates found for {pitcher}.")
            selected_dates_i = []

        throw_type_label = (
            f"Group {group_index} Throw Type{label_suffix}"
            if show_group_prefix else
            f"Throw Type{label_suffix}"
        )
        throw_types_i = st.sidebar.multiselect(
            throw_type_label,
            options=["Mound", "Pulldown"],
            default=["Mound"],
            key=f"group{group_index}_throw_types_{i}"
        )
        if not throw_types_i:
            throw_types_i = ["Mound"]

        vel_min_i, vel_max_i = get_velocity_bounds(pitcher, selected_dates_i)
        if vel_min_i is not None and vel_max_i is not None:
            velocity_label = (
                f"Group {group_index} Velocity Range{label_suffix} (mph)"
                if show_group_prefix else
                f"Velocity Range{label_suffix} (mph)"
            )
            vel_min_float = float(vel_min_i)
            vel_max_float = float(vel_max_i)
            if vel_min_float == vel_max_float:
                velocity_min_i = vel_min_float
                velocity_max_i = vel_max_float
                st.sidebar.caption(f"{velocity_label}: {vel_min_float:.1f}")
            else:
                velocity_range_i = st.sidebar.slider(
                    velocity_label,
                    min_value=vel_min_float,
                    max_value=vel_max_float,
                    value=(vel_min_float, vel_max_float),
                    step=0.1,
                    key=f"group{group_index}_velocity_range_{i}"
                )
                velocity_min_i, velocity_max_i = velocity_range_i
        else:
            velocity_min_i, velocity_max_i = None, None
            st.sidebar.info(f"Velocity data not available for {pitcher}.")

        group_pitcher_filters[pitcher] = {
            "selected_dates": selected_dates_i,
            "throw_types": throw_types_i,
            "velocity_min": velocity_min_i,
            "velocity_max": velocity_max_i,
        }

    return group_pitcher_filters

def get_filtered_takes_for_pitcher(pitcher, cfg):
    selected_dates_i = cfg.get("selected_dates", [])
    throw_types_i = cfg.get("throw_types", [])
    velocity_min_i = cfg.get("velocity_min")
    velocity_max_i = cfg.get("velocity_max")

    if velocity_min_i is None or velocity_max_i is None:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if "All Dates" in selected_dates_i or not selected_dates_i:
                cur.execute("""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.throw_type = ANY(%s)
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, (pitcher, throw_types_i, velocity_min_i, velocity_max_i))
            else:
                placeholders = ",".join(["%s"] * len(selected_dates_i))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.throw_type = ANY(%s)
                      AND t.take_date IN ({placeholders})
                      AND t.pitch_velo BETWEEN %s AND %s
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, (pitcher, throw_types_i, *selected_dates_i, velocity_min_i, velocity_max_i))
            return cur.fetchall()
    finally:
        conn.close()

def build_take_options_for_group(group_pitcher_filters):
    from collections import defaultdict

    takes_by_id = {}
    for pitcher, cfg in group_pitcher_filters.items():
        for take_id, velo, date, pitcher_name in get_filtered_takes_for_pitcher(pitcher, cfg):
            takes_by_id[take_id] = (take_id, velo, date, pitcher_name)

    if not takes_by_id:
        return [], {}

    sorted_rows = sorted(
        takes_by_id.values(),
        key=lambda row: (row[3], row[2], row[0])
    )

    date_groups = defaultdict(list)
    for tid, velo, date, pitcher_name in sorted_rows:
        date_groups[(pitcher_name, date)].append((tid, velo))

    options = []
    label_to_take_id = {}
    for (pitcher_name, date), items in date_groups.items():
        for order, (tid, velo) in enumerate(items, start=1):
            velo_text = f"{velo:.1f}" if velo is not None else "N/A"
            label = f"{pitcher_name} | {date.strftime('%Y-%m-%d')} – Pitch {order} ({velo_text} mph)"
            options.append(label)
            label_to_take_id[label] = tid

    return options, label_to_take_id


def exit_group_mode():
    st.session_state["create_groups_mode"] = False
    st.session_state["group_count"] = 1

    keys_to_clear = [
        key for key in st.session_state.keys()
        if key.startswith("group") or key.startswith("create_groups_mode")
    ]
    for key in keys_to_clear:
        if key in {"create_groups_mode", "group_count"}:
            continue
        del st.session_state[key]

    st.rerun()


def remove_control_group():
    st.session_state["show_control_group_velocity"] = False
    st.session_state["control_group_take_ids"] = []
    st.session_state["excluded_control_group_take_ids"] = []
    st.session_state["control_group_arm_slot_ids"] = []
    st.session_state["control_group_pitchers"] = []
    st.session_state["control_group_handedness"] = "Both"
    st.session_state["control_group_status_message"] = ""
    for key in ["control_group_velocity_range"]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


def render_control_group_arm_slot_category_checkboxes():
    st.markdown("Arm Slot Categories")
    checkbox_cols = st.columns(2)
    for idx, (category_label, _, _) in enumerate(control_group_arm_slot_categories):
        category_key = f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}"
        with checkbox_cols[idx % 2]:
            st.checkbox(
                category_label,
                key=category_key,
            )


def build_take_exclusion_options(take_ids):
    if not take_ids:
        return [], {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE t.take_id IN ({placeholders})
                ORDER BY a.athlete_name, t.take_date, t.take_id
            """, tuple(take_ids))
            rows = cur.fetchall()
    finally:
        conn.close()

    from collections import defaultdict

    requested_take_ids = set(take_ids)
    date_groups = defaultdict(list)
    take_labels = {}
    for tid, velo, date, pitcher in rows:
        if tid in requested_take_ids:
            date_groups[(pitcher, date)].append((tid, velo))

    for (pitcher, date), items in date_groups.items():
        for i, (tid, velo) in enumerate(items, start=1):
            date_label = date.strftime("%Y-%m-%d")
            velo_label = f"{float(velo):.1f}" if velo is not None else "N/A"
            take_labels[tid] = f"{pitcher} | {date_label} - Pitch {i} ({velo_label} mph)"

    take_options = [take_labels[tid] for tid in take_ids if tid in take_labels]
    label_to_take_id = {
        take_labels[tid]: tid
        for tid in take_ids
        if tid in take_labels
    }
    return take_options, label_to_take_id


def render_control_group_exclude_takes(key, container=st.sidebar):
    control_group_take_ids = st.session_state.get("control_group_take_ids", [])
    if not control_group_take_ids:
        st.session_state["excluded_control_group_take_ids"] = []
        return

    take_options, label_to_take_id = build_take_exclusion_options(control_group_take_ids)
    valid_excluded_take_ids = {
        tid for tid in st.session_state.get("excluded_control_group_take_ids", [])
        if tid in control_group_take_ids
    }
    excluded_labels = container.multiselect(
        "Exclude Takes",
        options=take_options,
        default=[
            label for label, tid in label_to_take_id.items()
            if tid in valid_excluded_take_ids
        ],
        key=key
    )
    st.session_state["excluded_control_group_take_ids"] = [
        label_to_take_id[label] for label in excluded_labels
    ]


def arm_slot_matches_control_group_categories(arm_slot_deg):
    if arm_slot_deg is None:
        return False

    selected_categories = [
        (min_slot, max_slot)
        for category_label, min_slot, max_slot in control_group_arm_slot_categories
        if st.session_state.get(
            f"control_group_arm_slot_category_{category_label.lower().replace(' ', '_').replace('/', '')}",
            False
        )
    ]

    if not selected_categories:
        return False

    arm_slot_value = float(arm_slot_deg)
    return any(min_slot <= arm_slot_value <= max_slot for min_slot, max_slot in selected_categories)

if not pitcher_names:
    st.sidebar.warning("No pitchers found in the database.")
else:
    if group_mode_enabled:
        group_count = max(1, int(st.session_state.get("group_count", 1)))
        st.session_state["group_count"] = group_count

        for group_idx in range(1, group_count + 1):
            st.sidebar.markdown(f"**Group {group_idx}**")
            selected_group_pitchers = st.sidebar.multiselect(
                f"Select Group {group_idx} Pitchers",
                options=pitcher_names,
                default=[pitcher_names[0]] if group_idx == 1 and pitcher_names else [],
                key=f"group{group_idx}_select_pitchers"
            )
            group_pitcher_filters = build_pitcher_filters_for_group(
                selected_group_pitchers,
                group_idx,
                show_group_prefix=True
            )
            group_take_options, group_label_to_take_id = build_take_options_for_group(group_pitcher_filters)
            selected_group_take_labels = st.sidebar.multiselect(
                f"Group {group_idx} Selected Takes",
                options=group_take_options,
                default=[],
                key=f"group{group_idx}_selected_takes"
            )
            selected_group_take_ids = [
                group_label_to_take_id[label]
                for label in selected_group_take_labels
                if label in group_label_to_take_id
            ]
            group_configs.append({
                "group_index": group_idx,
                "selected_pitchers": selected_group_pitchers,
                "pitcher_filters": group_pitcher_filters,
                "selected_take_ids": selected_group_take_ids,
            })

            if group_idx < group_count:
                st.sidebar.markdown("---")

        st.sidebar.markdown("---")
        if st.sidebar.button("Create Another Group", key="create_another_group_btn", use_container_width=True):
            st.session_state["group_count"] = group_count + 1
            st.rerun()
        if st.sidebar.button("Exit Group Mode", key="exit_group_mode_btn", use_container_width=True):
            exit_group_mode()
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Control Group**")
        if not st.session_state.get("show_control_group_velocity"):
            if st.sidebar.button(
                "Create Control Group",
                key="create_control_group_btn_group_mode",
                use_container_width=True
            ):
                st.session_state["show_control_group_velocity"] = True
                st.rerun()

        # Merge group filters into the existing downstream data model.
        for group_cfg in group_configs:
            for pitcher in group_cfg["selected_pitchers"]:
                if pitcher not in selected_pitchers:
                    selected_pitchers.append(pitcher)

            for pitcher, cfg in group_cfg["pitcher_filters"].items():
                if pitcher not in pitcher_filters:
                    pitcher_filters[pitcher] = {
                        "selected_dates": list(cfg["selected_dates"]),
                        "throw_types": list(cfg["throw_types"]),
                        "velocity_min": cfg["velocity_min"],
                        "velocity_max": cfg["velocity_max"],
                    }
                    continue

                existing = pitcher_filters[pitcher]
                existing_dates = existing.get("selected_dates", [])
                new_dates = cfg.get("selected_dates", [])
                if "All Dates" in existing_dates or "All Dates" in new_dates:
                    existing["selected_dates"] = ["All Dates"]
                else:
                    existing["selected_dates"] = sorted(set(existing_dates + new_dates))

                existing["throw_types"] = sorted(set(existing.get("throw_types", []) + cfg.get("throw_types", [])))

                vmins = [v for v in [existing.get("velocity_min"), cfg.get("velocity_min")] if v is not None]
                vmaxs = [v for v in [existing.get("velocity_max"), cfg.get("velocity_max")] if v is not None]
                existing["velocity_min"] = min(vmins) if vmins else None
                existing["velocity_max"] = max(vmaxs) if vmaxs else None
    else:
        selected_pitchers = st.sidebar.multiselect(
            "Select Pitcher(s)",
            options=pitcher_names,
            default=[pitcher_names[0]] if pitcher_names else [],
            key="select_pitchers"
        )
        pitcher_filters = build_pitcher_filters_for_group(
            selected_pitchers,
            group_index=0,
            show_group_prefix=False
        )
        group_configs = [{
            "group_index": 1,
            "selected_pitchers": selected_pitchers,
            "pitcher_filters": pitcher_filters,
            "selected_take_ids": [],
        }]

selected_take_ids_union = set()
if group_mode_enabled:
    for group_cfg in group_configs:
        selected_take_ids_union.update(group_cfg.get("selected_take_ids", []))

group_palette = [
    "#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD",
    "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"
]

def get_group_display_label(group_cfg):
    group_idx = group_cfg["group_index"]
    return f"Group {group_idx}"

def is_control_group_label(label):
    return label == "Control Group"

group_color_map = {}
take_group_map = {}
if group_mode_enabled:
    for idx, group_cfg in enumerate(group_configs):
        group_label = get_group_display_label(group_cfg)
        group_color_map[group_label] = group_palette[idx % len(group_palette)]
        for tid in group_cfg.get("selected_take_ids", []):
            if tid not in take_group_map:
                take_group_map[tid] = group_label

all_throw_types = sorted({
    t
    for cfg in pitcher_filters.values()
    for t in cfg["throw_types"]
})
multi_pitcher_mode = len(selected_pitchers) > 1
mound_only_sidebar = bool(pitcher_filters) and all(
    set(cfg["throw_types"]) == {"Mound"}
    for cfg in pitcher_filters.values()
)

def render_group_selection_summary():
    if not group_mode_enabled:
        return
    if not group_configs:
        st.caption("Group mode active. No groups configured.")
        return

    for group_cfg in group_configs:
        group_idx = group_cfg["group_index"]
        group_pitchers = group_cfg["selected_pitchers"]
        group_pitcher_filters = group_cfg["pitcher_filters"]
        group_label = get_group_display_label(group_cfg)

        if not group_pitchers:
            st.caption(f"Group {group_idx} | No pitchers selected.")
            continue

        throw_types = sorted({
            throw_type
            for cfg in group_pitcher_filters.values()
            for throw_type in cfg.get("throw_types", [])
        })
        throw_types_label = ", ".join(throw_types) if throw_types else "None"

        per_pitcher_ranges = []
        for pitcher in group_pitchers:
            cfg = group_pitcher_filters.get(pitcher, {})
            vmin = cfg.get("velocity_min")
            vmax = cfg.get("velocity_max")
            if vmin is None or vmax is None:
                continue
            per_pitcher_ranges.append(f"{pitcher}: {vmin:.1f}-{vmax:.1f}")

        velocity_label = "; ".join(per_pitcher_ranges) if per_pitcher_ranges else "N/A"
        st.caption(
            f"{group_label} | "
            f"{', '.join(group_pitchers)} | "
            f"Throw Type: {throw_types_label} | "
            f"Velocity Range (mph): {velocity_label}"
        )

def aggregate_curves(curves_dict, stat="Median"):
    """
    curves_dict: { take_id: { "frame": [...], "value": [...] } }
    Returns aggregated_x, aggregated_y, iqr_low, iqr_high
    """
    if (
        len(curves_dict) == 1
        and any("q1" in d and "q3" in d for d in curves_dict.values())
    ):
        curve = next(iter(curves_dict.values()))
        return (
            list(curve.get("frame", [])),
            list(curve.get("value", [])),
            list(curve.get("q1", [])),
            list(curve.get("q3", [])),
        )

    all_frames = sorted(set(
        f for d in curves_dict.values() for f in d["frame"]
    ))

    agg_y = []
    iqr_low = []
    iqr_high = []

    for f in all_frames:
        vals = [
            d["value"][i]
            for d in curves_dict.values()
            for i, fr in enumerate(d["frame"])
            if fr == f
        ]

        if not vals:
            continue

        if stat == "Mean":
            agg_y.append(np.mean(vals))
        else:
            agg_y.append(np.median(vals))

        iqr_low.append(np.percentile(vals, 25))
        iqr_high.append(np.percentile(vals, 75))

    return all_frames, agg_y, iqr_low, iqr_high

def build_shared_dashboard_state():
    pitcher_handedness = {
        p: get_pitcher_handedness(p)
        for p in selected_pitchers
    }

    shared_take_ids = []
    shared_take_pitcher_map = {}
    primary_take_ids = []
    control_take_ids = []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for pitcher, cfg in pitcher_filters.items():
                selected_dates_i = cfg["selected_dates"]
                throw_types_i = cfg["throw_types"]
                velocity_min_i = cfg["velocity_min"]
                velocity_max_i = cfg["velocity_max"]

                if velocity_min_i is None or velocity_max_i is None:
                    continue

                if "All Dates" in selected_dates_i or not selected_dates_i:
                    cur.execute("""
                        SELECT t.take_id
                        FROM takes t
                        JOIN athletes a ON a.athlete_id = t.athlete_id
                        WHERE a.athlete_name = %s
                          AND t.throw_type = ANY(%s)
                          AND t.pitch_velo BETWEEN %s AND %s
                    """, (pitcher, throw_types_i, velocity_min_i, velocity_max_i))
                else:
                    placeholders = ",".join(["%s"] * len(selected_dates_i))
                    cur.execute(f"""
                        SELECT t.take_id
                        FROM takes t
                        JOIN athletes a ON a.athlete_id = t.athlete_id
                        WHERE a.athlete_name = %s
                          AND t.throw_type = ANY(%s)
                          AND t.take_date IN ({placeholders})
                          AND t.pitch_velo BETWEEN %s AND %s
                    """, (pitcher, throw_types_i, *selected_dates_i, velocity_min_i, velocity_max_i))

                for (take_id,) in cur.fetchall():
                    if take_id not in shared_take_pitcher_map:
                        shared_take_pitcher_map[take_id] = pitcher
                        shared_take_ids.append(take_id)
    finally:
        conn.close()

    if group_mode_enabled:
        if selected_take_ids_union:
            shared_take_ids = [tid for tid in shared_take_ids if tid in selected_take_ids_union]
        else:
            shared_take_ids = []

    shared_take_handedness = {
        tid: pitcher_handedness.get(shared_take_pitcher_map.get(tid))
        for tid in shared_take_ids
    }
    shared_take_ids = [tid for tid in shared_take_ids if shared_take_handedness.get(tid) in ("R", "L")]

    shared_take_order = {}
    shared_take_velocity = {}
    shared_take_date_map = {}
    shared_take_pitcher_name_map = {}

    def merge_control_group_takes_into_shared_state():
        nonlocal shared_take_ids
        nonlocal shared_take_handedness
        nonlocal shared_take_order
        nonlocal shared_take_velocity
        nonlocal shared_take_date_map
        nonlocal shared_take_pitcher_name_map
        nonlocal primary_take_ids
        nonlocal control_take_ids

        primary_take_ids = list(shared_take_ids)
        excluded_control_group_take_ids = set(
            st.session_state.get("excluded_control_group_take_ids", [])
        )
        control_take_ids = [
            tid for tid in st.session_state.get("control_group_take_ids", [])
            if tid not in primary_take_ids and tid not in excluded_control_group_take_ids
        ]
        combined_take_ids = primary_take_ids + control_take_ids

        if not control_take_ids or not combined_take_ids:
            return

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(combined_take_ids))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name, a.handedness
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE t.take_id IN ({placeholders})
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, tuple(combined_take_ids))
                combined_rows = cur.fetchall()
        finally:
            conn.close()

        from collections import defaultdict

        shared_take_ids = []
        shared_take_handedness = {}
        shared_take_order = {}
        shared_take_velocity = {}
        shared_take_date_map = {}
        shared_take_pitcher_name_map = {}

        combined_date_groups = defaultdict(list)
        for tid, velo, date, pitcher, handedness in combined_rows:
            if handedness not in ("R", "L"):
                continue
            shared_take_ids.append(tid)
            shared_take_handedness[tid] = handedness
            shared_take_velocity[tid] = velo
            shared_take_date_map[tid] = date.strftime("%Y-%m-%d")
            shared_take_pitcher_name_map[tid] = pitcher
            combined_date_groups[(pitcher, date)].append((tid, velo))

        for (pitcher, date), items in combined_date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                shared_take_order[tid] = i

    if shared_take_ids:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(shared_take_ids))
                cur.execute(f"""
                    SELECT t.take_id, t.pitch_velo, t.take_date, a.athlete_name
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE t.take_id IN ({placeholders})
                    ORDER BY a.athlete_name, t.take_date, t.take_id
                """, tuple(shared_take_ids))
                rows = cur.fetchall()
        finally:
            conn.close()

        from collections import defaultdict

        date_groups = defaultdict(list)
        for tid, velo, date, pitcher in rows:
            date_groups[(pitcher, date)].append((tid, velo))

        for (pitcher, date), items in date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                shared_take_order[tid] = i
                shared_take_velocity[tid] = velo
                shared_take_date_map[tid] = date.strftime("%Y-%m-%d")
                shared_take_pitcher_name_map[tid] = pitcher

        if not group_mode_enabled:
            take_options = [
                (
                    f"{shared_take_pitcher_name_map[tid]} | {shared_take_date_map[tid]} - "
                    f"Pitch {shared_take_order[tid]} ({shared_take_velocity[tid]:.1f} mph)"
                )
                for tid in shared_take_ids
            ]

            label_to_take_id = {
                (
                    f"{shared_take_pitcher_name_map[tid]} | {shared_take_date_map[tid]} - "
                    f"Pitch {shared_take_order[tid]} ({shared_take_velocity[tid]:.1f} mph)"
                ): tid
                for tid in shared_take_ids
            }

            excluded_labels = st.sidebar.multiselect(
                "Exclude Takes",
                options=take_options,
                default=[
                    label for label, tid in label_to_take_id.items()
                    if tid in st.session_state["excluded_take_ids"]
                ],
                key="exclude_takes"
            )

            st.session_state["excluded_take_ids"] = [
                label_to_take_id[label] for label in excluded_labels
            ]
            if st.sidebar.button("Create Custom Groups", key="create_groups_mode_btn", use_container_width=True):
                st.session_state["create_groups_mode"] = True
                st.rerun()
            if not st.session_state.get("show_control_group_velocity"):
                if st.sidebar.button(
                    "Create Control Group",
                    key="create_control_group_btn",
                    use_container_width=True
                ):
                    st.session_state["show_control_group_velocity"] = True
                    st.rerun()
            shared_take_ids = [
                tid for tid in shared_take_ids
                if tid not in st.session_state["excluded_take_ids"]
            ]
            shared_take_handedness = {
                tid: shared_take_handedness[tid]
                for tid in shared_take_ids
                if tid in shared_take_handedness
            }
            shared_take_order = {
                tid: shared_take_order[tid]
                for tid in shared_take_ids
                if tid in shared_take_order
            }
            shared_take_velocity = {
                tid: shared_take_velocity[tid]
                for tid in shared_take_ids
                if tid in shared_take_velocity
            }
            shared_take_date_map = {
                tid: shared_take_date_map[tid]
                for tid in shared_take_ids
                if tid in shared_take_date_map
            }
            shared_take_pitcher_name_map = {
                tid: shared_take_pitcher_name_map[tid]
                for tid in shared_take_ids
                if tid in shared_take_pitcher_name_map
            }
            if st.session_state.get("show_control_group_velocity"):
                st.sidebar.markdown("**Control Group**")
                if st.sidebar.button(
                    "Remove Control Group",
                    key="remove_control_group_btn",
                    use_container_width=True
                ):
                    remove_control_group()
                with st.sidebar.form("control_group_filters_form"):
                    st.multiselect(
                        "Pitchers",
                        options=pitcher_names,
                        key="control_group_pitchers"
                    )
                    st.radio(
                        "Handedness",
                        options=["Both", "Left", "Right"],
                        key="control_group_handedness",
                        horizontal=True
                    )
                    st.slider(
                        "Velocity Range (mph)",
                        min_value=50.0,
                        max_value=100.0,
                        value=st.session_state.get("control_group_velocity_range", (50.0, 100.0)),
                        step=0.1,
                        key="control_group_velocity_range"
                    )
                    render_control_group_arm_slot_category_checkboxes()
                    render_control_group_exclude_takes("exclude_control_group_takes", st)
                    generate_control_group = st.form_submit_button(
                        "Generate Control Group",
                        use_container_width=True
                    )

                if generate_control_group:
                    handedness_filter = st.session_state.get("control_group_handedness", "Both")
                    pool_handedness = (
                        "L" if handedness_filter == "Left"
                        else "R" if handedness_filter == "Right"
                        else None
                    )
                    all_control_group_pool = get_control_group_take_pool(pool_handedness)
                    selected_pitcher_set = set(st.session_state.get("control_group_pitchers", []))
                    selected_velocity_range = st.session_state.get("control_group_velocity_range", (50.0, 100.0))

                    final_candidate_control_take_ids = []
                    for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                        if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                            continue
                        if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                            continue
                        if not arm_slot_matches_control_group_categories(arm_slot_deg):
                            continue
                        final_candidate_control_take_ids.append(take_id)

                    st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_status_message"] = (
                        f"Total Pitches: {len(final_candidate_control_take_ids)}"
                        if final_candidate_control_take_ids else
                        "No control-group takes found for the selected filters."
                    )
                    st.rerun()

                if st.session_state.get("control_group_status_message"):
                    st.sidebar.caption(st.session_state["control_group_status_message"])
                if st.session_state.get("show_control_group_velocity"):
                    merge_control_group_takes_into_shared_state()
        elif group_mode_enabled and st.session_state.get("show_control_group_velocity"):
            if st.sidebar.button(
                "Remove Control Group",
                key="remove_control_group_btn_group_mode",
                use_container_width=True
            ):
                remove_control_group()
            with st.sidebar.form("control_group_filters_form_group_mode"):
                st.multiselect(
                    "Pitchers",
                    options=pitcher_names,
                    key="control_group_pitchers"
                )
                st.radio(
                    "Handedness",
                    options=["Both", "Left", "Right"],
                    key="control_group_handedness",
                    horizontal=True
                )
                st.slider(
                    "Velocity Range (mph)",
                    min_value=50.0,
                    max_value=100.0,
                    value=st.session_state.get("control_group_velocity_range", (50.0, 100.0)),
                    step=0.1,
                    key="control_group_velocity_range"
                )
                render_control_group_arm_slot_category_checkboxes()
                render_control_group_exclude_takes("exclude_control_group_takes_group_mode", st)
                generate_control_group = st.form_submit_button(
                    "Generate Control Group",
                    use_container_width=True
                )

            if generate_control_group:
                handedness_filter = st.session_state.get("control_group_handedness", "Both")
                pool_handedness = (
                    "L" if handedness_filter == "Left"
                    else "R" if handedness_filter == "Right"
                    else None
                )
                all_control_group_pool = get_control_group_take_pool(pool_handedness)
                selected_pitcher_set = set(st.session_state.get("control_group_pitchers", []))
                selected_velocity_range = st.session_state.get("control_group_velocity_range", (50.0, 100.0))

                final_candidate_control_take_ids = []
                for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                    if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                        continue
                    if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                        continue
                    if not arm_slot_matches_control_group_categories(arm_slot_deg):
                        continue
                    final_candidate_control_take_ids.append(take_id)

                st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_status_message"] = (
                    f"Total Pitches: {len(final_candidate_control_take_ids)}"
                    if final_candidate_control_take_ids else
                    "No control-group takes found for the selected filters."
                )
                st.rerun()

            if st.session_state.get("control_group_status_message"):
                st.sidebar.caption(st.session_state["control_group_status_message"])

            merge_control_group_takes_into_shared_state()
        else:
            primary_take_ids = []
            control_take_ids = []
    else:
        primary_take_ids = []
        control_take_ids = []

    from collections import defaultdict

    shared_take_ids_by_handedness = defaultdict(list)
    for tid in shared_take_ids:
        hand = shared_take_handedness.get(tid)
        if hand in ("R", "L"):
            shared_take_ids_by_handedness[hand].append(tid)

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in shared_take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    shared_br_frames = {}
    shared_shoulder_er_max_frames = {}
    shared_knee_peak_frames = {}
    shared_foot_plant_zero_cross_frames = {}
    shared_knee_event_frames = []
    shared_fp_event_frames = []
    shared_mer_event_frames = []
    shared_window_start = -100

    if shared_take_ids:
        cg_data = load_by_handedness(get_hand_cg_velocity)
        shoulder_data = load_by_handedness(get_shoulder_er_angles)

        for take_id in shared_take_ids:
            if take_id in cg_data:
                cg_frames = cg_data[take_id]["frame"]
                cg_vals = cg_data[take_id]["x"]
                valid = [(i, v) for i, v in enumerate(cg_vals) if v is not None]
                if valid:
                    idx, _ = max(valid, key=lambda x: x[1])
                    shared_br_frames[take_id] = cg_frames[idx]

        for take_id, d in shoulder_data.items():
            frames = d["frame"]
            values = d["z"]
            valid = [(f, v) for f, v in zip(frames, values) if v is not None]
            if not valid:
                continue

            hand = shared_take_handedness.get(take_id)
            if hand == "R":
                er_frame, _ = min(valid, key=lambda x: x[1])
            else:
                er_frame, _ = max(valid, key=lambda x: x[1])
            shared_shoulder_er_max_frames[take_id] = er_frame

        ankle_prox_x_peak_frames = {}
        ankle_min_frames = {}
        heel_contact_frames = {}
        for hand, ids in shared_take_ids_by_handedness.items():
            if not ids:
                continue
            shared_knee_peak_frames.update(
                get_peak_glove_knee_pre_br(ids, hand, shared_br_frames)
            )
            hand_ankle_prox_x_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_prox_x_peak_frames.update(hand_ankle_prox_x_peak_frames)
            hand_ankle_min_frames = get_ankle_min_frame(
                ids,
                hand,
                hand_ankle_prox_x_peak_frames,
                shared_shoulder_er_max_frames
            )
            ankle_min_frames.update(hand_ankle_min_frames)
            ankle_zero_cross_frames = get_foot_plant_frame_zero_cross(
                ids,
                hand,
                hand_ankle_min_frames,
                shared_shoulder_er_max_frames
            )
            heel_anchor_frames = {
                take_id: ankle_zero_cross_frames.get(take_id, hand_ankle_min_frames.get(take_id))
                for take_id in ids
            }
            heel_contact_frames.update(
                get_lead_heel_contact_frame(
                    ids,
                    hand,
                    hand_ankle_prox_x_peak_frames,
                    shared_shoulder_er_max_frames,
                    heel_anchor_frames
                )
            )

            for take_id in ids:
                ankle_fp_frame = ankle_zero_cross_frames.get(take_id)
                heel_fp_frame = heel_contact_frames.get(take_id)
                ankle_min_frame = hand_ankle_min_frames.get(take_id)
                prox_peak_frame = hand_ankle_prox_x_peak_frames.get(take_id)

                if ankle_fp_frame is not None and heel_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(max(ankle_fp_frame, heel_fp_frame))
                elif ankle_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(ankle_fp_frame)
                elif heel_fp_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(heel_fp_frame)
                elif ankle_min_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(ankle_min_frame)
                elif prox_peak_frame is not None:
                    shared_foot_plant_zero_cross_frames[take_id] = int(prox_peak_frame)

        for take_id, fp_frame in shared_foot_plant_zero_cross_frames.items():
            if take_id in shared_shoulder_er_max_frames:
                er_frame = shared_shoulder_er_max_frames[take_id]
                if fp_frame > er_frame:
                    shared_foot_plant_zero_cross_frames[take_id] = er_frame

        for take_id, fp_frame in shared_foot_plant_zero_cross_frames.items():
            if take_id in shared_br_frames:
                shared_fp_event_frames.append(fp_frame - shared_br_frames[take_id])
        for take_id, knee_frame in shared_knee_peak_frames.items():
            if take_id in shared_br_frames:
                shared_knee_event_frames.append(knee_frame - shared_br_frames[take_id])

        for take_id, er_frame in shared_shoulder_er_max_frames.items():
            if take_id in shared_br_frames:
                shared_mer_event_frames.append(er_frame - shared_br_frames[take_id])

        if shared_fp_event_frames:
            shared_window_start = int(np.median(shared_fp_event_frames)) - 50

    return {
        "take_ids": shared_take_ids,
        "primary_take_ids": primary_take_ids,
        "control_take_ids": control_take_ids,
        "take_order": shared_take_order,
        "take_velocity": shared_take_velocity,
        "take_date_map": shared_take_date_map,
        "take_pitcher_map": shared_take_pitcher_name_map,
        "take_handedness": shared_take_handedness,
        "take_ids_by_handedness": shared_take_ids_by_handedness,
        "br_frames": shared_br_frames,
        "foot_plant_zero_cross_frames": shared_foot_plant_zero_cross_frames,
        "shoulder_er_max_frames": shared_shoulder_er_max_frames,
        "knee_peak_frames": shared_knee_peak_frames,
        "fp_event_frames": shared_fp_event_frames,
        "knee_event_frames": shared_knee_event_frames,
        "mer_event_frames": shared_mer_event_frames,
        "window_start": shared_window_start,
    }





@st.cache_data(ttl=300, show_spinner=False)
def get_reference_velocity_bounds(pitcher_names, handedness_filter):
    """
    Returns lightweight mound-throw velocity bounds without loading time-series
    data or calculating arm slot.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            params = []
            clauses = ["t.pitch_velo IS NOT NULL", "t.throw_type = 'Mound'"]
            if pitcher_names:
                clauses.append("a.athlete_name = ANY(%s)")
                params.append(list(pitcher_names))
            if handedness_filter in ("R", "L"):
                clauses.append("a.handedness = %s")
                params.append(handedness_filter)
            cur.execute(
                f"""
                SELECT MIN(t.pitch_velo), MAX(t.pitch_velo)
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE {" AND ".join(clauses)}
                """,
                tuple(params),
            )
            row = cur.fetchone()
            return row if row else (None, None)
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_report_reference_take_metadata():
    """
    Lightweight take metadata for Report tab filters.

    This intentionally avoids time_series_data joins so dropdowns/sliders can be
    populated quickly. Arm slot is read from precomputed take_biomech_metadata
    when that table exists.
    """
    ensure_report_filter_indexes()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.take_biomech_metadata')")
            metadata_table_exists = cur.fetchone()[0] is not None
            arm_slot_select = (
                "bm.arm_slot_deg, bm.arm_slot_bucket"
                if metadata_table_exists else
                "NULL::double precision AS arm_slot_deg, NULL::text AS arm_slot_bucket"
            )
            arm_slot_join = (
                "LEFT JOIN take_biomech_metadata bm ON bm.take_id = t.take_id"
                if metadata_table_exists else
                ""
            )
            cur.execute(f"""
                SELECT
                    t.take_id,
                    t.pitch_velo,
                    t.take_date,
                    COALESCE(t.throw_type, '') AS throw_type,
                    a.athlete_name,
                    a.handedness,
                    {arm_slot_select}
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                {arm_slot_join}
                WHERE t.pitch_velo IS NOT NULL
                  AND a.handedness IN ('R', 'L')
                ORDER BY a.athlete_name, t.take_date, t.take_id
            """)
            return [
                {
                    "take_id": row[0],
                    "pitch_velo": float(row[1]) if row[1] is not None else None,
                    "take_date": row[2],
                    "take_date_label": row[2].strftime("%Y-%m-%d") if row[2] else "",
                    "throw_type": row[3],
                    "athlete_name": row[4],
                    "handedness": row[5],
                    "arm_slot_deg": float(row[6]) if row[6] is not None else None,
                    "arm_slot_bucket": row[7],
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


def filter_reference_metadata(
    metadata_rows,
    pitchers=None,
    handedness=None,
    session_dates=None,
    throw_types=None,
    velocity_range=None,
    arm_slot_ranges=None,
    arm_slot_buckets=None,
    excluded_take_ids=None,
):
    pitchers = set(pitchers or [])
    session_dates = set(session_dates or [])
    throw_types = set(throw_types or [])
    arm_slot_buckets = set(arm_slot_buckets or [])
    excluded_take_ids = set(excluded_take_ids or [])

    filtered_rows = []
    for row in metadata_rows:
        if pitchers and row["athlete_name"] not in pitchers:
            continue
        if handedness in ("R", "L") and row["handedness"] != handedness:
            continue
        if session_dates and "All Dates" not in session_dates and row["take_date_label"] not in session_dates:
            continue
        if throw_types and row["throw_type"] not in throw_types:
            continue
        if velocity_range and velocity_range[0] is not None and velocity_range[1] is not None:
            velocity = row["pitch_velo"]
            if velocity is None or not (velocity_range[0] <= velocity <= velocity_range[1]):
                continue
        if arm_slot_ranges:
            arm_slot_deg = row.get("arm_slot_deg")
            if arm_slot_deg is None or not any(minimum <= arm_slot_deg <= maximum for minimum, maximum in arm_slot_ranges):
                continue
        if arm_slot_buckets and row.get("arm_slot_bucket") not in arm_slot_buckets:
            continue
        if row["take_id"] in excluded_take_ids:
            continue
        filtered_rows.append(row)
    return filtered_rows


def metadata_pitchers(metadata_rows):
    return sorted({
        row["athlete_name"]
        for row in metadata_rows
        if row.get("athlete_name")
    })


def reference_arm_slot_ranges_from_labels(arm_slot_labels, metadata_available=True):
    if not metadata_available or not arm_slot_labels or "All" in arm_slot_labels:
        return []
    return [
        (minimum, maximum)
        for label, minimum, maximum in control_group_arm_slot_categories
        if f"{label} ({minimum}° to {maximum}°)" in arm_slot_labels
    ]


def metadata_velocity_bounds(metadata_rows):
    velocities = [
        row["pitch_velo"]
        for row in metadata_rows
        if row.get("pitch_velo") is not None
    ]
    if not velocities:
        return None, None
    return min(velocities), max(velocities)


def metadata_session_dates(metadata_rows, pitcher):
    return sorted({
        row["take_date_label"]
        for row in metadata_rows
        if row["athlete_name"] == pitcher and row["take_date_label"]
    })


def build_take_options_from_metadata(metadata_rows):
    from collections import defaultdict

    if not metadata_rows:
        return [], {}

    sorted_rows = sorted(
        metadata_rows,
        key=lambda row: (row["athlete_name"], row["take_date"] or "", row["take_id"])
    )

    date_groups = defaultdict(list)
    for row in sorted_rows:
        date_groups[(row["athlete_name"], row["take_date_label"])].append(row)

    options = []
    label_to_take_id = {}
    for (pitcher_name, date_label), rows in date_groups.items():
        for order, row in enumerate(rows, start=1):
            velo = row["pitch_velo"]
            velo_text = f"{velo:.1f}" if velo is not None else "N/A"
            label = f"{pitcher_name} | {date_label} – Pitch {order} ({velo_text} mph)"
            options.append(label)
            label_to_take_id[label] = row["take_id"]

    return options, label_to_take_id


def ensure_report_filter_indexes():
    conn = get_connection()
    try:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_theia_report_takes_filters
                            ON takes (athlete_id, throw_type, pitch_velo, take_date)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_theia_report_athletes_filters
                            ON athletes (handedness, athlete_name)
                        """
                    )
                    cur.execute("SELECT to_regclass('public.take_biomech_metadata')")
                    if cur.fetchone()[0] is not None:
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_theia_take_biomech_metadata_arm_slot
                                ON take_biomech_metadata (arm_slot_bucket, arm_slot_deg, take_id)
                            """
                        )
        except Exception:
            conn.rollback()
    finally:
        conn.close()


def format_report_velocity_range(velocity_range):
    if not velocity_range or velocity_range[0] is None or velocity_range[1] is None:
        return "All"
    return f"{float(velocity_range[0]):.1f} - {float(velocity_range[1]):.1f} mph"


def format_report_list_label(values, all_label="All", max_items=4):
    values = [str(value) for value in (values or []) if value]
    if not values or "All" in values:
        return all_label
    if len(values) <= max_items:
        return ", ".join(values)
    return f"{', '.join(values[:max_items])} + {len(values) - max_items} more"


@st.cache_data(ttl=300, show_spinner=False)
def get_pelvis_angular_velocity_x(take_ids, handedness=None):
    """
    Returns pelvis angular velocity x_data over frames for given take_ids.
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, x in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_pelvis_angular_velocity_y(take_ids, handedness=None):
    """
    Returns pelvis angular velocity y_data over frames for given take_ids.
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, y in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(y)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_pelvis_angular_velocity_z(take_ids, handedness=None):
    """
    Returns pelvis angular velocity z_data over frames for given take_ids.
    """
    if not take_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_torso_angular_velocity_component(take_ids, component, handedness=None):
    """
    Returns one torso angular velocity component over frames for given take_ids.
    """
    if not take_ids:
        return {}
    component_columns = {
        "x": "x_data",
        "y": "y_data",
        "z": "z_data",
    }
    column = component_columns.get(component)
    if column is None:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.{column}
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, value in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(value)
            return data
    finally:
        conn.close()


def get_torso_angular_velocity_x(take_ids, handedness=None):
    return get_torso_angular_velocity_component(take_ids, "x", handedness)


def get_torso_angular_velocity_y(take_ids, handedness=None):
    return get_torso_angular_velocity_component(take_ids, "y", handedness)


def get_torso_angular_velocity_z(take_ids, handedness=None):
    return get_torso_angular_velocity_component(take_ids, "z", handedness)


@st.cache_data(ttl=300, show_spinner=False)
def get_torso_pelvis_angular_velocity_component(take_ids, component, handedness=None):
    """
    Returns torso-pelvis angular velocity component data.
    Category: ORIGINAL
    Segment: TORSO_PELVIS_ANGULAR_VELOCITY
    """
    if not take_ids:
        return {}
    component_columns = {
        "x": "x_data",
        "y": "y_data",
        "z": "z_data",
    }
    column = component_columns.get(component)
    if column is None:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.{column}
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGULAR_VELOCITY'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            data = {}
            for take_id, frame, value in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(value)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_pelvis_angle_components(take_ids):
    """
    Returns pelvis angle components from ORIGINAL / PELVIS_ANGLE.
    """
    if not take_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_pelvis_angle_component(take_ids, handedness, component):
    data = get_pelvis_angle_components(take_ids)
    return {
        take_id: {"frame": curve["frame"], "value": curve[component]}
        for take_id, curve in data.items()
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_joint_angle_rotation_component(take_ids, segment_name):
    """
    Returns JOINT_ANGLES z_data for report rotation convention.

    The report uses the 0-10 style offset:
      RHP -> z_data + 90
      LHP -> 90 - z_data
    """
    if not take_ids or segment_name not in {"PELVIS", "TORSO"}:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON c.category_id = ts.category_id
                JOIN segments s ON s.segment_id = ts.segment_id
                WHERE c.category_name = 'JOINT_ANGLES'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.z_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))
            data = {}
            for take_id, frame, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(z)
            return data
    finally:
        conn.close()


def normalize_report_rotation_values(values, handedness):
    return [
        None if value is None else (90 - value if handedness == "L" else value + 90)
        for value in values
    ]


@st.cache_data(ttl=300, show_spinner=False)
def get_hip_angle_components(take_ids, segment_name):
    """
    Returns hip angle components from ORIGINAL / LT_HIP_ANGLE or RT_HIP_ANGLE.
    """
    if not take_ids:
        return {}
    if segment_name not in {"LT_HIP_ANGLE", "RT_HIP_ANGLE"}:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *tuple(take_ids)))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_hip_segment_prefix(handedness, hip_role):
    if hip_role == "back":
        return "LT" if handedness == "L" else "RT"
    return "RT" if handedness == "L" else "LT"


def normalize_hip_angle_value(value, component, segment_prefix):
    if value is None:
        return value
    if component == "x":
        # Report convention: positive is hip flexion, negative is hip extension.
        return -value
    if component == "y" and segment_prefix == "RT":
        # Report convention: positive is hip abduction for both left and right hips.
        return -value
    if component == "z" and segment_prefix == "RT":
        # Report convention: positive is hip internal rotation for both left and right hips.
        return -value
    return value


def get_hip_angle_component(take_ids, handedness, hip_role, component):
    segment_prefix = get_hip_segment_prefix(handedness, hip_role)
    segment_name = f"{segment_prefix}_HIP_ANGLE"
    data = get_hip_angle_components(take_ids, segment_name)
    return {
        take_id: {
            "frame": curve["frame"],
            "value": [
                normalize_hip_angle_value(value, component, segment_prefix)
                for value in curve[component]
            ],
        }
        for take_id, curve in data.items()
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_hip_angular_velocity_components(take_ids, segment_name):
    """
    Returns hip angular velocity components from ORIGINAL / LT_HIP_ANGULAR_VELOCITY or RT_HIP_ANGULAR_VELOCITY.
    """
    if not take_ids:
        return {}
    if segment_name not in {"LT_HIP_ANGULAR_VELOCITY", "RT_HIP_ANGULAR_VELOCITY"}:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *tuple(take_ids)))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_hip_angular_velocity_component(take_ids, handedness, hip_role, component):
    segment_name = f"{get_hip_segment_prefix(handedness, hip_role)}_HIP_ANGULAR_VELOCITY"
    data = get_hip_angular_velocity_components(take_ids, segment_name)
    return {
        take_id: {"frame": curve["frame"], "value": curve[component]}
        for take_id, curve in data.items()
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_lower_extremity_angle_components(take_ids, segment_name):
    """
    Returns knee or ankle angle components from ORIGINAL.
    """
    if not take_ids:
        return {}
    if segment_name not in {"LT_KNEE_ANGLE", "RT_KNEE_ANGLE", "LT_ANKLE_ANGLE", "RT_ANKLE_ANGLE"}:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *tuple(take_ids)))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_lower_extremity_angle_component(take_ids, handedness, leg_role, joint, component):
    side = "LT" if (leg_role == "front") == (handedness == "R") else "RT"
    segment_name = f"{side}_{joint}_ANGLE"
    data = get_lower_extremity_angle_components(take_ids, segment_name)
    return {
        take_id: {"frame": curve["frame"], "value": curve[component]}
        for take_id, curve in data.items()
    }


def normalize_identity(value, handedness=None):
    return value


@st.cache_data(ttl=300, show_spinner=False)
def get_lower_extremity_angular_velocity_components(take_ids, segment_name):
    """
    Returns knee or ankle angular velocity components from ORIGINAL.
    """
    if not take_ids:
        return {}
    if segment_name not in {
        "LT_KNEE_ANGULAR_VELOCITY",
        "RT_KNEE_ANGULAR_VELOCITY",
        "LT_ANKLE_ANGULAR_VELOCITY",
        "RT_ANKLE_ANGULAR_VELOCITY",
    }:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *tuple(take_ids)))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_lower_extremity_angular_velocity_component(take_ids, handedness, leg_role, joint, component):
    side = "LT" if (leg_role == "front") == (handedness == "R") else "RT"
    segment_name = f"{side}_{joint}_ANGULAR_VELOCITY"
    data = get_lower_extremity_angular_velocity_components(take_ids, segment_name)
    return {
        take_id: {"frame": curve["frame"], "value": curve[component]}
        for take_id, curve in data.items()
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_torso_pelvis_angle_components(take_ids):
    """
    Returns torso-pelvis angle components from ORIGINAL / TORSO_PELVIS_ANGLE.
    """
    if not take_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT ts.take_id, ts.frame, ts.x_data, ts.y_data, ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = 'TORSO_PELVIS_ANGLE'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))
            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_hand_cg_velocity_components(take_ids, handedness):
    """
    Returns throwing-hand center-of-gravity velocity components.
    Category: KINETIC_KINEMATIC_CGVel
    Segments: RHA / LHA
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RHA" if handedness == "R" else "LHA"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data,
                    ts.z_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGVel'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            data = {}
            for take_id, frame, x, y, z in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "x": [], "y": [], "z": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["x"].append(x)
                data[take_id]["y"].append(y)
                data[take_id]["z"].append(z)
            return data
    finally:
        conn.close()


def get_hand_cg_velocity_component(take_ids, handedness, component):
    data = get_hand_cg_velocity_components(take_ids, handedness)
    return {
        take_id: {"frame": curve["frame"], "value": curve[component]}
        for take_id, curve in data.items()
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_center_of_mass_velocity_component(take_ids, component, handedness=None):
    """
    Returns Center of Mass velocity component data.

    Category: PROCESSED
    Segment: CenterOfMass_VELO
    """
    if not take_ids:
        return {}
    component_columns = {
        "x": "x_data",
        "y": "y_data",
        "z": "z_data",
    }
    column = component_columns.get(component)
    if column is None:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.{column}
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'PROCESSED'
                  AND s.segment_name = 'CenterOfMass_VELO'
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, tuple(take_ids))

            data = {}
            for take_id, frame, value in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(value)

            return data
    finally:
        conn.close()


st.title("Terra Sports Biomechanics Dashboard")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_labels = ["Kinematic Sequence", "Kinematics", "Energy Flow", "Report"]
tab_kinematic, tab_joint, tab_energy, tab_report = st.tabs(tab_labels)

# Shared state used across tabs; Kinematic Sequence populates these when data exists.
take_ids = []
take_order = {}
take_velocity = {}
take_date_map = {}
br_frames = {}
take_pitcher_map = {}
take_handedness = {}
take_ids_by_handedness = {}
foot_plant_zero_cross_frames = {}
shoulder_er_max_frames = {}
knee_peak_frames = {}
fp_event_frames = []
knee_event_frames = []
mer_event_frames = []
window_start = -100

# Workaround for Streamlit tab reset on rerun:
# persist active tab in URL query param and re-select it after rerender.
components.html(
    """
    <script>
    const TAB_PARAM = "active_tab";

    function getActiveTabFromUrl() {
      const url = new URL(parent.window.location.href);
      return url.searchParams.get(TAB_PARAM);
    }

    function setActiveTabInUrl(tabLabel) {
      const url = new URL(parent.window.location.href);
      url.searchParams.set(TAB_PARAM, tabLabel);
      parent.window.history.replaceState({}, "", url.toString());
    }

    function getTabButtons() {
      return Array.from(parent.document.querySelectorAll('button[role="tab"]'));
    }

    function bindTabClicks() {
      const buttons = getTabButtons();
      buttons.forEach((button) => {
        if (button.dataset.terraTabBound === "1") return;
        button.dataset.terraTabBound = "1";
        button.addEventListener("click", () => {
          setActiveTabInUrl(button.textContent.trim());
        });
      });
    }

    function restoreActiveTab() {
      const desiredTab = getActiveTabFromUrl();
      if (!desiredTab) return;

      const buttons = getTabButtons();
      const target = buttons.find(
        (button) => button.textContent.trim() === desiredTab
      );
      if (!target) return;

      if (target.getAttribute("aria-selected") !== "true") {
        target.click();
      }
    }

    function syncTabs() {
      bindTabClicks();
      restoreActiveTab();

      const selected = getTabButtons().find(
        (button) => button.getAttribute("aria-selected") === "true"
      );
      if (selected && !getActiveTabFromUrl()) {
        setActiveTabInUrl(selected.textContent.trim());
      }
    }

    syncTabs();
    let attempts = 0;
    const interval = setInterval(() => {
      syncTabs();
      attempts += 1;
      if (attempts > 30) clearInterval(interval);
    }, 200);
    </script>
    """,
    height=0,
)

shared_state = build_shared_dashboard_state()
take_ids = shared_state["take_ids"]
primary_take_ids = shared_state["primary_take_ids"]
control_take_ids = shared_state["control_take_ids"]
take_order = shared_state["take_order"]
take_velocity = shared_state["take_velocity"]
take_date_map = shared_state["take_date_map"]
br_frames = shared_state["br_frames"]
take_pitcher_map = shared_state["take_pitcher_map"]
take_handedness = shared_state["take_handedness"]
take_ids_by_handedness = shared_state["take_ids_by_handedness"]
foot_plant_zero_cross_frames = shared_state["foot_plant_zero_cross_frames"]
shoulder_er_max_frames = shared_state["shoulder_er_max_frames"]
knee_peak_frames = shared_state["knee_peak_frames"]
fp_event_frames = shared_state["fp_event_frames"]
knee_event_frames = shared_state["knee_event_frames"]
mer_event_frames = shared_state["mer_event_frames"]
window_start = shared_state["window_start"]
comparison_grouping_enabled = group_mode_enabled or bool(control_take_ids)

if control_take_ids:
    control_group_label = "Control Group"
    if not group_mode_enabled:
        selected_group_label = ", ".join(selected_pitchers) if selected_pitchers else "Selected Takes"
        group_color_map = {
            selected_group_label: group_palette[0],
            control_group_label: group_palette[1]
        }
        take_group_map = {
            **{tid: selected_group_label for tid in primary_take_ids},
            **{tid: control_group_label for tid in control_take_ids}
        }
    else:
        group_color_map[control_group_label] = group_palette[len(group_color_map) % len(group_palette)]
        for tid in control_take_ids:
            take_group_map[tid] = control_group_label

multi_pitcher_mode = len(set(take_pitcher_map.values())) > 1
group_mode_aggregate_across_pitchers = group_mode_enabled
show_group_pitcher_breakout = multi_pitcher_mode and not group_mode_aggregate_across_pitchers


with tab_kinematic:
    st.subheader("Kinematic Sequence")
    render_group_selection_summary()
    st.markdown(
        """
        <style>
        .ks-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .ks-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    controls_col, options_col, spacer_col = st.columns([1.45, 1.75, 2.2])
    with controls_col:
        st.markdown('<div class="ks-controls-label">Display Mode</div>', unsafe_allow_html=True)
        display_mode = st.segmented_control(
            "Display Mode",
            ["Individual Throws", "Grouped"],
            default="Grouped",
            key="ks_display_mode",
            label_visibility="collapsed",
        )
    with options_col:
        st.markdown('<div class="ks-controls-label ks-toggle-label">Options</div>', unsafe_allow_html=True)
        event_toggle_col, signal_toggle_col = st.columns(2)
        with event_toggle_col:
            show_ks_fp_iqr_band = st.toggle(
                "Event Bands",
                value=False,
                key="ks_show_fp_iqr_band",
                help="Shows the middle 50% range for event timing across selected throws.",
            )
        with signal_toggle_col:
            show_ks_signal_iqr_band = st.toggle(
                "Signal Bands",
                value=True,
                key="ks_show_signal_iqr_band",
                help="Shows the middle 50% range of angular velocity around each grouped mean line.",
            )
    with spacer_col:
        st.markdown("")
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] + div[data-testid="stVerticalBlock"] {
            margin-top: -0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not take_ids:
        st.info("No takes found for this selection.")
    else:
        def load_by_handedness(loader_fn):
            merged = {}
            for hand, ids in take_ids_by_handedness.items():
                if ids:
                    merged.update(loader_fn(ids, hand))
            return merged

        data = get_pelvis_angular_velocity(take_ids)
        cg_data = load_by_handedness(get_hand_cg_velocity)
        torso_data = get_torso_angular_velocity(take_ids)
        elbow_data = load_by_handedness(get_elbow_angular_velocity)
        shoulder_ir_data = load_by_handedness(get_shoulder_ir_velocity)
        pre_fp_frames = ms_to_rel_frame(100)
        post_br_frames = ms_to_rel_frame(150)
        kinematic_window_start = (
            int(np.median(fp_event_frames)) - pre_fp_frames
            if fp_event_frames else -pre_fp_frames
        )
        kinematic_window_end = post_br_frames
        window_start_ms = rel_frame_to_ms(kinematic_window_start)
        window_end_ms = rel_frame_to_ms(kinematic_window_end)

        fig = go.Figure()
        grouped_pelvis = {}
        grouped_torso = {}
        grouped_elbow = {}
        grouped_shoulder_ir = {}

        # --- Date-based dash style map (Kinematic Sequence) ---
        unique_dates = sorted(set(take_date_map.values()))
        dash_styles = ["solid", "dash", "dot", "dashdot"]
        date_dash_map = {
            d: dash_styles[i % len(dash_styles)]
            for i, d in enumerate(unique_dates)
        }

        # Track legend entries to avoid duplicates (for condensed legend)
        legend_keys_added = set()

        for take_id, d in data.items():
            frames = d["frame"]
            values = d["z"]
            take_hand = take_handedness.get(take_id)
            take_group_label = take_group_map.get(take_id, "")
            control_group_take = is_control_group_label(take_group_label)
            hover_pitcher_name = "" if control_group_take else take_pitcher_map.get(take_id, "")

            # -----------------------------
            # Ball Release Detection (CGVel)
            # -----------------------------
            if take_id not in cg_data:
                continue

            cg_frames = cg_data[take_id]["frame"]
            cg_values = cg_data[take_id]["x"]

            valid_cg = [(i, v) for i, v in enumerate(cg_values) if v is not None]
            if not valid_cg:
                continue

            br_idx, _ = max(valid_cg, key=lambda x: x[1])
            br_frame = cg_frames[br_idx]

            # -----------------------------
            # Peak Glove-Side Knee Height
            # -----------------------------
            knee_rel_frame = None
            if take_id in knee_peak_frames:
                knee_rel_frame = knee_peak_frames[take_id] - br_frame

            # MER defined as max shoulder external rotation prior to ball release
            # -----------------------------
            mer_rel_frame = None
            if take_id in shoulder_er_max_frames:
                mer_rel_frame = shoulder_er_max_frames[take_id] - br_frame

            # -----------------------------
            # Normalize time to Ball Release
            # -----------------------------
            norm_frames = []
            norm_values = []

            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel_frame = f - br_frame

                # Keep frames from 150 before median FP through +150 after BR
                if (
                    rel_frame >= kinematic_window_start
                    and rel_frame <= kinematic_window_end
                ):
                    norm_frames.append(rel_frame_to_ms(rel_frame))
                    # Handedness normalization for Pelvis AV (Kinematic Sequence only)
                    if take_hand == "L":
                        norm_values.append(-v)
                    else:
                        norm_values.append(v)

            grouped_pelvis[take_id] = {
                "frame": norm_frames,
                "value": norm_values
            }

            # -----------------------------
            # Normalize Torso Angular Velocity
            # -----------------------------
            if take_id in torso_data:
                torso_frames = torso_data[take_id]["frame"]
                torso_values = torso_data[take_id]["z"]

                norm_torso_frames = []
                norm_torso_values = []

                for f, v in zip(torso_frames, torso_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_torso_frames.append(rel_frame_to_ms(rel_frame))
                        # Handedness normalization for Torso AV (Kinematic Sequence only)
                        if take_hand == "L":
                            norm_torso_values.append(-v)
                        else:
                            norm_torso_values.append(v)

                grouped_torso[take_id] = {
                    "frame": norm_torso_frames,
                    "value": norm_torso_values
                }

                if norm_torso_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Torso" if control_group_take else f"Torso_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    trace_name = (
                        f"Control Group | Torso – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                    ) if control_group_take else (
                        f"{take_group_label} | Torso – {take_date_map[take_id]} | "
                        f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                    ) if comparison_grouping_enabled else None
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_torso_frames,
                            y=norm_torso_values,
                            mode="lines",
                            line=dict(
                                color="orange",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Torso", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_torso_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=trace_name,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Torso + Date)
                    legend_key = ("Control Group", "Torso") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="orange",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Torso AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Torso AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Torso AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Torso AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Torso AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)

            # -----------------------------
            # Normalize Elbow Angular Velocity (Extension)
            # -----------------------------
            if take_id in elbow_data:
                elbow_frames = elbow_data[take_id]["frame"]
                elbow_values = elbow_data[take_id]["x"]

                norm_elbow_frames = []
                norm_elbow_values = []

                for f, v in zip(elbow_frames, elbow_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_elbow_frames.append(rel_frame_to_ms(rel_frame))
                        # Flip sign so elbow extension is positive on the plot
                        norm_elbow_values.append(-v)

                grouped_elbow[take_id] = {
                    "frame": norm_elbow_frames,
                    "value": norm_elbow_values
                }

                if norm_elbow_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Elbow" if control_group_take else f"Elbow_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_elbow_frames,
                            y=norm_elbow_values,
                            mode="lines",
                            line=dict(
                                color="green",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Elbow", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_elbow_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=(
                                f"Control Group | Elbow – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if control_group_take else (
                                f"{take_group_label} | Elbow – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if comparison_grouping_enabled else None,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Elbow + Date)
                    legend_key = ("Control Group", "Elbow") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="green",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Elbow AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Elbow AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Elbow AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Elbow AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Elbow AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)

            # -----------------------------
            # Normalize Shoulder IR Angular Velocity
            # -----------------------------
            if take_id in shoulder_ir_data:
                sh_frames = shoulder_ir_data[take_id]["frame"]
                sh_values = shoulder_ir_data[take_id]["x"]

                norm_sh_frames = []
                norm_sh_values = []

                for f, v in zip(sh_frames, sh_values):
                    if v is None:
                        continue

                    rel_frame = f - br_frame
                    if (
                        rel_frame >= kinematic_window_start
                        and rel_frame <= kinematic_window_end
                    ):
                        norm_sh_frames.append(rel_frame_to_ms(rel_frame))
                        # Normalize so IR velocity is positive for both handedness
                        if take_hand == "L":
                            norm_sh_values.append(-v)
                        else:
                            norm_sh_values.append(v)

                grouped_shoulder_ir[take_id] = {
                    "frame": norm_sh_frames,
                    "value": norm_sh_values
                }

                if norm_sh_frames and display_mode == "Individual Throws":
                    legendgroup = "Control_Group_Shoulder_IR" if control_group_take else f"Shoulder IR_{take_date_map[take_id]}"
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    # Actual data trace (no legend)
                    fig.add_trace(
                        go.Scatter(
                            x=norm_sh_frames,
                            y=norm_sh_values,
                            mode="lines",
                            line=dict(
                                color="red",
                                dash=date_dash_map[take_date_map[take_id]]
                            ),
                            customdata=[[ "Shoulder", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_sh_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            name=(
                                f"Control Group | Shoulder – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if control_group_take else (
                                f"{take_group_label} | Shoulder – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                            ) if comparison_grouping_enabled else None,
                            showlegend=False,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    # Legend-only trace (once per Shoulder IR + Date)
                    legend_key = ("Control Group", "Shoulder IR") if control_group_take else None
                    if control_group_take and legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color="red",
                                    dash=date_dash_map[take_date_map[take_id]],
                                    width=4
                                ),
                                name=(
                                    f"Control Group | Shoulder IR AV"
                                    if (comparison_grouping_enabled and control_group_take) else
                                    f"{take_group_label} | Shoulder IR AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if (comparison_grouping_enabled and multi_pitcher_mode) else
                                    f"{take_group_label} | Shoulder IR AV | {take_date_map[take_id]}"
                                    if comparison_grouping_enabled else
                                    f"Shoulder IR AV | {take_date_map[take_id]} | {pitcher_name}"
                                    if multi_pitcher_mode else
                                    f"Shoulder IR AV | {take_date_map[take_id]}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup,
                                legendgrouptitle_text=None
                            )
                        )
                        legend_keys_added.add(legend_key)
            if not norm_frames:
                continue

            if display_mode == "Individual Throws":
                legendgroup = "Control_Group_Pelvis" if control_group_take else f"Pelvis_{take_date_map[take_id]}"
                pitcher_name = take_pitcher_map.get(take_id, "")
                # Actual data trace (no legend)
                fig.add_trace(
                    go.Scatter(
                        x=norm_frames,
                        y=norm_values,
                        mode="lines",
                        line=dict(
                            color="blue",
                            dash=date_dash_map[take_date_map[take_id]]
                        ),
                        customdata=[[ "Pelvis", take_date_map[take_id], take_order[take_id], take_velocity[take_id], hover_pitcher_name ]] * len(norm_frames),
                        hovertemplate=(
                            "%{customdata[0]} – %{customdata[1]} | "
                            "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                            + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                            + "<br>Angular Velocity: %{y:.1f}°/s"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        name=(
                            f"Control Group | Pelvis – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                        ) if control_group_take else (
                            f"{take_group_label} | Pelvis – {take_date_map[take_id]} | "
                            f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} MPH)"
                        ) if comparison_grouping_enabled else None,
                        showlegend=False,
                        legendgroup=legendgroup,
                        legendgrouptitle_text=None
                    )
                )
                # Legend-only trace (once per Pelvis + Date)
                legend_key = ("Control Group", "Pelvis") if control_group_take else None
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color="blue",
                                dash=date_dash_map[take_date_map[take_id]],
                                width=4
                            ),
                        name=(
                            f"Control Group | Pelvis AV"
                            if (comparison_grouping_enabled and control_group_take) else
                            f"{take_group_label} | Pelvis AV | {take_date_map[take_id]} | {pitcher_name}"
                            if (comparison_grouping_enabled and multi_pitcher_mode) else
                            f"{take_group_label} | Pelvis AV | {take_date_map[take_id]}"
                            if comparison_grouping_enabled else
                            f"Pelvis AV | {take_date_map[take_id]} | {pitcher_name}"
                            if multi_pitcher_mode else
                            f"Pelvis AV | {take_date_map[take_id]}"
                        ),
                            showlegend=True,
                            legendgroup=legendgroup,
                            legendgrouptitle_text=None
                        )
                    )
                    legend_keys_added.add(legend_key)

        # --- Store peak summary for table ---
        kinematic_peak_rows = []

        if display_mode == "Grouped":
            color_map = {
                "Pelvis": "blue",
                "Torso": "orange",
                "Elbow": "green",
                "Shoulder": "red"
            }
            grouped_peak_time_reference = {}

            # --- Condensed legend: track (Segment, Date) pairs ---
            legend_keys_added = set()
            peak_marker_traces = []
            peak_marker_annotations = []

            for label, curves in [
                ("Pelvis", grouped_pelvis),
                ("Torso", grouped_torso),
                ("Elbow", grouped_elbow),
                ("Shoulder", grouped_shoulder_ir)
            ]:
                if not curves:
                    continue

                # Group curves by date
                from collections import defaultdict
                curves_by_date = defaultdict(dict)
                for take_id, d in curves.items():
                    date = take_date_map[take_id]
                    pitcher_name = take_pitcher_map.get(take_id, "")
                    group_label = take_group_map.get(take_id, "")
                    if comparison_grouping_enabled and is_control_group_label(group_label):
                        date_key = group_label
                    elif comparison_grouping_enabled:
                        date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
                    else:
                        date_key = (pitcher_name, date) if multi_pitcher_mode else date
                    curves_by_date[date_key][take_id] = d
                for date_key, curves_date in curves_by_date.items():
                    if comparison_grouping_enabled and date_key == "Control Group":
                        group_label = "Control Group"
                        pitcher_name = ""
                        date = "Selected Takes"
                    elif comparison_grouping_enabled and show_group_pitcher_breakout:
                        group_label, pitcher_name, date = date_key
                    elif comparison_grouping_enabled:
                        group_label = date_key
                        date = "Selected Takes"
                        pitcher_name = ""
                    elif multi_pitcher_mode and not comparison_grouping_enabled:
                        pitcher_name, date = date_key
                        group_label = ""
                    else:
                        date = date_key
                        pitcher_name = ""
                        group_label = ""
                    x_date, y_date, q1_date, q3_date = aggregate_curves(curves_date, "Mean")
                    avg_velocity = (
                        float(np.mean([
                            take_velocity[tid]
                            for tid in curves_date.keys()
                            if tid in take_velocity and take_velocity[tid] is not None
                        ]))
                        if any(
                            tid in take_velocity and take_velocity[tid] is not None
                            for tid in curves_date.keys()
                        ) else None
                    )
                    color = color_map[label]
                    # Smoothing
                    if len(y_date) >= 11:
                        y_date = savgol_filter(y_date, window_length=7, polyorder=3)
                    dash = date_dash_map.get(date, "solid")
                    legendgroup = f"{label}_{date}_{pitcher_name}" if show_group_pitcher_breakout else f"{label}_{date}"
                    # --- IQR band (draw first so the line color stays visually true on top) ---
                    if show_ks_signal_iqr_band:
                        fig.add_trace(
                            go.Scatter(
                                x=x_date + x_date[::-1],
                                y=q3_date + q1_date[::-1],
                                fill="toself",
                                fillcolor=to_rgba(color, alpha=0.30),
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=legendgroup
                            )
                        )
                    # --- Grouped curve (no legend, but legendgroup set) ---
                    fig.add_trace(
                        go.Scatter(
                            x=x_date,
                            y=y_date,
                            mode="lines",
                            line=dict(
                                width=4,
                                color=color,
                                dash=dash,
                            ),
                            customdata=[[label, date, group_label, pitcher_name]] * len(x_date),
                            hovertemplate=(
                                (f"{group_label}<br>" if comparison_grouping_enabled else "")
                                + ("%{customdata[0]}" if comparison_grouping_enabled else "%{customdata[0]} | %{customdata[1]}")
                                + (" | %{customdata[3]}" if show_group_pitcher_breakout else "")
                                + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                                + "<br>Angular Velocity: %{y:.1f}°/s"
                                + "<br>Time: %{x:.0f} ms rel BR"
                                + "<extra></extra>"
                            ),
                            showlegend=False,
                            legendgroup=legendgroup
                        )
                    )
                    # --- Legend-only trace (once per Segment + Date, legendgroup set) ---
                    legend_key = (label, date, pitcher_name) if show_group_pitcher_breakout else (label, date)
                    if legend_key not in legend_keys_added:
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                line=dict(
                                    color=color,
                                    dash=dash,
                                    width=4
                                ),
                            name=(
                                    f"{group_label} | {label} AV | {date} | {pitcher_name}"
                                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                                    f"{group_label} | {label} AV | {date}"
                                    if comparison_grouping_enabled else
                                    f"{label} AV | {date} | {pitcher_name}"
                                    if show_group_pitcher_breakout else
                                    f"{label} AV | {date}"
                                ),
                                showlegend=True,
                                legendgroup=legendgroup
                            )
                        )
                        legend_keys_added.add(legend_key)
                    # --- Peak arrow and marker for this grouped curve ---
                    if len(y_date) > 0:
                        # Restrict pelvis & torso peak search to FP → BR
                        if label in ["Pelvis", "Torso"] and fp_event_frames:
                            fp_rel = rel_frame_to_ms(int(np.median(fp_event_frames)))
                            valid_idxs = [
                                i for i, xf in enumerate(x_date)
                                if fp_rel <= xf <= 0
                            ]
                            if not valid_idxs:
                                continue
                            max_idx = max(valid_idxs, key=lambda i: y_date[i])
                        else:
                            # Elbow / Shoulder IR use full window
                            max_idx = int(np.argmax(y_date))
                        max_x = x_date[max_idx]
                        max_y = y_date[max_idx]
                        reference_time_ms_grouped = None
                        if label == "Pelvis" and fp_event_frames:
                            fp_rel = rel_frame_to_ms(int(np.median(fp_event_frames)))
                            reference_time_ms_grouped = max_x - fp_rel
                        elif label == "Torso":
                            pelvis_peak_time = grouped_peak_time_reference.get((date_key, "Pelvis"))
                            if pelvis_peak_time is not None:
                                reference_time_ms_grouped = max_x - pelvis_peak_time
                        elif label == "Elbow":
                            torso_peak_time = grouped_peak_time_reference.get((date_key, "Torso"))
                            if torso_peak_time is not None:
                                reference_time_ms_grouped = max_x - torso_peak_time
                        elif label == "Shoulder":
                            elbow_peak_time = grouped_peak_time_reference.get((date_key, "Elbow"))
                            if elbow_peak_time is not None:
                                reference_time_ms_grouped = max_x - elbow_peak_time

                        grouped_peak_time_reference[(date_key, label)] = max_x

                        local_y_min = min(y_date) if len(y_date) > 0 else max_y
                        local_y_max = max(y_date) if len(y_date) > 0 else max_y
                        local_y_span = max(local_y_max - local_y_min, 1)
                        peak_marker_y = max_y + max(0.07 * local_y_span, 55)

                        kinematic_peak_rows.append({
                            **({"Group": group_label} if comparison_grouping_enabled else {}),
                            **({"Pitcher": pitcher_name} if show_group_pitcher_breakout else {}),
                            "Session Date": date,
                            "Velocity (mph)": (
                                float(np.mean([
                                    take_velocity[tid]
                                    for tid in curves_date.keys()
                                    if tid in take_velocity and take_velocity[tid] is not None
                                ]))
                                if any(
                                    tid in take_velocity and take_velocity[tid] is not None
                                    for tid in curves_date.keys()
                                ) else None
                            ),
                            "Segment": segment_display_name(label),
                            "Peak Value (°/s)": max_y,
                            "Peak Time from Reference (ms)": reference_time_ms_grouped
                        })
                        peak_marker_traces.append(
                            go.Scatter(
                                x=[max_x],
                                y=[max_y],
                                mode="markers",
                                marker=dict(
                                    symbol="circle",
                                    size=10,
                                    color=color,
                                    opacity=0,
                                ),
                                showlegend=False,
                                legendgroup=legendgroup,
                                customdata=[[
                                    label,
                                    date,
                                    group_label,
                                    pitcher_name,
                                    max_y,
                                    max_x,
                                    peak_marker_y,
                                ]],
                                hovertemplate=(
                                    ("%{customdata[2]} | " if comparison_grouping_enabled else "")
                                    + "%{customdata[0]} | %{customdata[1]}"
                                    + (" | %{customdata[3]}" if show_group_pitcher_breakout else "")
                                    + "<br>Peak Angular Velocity: %{customdata[4]:.1f}°/s"
                                    + "<br>Peak Time: %{customdata[5]:.0f} ms rel BR"
                                    + "<extra></extra>"
                                ),
                            )
                        )
                        peak_marker_annotations.append(
                            dict(
                                x=max_x,
                                y=max_y,
                                xref="x",
                                yref="y",
                                text="",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.2,
                                arrowwidth=2,
                                arrowcolor=color,
                                ax=0,
                                ay=-40,
                            )
                        )
            for peak_marker_trace in peak_marker_traces:
                fig.add_trace(peak_marker_trace)
            for peak_marker_annotation in peak_marker_annotations:
                fig.add_annotation(**peak_marker_annotation)

        # Median Refined Foot Plant (zero-cross) event
        if fp_event_frames:
            add_event_iqr_band(fig, fp_event_frames, "green", show_ks_fp_iqr_band)
            median_fp_frame = rel_frame_to_ms(int(np.median(fp_event_frames)))

            fig.add_vline(
                x=median_fp_frame,
                line_width=3,
                line_dash="dash",
                line_color="green",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_fp_frame,
                y=1.055,
                xref="x",
                yref="paper",
                text="FP",
                showarrow=False,
                font=dict(color="green", size=14),
                align="center"
            )

        # Median Max Shoulder ER event
        if mer_event_frames:
            add_event_iqr_band(fig, mer_event_frames, "red", show_ks_fp_iqr_band)
            median_mer_frame = rel_frame_to_ms(int(np.median(mer_event_frames)))

            fig.add_vline(
                x=median_mer_frame,
                line_width=3,
                line_dash="dash",
                line_color="red",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_mer_frame,
                y=1.055,
                xref="x",
                yref="paper",
                text="MER",
                showarrow=False,
                font=dict(color="red", size=14),
                align="center"
            )

        # Normalized Ball Release reference line
        add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_ks_fp_iqr_band)
        fig.add_vline(
            x=0,
            line_width=3,
            line_dash="dash",
            line_color="blue",
            opacity=0.9
        )
        fig.add_annotation(
            x=0,
            y=1.055,
            xref="x",
            yref="paper",
            text="BR",
            showarrow=False,
            font=dict(color="blue", size=14),
            align="center"
        )

        grouped_visible_y_vals = []
        yaxis_range = None
        if display_mode == "Grouped":
            for trace in fig.data:
                if getattr(trace, "type", None) != "scatter":
                    continue
                if getattr(trace, "mode", None) != "lines":
                    continue
                if getattr(trace, "fill", None) == "toself":
                    continue

                trace_y = getattr(trace, "y", None)
                if trace_y is None:
                    continue

                grouped_visible_y_vals.extend(
                    v for v in trace_y
                    if v is not None and np.isfinite(v)
                )

            if grouped_visible_y_vals:
                y_min = min(grouped_visible_y_vals)
                y_max = max(grouped_visible_y_vals)
                y_span = max(y_max - y_min, 1)
                yaxis_range = [y_min - (0.10 * y_span), y_max + (0.22 * y_span)]

        fig.update_layout(
            xaxis_title="Time Relative to Ball Release (ms)",
            yaxis_title="Angular Velocity",
            yaxis=dict(
                ticksuffix="°/s",
                range=yaxis_range,
            ),
            xaxis_range=[window_start_ms, window_end_ms],
            height=600,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="center",
                x=0.5,
                groupclick="togglegroup"
            ),
            hoverlabel=dict(
                namelength=-1,
                font_size=13
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="kinematic_sequence_plot",
            config={
                "toImageButtonOptions": {
                    "filename": "kinematic_sequence"
                }
            },
        )

        # --- Kinematic Sequence Peak Summary Table (Individual Throws) ---
        if display_mode == "Individual Throws":

            individual_rows = []

            for take_id in take_ids:
                if take_id not in br_frames:
                    continue

                br_frame = br_frames[take_id]

                # Helper to compute peak and frame
                def peak_and_frame(curves, invert=False):
                    if take_id not in curves:
                        return None, None

                    vals = curves[take_id]["value"]
                    frames = curves[take_id]["frame"]
                    if not vals:
                        return None, None

                    if invert:
                        idx = int(np.argmax(vals))
                    else:
                        idx = int(np.argmax(vals))

                    return vals[idx], frames[idx]

                pelvis_peak, pelvis_frame = peak_and_frame(grouped_pelvis)
                # Pelvis peak timing from Foot Plant (zero-cross), in ms (250 Hz)
                pelvis_time_ms = None
                fp_abs = foot_plant_zero_cross_frames.get(take_id)  # absolute frame
                br_abs = br_frames.get(take_id)  # absolute frame

                if pelvis_frame is not None and fp_abs is not None and br_abs is not None:
                    fp_rel = fp_abs - br_abs  # FP relative to BR (frames)
                    pelvis_time_ms = pelvis_frame - rel_frame_to_ms(fp_rel)
                torso_peak, torso_frame = peak_and_frame(grouped_torso)
                elbow_peak, elbow_frame = peak_and_frame(grouped_elbow)
                shoulder_peak, shoulder_frame = peak_and_frame(grouped_shoulder_ir)
                torso_time_from_pelvis_ms = (
                    torso_frame - pelvis_frame
                    if torso_frame is not None and pelvis_frame is not None else None
                )
                elbow_time_from_torso_ms = (
                    elbow_frame - torso_frame
                    if elbow_frame is not None and torso_frame is not None else None
                )
                shoulder_time_from_elbow_ms = (
                    shoulder_frame - elbow_frame
                    if shoulder_frame is not None and elbow_frame is not None else None
                )

                individual_rows.append({
                    **({"Group": take_group_map.get(take_id, "")} if comparison_grouping_enabled else {}),
                    **({"Pitcher": take_pitcher_map.get(take_id)} if multi_pitcher_mode else {}),
                    "Session Date": take_date_map[take_id],
                    "Pitch": take_order[take_id],
                    "Velocity (mph)": take_velocity[take_id],
                    "Pelvis Rotation Peak (°/s)": pelvis_peak,
                    "Pelvis Rotation Time from FP (ms)": pelvis_time_ms,
                    "Torso Rotation Peak (°/s)": torso_peak,
                    "Torso Rotation Time from Peak Pelvis (ms)": torso_time_from_pelvis_ms,
                    "Elbow Extension Peak (°/s)": elbow_peak,
                    "Elbow Extension Time from Peak Torso (ms)": elbow_time_from_torso_ms,
                    "Shoulder Internal Rotation Peak (°/s)": shoulder_peak,
                    "Shoulder Internal Rotation Time from Peak Elbow (ms)": shoulder_time_from_elbow_ms
                })

            if individual_rows:
                import pandas as pd

                st.markdown("### Kinematic Sequence - Individual Throws")

                df_individual = pd.DataFrame(individual_rows)

                # Sort logically: date → pitch order
                sort_cols = ["Session Date", "Pitch"]
                if comparison_grouping_enabled and "Group" in df_individual.columns:
                    sort_cols = ["Group"] + sort_cols
                if multi_pitcher_mode and "Pitcher" in df_individual.columns:
                    sort_cols = ["Pitcher"] + sort_cols
                df_individual = df_individual.sort_values(sort_cols)

                index_cols = ["Session Date", "Velocity (mph)"]
                if comparison_grouping_enabled and "Group" in df_individual.columns:
                    index_cols = ["Group"] + index_cols
                if multi_pitcher_mode and "Pitcher" in df_individual.columns:
                    index_cols = (
                        (["Group"] if comparison_grouping_enabled and "Group" in df_individual.columns else [])
                        + ["Pitcher", "Session Date", "Velocity (mph)"]
                    )

                segment_metric_map = {
                    "Pelvis Rotation Peak (°/s)": ("Pelvis Rotation", "Peak (°/s)"),
                    "Pelvis Rotation Time from FP (ms)": ("Pelvis Rotation", "Peak Time from Foot Plant (ms)"),
                    "Torso Rotation Peak (°/s)": ("Torso Rotation", "Peak (°/s)"),
                    "Torso Rotation Time from Peak Pelvis (ms)": ("Torso Rotation", "Peak Time from Peak Pelvis (ms)"),
                    "Elbow Extension Peak (°/s)": ("Elbow Extension", "Peak (°/s)"),
                    "Elbow Extension Time from Peak Torso (ms)": ("Elbow Extension", "Peak Time from Peak Torso (ms)"),
                    "Shoulder Internal Rotation Peak (°/s)": ("Shoulder Internal Rotation", "Peak (°/s)"),
                    "Shoulder Internal Rotation Time from Peak Elbow (ms)": ("Shoulder Internal Rotation", "Peak Time from Peak Elbow (ms)"),
                }
                value_cols = list(segment_metric_map.keys())
                df_individual_display = df_individual[index_cols + value_cols].set_index(index_cols)
                df_individual_display.columns = pd.MultiIndex.from_tuples(
                    [segment_metric_map[col] for col in value_cols]
                )

                segment_order = [
                    "Pelvis Rotation",
                    "Torso Rotation",
                    "Elbow Extension",
                    "Shoulder Internal Rotation",
                ]
                ordered_cols = []
                for seg in segment_order:
                    segment_metrics = [
                        "Peak (°/s)",
                        {
                            "Pelvis Rotation": "Peak Time from Foot Plant (ms)",
                            "Torso Rotation": "Peak Time from Peak Pelvis (ms)",
                            "Elbow Extension": "Peak Time from Peak Torso (ms)",
                            "Shoulder Internal Rotation": "Peak Time from Peak Elbow (ms)",
                        }[seg],
                    ]
                    for metric_name in segment_metrics:
                        if (seg, metric_name) in df_individual_display.columns:
                            ordered_cols.append((seg, metric_name))
                if ordered_cols:
                    df_individual_display = df_individual_display[ordered_cols]

                segment_colors = {
                    "Pelvis Rotation": "#DBEAFE",
                    "Torso Rotation": "#FED7AA",
                    "Elbow Extension": "#DCFCE7",
                    "Shoulder Internal Rotation": "#FEE2E2",
                }

                def style_segment_headers(headers):
                    return [
                        f"background-color: {segment_colors.get(header, '#FFFFFF')}; color: #111827;"
                        if header in segment_colors else ""
                        for header in headers
                    ]

                def fmt(val, decimals=2):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    return f"{val:.{decimals}f}"

                styled_individual = (
                    df_individual_display
                    .style
                    .format(lambda x: fmt(x, 1) if isinstance(x, (int, float, np.floating)) else x)
                    .apply_index(style_segment_headers, axis="columns", level=0)
                    .set_table_styles([
                        {"selector": "th", "props": [("text-align", "center")]},
                        {"selector": "th.row_heading", "props": [("text-align", "center")]},
                        {"selector": "th.index_name", "props": [("text-align", "center")]},
                        *(
                            [
                                {
                                    "selector": "th.row_heading.level0",
                                    "props": [("min-width", "80px"), ("max-width", "80px")]
                                }
                            ]
                            if df_individual_display.index.names and df_individual_display.index.names[0] == "Group"
                            else []
                        ),
                    ])
                    .set_properties(**{"text-align": "center", "font-weight": "500"})
                )

                try:
                    st.dataframe(styled_individual, use_container_width=True)
                except KeyError:
                    # Streamlit Cloud can error on some Styler/MultiIndex combinations;
                    # fall back to the plain dataframe instead of breaking the page.
                    st.dataframe(df_individual_display, use_container_width=True)

        # --- Kinematic Sequence Peak Summary Table (Segment-Grouped) ---
        if display_mode == "Grouped" and kinematic_peak_rows:
            import pandas as pd

            st.markdown("### Kinematic Sequence - Grouped")

            df = pd.DataFrame(kinematic_peak_rows)
            index_cols = ["Session Date", "Velocity (mph)"]
            if comparison_grouping_enabled and "Group" in df.columns:
                index_cols = ["Group"] + index_cols
            if multi_pitcher_mode and "Pitcher" in df.columns:
                index_cols = (
                    (["Group"] if comparison_grouping_enabled and "Group" in df.columns else [])
                    + ["Pitcher", "Session Date", "Velocity (mph)"]
                )

            df_pivot = df.pivot_table(
                index=index_cols,
                columns="Segment",
                values=["Peak Value (°/s)", "Peak Time from Reference (ms)"],
                aggfunc="first"
            )

            # Reorder to (Segment, Metric) like the original grouped summary layout
            df_pivot = df_pivot.swaplevel(0, 1, axis=1)
            metric_map = {
                "Peak Value (°/s)": "Peak (°/s)",
            }
            segment_reference_metric_map = {
                "Pelvis Rotation": "Peak Time from Foot Plant (ms)",
                "Torso Rotation": "Peak Time from Peak Pelvis (ms)",
                "Elbow Extension": "Peak Time from Peak Torso (ms)",
                "Shoulder Internal Rotation": "Peak Time from Peak Elbow (ms)",
            }
            df_pivot.columns = pd.MultiIndex.from_tuples(
                [
                    (
                        seg,
                        metric_map.get(metric, segment_reference_metric_map.get(seg, metric))
                    )
                    for seg, metric in df_pivot.columns
                ]
            )
            segment_order = [
                "Pelvis Rotation",
                "Torso Rotation",
                "Elbow Extension",
                "Shoulder Internal Rotation",
            ]
            ordered_cols = []
            for seg in segment_order:
                segment_metrics = [
                    "Peak (°/s)",
                    segment_reference_metric_map.get(seg, "Peak Time from Reference (ms)"),
                ]
                for metric_name in segment_metrics:
                    if (seg, metric_name) in df_pivot.columns:
                        ordered_cols.append((seg, metric_name))
            if ordered_cols:
                df_pivot = df_pivot[ordered_cols]

            segment_colors = {
                "Pelvis Rotation": "#DBEAFE",
                "Torso Rotation": "#FED7AA",
                "Elbow Extension": "#DCFCE7",
                "Shoulder Internal Rotation": "#FEE2E2",
            }

            def style_segments(col):
                seg = col[0]
                if seg in segment_colors:
                    return [f"background-color: {segment_colors[seg]}"] * len(df_pivot)
                return [""] * len(df_pivot)

            def style_segment_headers(headers):
                return [
                    f"background-color: {segment_colors.get(header, '#FFFFFF')}; color: #111827;"
                    if header in segment_colors else ""
                    for header in headers
                ]

            df_display = df_pivot.copy()
            if "Velocity (mph)" in df_display.index.names:
                velocity_level = df_display.index.names.index("Velocity (mph)")
                formatted_index = []
                for idx in df_display.index:
                    idx_list = list(idx) if isinstance(idx, tuple) else [idx]
                    velocity_value = idx_list[velocity_level]
                    if velocity_value is not None and not pd.isna(velocity_value):
                        idx_list[velocity_level] = f"{velocity_value:.1f}"
                    formatted_index.append(tuple(idx_list) if isinstance(idx, tuple) else idx_list[0])
                df_display.index = pd.MultiIndex.from_tuples(formatted_index, names=df_display.index.names)
            for col in df_display.columns:
                if col[1] == "Peak (°/s)":
                    df_display[col] = df_display[col].map(lambda x: "" if x is None or pd.isna(x) else f"{x:.0f}")
                elif "Time" in col[1]:
                    df_display[col] = df_display[col].map(lambda x: "" if x is None or pd.isna(x) else f"{x:.0f}")

            styled = (
                df_display
                .style
                .apply(style_segments, axis=0)
                .apply_index(style_segment_headers, axis="columns", level=0)
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "center")]},
                    {"selector": "th.row_heading", "props": [("text-align", "center")]},
                    {"selector": "th.index_name", "props": [("text-align", "center")]},
                    *(
                        [
                            {
                                "selector": "th.row_heading.level0",
                                "props": [("min-width", "80px"), ("max-width", "80px")]
                            }
                        ]
                        if df_display.index.names and df_display.index.names[0] == "Group"
                        else []
                    ),
                ])
                .set_properties(**{"text-align": "center", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True)

        kinematic_sequence_definitions = {
            "Pelvis Angular Velocity": "how fast the hips are rotating.",
            "Torso Angular Velocity": "how fast the shoulders are rotating.",
            "Elbow Angular Velocity": "how fast the elbow is straightening.",
            "Shoulder Angular Velocity": "how fast the shoulder rotates during the throw.",
        }
        st.markdown("### Kinematic Sequence Definitions")
        for segment, definition in kinematic_sequence_definitions.items():
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{segment}:</strong> {definition}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )


with tab_joint:
    st.subheader("Kinematics")
    render_group_selection_summary()
    st.markdown(
        """
        <style>
        .joint-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .joint-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    joint_view_mode = st.session_state.get("joint_view_mode", "Single")

    kinematic_options = [
        # Arm and hand
        "Elbow Extension Velocity",
        "Elbow Flexion",
        "Forearm Pronation/Supination",
        "Hand Speed",

        # Lower body
        "Hip-Shoulder Separation",
        "Lead Knee Flexion",
        "Lead Knee Flexion/Extension Velocity",
        "Pelvic Lateral Tilt",
        "Pelvis Rotation",
        "Pelvis Rotational Velocity",

        # Shoulder
        "Shoulder Abduction",
        "Shoulder Horizontal Abduction",
        "Shoulder Rotation",
        "Shoulder Rotation Velocity",

        # Trunk and whole-body movement
        "Center of Mass Velocity (Anterior/Posterior)",
        "Torso-Pelvis Rotational Velocity",
        "Trunk Forward Tilt",
        "Trunk Lateral Tilt",
        "Trunk Rotation",
        "Trunk Rotational Velocity",
    ]

    kinematic_definitions = {
        "Elbow Flexion": {
            "definition": "Angle of the elbow (forearm relative to upper arm) showing how bent the arm is during the throw. 0° = fully straight; >90° at foot plant = inside 90.",
        },
        "Hand Speed": {
            "definition": "How fast the throwing hand is moving in space during the throw (total speed in all directions).",
        },
        "Center of Mass Velocity (Anterior/Posterior)": {
            "definition": "Forward/backward speed of the body's center of mass toward or away from home plate. Positive = moving toward home plate; negative = moving back toward the mound.",
        },
        "Shoulder Rotation": {
            "definition": "How much the throwing arm rotates back and forward at the shoulder.",
        },
        "Shoulder Rotation Velocity": {
            "definition": "How fast the shoulder rotates during the throw. Peak internal rotation velocity = how quickly the arm turns forward. ~90° = goalpost position; moving forward toward 0° = arm rotating forward.",
        },
        "Shoulder Abduction": {
            "definition": "How far the arm is lifted away from the body. 0° = arms at your side; 90° = straight out (T-pose).",
        },
        "Shoulder Horizontal Abduction": {
            "definition": "How far the upper arm moves forward or backward relative to the trunk. 0° = T-pose; positive = arm moves behind you.",
        },
        "Lead Knee Flexion": {
            "definition": "Angle of the front knee (lower leg relative to upper leg). 0° = fully straight; higher values = more bend.",
        },
        "Lead Knee Flexion/Extension Velocity": {
            "definition": "How fast the front knee is bending or straightening.",
        },
        "Trunk Forward Tilt": {
            "definition": "Forward/backward lean of the upper body. 0° = upright; positive = leaning forward; negative = leaning back.",
        },
        "Trunk Lateral Tilt": {
            "definition": "Side-to-side lean of the upper body. 0° = upright; positive = leaning toward the lead leg side.",
        },
        "Trunk Rotation": {
            "definition": "How much the shoulders are turned toward home plate. -90° = open/sideways; 0° = square to home plate.",
        },
        "Pelvis Rotation": {
            "definition": "How much the hips are turned toward home plate. -90° = open/sideways; 0° = square to home plate.",
        },
        "Pelvic Lateral Tilt": {
            "definition": "Side-to-side tilt of the hips. 0° = level; positive = lead leg side drops; negative = trail side drops.",
        },
        "Hip-Shoulder Separation": {
            "definition": "Difference between hip and shoulder rotation. Positive = hips are opening ahead of the shoulders.",
        },
        "Pelvis Rotational Velocity": {
            "definition": "How fast the hips are rotating.",
        },
        "Trunk Rotational Velocity": {
            "definition": "How fast the shoulders are rotating.",
        },
        "Torso-Pelvis Rotational Velocity": {
            "definition": "How fast the shoulders are rotating relative to the hips.",
        },
        "Elbow Extension Velocity": {
            "definition": "How fast the elbow is straightening.",
        },
        "Forearm Pronation/Supination": {
            "definition": "Rotation of the forearm (palm turning down vs. up).",
        },
    }

    energy_definitions = {
        "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": {
            "definition": (
                "Measures how the trunk loads and then transfers energy to the throwing arm. "
                "Negative = the trunk is absorbing energy (loading), positive = the trunk is "
                "sending energy to the arm (throwing)."
            ),
        },
        "Arm Energy Flow (LAR_PROX | RAR_PROX)": {
            "definition": (
                "Measures how the throwing arm receives and responds to energy from the trunk "
                "at the shoulder connection. Positive values -> the arm is loading "
                "(receiving energy). Negative values -> the arm is being accelerated by the trunk."
            ),
        },
        "Glove Side Trunk-Shoulder Energy Flow": {
            "definition": (
                "Measures how the trunk loads and then transfers energy to the glove-side arm."
            ),
        },
        "Glove Arm Energy Flow": {
            "definition": (
                "Measures how the glove-side arm receives and responds to energy from the trunk "
                "at the shoulder connection."
            ),
        },
        "Trunk-Shoulder Elevation/Depression Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "elevation/depression (vertical abduction/adduction)."
            ),
        },
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "horizontal abduction/adduction (scap load)."
            ),
        },
        "Trunk-Shoulder Rotational Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the trunk at the shoulder due to shoulder "
                "internal/external rotation."
            ),
        },
        "Arm Elevation/Depression Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder elevation/depression (vertical abduction/adduction)."
            ),
        },
        "Arm Horizontal Abd/Add Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder horizontal abduction/adduction (scap load)."
            ),
        },
        "Arm Rotational Energy Flow": {
            "definition": (
                "Energy flow into (+) or out of (-) the upper arm at the shoulder due to "
                "shoulder internal/external rotation."
            ),
        },
        "Throwing Shoulder Rotational Torque (Relative to Trunk)": {
            "definition": (
                "The rotational torque at the throwing shoulder over time, measured relative "
                "to the trunk, representing the rotational load acting at the shoulder joint "
                "throughout the pitching motion."
            ),
        },
    }

    def get_kinematic_unit(kinematic_name):
        if "Velocity" in kinematic_name and "Hand Speed" not in kinematic_name and "Center of Mass Velocity" not in kinematic_name:
            return "°/s"
        if kinematic_name in {"Hand Speed", "Center of Mass Velocity (Anterior/Posterior)"}:
            return "m/s"
        return "°"

    compare_energy_metrics = []
    compare_energy_display_mode = "Grouped"
    compare_energy_window_mode = "Peak Knee Height View"

    if joint_view_mode == "Comparison":
        compare_top_left, compare_top_right = st.columns([1.3, 4.7])
        with compare_top_left:
            st.markdown('<div class="joint-controls-label">View Mode</div>', unsafe_allow_html=True)
            joint_view_mode = st.segmented_control(
                "View Mode",
                ["Single", "Comparison"],
                default="Comparison",
                key="joint_view_mode",
                label_visibility="collapsed",
            )
        with compare_top_right:
            st.markdown("")
        control_left_col, control_right_col = st.columns(2)
        with control_left_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            display_mode = st.segmented_control(
                "Kinematics Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_display_mode_compare",
                label_visibility="collapsed",
            )
            joint_window_mode = "Foot Plant to Ball Release View"
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            left_event_col, left_signal_col = st.columns(2)
            with left_event_col:
                show_joint_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_show_fp_iqr_band_compare",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with left_signal_col:
                show_joint_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_show_signal_iqr_band_compare",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
            selected_kinematics = st.multiselect(
                "Select Kinematics",
                options=kinematic_options,
                default=[],
                help=(
                    "Select one or more kinematics to plot. "
                    "Hover any line in the chart to see that metric's definition."
                ),
                key="joint_angles_select_compare"
            )
        with control_right_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            compare_energy_display_mode = st.segmented_control(
                "Energy Flow Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_energy_display_mode_compare"
                ,
                label_visibility="collapsed",
            )
            st.markdown('<div class="joint-controls-label">View Window</div>', unsafe_allow_html=True)
            compare_energy_window_mode = st.segmented_control(
                "Energy Flow View",
                ["Peak Knee Height View", "Foot Plant to Ball Release View"],
                default="Peak Knee Height View",
                key="joint_energy_window_mode_compare",
                label_visibility="collapsed",
            )
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            right_event_col, right_signal_col = st.columns(2)
            with right_event_col:
                show_compare_energy_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_energy_show_fp_iqr_band_compare",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with right_signal_col:
                show_compare_energy_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_energy_show_signal_iqr_band_compare",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
            compare_energy_metrics = st.multiselect(
                "Select Energy Flow Metrics",
                [
                    "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)",
                    "Arm Energy Flow (LAR_PROX | RAR_PROX)",
                    "Glove Side Trunk-Shoulder Energy Flow",
                    "Glove Arm Energy Flow",
                    "Trunk-Shoulder Rotational Energy Flow",
                    "Trunk-Shoulder Elevation/Depression Energy Flow",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                    "Arm Rotational Energy Flow",
                    "Arm Elevation/Depression Energy Flow",
                    "Arm Horizontal Abd/Add Energy Flow",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)",
                    *NEW_TRUNK_PELVIS_ENERGY_METRICS,
                ],
                default=[],
                key="joint_energy_metrics_compare"
            )
    else:
        display_col, options_col, spacer_col = st.columns([1.45, 1.75, 2.2])
        with display_col:
            st.markdown('<div class="joint-controls-label">Display Mode</div>', unsafe_allow_html=True)
            display_mode = st.segmented_control(
                "Select Display Mode",
                ["Individual Throws", "Grouped"],
                default="Grouped",
                key="joint_display_mode",
                label_visibility="collapsed",
            )
        with options_col:
            st.markdown('<div class="joint-controls-label joint-toggle-label">Options</div>', unsafe_allow_html=True)
            joint_event_col, joint_signal_col = st.columns(2)
            with joint_event_col:
                show_joint_fp_iqr_band = st.toggle(
                    "Event Bands",
                    value=False,
                    key="joint_show_fp_iqr_band",
                    help="Shows the middle 50% range for event timing across selected throws.",
                )
            with joint_signal_col:
                show_joint_signal_iqr_band = st.toggle(
                    "Signal Bands",
                    value=True,
                    key="joint_show_signal_iqr_band",
                    help="Shows the middle 50% range around each grouped mean line.",
                )
        with spacer_col:
            st.markdown("")
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] + div[data-testid="stVerticalBlock"] {
                margin-top: -0.2rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        window_col, mode_col, second_row_spacer = st.columns([2.35, 1.15, 2.75])
        with window_col:
            st.markdown('<div class="joint-controls-label">View Window</div>', unsafe_allow_html=True)
            joint_window_mode = st.segmented_control(
                "Kinematics View",
                ["Peak Knee Height View", "Foot Plant to Ball Release View"],
                default="Peak Knee Height View",
                key="joint_window_mode",
                label_visibility="collapsed",
            )
        with mode_col:
            st.markdown('<div class="joint-controls-label">View Mode</div>', unsafe_allow_html=True)
            joint_view_mode = st.segmented_control(
                "View Mode",
                ["Single", "Comparison"],
                default="Single",
                key="joint_view_mode",
                label_visibility="collapsed",
            )
        with second_row_spacer:
            st.markdown("")
        kinematics_select_col, kinematics_select_spacer = st.columns([2.35, 3.65])
        with kinematics_select_col:
            selected_kinematics = st.multiselect(
                "Select Kinematics",
                options=kinematic_options,
                default=[],
                help=(
                    "Select one or more kinematics to plot. "
                    "Hover any line in the chart to see that metric's definition."
                ),
                key="joint_angles_select"
            )
        with kinematics_select_spacer:
            st.markdown("")

    has_kinematics_selection = bool(selected_kinematics)
    show_single_kinematics_empty_state = not has_kinematics_selection and joint_view_mode == "Single"

    if show_single_kinematics_empty_state:
        kinematics_empty_col, kinematics_empty_spacer = st.columns([2.35, 3.65])
        with kinematics_empty_col:
            st.info("Select at least one kinematic.")
        with kinematics_empty_spacer:
            st.markdown("")

    # --- Color map for joint types ---
    joint_color_map = {
        "Elbow Flexion": "purple",
        "Hand Speed": "deeppink",
        "Center of Mass Velocity (Anterior/Posterior)": "cyan",
        "Shoulder Rotation": "teal",
        "Shoulder Rotation Velocity": "magenta",
        "Shoulder Abduction": "orange",
        "Shoulder Horizontal Abduction": "brown",
        "Forearm Pronation/Supination": "crimson",
        "Pelvis Rotational Velocity": "navy",
        "Trunk Rotational Velocity": "darkorange",
        "Torso-Pelvis Rotational Velocity": "dodgerblue",
        "Elbow Extension Velocity": "limegreen",
    }
    joint_color_map.update({
        "Trunk Forward Tilt": "blue",
        "Trunk Lateral Tilt": "green",
        "Trunk Rotation": "#E9FF70"
    })
    joint_color_map.update({
        "Pelvis Rotation": "darkblue",
        "Pelvic Lateral Tilt": "#FF4FA3",
        "Hip-Shoulder Separation": "darkred"
    })
    joint_color_map.update({
        "Lead Knee Flexion": "darkgreen"
    })
    joint_color_map.update({
        "Lead Knee Flexion/Extension Velocity": "olive"
    })

    # --- Load joint data conditionally ---
    joint_data = {}

    def load_joint_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    # --- Pelvis / Trunk rotational velocity (z_data) ---
    if "Pelvis Rotational Velocity" in selected_kinematics:
        joint_data["Pelvis Rotational Velocity"] = get_pelvis_angular_velocity(take_ids)

    if "Trunk Rotational Velocity" in selected_kinematics:
        joint_data["Trunk Rotational Velocity"] = get_torso_angular_velocity(take_ids)

    if "Torso-Pelvis Rotational Velocity" in selected_kinematics:
        joint_data["Torso-Pelvis Rotational Velocity"] = get_torso_pelvis_angular_velocity(take_ids)

    if "Elbow Flexion" in selected_kinematics:
        joint_data["Elbow Flexion"] = load_joint_by_handedness(get_elbow_flexion_angle)

    if "Hand Speed" in selected_kinematics:
        joint_data["Hand Speed"] = load_joint_by_handedness(get_hand_speed)

    if "Center of Mass Velocity (Anterior/Posterior)" in selected_kinematics:
        joint_data["Center of Mass Velocity (Anterior/Posterior)"] = get_center_of_mass_velocity_x(take_ids)

    if "Shoulder Rotation" in selected_kinematics:
        joint_data["Shoulder Rotation"] = load_joint_by_handedness(get_shoulder_er_angle)

    if "Shoulder Rotation Velocity" in selected_kinematics:
        joint_data["Shoulder Rotation Velocity"] = load_joint_by_handedness(get_shoulder_ir_velocity)

    if "Shoulder Abduction" in selected_kinematics:
        joint_data["Shoulder Abduction"] = load_joint_by_handedness(get_shoulder_abduction_angle)

    if "Shoulder Horizontal Abduction" in selected_kinematics:
        joint_data["Shoulder Horizontal Abduction"] = load_joint_by_handedness(
            get_shoulder_horizontal_abduction_angle
        )

    if "Forearm Pronation/Supination" in selected_kinematics:
        joint_data["Forearm Pronation/Supination"] = load_joint_by_handedness(
            get_forearm_pron_sup_angle
        )

    if "Lead Knee Flexion" in selected_kinematics:
        joint_data["Lead Knee Flexion"] = load_joint_by_handedness(get_front_knee_flexion_angle)

    if "Lead Knee Flexion/Extension Velocity" in selected_kinematics:
        joint_data["Lead Knee Flexion/Extension Velocity"] = load_joint_by_handedness(
            get_front_knee_extension_velocity
        )

    # --- Load Torso Angle components conditionally ---
    needs_torso_angle_data = any(
        metric in selected_kinematics
        for metric in ["Trunk Forward Tilt", "Trunk Lateral Tilt", "Trunk Rotation"]
    )
    torso_angle_data = get_torso_angle_components(take_ids) if needs_torso_angle_data else {}

    if "Trunk Forward Tilt" in selected_kinematics:
        joint_data["Trunk Forward Tilt"] = {
            k: {"frame": v["frame"], "value": v["x"]}
            for k, v in torso_angle_data.items()
        }

    if "Trunk Lateral Tilt" in selected_kinematics:
        joint_data["Trunk Lateral Tilt"] = {
            k: {"frame": v["frame"], "value": v["y"]}
            for k, v in torso_angle_data.items()
        }

    if "Trunk Rotation" in selected_kinematics:
        joint_data["Trunk Rotation"] = {
            k: {"frame": v["frame"], "value": v["z"]}
            for k, v in torso_angle_data.items()
        }

    if "Pelvis Rotation" in selected_kinematics:
        joint_data["Pelvis Rotation"] = get_pelvis_angle(take_ids)

    if "Pelvic Lateral Tilt" in selected_kinematics:
        joint_data["Pelvic Lateral Tilt"] = get_pelvic_lateral_tilt(take_ids)

    if "Hip-Shoulder Separation" in selected_kinematics:
        joint_data["Hip-Shoulder Separation"] = get_hip_shoulder_separation(take_ids)

    if "Elbow Extension Velocity" in selected_kinematics:
        joint_data["Elbow Extension Velocity"] = load_joint_by_handedness(get_elbow_angular_velocity)

    # --- Helper for extracting value at a specific time (ms) ---
    def value_at_time_ms(times_ms, values, target_time_ms):
        if target_time_ms in times_ms:
            return values[times_ms.index(target_time_ms)]
        return None

    import pandas as pd
    summary_rows = []
    compare_energy_summary_rows = []

    fig = go.Figure()

    # --- Date-based colors (Joint Angles ONLY) ---
    unique_dates = sorted(set(take_date_map.values()))
    date_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    date_color_map = {
        d: date_palette[i % len(date_palette)]
        for i, d in enumerate(unique_dates)
    }

    # --- Date-based line dash style map (for visual distinction) ---
    date_dash_map = {}
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    for i, d in enumerate(unique_dates):
        date_dash_map[d] = dash_styles[i % len(dash_styles)]
    use_group_colors_joint = (
        comparison_grouping_enabled
        and len(selected_kinematics) == 1
        and len(group_color_map) >= 2
    )

    # --- Per-take normalization and plotting ---
    grouped = {}
    grouped_by_date = {}
    mound_only_selected = mound_only_sidebar
    median_pkh_frame = None
    if mound_only_selected and knee_event_frames:
        median_pkh_frame = int(np.median(knee_event_frames))

    if joint_window_mode == "Foot Plant to Ball Release View":
        median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
        joint_window_start = (median_fp_frame - 25) if median_fp_frame is not None else window_start
        joint_window_end = 25
    else:
        joint_window_start = window_start
        joint_window_end = 50
        # For mound throws, ensure the window includes PKH and 20 frames before it.
        if median_pkh_frame is not None:
            joint_window_start = min(window_start, median_pkh_frame - 20)

    joint_window_start_ms = rel_frame_to_ms(joint_window_start)
    joint_window_end_ms = rel_frame_to_ms(joint_window_end)

    # For condensed legend: track which (kinematic, date) pairs have legend entries
    legend_keys_added = set()
    summary_knee_frame = None
    if median_pkh_frame is not None:
        summary_knee_frame = median_pkh_frame
    elif knee_event_frames:
        summary_knee_frame = int(np.median(knee_event_frames))

    # Reuse take_order and take_velocity from Kinematic Sequence section if available
    peak_positive_kinematics = {
        "Shoulder Rotation Velocity",
        "Trunk Rotational Velocity",
        "Torso-Pelvis Rotational Velocity",
        "Pelvis Rotational Velocity",
        "Elbow Extension Velocity",
    }
    collapse_control_group_in_comparison = joint_view_mode == "Comparison" and bool(control_take_ids)
    right_hand_mirror_kinematics = {
        "Shoulder Horizontal Abduction",
        "Shoulder Rotation",
    }
    left_hand_mirror_kinematics = {
        "Trunk Forward Tilt",
        "Trunk Lateral Tilt",
        "Trunk Rotation",
        "Pelvic Lateral Tilt",
        "Pelvis Rotation",
        "Hip-Shoulder Separation",
    }
    for kinematic, data_dict in joint_data.items():
        grouped[kinematic] = {}

        for take_id in take_ids:
            if take_id not in data_dict or take_id not in br_frames:
                continue

            # --- Support both "value" (angles) and "z" (rotational velocities) dicts ---
            if "value" in data_dict[take_id]:
                values = data_dict[take_id]["value"]
                frames = data_dict[take_id]["frame"]
            elif "x" in data_dict[take_id]:
                values = data_dict[take_id]["x"]
                frames = data_dict[take_id]["frame"]
            elif "z" in data_dict[take_id]:
                values = data_dict[take_id]["z"]
                frames = data_dict[take_id]["frame"]
            else:
                continue
            br = br_frames[take_id]
            sign_flip = 1.0
            if kinematic in peak_positive_kinematics:
                valid_vals = [v for v in values if v is not None]
                if valid_vals:
                    dominant_peak = max(valid_vals, key=lambda x: abs(x))
                    if dominant_peak < 0:
                        sign_flip = -1.0

            norm_f, norm_v = [], []
            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel = f - br
                if joint_window_start <= rel <= joint_window_end:
                    norm_f.append(rel_frame_to_ms(rel))

                    # --- Handedness normalization ---
                    take_hand = take_handedness.get(take_id)
                    handedness_factor = 1.0

                    # Keep selected angle directions aligned to a shared orientation.
                    if "Velocity" not in kinematic and take_hand == "R" and kinematic in right_hand_mirror_kinematics:
                        handedness_factor = -1.0

                    # Mirror left-handed trunk tilt curves to right-handed orientation.
                    if take_hand == "L" and kinematic in left_hand_mirror_kinematics:
                        handedness_factor = -1.0

                    norm_v.append(sign_flip * handedness_factor * v)

            grouped[kinematic][take_id] = {"frame": norm_f, "value": norm_v}

            # --- Store by date for grouped plotting ---
            date = take_date_map[take_id]
            group_label = take_group_map.get(take_id, "Ungrouped")
            pitcher_name = take_pitcher_map.get(take_id, "")
            control_group_take = is_control_group_label(group_label)
            hover_pitcher_name = "" if control_group_take else pitcher_name
            if comparison_grouping_enabled and control_group_take:
                date_key = group_label
            elif comparison_grouping_enabled:
                date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
            else:
                date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {}).setdefault(kinematic, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }
            trace_color = (
                group_color_map.get(group_label, joint_color_map[kinematic])
                if use_group_colors_joint else
                joint_color_map[kinematic]
            )

            if display_mode == "Individual Throws":
                if collapse_control_group_in_comparison and control_group_take:
                    continue
                # Use kinematic color and date-based dash for individual throws
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        customdata=[[hover_pitcher_name]] * len(norm_f),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                            "Time: %{x:.1f} ms"
                            + ("<br>Pitcher: %{customdata[0]}" if show_group_pitcher_breakout else "")
                            + "<extra></extra>"
                        ),
                        line=dict(
                            color=trace_color,
                            dash=date_dash_map[take_date_map[take_id]]
                        ),
                        name=(
                            f"Control Group | {kinematic} – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if (comparison_grouping_enabled and control_group_take) else
                            (
                                f"{group_label} | {kinematic} – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            ) if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            (
                                f"{group_label} | {kinematic} – {take_date_map[take_id]} | "
                                f"Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            ) if comparison_grouping_enabled else
                            (
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} "
                            f"({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            )
                        ),
                        showlegend=False
                    )
                )
                # Add one legend-only trace per (kinematic, date) (shows color + dash)
                legend_key = (kinematic, date_key)
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color=trace_color,
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"Control Group | {kinematic}"
                                if (comparison_grouping_enabled and control_group_take) else
                                f"{group_label} | {kinematic} | {date} | {pitcher_name}"
                                if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                                f"{group_label} | {kinematic} | {date}"
                                if comparison_grouping_enabled else
                                f"{kinematic} | {date} | {pitcher_name}"
                                if show_group_pitcher_breakout else
                                f"{kinematic} | {date}"
                            ),
                            showlegend=True
                        )
                    )
                    legend_keys_added.add(legend_key)

    if display_mode == "Individual Throws" and collapse_control_group_in_comparison:
        control_group_curves = grouped_by_date.get("Control Group", {})
        for kinematic, curves in control_group_curves.items():
            if not curves:
                continue

            x, y, q1, q3 = aggregate_curves(curves, "Mean")
            if len(y) >= 11:
                y = savgol_filter(y, window_length=11, polyorder=3)

            color = (
                group_color_map.get("Control Group", joint_color_map.get(kinematic, "#444"))
                if use_group_colors_joint else
                joint_color_map.get(kinematic, "#444")
            )
            legendgroup = f"Control_Group_{kinematic}"

            if show_joint_signal_iqr_band:
                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=q3 + q1[::-1],
                        fill="toself",
                        fillcolor=to_rgba(color, 0.35),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=legendgroup
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=4, color=color),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                        "Time: %{x:.1f} ms<extra></extra>"
                    ),
                    name=f"Control Group | {kinematic}",
                    showlegend=True,
                    legendgroup=legendgroup
                )
            )

    # --- Summary table: Individual Throws ---
    if display_mode == "Individual Throws":
        for kinematic, curves in grouped.items():
            for take_id, d in curves.items():
                frames = d["frame"]
                values = d["value"]
                if not values:
                    continue

                max_val = np.max(values)
                # sd_val = np.std(values)  # removed as not used below

                br_val = value_at_time_ms(frames, values, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                    fp_val = value_at_time_ms(frames, values, median_fp)

                # value at MER (same frame used in plot)
                mer_val = None
                if take_id in shoulder_er_max_frames:
                    mer_frame_rel = shoulder_er_max_frames[take_id] - br_frames[take_id]
                    mer_val = value_at_time_ms(frames, values, rel_frame_to_ms(mer_frame_rel))

                # value at per-take PKH frame (fallback to summary knee frame)
                pkh_val = None
                if take_id in knee_peak_frames:
                    pkh_frame_rel = knee_peak_frames[take_id] - br_frames[take_id]
                    pkh_val = value_at_time_ms(frames, values, rel_frame_to_ms(pkh_frame_rel))
                elif summary_knee_frame is not None:
                    pkh_val = value_at_time_ms(frames, values, rel_frame_to_ms(summary_knee_frame))

                summary_rows.append({
                    **({"Group": take_group_map.get(take_id, "")} if comparison_grouping_enabled else {}),
                    **({"Pitcher": take_pitcher_map.get(take_id)} if show_group_pitcher_breakout else {}),
                    "Kinematic": kinematic + (" (°/s)" if "Velocity" in kinematic else ""),
                    "Session Date": take_date_map[take_id],
                    "Average Velocity": take_velocity[take_id],
                    "Max": max_val,
                    "Peak Knee Height": pkh_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val
                })

    # --- Grouped plot (mean + IQR per date) ---
    if display_mode == "Grouped":
        for date_key, kin_dict in grouped_by_date.items():
            if comparison_grouping_enabled and date_key == "Control Group":
                group_label = "Control Group"
                pitcher_name = ""
                date = "Selected Takes"
            elif comparison_grouping_enabled and show_group_pitcher_breakout:
                group_label, pitcher_name, date = date_key
            elif comparison_grouping_enabled:
                group_label = date_key
                date = "Selected Takes"
                pitcher_name = ""
            elif multi_pitcher_mode and not comparison_grouping_enabled:
                pitcher_name, date = date_key
                group_label = ""
            else:
                date = date_key
                pitcher_name = ""
                group_label = ""
            for kinematic, curves in kin_dict.items():
                if not curves:
                    continue

                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])

                # Smooth grouped curve ONLY
                if len(y) >= 11:
                    y = savgol_filter(y, window_length=11, polyorder=3)

                color = (
                    group_color_map.get(group_label, joint_color_map.get(kinematic, "#444"))
                    if use_group_colors_joint else
                    joint_color_map.get(kinematic, "#444")
                )
                dash = date_dash_map.get(date, "solid")

                # IQR band (draw first so the line color stays visually true on top)
                if show_joint_signal_iqr_band:
                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=q3 + q1[::-1],
                            fill="toself",
                            fillcolor=to_rgba(color, 0.35),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip"
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        hovertemplate=(
                            (f"<b>{group_label}</b><br>" if comparison_grouping_enabled else "<b>%{fullData.name}</b><br>")
                            + (f"Avg Velocity: {avg_velocity:.1f} mph<br>" if avg_velocity is not None else "")
                            +
                            f"{kinematic}: %{{y:.1f}}{get_kinematic_unit(kinematic)}<br>"
                            "Time: %{x:.1f} ms<extra></extra>"
                        ),
                        line=dict(width=4, color=color, dash=dash),
                        name=(
                            f"{group_label} | {kinematic} – {date} | {pitcher_name}"
                            if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            f"{group_label} | {kinematic} – {date}"
                            if comparison_grouping_enabled else
                            f"{kinematic} – {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{kinematic} – {date}"
                        )
                    )
                )

                max_val = np.max(y)
                br_val = value_at_time_ms(x, y, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                    fp_val = value_at_time_ms(x, y, median_fp)

                max_vals = [np.max(d["value"]) for d in curves.values() if d["value"]]
                sd_val = np.std(max_vals)

                # value at MER from grouped mean curve
                mer_val = None
                if mer_event_frames:
                    median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                    mer_val = value_at_time_ms(x, y, median_mer)

                # value at summary PKH frame from grouped mean curve
                pkh_val = None
                if summary_knee_frame is not None:
                    pkh_val = value_at_time_ms(x, y, rel_frame_to_ms(summary_knee_frame))

                summary_rows.append({
                    **({"Group": group_label} if comparison_grouping_enabled else {}),
                    **({"Pitcher": pitcher_name} if show_group_pitcher_breakout else {}),
                    "Kinematic": kinematic + (" (°/s)" if "Velocity" in kinematic else ""),
                    "Session Date": date,
                    "Average Velocity": np.mean([take_velocity[tid] for tid in curves.keys()]),
                    "Max": max_val,
                    "Peak Knee Height": pkh_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val,
                    "Standard Deviation": sd_val
                })

    # --- Event lines and annotations (match Kinematic Sequence styling) ---
    if median_pkh_frame is not None:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_joint_fp_iqr_band)
        median_pkh_time_ms = rel_frame_to_ms(median_pkh_frame)
        fig.add_vline(
            x=median_pkh_time_ms,
            line_width=3,
            line_dash="dash",
            line_color="gold",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_pkh_time_ms,
            y=1.055,
            xref="x",
            yref="paper",
            text="PKH",
            showarrow=False,
            font=dict(color="gold", size=14),
            align="center"
        )
    elif knee_event_frames:
        # Non-mound fallback: keep a single knee marker when PKH is not enabled.
        add_event_iqr_band(fig, knee_event_frames, "gold", show_joint_fp_iqr_band)
        median_knee_frame = rel_frame_to_ms(int(np.median(knee_event_frames)))
        fig.add_vline(
            x=median_knee_frame,
            line_width=3,
            line_dash="dash",
            line_color="gold",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_knee_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="Knee",
            showarrow=False,
            font=dict(color="gold", size=14),
            align="center"
        )

    if fp_event_frames:
        add_event_iqr_band(fig, fp_event_frames, "green", show_joint_fp_iqr_band)
        median_fp_frame = rel_frame_to_ms(int(np.median(fp_event_frames)))
        fig.add_vline(
            x=median_fp_frame,
            line_width=3,
            line_dash="dash",
            line_color="green",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_fp_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="FP",
            showarrow=False,
            font=dict(color="green", size=14),
            align="center"
        )

    if mer_event_frames:
        add_event_iqr_band(fig, mer_event_frames, "red", show_joint_fp_iqr_band)
        median_mer_frame = rel_frame_to_ms(int(np.median(mer_event_frames)))
        fig.add_vline(
            x=median_mer_frame,
            line_width=3,
            line_dash="dash",
            line_color="red",
            opacity=0.9
        )
        fig.add_annotation(
            x=median_mer_frame,
            y=1.055,
            xref="x",
            yref="paper",
            text="MER",
            showarrow=False,
            font=dict(color="red", size=14),
            align="center"
        )

    # Ball Release reference
    add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_joint_fp_iqr_band)
    fig.add_vline(
        x=0,
        line_width=3,
        line_dash="dash",
        line_color="blue",
        opacity=0.9
    )
    fig.add_annotation(
        x=0,
        y=1.055,
        xref="x",
        yref="paper",
        text="BR",
        showarrow=False,
        font=dict(color="blue", size=14),
        align="center"
    )

    fig.update_layout(
        xaxis_title="Time Relative to Ball Release (ms)",
        yaxis_title="Kinematics",
        yaxis=dict(),
        xaxis_range=[joint_window_start_ms, joint_window_end_ms],
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(
            namelength=-1,
            font_size=13
        )
    )

    if joint_view_mode == "Comparison":
        plot_left_col, plot_right_col = st.columns(2)
        with plot_left_col:
            if has_kinematics_selection:
                st.markdown("#### Kinematics")
                st.plotly_chart(fig, use_container_width=True, key="joint_plot_compare_left")
            else:
                st.info("Select at least one kinematic to render the left-side plot.")

        with plot_right_col:
            if compare_energy_metrics:
                st.markdown("#### Energy Flow")
            if not compare_energy_metrics:
                st.info("Select at least one energy flow metric to render the right-side plot.")
            elif not take_ids:
                st.info("No takes available for Energy Flow.")
            else:
                energy_color_map = {
                    "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": "#4C1D95",
                    "Arm Energy Flow (LAR_PROX | RAR_PROX)": "#7C2D12",
                    "Glove Side Trunk-Shoulder Energy Flow": "#E11D48",
                    "Glove Arm Energy Flow": "#14B8A6",
                    "Trunk-Shoulder Rotational Energy Flow": "#DC2626",
                    "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",
                    "Arm Rotational Energy Flow": "#F59E0B",
                    "Arm Elevation/Depression Energy Flow": "#06B6D4",
                    "Arm Horizontal Abd/Add Energy Flow": "#9333EA",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00",
                    **NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP,
                }

                compare_energy_data_by_metric = {}

                def load_compare_energy_by_handedness(loader_fn):
                    merged = {}
                    for hand, ids in take_ids_by_handedness.items():
                        if ids:
                            merged.update(loader_fn(ids, hand))
                    return merged

                for metric in compare_energy_metrics:
                    if metric == "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_distal_arm_segment_power)
                    elif metric == "Arm Energy Flow (LAR_PROX | RAR_PROX)":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_proximal_energy_transfer)
                    elif metric == "Glove Side Trunk-Shoulder Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_glove_side_trunk_shoulder_energy_flow)
                    elif metric == "Glove Arm Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_glove_arm_energy_flow)
                    elif metric == "Trunk-Shoulder Rotational Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_rot_energy_flow)
                    elif metric == "Trunk-Shoulder Elevation/Depression Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_elev_energy_flow)
                    elif metric == "Trunk-Shoulder Horizontal Abd/Add Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_trunk_shoulder_horizabd_energy_flow)
                    elif metric == "Arm Rotational Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_rot_energy_flow)
                    elif metric == "Arm Elevation/Depression Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_elev_energy_flow)
                    elif metric == "Arm Horizontal Abd/Add Energy Flow":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_horizabd_energy_flow)
                    elif metric == "Throwing Shoulder Rotational Torque (Relative to Trunk)":
                        mmt_data = {}
                        if take_ids_by_handedness.get("R"):
                            mmt_data.update(
                                get_energy_flow_from_segment(
                                    take_ids_by_handedness["R"],
                                    "RT_SHOULDER_RTA_MMT",
                                    component="z"
                                )
                            )
                        if take_ids_by_handedness.get("L"):
                            mmt_data.update(
                                get_energy_flow_from_segment(
                                    take_ids_by_handedness["L"],
                                    "LT_SHOULDER_RTA_MMT",
                                    component="z"
                                )
                            )
                        compare_energy_data_by_metric[metric] = mmt_data
                    elif metric in NEW_TRUNK_PELVIS_ENERGY_METRICS:
                        segment_name, category_name = NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP[metric]
                        compare_energy_data_by_metric[metric] = get_energy_flow_from_category_segment(
                            take_ids,
                            category_name,
                            segment_name,
                            component="x",
                        )

                compare_energy_data_by_metric = {
                    k: v for k, v in compare_energy_data_by_metric.items() if v
                }

                if not compare_energy_data_by_metric:
                    st.warning("No energy flow data found for the selected metrics.")
                else:
                    energy_fig = go.Figure()
                    collapse_control_group_energy = bool(control_take_ids)
                    unique_dates = sorted(set(take_date_map.values()))
                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    date_dash_map = {
                        d: dash_styles[i % len(dash_styles)]
                        for i, d in enumerate(unique_dates)
                    }

                    energy_legend_keys = set()
                    compare_energy_median_pkh_frame = None
                    if mound_only_sidebar and knee_event_frames:
                        compare_energy_median_pkh_frame = int(np.median(knee_event_frames))

                    if compare_energy_window_mode == "Foot Plant to Ball Release View":
                        compare_energy_median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
                        energy_window_start = (
                            compare_energy_median_fp_frame - 25
                            if compare_energy_median_fp_frame is not None else
                            window_start
                        )
                        energy_window_end = 25
                    else:
                        energy_window_start = window_start
                        energy_window_end = 50
                        if compare_energy_median_pkh_frame is not None:
                            energy_window_start = min(window_start, compare_energy_median_pkh_frame - 20)

                    energy_window_start_ms = rel_frame_to_ms(energy_window_start)
                    energy_window_end_ms = rel_frame_to_ms(energy_window_end)

                    for metric, energy_data in compare_energy_data_by_metric.items():
                        metric_color = energy_color_map.get(metric, "#444")
                        grouped_by_date = {}

                        for take_id, d in energy_data.items():
                            if take_id not in br_frames:
                                continue

                            frames = d["frame"]
                            values = d["value"]
                            br = br_frames[take_id]

                            norm_f, norm_v = [], []
                            for f, v in zip(frames, values):
                                rel = f - br
                                if energy_window_start <= rel <= energy_window_end:
                                    norm_f.append(rel_frame_to_ms(rel))
                                    norm_v.append(v)

                            date = take_date_map[take_id]
                            pitcher_name = take_pitcher_map.get(take_id, "")
                            group_label = take_group_map.get(take_id, "")
                            control_group_take = is_control_group_label(group_label)
                            if comparison_grouping_enabled and control_group_take:
                                date_key = "Control Group"
                            else:
                                date_key = (pitcher_name, date) if multi_pitcher_mode else date
                            grouped_by_date.setdefault(date_key, {})[take_id] = {
                                "frame": norm_f,
                                "value": norm_v
                            }

                            peak_val = None
                            if norm_v:
                                peak_idx = int(np.argmax(np.abs(np.array(norm_v, dtype=float))))
                                peak_val = norm_v[peak_idx]

                            br_val = value_at_time_ms(norm_f, norm_v, 0)
                            fp_val = None
                            if fp_event_frames:
                                median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                                fp_val = value_at_time_ms(norm_f, norm_v, median_fp)

                            mer_val = None
                            if mer_event_frames:
                                median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                                mer_val = value_at_time_ms(norm_f, norm_v, median_mer)

                            if compare_energy_display_mode == "Individual Throws":
                                compare_energy_summary_rows.append({
                                    **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
                                    "Metric": metric,
                                    "Session Date": date,
                                    "Average Velocity": take_velocity[take_id],
                                    "Peak": peak_val,
                                    "Foot Plant": fp_val,
                                    "Ball Release": br_val,
                                    "Max External Rotation": mer_val,
                                })

                            if compare_energy_display_mode == "Individual Throws":
                                if collapse_control_group_energy and control_group_take:
                                    continue
                                legendgroup = f"{metric}_{pitcher_name}_{date}" if multi_pitcher_mode else f"{metric}_{date}"
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=norm_f,
                                        y=norm_v,
                                        mode="lines",
                                        line=dict(
                                            color=metric_color,
                                            dash=date_dash_map[date]
                                        ),
                                        customdata=[[metric, date, take_order[take_id], take_velocity[take_id], pitcher_name]] * len(norm_f),
                                        hovertemplate=(
                                            ("%{customdata[4]} | %{customdata[1]}" if multi_pitcher_mode else "%{customdata[1]}")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
                                            + "<br>Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        showlegend=False,
                                        legendgroup=legendgroup
                                    )
                                )
                                legend_key = (metric, date_key)
                                if legend_key not in energy_legend_keys:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=[None],
                                            y=[None],
                                            mode="lines",
                                            line=dict(
                                                color=metric_color,
                                                dash=date_dash_map[date],
                                                width=4
                                            ),
                                            name=(
                                                f"{metric} | {date} | {pitcher_name}"
                                                if multi_pitcher_mode else
                                                f"{metric} | {date}"
                                            ),
                                            showlegend=True,
                                            legendgroup=legendgroup
                                        )
                                    )
                                    energy_legend_keys.add(legend_key)

                        if compare_energy_display_mode == "Individual Throws" and collapse_control_group_energy:
                            control_curves = grouped_by_date.get("Control Group", {})
                            if control_curves:
                                x, y, q1, q3 = aggregate_curves(control_curves, "Mean")
                                legendgroup = f"{metric}_Control_Group"

                                if show_compare_energy_signal_iqr_band:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=x + x[::-1],
                                            y=q3 + q1[::-1],
                                            fill="toself",
                                            fillcolor=to_rgba(metric_color, alpha=0.35),
                                            line=dict(width=0),
                                            showlegend=False,
                                            hoverinfo="skip",
                                            legendgroup=legendgroup
                                        )
                                    )

                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode="lines",
                                        line=dict(width=4, color=metric_color),
                                        hovertemplate=(
                                            "Control Group"
                                            + "<br>%{fullData.name}: %{y:.1f}"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        name=f"Control Group | {metric}",
                                        showlegend=True,
                                        legendgroup=legendgroup
                                    )
                                )

                        if compare_energy_display_mode == "Grouped":
                            for date_key, curves in grouped_by_date.items():
                                control_group_curves = comparison_grouping_enabled and date_key == "Control Group"
                                if control_group_curves:
                                    pitcher_name = ""
                                    date = "Selected Takes"
                                elif multi_pitcher_mode:
                                    pitcher_name, date = date_key
                                else:
                                    date = date_key
                                    pitcher_name = ""
                                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                                dash_style = date_dash_map.get(date, "solid")
                                legendgroup = (
                                    f"{metric}_Control_Group"
                                    if control_group_curves else
                                    f"{metric}_{pitcher_name}_{date}" if multi_pitcher_mode else f"{metric}_{date}"
                                )

                                peak_val = None
                                if y:
                                    peak_idx = int(np.argmax(np.abs(np.array(y, dtype=float))))
                                    peak_val = y[peak_idx]

                                br_val = value_at_time_ms(x, y, 0)
                                fp_val = None
                                if fp_event_frames:
                                    median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                                    fp_val = value_at_time_ms(x, y, median_fp)

                                mer_val = None
                                if mer_event_frames:
                                    median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                                    mer_val = value_at_time_ms(x, y, median_mer)

                                peak_vals = []
                                for curve in curves.values():
                                    if curve["value"]:
                                        curve_arr = np.array(curve["value"], dtype=float)
                                        peak_vals.append(float(curve_arr[np.argmax(np.abs(curve_arr))]))

                                compare_energy_summary_rows.append({
                                    **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
                                    "Metric": metric,
                                    "Session Date": date,
                                    "Average Velocity": np.mean([take_velocity[tid] for tid in curves.keys()]),
                                    "Peak": peak_val,
                                    "Foot Plant": fp_val,
                                    "Ball Release": br_val,
                                    "Max External Rotation": mer_val,
                                    "Standard Deviation": (np.std(peak_vals) if peak_vals else None),
                                })
                                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])

                                if show_compare_energy_signal_iqr_band:
                                    energy_fig.add_trace(
                                        go.Scatter(
                                            x=x + x[::-1],
                                            y=q3 + q1[::-1],
                                            fill="toself",
                                            fillcolor=to_rgba(metric_color, alpha=0.35),
                                            line=dict(width=0),
                                            showlegend=False,
                                            hoverinfo="skip",
                                            legendgroup=legendgroup
                                        )
                                    )
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode="lines",
                                        line=dict(width=4, color=metric_color, dash=dash_style),
                                        customdata=[[metric, date, pitcher_name]] * len(x),
                                        hovertemplate=(
                                            ("Control Group" if control_group_curves else "%{customdata[2]} | %{customdata[1]}" if multi_pitcher_mode else "%{customdata[1]}")
                                            + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        showlegend=False,
                                        legendgroup=legendgroup
                                    )
                                )
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=[None],
                                        y=[None],
                                        mode="lines",
                                        line=dict(color=metric_color, dash=dash_style, width=4),
                                        name=(
                                            f"Control Group | {metric}"
                                            if control_group_curves else
                                            f"{metric} | {date} | {pitcher_name}"
                                            if multi_pitcher_mode else
                                            f"{metric} | {date}"
                                        ),
                                        showlegend=True,
                                        legendgroup=legendgroup
                                    )
                    )

                    if compare_energy_median_pkh_frame is not None:
                        add_event_iqr_band(energy_fig, knee_event_frames, "gold", show_compare_energy_fp_iqr_band)
                        median_pkh = rel_frame_to_ms(compare_energy_median_pkh_frame)
                        energy_fig.add_vline(x=median_pkh, line_width=3, line_dash="dash", line_color="gold")
                        energy_fig.add_annotation(
                            x=median_pkh,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="PKH",
                            showarrow=False,
                            font=dict(color="gold", size=13, family="Arial"),
                            align="center"
                        )
                    elif knee_event_frames:
                        add_event_iqr_band(energy_fig, knee_event_frames, "gold", show_compare_energy_fp_iqr_band)
                        median_knee = rel_frame_to_ms(int(np.median(knee_event_frames)))
                        energy_fig.add_vline(x=median_knee, line_width=3, line_dash="dash", line_color="gold")
                        energy_fig.add_annotation(
                            x=median_knee,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="Knee",
                            showarrow=False,
                            font=dict(color="gold", size=13, family="Arial"),
                            align="center"
                        )

                    if fp_event_frames:
                        add_event_iqr_band(energy_fig, fp_event_frames, "green", show_compare_energy_fp_iqr_band)
                        median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
                        energy_fig.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")
                        energy_fig.add_annotation(
                            x=median_fp,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="FP",
                            showarrow=False,
                            font=dict(color="green", size=13, family="Arial"),
                            align="center"
                        )
                    if mer_event_frames:
                        add_event_iqr_band(energy_fig, mer_event_frames, "red", show_compare_energy_fp_iqr_band)
                        median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
                        energy_fig.add_vline(x=median_mer, line_width=3, line_dash="dash", line_color="red")
                        energy_fig.add_annotation(
                            x=median_mer,
                            y=1.06,
                            xref="x",
                            yref="paper",
                            text="MER",
                            showarrow=False,
                            font=dict(color="red", size=13, family="Arial"),
                            align="center"
                        )
                    add_event_iqr_band(energy_fig, [0] * max(len(take_ids), 1), "blue", show_compare_energy_fp_iqr_band)
                    energy_fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")
                    energy_fig.add_annotation(
                        x=0,
                        y=1.06,
                        xref="x",
                        yref="paper",
                        text="BR",
                        showarrow=False,
                        font=dict(color="blue", size=13, family="Arial"),
                        align="center"
                    )

                    energy_fig.update_layout(
                        xaxis_title="Time Relative to Ball Release (ms)",
                        yaxis_title=get_energy_yaxis_title(compare_energy_data_by_metric.keys()),
                        xaxis_range=[energy_window_start_ms, energy_window_end_ms],
                        height=600,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.30,
                            xanchor="center",
                            x=0.5,
                            groupclick="togglegroup"
                        ),
                        hoverlabel=dict(
                            namelength=-1,
                            font_size=13
                        )
                    )
                    st.plotly_chart(energy_fig, use_container_width=True, key="joint_plot_compare_right_energy")
    else:
        if show_single_kinematics_empty_state:
            st.markdown("")
        else:
            st.plotly_chart(fig, use_container_width=True, key="joint_plot_single")

    # --- Kinematics Table ---
    has_compare_energy_summary = (
        joint_view_mode == "Comparison"
        and bool(compare_energy_metrics)
        and bool(compare_energy_summary_rows)
    )
    combined_summary_mode = (
        not show_single_kinematics_empty_state
        and bool(summary_rows)
        and has_compare_energy_summary
    )
    rendered_summary_heading = False
    if not show_single_kinematics_empty_state and summary_rows:
        st.markdown("### Summary" if combined_summary_mode else "### Kinematics Summary")
        rendered_summary_heading = True
        df_summary = pd.DataFrame(summary_rows)
        # Reorder columns explicitly
        base_columns = [
            "Kinematic",
            "Session Date",
            "Average Velocity",
            "Max",
            "Peak Knee Height",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation"
        ]
        if joint_window_mode == "Foot Plant to Ball Release View":
            base_columns.remove("Peak Knee Height")
        if comparison_grouping_enabled:
            base_columns = ["Group"] + base_columns
        if show_group_pitcher_breakout:
            base_columns = ["Pitcher"] + base_columns

        if display_mode == "Grouped":
            column_order = base_columns + ["Standard Deviation"]
        else:
            column_order = base_columns

        df_summary = df_summary[column_order]

        import numpy as np
        def fmt(val, decimals=2):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{val:.{decimals}f}"

        def normalize_kinematic_name(display_name):
            return display_name.replace(" (°/s)", "")

        def format_with_unit(val, unit, decimals=2, prefix=""):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{prefix}{val:.{decimals}f} {unit}".strip()

        measurement_columns = [
            "Max",
            "Peak Knee Height",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation",
        ]
        if joint_window_mode == "Foot Plant to Ball Release View":
            measurement_columns.remove("Peak Knee Height")

        for idx, row in df_summary.iterrows():
            kinematic_name = normalize_kinematic_name(row["Kinematic"])
            kinematic_unit = get_kinematic_unit(kinematic_name)

            if "Average Velocity" in df_summary.columns:
                df_summary.at[idx, "Average Velocity"] = fmt(row["Average Velocity"], 1)

            for col in measurement_columns:
                if col in df_summary.columns:
                    df_summary.at[idx, col] = format_with_unit(
                        row[col], kinematic_unit, decimals=2
                    )

            if display_mode == "Grouped" and "Standard Deviation" in df_summary.columns:
                df_summary.at[idx, "Standard Deviation"] = format_with_unit(
                    row["Standard Deviation"], kinematic_unit, decimals=2, prefix="±"
                )

        styled_summary = (
            df_summary
            .style
            .hide(axis="index")
            # Center headers
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])
            # Center label columns
            .set_properties(
                subset=["Kinematic", "Session Date"],
                **{"text-align": "center"}
            )
            # Center numeric columns
            .set_properties(
                subset=[c for c in df_summary.columns if c not in ["Kinematic", "Session Date"]],
                **{"text-align": "center"}
            )
        )
        summary_column_config = {}
        if "Group" in df_summary.columns:
            summary_column_config["Group"] = st.column_config.TextColumn(
                "Group",
                width="small",
            )

        st.dataframe(
            styled_summary,
            use_container_width=True,
            column_config=summary_column_config or None,
            hide_index=True,
        )

    if has_compare_energy_summary:
        if not rendered_summary_heading:
            st.markdown("### Energy Flow Summary")
            rendered_summary_heading = True
        df_energy_summary = pd.DataFrame(compare_energy_summary_rows)

        def fmt_energy(val, decimals=2):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return f"{val:.{decimals}f}"

        energy_base_columns = [
            "Metric",
            "Session Date",
            "Average Velocity",
            "Peak",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation",
        ]
        if multi_pitcher_mode and "Pitcher" in df_energy_summary.columns:
            energy_base_columns = ["Pitcher"] + energy_base_columns
        if compare_energy_display_mode == "Grouped" and "Standard Deviation" in df_energy_summary.columns:
            energy_column_order = energy_base_columns + ["Standard Deviation"]
        else:
            energy_column_order = energy_base_columns

        df_energy_summary = df_energy_summary[energy_column_order]

        for idx, row in df_energy_summary.iterrows():
            if "Average Velocity" in df_energy_summary.columns:
                df_energy_summary.at[idx, "Average Velocity"] = fmt_energy(row["Average Velocity"], 1)
            for col in ["Peak", "Foot Plant", "Ball Release", "Max External Rotation"]:
                if col in df_energy_summary.columns:
                    df_energy_summary.at[idx, col] = fmt_energy(row[col], 1)
            if compare_energy_display_mode == "Grouped" and "Standard Deviation" in df_energy_summary.columns:
                df_energy_summary.at[idx, "Standard Deviation"] = (
                    f"±{row['Standard Deviation']:.1f}" if row["Standard Deviation"] is not None and not (isinstance(row["Standard Deviation"], float) and np.isnan(row["Standard Deviation"])) else ""
                )

        styled_energy_summary = (
            df_energy_summary
            .style
            .hide(axis="index")
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])
            .set_properties(
                subset=[c for c in df_energy_summary.columns],
                **{"text-align": "center"}
            )
        )

        st.dataframe(
            styled_energy_summary,
            use_container_width=True,
            hide_index=True,
        )

    has_compare_energy_definitions = (
        joint_view_mode == "Comparison"
        and bool(compare_energy_metrics)
        and any(metric in energy_definitions for metric in compare_energy_metrics)
    )
    combined_definitions_mode = (
        not show_single_kinematics_empty_state
        and bool(selected_kinematics)
        and has_compare_energy_definitions
    )
    rendered_definitions_heading = False
    if not show_single_kinematics_empty_state:
        st.markdown("### Definitions" if combined_definitions_mode else "### Kinematic Definitions")
        rendered_definitions_heading = True
        for metric in selected_kinematics:
            metric_info = kinematic_definitions.get(metric, {})
            if not metric_info:
                continue
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

    if joint_view_mode == "Comparison" and compare_energy_metrics:
        defined_compare_energy_metrics = [
            metric for metric in compare_energy_metrics if metric in energy_definitions
        ]
        if defined_compare_energy_metrics:
            if not rendered_definitions_heading:
                st.markdown("### Energy Flow Definitions")
                rendered_definitions_heading = True
            for metric in defined_compare_energy_metrics:
                metric_info = energy_definitions.get(metric, {})
                st.markdown(
                    (
                        f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                        f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                        f"</div>"
                    ),
                    unsafe_allow_html=True,
                )

# --------------------------------------------------
# Energy Flow Tab
# --------------------------------------------------


# --------------------------------------------------
# Helper: Compute peak distal arm segment power (W) per take
# --------------------------------------------------
def compute_peak_segment_power(energy_data, br_frames, fp_event_frames):
    """
    Compute peak distal arm segment power (W) per take
    restricted to Foot Plant → Ball Release.
    Uses the most negative (minimum) value in the window.
    """
    peak_map = {}

    if not fp_event_frames:
        return peak_map

    median_fp_rel = int(np.median(fp_event_frames))

    for take_id, d in energy_data.items():
        if take_id not in br_frames:
            continue

        br = br_frames[take_id]
        frames = np.array(d["frame"], dtype=int)
        values = np.array(d["value"], dtype=float)

        rel = frames - br

        # STRICT biomechanical window: Foot Plant → Ball Release
        mask = (rel >= median_fp_rel) & (rel <= 0)

        if not np.any(mask):
            continue

        peak_map[take_id] = float(np.nanmin(values[mask]))

    return peak_map


# --------------------------------------------------
# Report Tab
# --------------------------------------------------


def get_report_take_rows(athlete_name, session_dates, throw_types=None, velocity_range=None, excluded_take_ids=None):
    throw_types = throw_types or ["Mound"]
    excluded_take_ids = excluded_take_ids or []
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            params = [athlete_name, throw_types]
            date_clause = ""
            velocity_clause = ""
            exclusion_clause = ""
            if session_dates:
                date_placeholders = ",".join(["%s"] * len(session_dates))
                date_clause = f"AND t.take_date IN ({date_placeholders})"
                params.extend(session_dates)
            if velocity_range and velocity_range[0] is not None and velocity_range[1] is not None:
                velocity_clause = "AND t.pitch_velo BETWEEN %s AND %s"
                params.extend(velocity_range)
            if excluded_take_ids:
                exclusion_clause = "AND NOT (t.take_id = ANY(%s))"
                params.append(excluded_take_ids)
            cur.execute(
                f"""
                SELECT
                    t.take_id,
                    t.pitch_velo,
                    t.take_date,
                    a.athlete_name,
                    a.handedness,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.athlete_name, t.take_date
                        ORDER BY t.take_id
                    ) AS pitch_number
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE a.athlete_name = %s
                  AND t.throw_type = ANY(%s)
                  {date_clause}
                  {velocity_clause}
                  {exclusion_clause}
                  AND t.pitch_velo IS NOT NULL
                ORDER BY t.take_id
                """,
                tuple(params),
            )
            return cur.fetchall()
    finally:
        conn.close()


def build_report_kinematic_summary(report_rows):
    if not report_rows:
        return [], {}

    take_ids = [row[0] for row in report_rows]
    take_velocity = {row[0]: row[1] for row in report_rows}
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        "R": [take_id for take_id in take_ids if take_handedness.get(take_id) == "R"],
        "L": [take_id for take_id in take_ids if take_handedness.get(take_id) == "L"],
    }

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    cg_data = load_by_handedness(get_hand_cg_velocity)
    shoulder_er_data = load_by_handedness(get_shoulder_er_angles)
    br_frames = {}
    for take_id, d in cg_data.items():
        valid = [(i, v) for i, v in enumerate(d["x"]) if v is not None]
        if valid:
            idx, _ = max(valid, key=lambda item: item[1])
            br_frames[take_id] = d["frame"][idx] + 4

    shoulder_er_max_frames = {}
    for take_id, d in shoulder_er_data.items():
        valid = [(frame, value) for frame, value in zip(d["frame"], d["z"]) if value is not None]
        if not valid:
            continue

        if take_handedness.get(take_id) == "R":
            shoulder_er_max_frames[take_id] = min(valid, key=lambda item: item[1])[0]
        else:
            shoulder_er_max_frames[take_id] = max(valid, key=lambda item: item[1])[0]

    foot_plant_frames = {}
    for hand, ids in take_ids_by_handedness.items():
        if not ids:
            continue

        ankle_prox_x_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
        ankle_min_frames = get_ankle_min_frame(ids, hand, ankle_prox_x_peak_frames, shoulder_er_max_frames)
        ankle_zero_cross_frames = get_foot_plant_frame_zero_cross(
            ids,
            hand,
            ankle_min_frames,
            shoulder_er_max_frames,
        )
        heel_anchor_frames = {
            take_id: ankle_zero_cross_frames.get(take_id, ankle_min_frames.get(take_id))
            for take_id in ids
        }
        heel_contact_frames = get_lead_heel_contact_frame(
            ids,
            hand,
            ankle_prox_x_peak_frames,
            shoulder_er_max_frames,
            heel_anchor_frames,
        )

        for take_id in ids:
            candidates = [
                value
                for value in [
                    ankle_zero_cross_frames.get(take_id),
                    heel_contact_frames.get(take_id),
                    ankle_min_frames.get(take_id),
                    ankle_prox_x_peak_frames.get(take_id),
                ]
                if value is not None
            ]
            if candidates:
                foot_plant_frames[take_id] = int(max(candidates[:2]) if len(candidates[:2]) == 2 else candidates[0])

    for take_id, fp_frame in list(foot_plant_frames.items()):
        er_frame = shoulder_er_max_frames.get(take_id)
        if er_frame is not None and fp_frame > er_frame:
            foot_plant_frames[take_id] = er_frame

    pelvis_data = get_pelvis_angular_velocity(take_ids)
    torso_data = get_torso_angular_velocity(take_ids)
    elbow_data = load_by_handedness(get_elbow_angular_velocity)
    shoulder_ir_data = load_by_handedness(get_shoulder_ir_velocity)

    segment_sources = {
        "Pelvis Rotation": (pelvis_data, "z"),
        "Torso Rotation": (torso_data, "z"),
        "Elbow Extension": (elbow_data, "x"),
        "Shoulder Internal Rotation": (shoulder_ir_data, "x"),
    }
    curves_by_segment = {segment: {} for segment in segment_sources}

    for segment, (source, axis_key) in segment_sources.items():
        for take_id, d in source.items():
            if take_id not in br_frames:
                continue

            norm_frames = []
            norm_values = []
            for frame, value in zip(d["frame"], d[axis_key]):
                if value is None:
                    continue

                rel_frame = frame - br_frames[take_id]
                norm_frames.append(rel_frame_to_ms(rel_frame))
                normalized_value = value
                if segment in {"Pelvis Rotation", "Torso Rotation", "Shoulder Internal Rotation"} and take_handedness.get(take_id) == "L":
                    normalized_value = -normalized_value
                elif segment == "Elbow Extension":
                    normalized_value = -normalized_value
                norm_values.append(normalized_value)

            if norm_frames:
                curves_by_segment[segment][take_id] = {
                    "frame": norm_frames,
                    "value": norm_values,
                }

    summary_rows = []
    peak_times = {}
    fp_event_frames = [
        foot_plant_frames[take_id] - br_frames[take_id]
        for take_id in take_ids
        if take_id in foot_plant_frames and take_id in br_frames
    ]
    mer_event_frames = [
        shoulder_er_max_frames[take_id] - br_frames[take_id]
        for take_id in take_ids
        if take_id in shoulder_er_max_frames and take_id in br_frames
    ]

    median_fp_ms = None
    if fp_event_frames:
        median_fp_ms = rel_frame_to_ms(int(np.median(fp_event_frames)))

    for segment in ["Pelvis Rotation", "Torso Rotation", "Elbow Extension", "Shoulder Internal Rotation"]:
        curves = curves_by_segment.get(segment, {})
        if not curves:
            continue

        x, y, q1, q3 = aggregate_curves(curves, "Mean")
        if not y:
            continue

        valid_idxs = [i for i, value in enumerate(y) if value is not None and np.isfinite(value)]
        if segment in {"Pelvis Rotation", "Torso Rotation"} and median_fp_ms is not None:
            valid_idxs = [i for i in valid_idxs if median_fp_ms <= x[i] <= 0]
        if not valid_idxs:
            continue

        peak_idx = max(valid_idxs, key=lambda i: y[i])
        peak_time = x[peak_idx]
        peak_times[segment] = peak_time

        reference_time = None
        if segment == "Pelvis Rotation" and median_fp_ms is not None:
            reference_time = peak_time - median_fp_ms
        elif segment == "Torso Rotation" and "Pelvis Rotation" in peak_times:
            reference_time = peak_time - peak_times["Pelvis Rotation"]
        elif segment == "Elbow Extension" and "Torso Rotation" in peak_times:
            reference_time = peak_time - peak_times["Torso Rotation"]
        elif segment == "Shoulder Internal Rotation" and "Elbow Extension" in peak_times:
            reference_time = peak_time - peak_times["Elbow Extension"]

        summary_rows.append({
            "Segment": segment,
            "Peak (deg/s)": y[peak_idx],
            "Peak Time (ms rel BR)": peak_time,
            "Peak Time from Reference (ms)": reference_time,
            "Average Velocity (mph)": float(np.mean([take_velocity[take_id] for take_id in curves.keys() if take_velocity.get(take_id) is not None])),
        })

    report_events = {
        "fp_event_frames": fp_event_frames,
        "mer_event_frames": mer_event_frames,
        "take_count": len(take_ids),
    }

    return summary_rows, curves_by_segment, report_events


def get_report_take_rows_by_ids(take_ids):
    if not take_ids:
        return []
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(
                f"""
                SELECT
                    t.take_id,
                    t.pitch_velo,
                    t.take_date,
                    a.athlete_name,
                    a.handedness,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.athlete_name, t.take_date
                        ORDER BY t.take_id
                    ) AS pitch_number
                FROM takes t
                JOIN athletes a ON a.athlete_id = t.athlete_id
                WHERE t.take_id IN ({placeholders})
                ORDER BY t.take_id
                """,
                tuple(take_ids),
            )
            return cur.fetchall()
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_take_report_metric_summaries(take_ids, metric_keys, logic_version=REPORT_METRIC_LOGIC_VERSION):
    if not take_ids or not metric_keys:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.take_report_metrics')")
            if cur.fetchone()[0] is None:
                return {}
            cur.execute(
                """
                SELECT
                    metric_key,
                    event_label,
                    AVG(metric_value)::double precision AS mean_value,
                    CASE
                        WHEN COUNT(metric_value) > 1 THEN STDDEV_SAMP(metric_value)::double precision
                        WHEN COUNT(metric_value) = 1 THEN 0::double precision
                        ELSE NULL::double precision
                    END AS sd_value
                FROM take_report_metrics
                WHERE take_id = ANY(%s)
                  AND metric_key = ANY(%s)
                  AND event_label = ANY(%s)
                  AND logic_version = %s
                  AND metric_value IS NOT NULL
                GROUP BY metric_key, event_label
                """,
                (
                    list(take_ids),
                    list(metric_keys),
                    ["FP", "MER", "BR", "Max", "Average"],
                    logic_version,
                ),
            )
            summaries = {}
            for metric_key, event_label, mean_value, sd_value in cur.fetchall():
                summaries.setdefault(metric_key, {})[event_label] = {
                    "mean": float(mean_value) if mean_value is not None else None,
                    "std": float(sd_value) if sd_value is not None else None,
                }
            return summaries
    finally:
        conn.close()


def apply_precomputed_metric_summary(report_metric_data, metric_summaries, metric_key):
    precomputed_metrics = metric_summaries.get(metric_key)
    if not precomputed_metrics:
        return report_metric_data

    merged_data = dict(report_metric_data or {})
    merged_metrics = dict(merged_data.get("metrics", {}))
    for event_label, values in precomputed_metrics.items():
        merged_metrics[event_label] = values
    merged_data["metrics"] = merged_metrics
    return merged_data


def summarize_metric_values(values):
    clean_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "mean": float(np.mean(clean_values)) if clean_values else None,
        "std": float(np.std(clean_values, ddof=1 if len(clean_values) > 1 else 0)) if clean_values else None,
    }


def build_metric_summary_from_take_metrics(take_metrics):
    metric_values = {}
    for event_values in (take_metrics or {}).values():
        for event_label, payload in event_values.items():
            if isinstance(payload, dict):
                value = payload.get("value")
            else:
                value = payload
            metric_values.setdefault(event_label, []).append(value)
    return {
        event_label: summarize_metric_values(values)
        for event_label, values in metric_values.items()
    }


def ensure_report_metric_cache_schema(cur):
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'takes'
          AND n.nspname = 'public'
          AND a.attname = 'take_id'
          AND NOT a.attisdropped
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        return
    take_id_type = row[0]
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS take_report_metrics (
            take_id {take_id_type} NOT NULL REFERENCES takes(take_id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            metric_label TEXT NOT NULL,
            metric_group TEXT NOT NULL,
            event_label TEXT NOT NULL,
            metric_value DOUBLE PRECISION,
            source_frame INTEGER,
            unit TEXT NOT NULL,
            source_category TEXT,
            source_segment TEXT,
            source_axis TEXT,
            logic_version TEXT NOT NULL DEFAULT '{REPORT_METRIC_LOGIC_VERSION}',
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (take_id, metric_key, event_label, logic_version)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_take_report_metrics_metric_event
            ON take_report_metrics (metric_key, event_label)
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS take_report_metric_curves (
            take_id {take_id_type} NOT NULL REFERENCES takes(take_id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            time_ms DOUBLE PRECISION NOT NULL,
            metric_value DOUBLE PRECISION,
            logic_version TEXT NOT NULL DEFAULT '{REPORT_METRIC_LOGIC_VERSION}',
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (take_id, metric_key, time_ms, logic_version)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_take_report_metric_curves_metric
            ON take_report_metric_curves (metric_key, logic_version, take_id)
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS take_report_metric_cache_status (
            take_id {take_id_type} NOT NULL REFERENCES takes(take_id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            logic_version TEXT NOT NULL DEFAULT '{REPORT_METRIC_LOGIC_VERSION}',
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (take_id, metric_key, logic_version)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_take_report_metric_cache_status_metric
            ON take_report_metric_cache_status (metric_key, logic_version, take_id)
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS take_report_events (
            take_id {take_id_type} NOT NULL REFERENCES takes(take_id) ON DELETE CASCADE,
            event_label TEXT NOT NULL,
            source_frame INTEGER,
            relative_frame INTEGER,
            relative_ms DOUBLE PRECISION,
            logic_version TEXT NOT NULL DEFAULT '{REPORT_METRIC_LOGIC_VERSION}',
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (take_id, event_label, logic_version)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_take_report_events_label
            ON take_report_events (event_label, logic_version, take_id)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS report_reference_metric_aggregates (
            aggregate_hash TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            logic_version TEXT NOT NULL,
            take_count INTEGER NOT NULL,
            metrics_json JSONB NOT NULL,
            curve_json JSONB NOT NULL,
            events_json JSONB NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (aggregate_hash, metric_key, logic_version)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_report_reference_metric_aggregates_lookup
            ON report_reference_metric_aggregates (logic_version, aggregate_hash, metric_key)
        """
    )


def report_reference_aggregate_hash(take_ids, logic_version=REPORT_METRIC_LOGIC_VERSION):
    unique_take_ids = sorted({str(take_id) for take_id in take_ids if take_id is not None})
    payload = json.dumps(
        {"logic_version": logic_version, "take_ids": unique_take_ids},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_report_reference_aggregate_cache(cur, aggregate_hash, metric_keys, logic_version=REPORT_METRIC_LOGIC_VERSION):
    if not aggregate_hash or not metric_keys:
        return {}
    cur.execute("SELECT to_regclass('public.report_reference_metric_aggregates')")
    if cur.fetchone()[0] is None:
        return {}
    cur.execute(
        """
        SELECT metric_key, metrics_json, curve_json, events_json
        FROM report_reference_metric_aggregates
        WHERE aggregate_hash = %s
          AND metric_key = ANY(%s)
          AND logic_version = %s
        """,
        (aggregate_hash, list(metric_keys), logic_version),
    )
    cached = {}
    for metric_key, metrics_json, curve_json, events_json in cur.fetchall():
        cached[metric_key] = {
            "curves": {"__aggregate__": curve_json} if curve_json and curve_json.get("frame") else {},
            "events": events_json or {},
            "metrics": metrics_json or {},
            "take_metrics": {},
        }
    return cached


def store_report_reference_aggregate_cache(cur, aggregate_hash, take_count, bundle, metric_keys, logic_version=REPORT_METRIC_LOGIC_VERSION):
    if not aggregate_hash or not bundle or not metric_keys:
        return
    from psycopg2.extras import Json, execute_values

    rows = []
    for metric_key in metric_keys:
        metric_bundle = bundle.get(metric_key)
        if not metric_bundle:
            continue
        aggregate_curve = (metric_bundle.get("curves") or {}).get("__aggregate__", {})
        rows.append((
            aggregate_hash,
            metric_key,
            logic_version,
            int(take_count),
            Json(metric_bundle.get("metrics") or {}),
            Json(aggregate_curve or {}),
            Json(metric_bundle.get("events") or {}),
        ))
    if not rows:
        return
    execute_values(
        cur,
        """
        INSERT INTO report_reference_metric_aggregates (
            aggregate_hash, metric_key, logic_version, take_count,
            metrics_json, curve_json, events_json, computed_at
        )
        VALUES %s
        ON CONFLICT (aggregate_hash, metric_key, logic_version)
        DO UPDATE SET
            take_count = EXCLUDED.take_count,
            metrics_json = EXCLUDED.metrics_json,
            curve_json = EXCLUDED.curve_json,
            events_json = EXCLUDED.events_json,
            computed_at = NOW()
        """,
        rows,
        template="(%s,%s,%s,%s,%s,%s,%s,NOW())",
        page_size=250,
    )


def build_event_payload_from_rows(rows):
    event_frames = {"fp_event_frames": [], "mer_event_frames": [], "pkh_event_frames": []}
    frames_by_take = {}
    for take_id, event_label, source_frame, relative_frame, relative_ms in rows:
        frames_by_take.setdefault(take_id, {})[event_label] = {
            "source_frame": int(source_frame) if source_frame is not None else None,
            "relative_frame": int(relative_frame) if relative_frame is not None else None,
            "relative_ms": float(relative_ms) if relative_ms is not None else None,
        }
        if event_label == "FP" and relative_frame is not None:
            event_frames["fp_event_frames"].append(int(relative_frame))
        elif event_label == "MER" and relative_frame is not None:
            event_frames["mer_event_frames"].append(int(relative_frame))
        elif event_label == "PKH" and relative_frame is not None:
            event_frames["pkh_event_frames"].append(int(relative_frame))
    event_frames["events_by_take"] = frames_by_take
    return event_frames


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_report_events(take_ids, logic_version=REPORT_METRIC_LOGIC_VERSION):
    take_ids = [take_id for take_id in take_ids if take_id is not None]
    if not take_ids:
        return {"fp_event_frames": [], "mer_event_frames": [], "pkh_event_frames": [], "events_by_take": {}}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.take_report_events')")
            if cur.fetchone()[0] is None:
                return {"fp_event_frames": [], "mer_event_frames": [], "pkh_event_frames": [], "events_by_take": {}}
            cur.execute(
                """
                SELECT take_id, event_label, source_frame, relative_frame, relative_ms
                FROM take_report_events
                WHERE take_id = ANY(%s)
                  AND logic_version = %s
                ORDER BY event_label
                """,
                (take_ids, logic_version),
            )
            return build_event_payload_from_rows(cur.fetchall())
    finally:
        conn.close()


def cache_report_events_from_metric_data(cur, report_metric_data, logic_version=REPORT_METRIC_LOGIC_VERSION):
    take_metrics = (report_metric_data or {}).get("take_metrics", {})
    event_rows = []
    for take_id, event_values in take_metrics.items():
        br_payload = event_values.get("BR", {}) if isinstance(event_values, dict) else {}
        br_frame = br_payload.get("source_frame") if isinstance(br_payload, dict) else None
        for event_label in ("FP", "MER", "BR", "PKH"):
            payload = event_values.get(event_label, {}) if isinstance(event_values, dict) else {}
            if not isinstance(payload, dict):
                continue
            source_frame = payload.get("source_frame")
            if source_frame is None:
                continue
            relative_frame = int(source_frame) - int(br_frame) if br_frame is not None else None
            relative_ms = rel_frame_to_ms(relative_frame) if relative_frame is not None else None
            event_rows.append((
                take_id,
                event_label,
                int(source_frame),
                int(relative_frame) if relative_frame is not None else None,
                float(relative_ms) if relative_ms is not None else None,
                logic_version,
            ))
    if not event_rows:
        return
    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO take_report_events (
            take_id, event_label, source_frame, relative_frame,
            relative_ms, logic_version, computed_at
        )
        VALUES %s
        ON CONFLICT (take_id, event_label, logic_version)
        DO UPDATE SET
            source_frame = EXCLUDED.source_frame,
            relative_frame = EXCLUDED.relative_frame,
            relative_ms = EXCLUDED.relative_ms,
            computed_at = NOW()
        """,
        event_rows,
        template="(%s,%s,%s,%s,%s,%s,NOW())",
        page_size=1000,
    )


def get_report_event_frame_maps(take_ids):
    cached_events = get_cached_report_events(take_ids)
    events_by_take = cached_events.get("events_by_take", {})

    def frame_map(event_label):
        out = {}
        for take_id, events in events_by_take.items():
            payload = events.get(event_label, {})
            source_frame = payload.get("source_frame") if isinstance(payload, dict) else None
            if source_frame is not None:
                out[take_id] = int(source_frame)
        return out

    return {
        "FP": frame_map("FP"),
        "MER": frame_map("MER"),
        "BR": frame_map("BR"),
        "PKH": frame_map("PKH"),
        "events": cached_events,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_report_metric_data(take_ids, metric_key, logic_version=REPORT_METRIC_LOGIC_VERSION):
    take_ids = [take_id for take_id in take_ids if take_id is not None]
    if not take_ids or not metric_key:
        return None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.take_report_metric_cache_status')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                """
                SELECT COUNT(DISTINCT take_id)::int
                FROM take_report_metric_cache_status
                WHERE take_id = ANY(%s)
                  AND metric_key = %s
                  AND logic_version = %s
                """,
                (take_ids, metric_key, logic_version),
            )
            processed_count = cur.fetchone()[0] or 0
            if processed_count != len(set(take_ids)):
                return None

            cur.execute("SELECT to_regclass('public.take_report_metrics')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                """
                SELECT
                    event_label,
                    AVG(metric_value)::double precision AS mean_value,
                    CASE
                        WHEN COUNT(metric_value) > 1 THEN STDDEV_SAMP(metric_value)::double precision
                        WHEN COUNT(metric_value) = 1 THEN 0::double precision
                        ELSE NULL::double precision
                    END AS sd_value
                FROM take_report_metrics
                WHERE take_id = ANY(%s)
                  AND metric_key = %s
                  AND logic_version = %s
                  AND metric_value IS NOT NULL
                GROUP BY event_label
                ORDER BY event_label
                """,
                (take_ids, metric_key, logic_version),
            )
            metrics = {
                event_label: {
                    "mean": float(mean_value) if mean_value is not None else None,
                    "std": float(sd_value) if sd_value is not None else None,
                }
                for event_label, mean_value, sd_value in cur.fetchall()
            }

            curves = {}
            cur.execute("SELECT to_regclass('public.take_report_metric_curves')")
            if cur.fetchone()[0] is not None:
                cur.execute(
                    """
                    SELECT
                        time_ms,
                        AVG(metric_value)::double precision AS mean_value,
                        percentile_cont(0.25) WITHIN GROUP (ORDER BY metric_value)::double precision AS q1_value,
                        percentile_cont(0.75) WITHIN GROUP (ORDER BY metric_value)::double precision AS q3_value
                    FROM take_report_metric_curves
                    WHERE take_id = ANY(%s)
                      AND metric_key = %s
                      AND logic_version = %s
                      AND metric_value IS NOT NULL
                    GROUP BY time_ms
                    ORDER BY time_ms
                    """,
                    (take_ids, metric_key, logic_version),
                )
                aggregate_curve = {"frame": [], "value": [], "q1": [], "q3": []}
                for time_ms, mean_value, q1_value, q3_value in cur.fetchall():
                    aggregate_curve["frame"].append(float(time_ms))
                    aggregate_curve["value"].append(float(mean_value) if mean_value is not None else None)
                    aggregate_curve["q1"].append(float(q1_value) if q1_value is not None else None)
                    aggregate_curve["q3"].append(float(q3_value) if q3_value is not None else None)
                if aggregate_curve["frame"]:
                    curves["__aggregate__"] = aggregate_curve

            cached_events = get_cached_report_events(take_ids, logic_version)
            fp_event_frames = list(cached_events.get("fp_event_frames", []))
            mer_event_frames = list(cached_events.get("mer_event_frames", []))
            if not fp_event_frames or not mer_event_frames:
                take_metrics = {}
                cur.execute(
                    """
                    SELECT take_id, event_label, metric_value, source_frame
                    FROM take_report_metrics
                    WHERE take_id = ANY(%s)
                      AND metric_key = %s
                      AND logic_version = %s
                    ORDER BY take_id, event_label
                    """,
                    (take_ids, metric_key, logic_version),
                )
                for take_id, event_label, metric_value, source_frame in cur.fetchall():
                    take_metrics.setdefault(take_id, {})[event_label] = {
                        "value": float(metric_value) if metric_value is not None else None,
                        "source_frame": int(source_frame) if source_frame is not None else None,
                    }
                fallback_fp_event_frames = []
                fallback_mer_event_frames = []
                for event_values in take_metrics.values():
                    br_frame = event_values.get("BR", {}).get("source_frame")
                    fp_frame = event_values.get("FP", {}).get("source_frame")
                    mer_frame = event_values.get("MER", {}).get("source_frame")
                    if br_frame is not None and fp_frame is not None:
                        fallback_fp_event_frames.append(fp_frame - br_frame)
                    if br_frame is not None and mer_frame is not None:
                        fallback_mer_event_frames.append(mer_frame - br_frame)
                if not fp_event_frames:
                    fp_event_frames = fallback_fp_event_frames
                if not mer_event_frames:
                    mer_event_frames = fallback_mer_event_frames

            return {
                "curves": curves,
                "events": {
                    "fp_event_frames": fp_event_frames,
                    "mer_event_frames": mer_event_frames,
                    "pkh_event_frames": cached_events.get("pkh_event_frames", []),
                    "events_by_take": cached_events.get("events_by_take", {}),
                },
                "metrics": metrics,
                "take_metrics": {},
            }
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_report_metric_bundle(take_ids, metric_keys, logic_version=REPORT_METRIC_LOGIC_VERSION):
    take_ids = sorted({take_id for take_id in take_ids if take_id is not None})
    metric_keys = [metric_key for metric_key in metric_keys if metric_key]
    if not take_ids or not metric_keys:
        return {}

    unique_take_count = len(take_ids)
    aggregate_hash = report_reference_aggregate_hash(take_ids, logic_version)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            ensure_report_metric_cache_schema(cur)
            conn.commit()
            for table_name in (
                "take_report_metric_cache_status",
                "take_report_metrics",
                "take_report_metric_curves",
            ):
                cur.execute(f"SELECT to_regclass('public.{table_name}')")
                if cur.fetchone()[0] is None:
                    return {}

            cur.execute(
                """
                SELECT metric_key, COUNT(DISTINCT take_id)::int
                FROM take_report_metric_cache_status
                WHERE take_id = ANY(%s)
                  AND metric_key = ANY(%s)
                  AND logic_version = %s
                GROUP BY metric_key
                """,
                (take_ids, metric_keys, logic_version),
            )
            complete_metric_keys = {
                metric_key
                for metric_key, processed_count in cur.fetchall()
                if processed_count == unique_take_count
            }
            if not complete_metric_keys:
                return {}

            cached_bundle = load_report_reference_aggregate_cache(
                cur,
                aggregate_hash,
                complete_metric_keys,
                logic_version,
            )
            missing_metric_keys = sorted(complete_metric_keys - set(cached_bundle.keys()))
            if not missing_metric_keys:
                return cached_bundle

            cached_events = get_cached_report_events(take_ids, logic_version)
            shared_events = {
                "fp_event_frames": list(cached_events.get("fp_event_frames", [])),
                "mer_event_frames": list(cached_events.get("mer_event_frames", [])),
                "pkh_event_frames": list(cached_events.get("pkh_event_frames", [])),
                "events_by_take": cached_events.get("events_by_take", {}),
            }
            bundle = dict(cached_bundle)
            bundle.update({
                metric_key: {
                    "curves": {},
                    "events": shared_events,
                    "metrics": {},
                    "take_metrics": {},
                }
                for metric_key in missing_metric_keys
            })

            cur.execute(
                """
                SELECT
                    metric_key,
                    event_label,
                    AVG(metric_value)::double precision AS mean_value,
                    CASE
                        WHEN COUNT(metric_value) > 1 THEN STDDEV_SAMP(metric_value)::double precision
                        WHEN COUNT(metric_value) = 1 THEN 0::double precision
                        ELSE NULL::double precision
                    END AS sd_value
                FROM take_report_metrics
                WHERE take_id = ANY(%s)
                  AND metric_key = ANY(%s)
                  AND logic_version = %s
                  AND metric_value IS NOT NULL
                GROUP BY metric_key, event_label
                ORDER BY metric_key, event_label
                """,
                (take_ids, missing_metric_keys, logic_version),
            )
            for metric_key, event_label, mean_value, sd_value in cur.fetchall():
                bundle[metric_key]["metrics"][event_label] = {
                    "mean": float(mean_value) if mean_value is not None else None,
                    "std": float(sd_value) if sd_value is not None else None,
                }

            cur.execute(
                """
                SELECT
                    metric_key,
                    time_ms,
                    AVG(metric_value)::double precision AS mean_value,
                    percentile_cont(0.25) WITHIN GROUP (ORDER BY metric_value)::double precision AS q1_value,
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY metric_value)::double precision AS q3_value
                FROM take_report_metric_curves
                WHERE take_id = ANY(%s)
                  AND metric_key = ANY(%s)
                  AND logic_version = %s
                  AND metric_value IS NOT NULL
                GROUP BY metric_key, time_ms
                ORDER BY metric_key, time_ms
                """,
                (take_ids, missing_metric_keys, logic_version),
            )
            for metric_key, time_ms, mean_value, q1_value, q3_value in cur.fetchall():
                aggregate_curve = bundle[metric_key]["curves"].setdefault(
                    "__aggregate__",
                    {"frame": [], "value": [], "q1": [], "q3": []},
                )
                aggregate_curve["frame"].append(float(time_ms))
                aggregate_curve["value"].append(float(mean_value) if mean_value is not None else None)
                aggregate_curve["q1"].append(float(q1_value) if q1_value is not None else None)
                aggregate_curve["q3"].append(float(q3_value) if q3_value is not None else None)

            store_report_reference_aggregate_cache(
                cur,
                aggregate_hash,
                unique_take_count,
                bundle,
                missing_metric_keys,
                logic_version,
            )
            conn.commit()
            return bundle
    finally:
        conn.close()


def cache_report_metric_data(report_rows, metric_key, metric_label, metric_group, unit, source_category, source_segment, source_axis, report_metric_data, logic_version=REPORT_METRIC_LOGIC_VERSION):
    if not report_rows or not metric_key or not report_metric_data:
        return
    take_ids = [row[0] for row in report_rows]
    take_metrics = report_metric_data.get("take_metrics", {})
    curves = report_metric_data.get("curves", {})

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_report_metric_cache_schema(cur)
                cur.execute("SELECT to_regclass('public.take_report_metric_cache_status')")
                if cur.fetchone()[0] is None:
                    return
                cache_report_events_from_metric_data(cur, report_metric_data, logic_version)
                metric_rows = []
                for take_id, event_values in take_metrics.items():
                    for event_label, payload in event_values.items():
                        value = payload.get("value") if isinstance(payload, dict) else payload
                        source_frame = payload.get("source_frame") if isinstance(payload, dict) else None
                        metric_rows.append((
                            take_id,
                            metric_key,
                            metric_label,
                            metric_group,
                            event_label,
                            float(value) if value is not None and np.isfinite(value) else None,
                            int(source_frame) if source_frame is not None else None,
                            unit,
                            source_category,
                            source_segment,
                            source_axis,
                            logic_version,
                        ))
                if metric_rows:
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        """
                        INSERT INTO take_report_metrics (
                            take_id, metric_key, metric_label, metric_group, event_label,
                            metric_value, source_frame, unit, source_category, source_segment,
                            source_axis, logic_version, computed_at
                        )
                        VALUES %s
                        ON CONFLICT (take_id, metric_key, event_label, logic_version)
                        DO UPDATE SET
                            metric_label = EXCLUDED.metric_label,
                            metric_group = EXCLUDED.metric_group,
                            metric_value = EXCLUDED.metric_value,
                            source_frame = EXCLUDED.source_frame,
                            unit = EXCLUDED.unit,
                            source_category = EXCLUDED.source_category,
                            source_segment = EXCLUDED.source_segment,
                            source_axis = EXCLUDED.source_axis,
                            computed_at = NOW()
                        """,
                        metric_rows,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())",
                        page_size=1000,
                    )

                curve_rows = []
                for take_id, curve in curves.items():
                    for time_ms, value in zip(curve.get("frame", []), curve.get("value", [])):
                        if value is None or not np.isfinite(value):
                            continue
                        curve_rows.append((take_id, metric_key, float(time_ms), float(value), logic_version))
                if curve_rows:
                    cur.execute(
                        """
                        DELETE FROM take_report_metric_curves
                        WHERE take_id = ANY(%s)
                          AND metric_key = %s
                          AND logic_version = %s
                        """,
                        (take_ids, metric_key, logic_version),
                    )
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        """
                        INSERT INTO take_report_metric_curves (
                            take_id, metric_key, time_ms, metric_value, logic_version, computed_at
                        )
                        VALUES %s
                        ON CONFLICT (take_id, metric_key, time_ms, logic_version)
                        DO UPDATE SET
                            metric_value = EXCLUDED.metric_value,
                            computed_at = NOW()
                        """,
                        curve_rows,
                        template="(%s,%s,%s,%s,%s,NOW())",
                        page_size=5000,
                    )

                status_rows = [(take_id, metric_key, logic_version) for take_id in set(take_ids)]
                if status_rows:
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        """
                        INSERT INTO take_report_metric_cache_status (
                            take_id, metric_key, logic_version, computed_at
                        )
                        VALUES %s
                        ON CONFLICT (take_id, metric_key, logic_version)
                        DO UPDATE SET computed_at = NOW()
                        """,
                        status_rows,
                        template="(%s,%s,%s,NOW())",
                        page_size=1000,
                    )
    finally:
        conn.close()


def build_report_arm_metric_data(report_rows, loader_fn, max_selector=None, max_window="fp_to_br"):
    if not report_rows:
        return {"curves": {}, "events": {}, "metrics": {}}

    take_ids = [row[0] for row in report_rows]
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        "R": [take_id for take_id in take_ids if take_handedness.get(take_id) == "R"],
        "L": [take_id for take_id in take_ids if take_handedness.get(take_id) == "L"],
    }

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    metric_data = load_by_handedness(loader_fn)
    event_maps = get_report_event_frame_maps(take_ids)
    br_frames = event_maps["BR"]
    mer_frames = event_maps["MER"]
    fp_frames = event_maps["FP"]
    pkh_frames = event_maps["PKH"]

    if any(take_id not in br_frames or take_id not in mer_frames or take_id not in fp_frames for take_id in take_ids):
        cg_data = load_by_handedness(get_hand_cg_velocity)
        shoulder_er_data = load_by_handedness(get_shoulder_er_angles)
        br_frames = {}
        for take_id, d in cg_data.items():
            valid = [(i, value) for i, value in enumerate(d["x"]) if value is not None]
            if valid:
                idx, _ = max(valid, key=lambda item: item[1])
                br_frames[take_id] = d["frame"][idx] + 4

        mer_frames = {}
        for take_id, d in shoulder_er_data.items():
            valid = [(frame, value) for frame, value in zip(d["frame"], d["z"]) if value is not None]
            if valid:
                mer_frames[take_id] = (
                    min(valid, key=lambda item: item[1])[0]
                    if take_handedness.get(take_id) == "R"
                    else max(valid, key=lambda item: item[1])[0]
                )

        fp_frames = {}
        for hand, ids in take_ids_by_handedness.items():
            if not ids:
                continue
            ankle_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_min_frames = get_ankle_min_frame(ids, hand, ankle_peak_frames, mer_frames)
            ankle_zero_frames = get_foot_plant_frame_zero_cross(ids, hand, ankle_min_frames, mer_frames)
            heel_anchor_frames = {
                take_id: ankle_zero_frames.get(take_id, ankle_min_frames.get(take_id))
                for take_id in ids
            }
            heel_frames = get_lead_heel_contact_frame(ids, hand, ankle_peak_frames, mer_frames, heel_anchor_frames)
            for take_id in ids:
                candidates = [
                    value
                    for value in [
                        ankle_zero_frames.get(take_id),
                        heel_frames.get(take_id),
                        ankle_min_frames.get(take_id),
                        ankle_peak_frames.get(take_id),
                    ]
                    if value is not None
                ]
                if candidates:
                    fp_frames[take_id] = int(max(candidates[:2]) if len(candidates[:2]) == 2 else candidates[0])
        for take_id, fp_frame in list(fp_frames.items()):
            if take_id in mer_frames and fp_frame > mer_frames[take_id]:
                fp_frames[take_id] = mer_frames[take_id]

        pkh_frames = {}
        if max_window == "pkh_to_br":
            for hand, ids in take_ids_by_handedness.items():
                if ids:
                    pkh_frames.update(get_peak_glove_knee_pre_br(ids, hand, br_frames))

    def value_near_frame(curve, frame):
        candidates = [
            (abs(curve_frame - frame), value, curve_frame)
            for curve_frame, value in zip(curve.get("frame", []), curve.get("value", []))
            if value is not None
        ]
        if not candidates:
            return None, None
        _distance, value, actual_frame = min(candidates, key=lambda item: item[0])
        return value, actual_frame

    curves = {}
    metric_values = {"FP": [], "MER": [], "BR": [], "Max": []}
    take_metrics = {}
    for take_id, curve in metric_data.items():
        br_frame = br_frames.get(take_id)
        if br_frame is None:
            continue
        normalized = {
            "frame": [
                rel_frame_to_ms(frame - br_frame)
                for frame, value in zip(curve["frame"], curve["value"])
                if value is not None and np.isfinite(value)
            ],
            "value": [
                value
                for value in curve["value"]
                if value is not None and np.isfinite(value)
            ],
        }
        curves[take_id] = normalized
        event_frame_map = {"FP": fp_frames.get(take_id), "MER": mer_frames.get(take_id), "BR": br_frame}
        for event_label, event_frame in event_frame_map.items():
            if event_frame is not None:
                value, actual_frame = value_near_frame(curve, event_frame)
                if value is not None:
                    metric_values[event_label].append(value)
                    take_metrics.setdefault(take_id, {})[event_label] = {
                        "value": float(value),
                        "source_frame": int(actual_frame if actual_frame is not None else event_frame),
                    }
        fp_frame = fp_frames.get(take_id)
        if max_window == "pre_br":
            finite_frames = [
                frame
                for frame, value in zip(curve["frame"], curve["value"])
                if value is not None and np.isfinite(value) and frame <= br_frame
            ]
            window_start = min(finite_frames) if finite_frames else br_frame - ms_to_rel_frame(200)
            window_end = br_frame
        elif max_window == "pkh_to_br":
            window_start = pkh_frames.get(take_id)
            if window_start is None:
                window_start = fp_frame if fp_frame is not None else br_frame - ms_to_rel_frame(200)
            window_end = br_frame
        else:
            window_start = fp_frame if fp_frame is not None else br_frame - ms_to_rel_frame(200)
            window_end = mer_frames.get(take_id) if max_window == "fp_to_mer" else br_frame
        if window_end is None:
            window_end = br_frame
        valid_samples = [
            (frame, value)
            for frame, value in zip(curve["frame"], curve["value"])
            if value is not None
            and np.isfinite(value)
            and window_start <= frame <= window_end
        ]
        if valid_samples:
            valid_values = [value for _frame, value in valid_samples]
            max_value = max_selector(valid_values) if max_selector else max(valid_values)
            max_frame, _raw_value = min(
                valid_samples,
                key=lambda item: abs(float(item[1]) - float(max_value)),
            )
            metric_values["Max"].append(max_value)
            take_metrics.setdefault(take_id, {})["Max"] = {
                "value": float(max_value),
                "source_frame": int(max_frame),
            }

    metrics = {}
    for label, values in metric_values.items():
        metrics[label] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)) if values else None,
        }
    return {
        "curves": curves,
        "events": {
            "fp_event_frames": [fp_frames[take_id] - br_frames[take_id] for take_id in fp_frames if take_id in br_frames],
            "mer_event_frames": [mer_frames[take_id] - br_frames[take_id] for take_id in mer_frames if take_id in br_frames],
        },
        "metrics": metrics,
        "take_metrics": take_metrics,
    }


def build_report_elbow_flexion_data(report_rows):
    return build_report_arm_metric_data(report_rows, get_elbow_flexion_angle)


def build_report_shoulder_abduction_data(report_rows):
    return build_report_arm_metric_data(report_rows, get_shoulder_abduction_angle)


def build_report_shoulder_horizontal_abduction_data(report_rows):
    return build_report_arm_metric_data(report_rows, get_shoulder_horizontal_abduction_angle)


def build_report_shoulder_rotation_data(report_rows):
    data = build_report_arm_metric_data(report_rows, get_shoulder_er_angle)
    metrics = dict(data.get("metrics", {}))
    if "MER" in metrics:
        metrics["Max"] = dict(metrics["MER"])
        data["metrics"] = metrics
    return data


def build_report_pelvis_component_data(report_rows, component):
    if component == "z":
        return build_report_arm_metric_data(
            report_rows,
            lambda take_ids, handedness: {
                take_id: {
                    "frame": curve["frame"],
                    "value": normalize_report_rotation_values(curve["value"], handedness),
                }
                for take_id, curve in get_joint_angle_rotation_component(take_ids, "PELVIS").items()
            },
        )
    return build_report_arm_metric_data(
        report_rows,
        lambda take_ids, handedness: {
            take_id: {
                "frame": curve["frame"],
                "value": normalize_pelvis_torso_component_values(curve["value"], component, handedness),
            }
            for take_id, curve in get_pelvis_angle_component(take_ids, handedness, component).items()
        },
        max_window="fp_to_mer" if component == "x" else "fp_to_br",
    )


def normalize_pelvis_torso_component_values(values, component, handedness):
    normalized_values = []
    for value in values:
        if value is None:
            normalized_values.append(value)
        elif component == "x":
            # Theia x tilt is negative into forward tilt; report positive means forward tilt.
            normalized_values.append(-value)
        elif component == "y" and handedness == "R":
            # Report positive lateral tilt means glove-side tilt for both RHP and LHP.
            normalized_values.append(-value)
        else:
            normalized_values.append(value)
    return normalized_values


def build_report_torso_component_data(report_rows, component):
    if component == "z":
        return build_report_arm_metric_data(
            report_rows,
            lambda take_ids, handedness: {
                take_id: {
                    "frame": curve["frame"],
                    "value": normalize_report_rotation_values(curve["value"], handedness),
                }
                for take_id, curve in get_joint_angle_rotation_component(take_ids, "TORSO").items()
            },
            max_selector=lambda values: np.nanmin(values),
        )
    return build_report_arm_metric_data(
        report_rows,
        lambda take_ids, handedness: {
            take_id: {
                "frame": curve["frame"],
                "value": normalize_pelvis_torso_component_values(curve[component], component, handedness),
            }
            for take_id, curve in get_torso_angle_components(take_ids).items()
        },
    )


def build_report_torso_pelvis_component_data(report_rows, component):
    return build_report_arm_metric_data(
        report_rows,
        lambda take_ids, handedness: {
            take_id: {
                "frame": curve["frame"],
                "value": (
                    normalize_pelvis_torso_component_values(curve[component], component, handedness)
                    if component in ("x", "y") else
                    [
                        normalize_torso_pelvis_rotation_value(value, handedness)
                        for value in curve[component]
                    ]
                    if component == "z" else curve[component]
                ),
            }
            for take_id, curve in get_torso_pelvis_angle_components(take_ids).items()
        },
    )


def build_report_hip_component_data(report_rows, hip_role, component):
    max_window = "pkh_to_br" if hip_role == "lead" and component == "x" else "pre_br" if component in ("x", "y", "z") else "fp_to_br"
    max_selector = (lambda values: abs(float(np.nanmin(values)))) if hip_role == "lead" and component == "x" else None
    return build_report_arm_metric_data(
        report_rows,
        lambda take_ids, handedness: get_hip_angle_component(take_ids, handedness, hip_role, component),
        max_selector=max_selector,
        max_window=max_window,
    )


def build_report_cg_velocity_data(report_rows, component):
    def load_cg_component(take_ids, handedness):
        data = get_center_of_mass_velocity_component(take_ids, component, handedness)
        if component != "y":
            return data
        return {
            take_id: {
                "frame": curve["frame"],
                "value": [
                    -value if handedness == "R" and value is not None else value
                    for value in curve["value"]
                ],
            }
            for take_id, curve in data.items()
        }

    if component == "z":
        max_selector = lambda values: abs(float(np.nanmin(values)))
    else:
        max_selector = lambda values: float(np.nanmax(values))

    return build_report_arm_metric_data(
        report_rows,
        load_cg_component,
        max_selector=max_selector,
        max_window="pre_br",
    )


def build_report_lower_extremity_component_data(report_rows, leg_role, joint, component):
    return build_report_arm_metric_data(
        report_rows,
        lambda take_ids, handedness: get_lower_extremity_angle_component(take_ids, handedness, leg_role, joint, component),
        max_window="pre_br" if component == "x" else "fp_to_br",
    )


def build_report_lower_extremity_velocity_data(report_rows, leg_role, joint, component):
    if component == "x":
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_lower_extremity_angular_velocity_component(take_ids, handedness, leg_role, joint, component),
            lambda values, handedness: np.nanmin(values),
            window_frames=50,
            component="x",
            peak_before_br_sign="negative",
            normalizer_fn=normalize_identity,
        )
    return build_report_arm_velocity_data(
        report_rows,
        lambda take_ids, handedness: get_lower_extremity_angular_velocity_component(take_ids, handedness, leg_role, joint, component),
        peak_mode="absolute",
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_stride_foot_positions(take_ids, handedness, fp_frames, knee_peak_frames):
    """
    Returns lead-foot positions used for stride metrics.

    Mirrors pitcher_query.sql: KINETIC_KINEMATIC_CGPos / LFT or RFT, using
    the lead foot at FP and the same foot at the pre-BR knee peak frame.
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "LFT" if handedness == "R" else "RFT"
    needed_frames = {
        take_id: {frame for frame in (fp_frames.get(take_id), knee_peak_frames.get(take_id)) if frame is not None}
        for take_id in take_ids
    }
    needed_frames = {take_id: frames for take_id, frames in needed_frames.items() if frames}
    if not needed_frames:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data,
                    ts.y_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'KINETIC_KINEMATIC_CGPos'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *tuple(take_ids)))

            positions = {}
            for take_id, frame, x, y in cur.fetchall():
                if frame not in needed_frames.get(take_id, set()):
                    continue
                positions.setdefault(take_id, {})[int(frame)] = (x, y)

            out = {}
            for take_id in take_ids:
                fp_frame = fp_frames.get(take_id)
                trail_frame = knee_peak_frames.get(take_id)
                lead_pos = positions.get(take_id, {}).get(fp_frame)
                trail_pos = positions.get(take_id, {}).get(trail_frame)
                if lead_pos is None or trail_pos is None:
                    continue
                out[take_id] = {
                    "lead_foot_x": lead_pos[0],
                    "lead_foot_y": lead_pos[1],
                    "trail_foot_x": trail_pos[0],
                    "trail_foot_y": trail_pos[1],
                }
            return out
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_take_heights(take_ids):
    if not take_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT take_id, height
                FROM takes
                WHERE take_id IN ({placeholders})
            """, tuple(take_ids))
            return {take_id: height for take_id, height in cur.fetchall()}
    finally:
        conn.close()


def summarize_values(values):
    clean_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "mean": float(np.mean(clean_values)) if clean_values else None,
        "std": float(np.std(clean_values, ddof=1 if len(clean_values) > 1 else 0)) if clean_values else None,
    }


def build_report_stride_data(report_rows):
    if not report_rows:
        return {"metrics": {}}

    take_ids = [row[0] for row in report_rows]
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        hand: [take_id for take_id in take_ids if take_handedness.get(take_id) == hand]
        for hand in ("R", "L")
    }

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    event_maps = get_report_event_frame_maps(take_ids)
    br_frames = event_maps["BR"]
    mer_frames = event_maps["MER"]
    fp_frames = event_maps["FP"]
    knee_peak_frames = event_maps["PKH"]

    if any(take_id not in br_frames or take_id not in mer_frames or take_id not in fp_frames or take_id not in knee_peak_frames for take_id in take_ids):
        cg_data = load_by_handedness(get_hand_cg_velocity)
        br_frames = {}
        for take_id, curve in cg_data.items():
            valid = [(idx, value) for idx, value in enumerate(curve["x"]) if value is not None]
            if valid:
                idx, _ = max(valid, key=lambda item: item[1])
                br_frames[take_id] = curve["frame"][idx] + 4

        shoulder_er_data = load_by_handedness(get_shoulder_er_angles)
        mer_frames = {}
        for take_id, curve in shoulder_er_data.items():
            valid = [(frame, value) for frame, value in zip(curve["frame"], curve["z"]) if value is not None]
            if valid:
                mer_frames[take_id] = (
                    min(valid, key=lambda item: item[1])[0]
                    if take_handedness.get(take_id) == "R"
                    else max(valid, key=lambda item: item[1])[0]
                )

        fp_frames = {}
        knee_peak_frames = {}
        for hand, ids in take_ids_by_handedness.items():
            if not ids:
                continue
            hand_knee_peak_frames = get_peak_glove_knee_pre_br(ids, hand, br_frames)
            knee_peak_frames.update(hand_knee_peak_frames)
            ankle_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_min_frames = get_ankle_min_frame(ids, hand, ankle_peak_frames, mer_frames)
            ankle_zero_frames = get_foot_plant_frame_zero_cross(ids, hand, ankle_min_frames, mer_frames)
            for take_id in ids:
                candidates = [
                    value
                    for value in [
                        ankle_zero_frames.get(take_id),
                        ankle_min_frames.get(take_id),
                        ankle_peak_frames.get(take_id),
                    ]
                    if value is not None
                ]
                if candidates:
                    fp_frames[take_id] = int(candidates[0])

    heights = get_take_heights(take_ids)
    stride_values = {
        "stride_length_in": [],
        "stride_length_pct_height": [],
        "stride_angle_deg": [],
    }
    take_athlete = {row[0]: row[3] for row in report_rows}
    take_session_date = {row[0]: row[2] for row in report_rows}
    stride_values_by_athlete_session = {}
    for hand, ids in take_ids_by_handedness.items():
        if not ids:
            continue
        positions = get_stride_foot_positions(ids, hand, fp_frames, knee_peak_frames)
        for take_id, pos in positions.items():
            if any(pos.get(key) is None for key in ("lead_foot_x", "trail_foot_x", "lead_foot_y", "trail_foot_y")):
                continue
            dx = float(pos["lead_foot_x"]) - float(pos["trail_foot_x"])
            dy = float(pos["lead_foot_y"]) - float(pos["trail_foot_y"])
            stride_length_in = dx * 39.37
            stride_angle = float(np.degrees(np.arctan2(abs(dy), dx))) if dx else None
            height = heights.get(take_id)
            stride_pct_height = (
                (stride_length_in / float(height) * 100)
                if height not in (None, 0)
                else None
            )
            athlete_name = take_athlete.get(take_id)
            values_for_take = {
                "stride_length_in": stride_length_in,
                "stride_length_pct_height": stride_pct_height,
                "stride_angle_deg": stride_angle,
            }
            for label, value in values_for_take.items():
                stride_values[label].append(value)
                session_key = (athlete_name, take_session_date.get(take_id))
                if session_key[0] is not None:
                    stride_values_by_athlete_session.setdefault(
                        session_key,
                        {
                            "stride_length_in": [],
                            "stride_length_pct_height": [],
                            "stride_angle_deg": [],
                        },
                    )[label].append(value)

    metrics = {}
    for label, values in stride_values.items():
        summary = summarize_values(values)
        athlete_session_sds = [
            summarize_values(session_values[label]).get("std")
            for session_values in stride_values_by_athlete_session.values()
        ]
        athlete_session_sds = [value for value in athlete_session_sds if value is not None and np.isfinite(value)]
        if athlete_session_sds:
            summary["std"] = float(np.mean(athlete_session_sds))
        metrics[label] = summary

    return {"metrics": metrics}


def normalize_hip_flexion_extension_velocity(value, handedness):
    if value is None:
        return value
    # Report convention: positive is hip flexion, negative is hip extension.
    return -value


def make_hip_abduction_adduction_velocity_normalizer(hip_role):
    def normalize(value, handedness):
        if value is None:
            return value
        segment_prefix = get_hip_segment_prefix(handedness, hip_role)
        return -value if segment_prefix == "RT" else value

    return normalize


def make_hip_internal_external_rotation_velocity_normalizer(hip_role):
    def normalize(value, handedness):
        if value is None:
            return value
        segment_prefix = get_hip_segment_prefix(handedness, hip_role)
        return -value if segment_prefix == "RT" else value

    return normalize


def build_report_hip_velocity_data(report_rows, hip_role, component, peak_direction=None):
    if component == "x":
        if peak_direction == "extension":
            return build_report_pelvis_velocity_window_data(
                report_rows,
                lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
                lambda values, handedness: np.nanmin(values),
                window_frames=50,
                component="x",
                peak_before_br_sign="negative",
                normalizer_fn=normalize_hip_flexion_extension_velocity,
                peak_window="pkh_to_br",
            )
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
            lambda values, handedness: np.nanmax(values),
            window_frames=50,
            component="x",
            peak_before_br_sign="positive",
            normalizer_fn=normalize_hip_flexion_extension_velocity,
            peak_window="pkh_to_br",
        )
    if component == "y":
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
            lambda values, handedness: np.nanmin(values),
            window_frames=50,
            component="y",
            peak_before_br_sign="negative",
            normalizer_fn=make_hip_abduction_adduction_velocity_normalizer(hip_role),
        )
    if component == "z":
        normalizer = make_hip_internal_external_rotation_velocity_normalizer(hip_role)
        if peak_direction == "external":
            return build_report_pelvis_velocity_window_data(
                report_rows,
                lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
                lambda values, handedness: np.nanmin(values),
                window_frames=50,
                component="z",
                peak_before_br_sign="negative",
                normalizer_fn=normalizer,
            )
        if peak_direction == "internal":
            return build_report_pelvis_velocity_window_data(
                report_rows,
                lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
                lambda values, handedness: np.nanmax(values),
                window_frames=50,
                component="z",
                peak_before_br_sign="positive",
                normalizer_fn=normalizer,
            )
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
            lambda values, handedness: values[int(np.nanargmax(np.abs(values)))],
            window_frames=50,
            component="z",
            normalizer_fn=normalizer,
        )
    selectors = {
    }
    return build_report_pelvis_velocity_window_data(
        report_rows,
        lambda take_ids, handedness: get_hip_angular_velocity_component(take_ids, handedness, hip_role, component),
        selectors[component],
    )


def get_shoulder_horizontal_abduction_velocity(take_ids, handedness):
    """
    Returns throwing-side shoulder RTA angular velocity x_data.

    This matches the 0-10 report's source for Max Scap Retraction Velocity:
      RHP -> RT_SHOULDER_RTA_ANGULAR_VELOCITY
      LHP -> LT_SHOULDER_RTA_ANGULAR_VELOCITY
      Category -> ORIGINAL
      Axis -> x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_SHOULDER_RTA_ANGULAR_VELOCITY" if handedness == "R" else "LT_SHOULDER_RTA_ANGULAR_VELOCITY"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            data = {}
            for take_id, frame, x in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def get_original_shoulder_horizontal_angle(take_ids, handedness):
    """
    Raw shoulder angle x_data used only to locate max scap retraction.

    Mirrors 0-10 report logic:
      RHP -> RT_SHOULDER_ANGLE
      LHP -> LT_SHOULDER_ANGLE
      Category -> ORIGINAL
      Axis -> x_data
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = "RT_SHOULDER_ANGLE" if handedness == "R" else "LT_SHOULDER_ANGLE"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(take_ids))
            cur.execute(f"""
                SELECT
                    ts.take_id,
                    ts.frame,
                    ts.x_data
                FROM time_series_data ts
                JOIN categories c ON ts.category_id = c.category_id
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
                  AND ts.x_data IS NOT NULL
                ORDER BY ts.take_id, ts.frame
            """, (segment_name, *take_ids))

            data = {}
            for take_id, frame, x in cur.fetchall():
                data.setdefault(take_id, {"frame": [], "value": []})
                data[take_id]["frame"].append(frame)
                data[take_id]["value"].append(x)
            return data
    finally:
        conn.close()


def build_report_arm_velocity_data(report_rows, loader_fn, value_key="value", invert_for_all=False, invert_left=False, peak_mode="max"):
    if not report_rows:
        return {"curves": {}, "events": {}, "metrics": {}}
    take_ids = [row[0] for row in report_rows]
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        hand: [take_id for take_id in take_ids if take_handedness.get(take_id) == hand]
        for hand in ("R", "L")
    }
    raw_data = {}
    for hand, ids in take_ids_by_handedness.items():
        if ids:
            raw_data.update(loader_fn(ids, hand))
    event_maps = get_report_event_frame_maps(take_ids)
    br_frames = event_maps["BR"]
    if any(take_id not in br_frames for take_id in take_ids):
        cg_data = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                cg_data.update(get_hand_cg_velocity(ids, hand))
        br_frames = {}
        for take_id, curve in cg_data.items():
            valid = [(i, value) for i, value in enumerate(curve["x"]) if value is not None]
            if valid:
                idx, _ = max(valid, key=lambda item: item[1])
                br_frames[take_id] = curve["frame"][idx] + 4
    curves = {}
    peaks = []
    take_metrics = {}
    for take_id, curve in raw_data.items():
        if take_id not in br_frames:
            continue
        normalized_frames = []
        normalized_values = []
        normalized_samples = []
        for frame, value in zip(curve.get("frame", []), curve.get(value_key, [])):
            if value is None or not np.isfinite(value):
                continue
            normalized = -value if invert_for_all or (invert_left and take_handedness.get(take_id) == "L") else value
            normalized_frames.append(rel_frame_to_ms(frame - br_frames[take_id]))
            normalized_values.append(normalized)
            normalized_samples.append((frame, normalized))
        if normalized_values:
            curves[take_id] = {"frame": normalized_frames, "value": normalized_values}
            if peak_mode == "external":
                peak_value = abs(min(normalized_values))
                peak_frame, _ = min(normalized_samples, key=lambda item: item[1])
            elif peak_mode == "absolute":
                peak_frame, peak_raw_value = max(normalized_samples, key=lambda item: abs(item[1]))
                peak_value = abs(peak_raw_value)
            else:
                peak_frame, peak_value = max(normalized_samples, key=lambda item: item[1])
            peaks.append(float(peak_value))
            take_metrics.setdefault(take_id, {})["Max"] = {
                "value": float(peak_value),
                "source_frame": int(peak_frame),
            }
    return {
        "curves": curves,
        "events": event_maps.get("events", {}),
        "metrics": {
            "Max": {
                "mean": float(np.mean(peaks)) if peaks else None,
                "std": float(np.std(peaks, ddof=1 if len(peaks) > 1 else 0)) if peaks else None,
            }
        },
        "take_metrics": take_metrics,
    }


def build_report_shoulder_horizontal_abduction_velocity_data(report_rows):
    if not report_rows:
        return {"curves": {}, "events": {}, "metrics": {}}

    take_ids = [row[0] for row in report_rows]
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        hand: [take_id for take_id in take_ids if take_handedness.get(take_id) == hand]
        for hand in ("R", "L")
    }

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    velocity_data = load_by_handedness(get_shoulder_horizontal_abduction_velocity)
    raw_angle_data = load_by_handedness(get_original_shoulder_horizontal_angle)

    event_maps = get_report_event_frame_maps(take_ids)
    br_frames = event_maps["BR"]
    mer_frames = event_maps["MER"]
    fp_frames = event_maps["FP"]

    if any(take_id not in br_frames or take_id not in mer_frames or take_id not in fp_frames for take_id in take_ids):
        cg_data = load_by_handedness(get_hand_cg_velocity)
        shoulder_er_data = load_by_handedness(get_shoulder_er_angles)
        br_frames = {}
        for take_id, curve in cg_data.items():
            valid = [(idx, value) for idx, value in enumerate(curve.get("x", [])) if value is not None]
            if valid:
                idx, _ = max(valid, key=lambda item: item[1])
                br_frames[take_id] = curve["frame"][idx] + 4

        mer_frames = {}
        for take_id, curve in shoulder_er_data.items():
            valid = [(frame, value) for frame, value in zip(curve.get("frame", []), curve.get("z", [])) if value is not None]
            if valid:
                mer_frames[take_id] = (
                    min(valid, key=lambda item: item[1])[0]
                    if take_handedness.get(take_id) == "R"
                    else max(valid, key=lambda item: item[1])[0]
                )

        fp_frames = {}
        for hand, ids in take_ids_by_handedness.items():
            if not ids:
                continue
            ankle_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_min_frames = get_ankle_min_frame(ids, hand, ankle_peak_frames, mer_frames)
            ankle_zero_frames = get_foot_plant_frame_zero_cross(ids, hand, ankle_min_frames, mer_frames)
            heel_anchor_frames = {
                take_id: ankle_zero_frames.get(take_id, ankle_min_frames.get(take_id))
                for take_id in ids
            }
            heel_frames = get_lead_heel_contact_frame(ids, hand, ankle_peak_frames, mer_frames, heel_anchor_frames)
            for take_id in ids:
                candidates = [
                    value
                    for value in [
                        ankle_zero_frames.get(take_id),
                        heel_frames.get(take_id),
                        ankle_min_frames.get(take_id),
                        ankle_peak_frames.get(take_id),
                    ]
                    if value is not None
                ]
                if candidates:
                    fp_frames[take_id] = int(max(candidates[:2]) if len(candidates[:2]) == 2 else candidates[0])
        for take_id, fp_frame in list(fp_frames.items()):
            if take_id in mer_frames and fp_frame > mer_frames[take_id]:
                fp_frames[take_id] = mer_frames[take_id]

    curves = {}
    peaks = []
    take_metrics = {}
    for take_id, velocity_curve in velocity_data.items():
        br_frame = br_frames.get(take_id)
        if br_frame is None:
            continue

        valid_velocity = [
            (int(frame), float(value))
            for frame, value in zip(velocity_curve.get("frame", []), velocity_curve.get("value", []))
            if value is not None and np.isfinite(value)
        ]
        if valid_velocity:
            curves[take_id] = {
                "frame": [rel_frame_to_ms(frame - br_frame) for frame, _ in valid_velocity],
                "value": [value for _, value in valid_velocity],
            }

        mer_frame = mer_frames.get(take_id)
        angle_curve = raw_angle_data.get(take_id, {})
        valid_angle = [
            (int(frame), float(value))
            for frame, value in zip(angle_curve.get("frame", []), angle_curve.get("value", []))
            if value is not None and np.isfinite(value)
        ]
        if mer_frame is None or not valid_angle or not valid_velocity:
            continue

        scap_window = [(frame, value) for frame, value in valid_angle if mer_frame - 50 <= frame <= mer_frame]
        if not scap_window:
            continue
        max_scap_frame, _ = (
            min(scap_window, key=lambda item: item[1])
            if take_handedness.get(take_id) == "R"
            else max(scap_window, key=lambda item: item[1])
        )

        preceding_angles = [(frame, value) for frame, value in valid_angle if frame <= max_scap_frame]
        start_frame = None
        if take_handedness.get(take_id) == "R":
            for (prev_frame, prev_value), (curr_frame, curr_value) in zip(preceding_angles, preceding_angles[1:]):
                if prev_value > 0 and curr_value <= 0:
                    start_frame = curr_frame
        else:
            for (prev_frame, prev_value), (curr_frame, curr_value) in zip(preceding_angles, preceding_angles[1:]):
                if prev_value < 0 and curr_value >= 0:
                    start_frame = curr_frame
        if start_frame is None and preceding_angles:
            start_frame, _ = min(preceding_angles, key=lambda item: abs(item[1]))
        if start_frame is None:
            continue

        velocity_window = [
            (frame, value)
            for frame, value in valid_velocity
            if start_frame <= frame <= max_scap_frame
        ]
        if velocity_window:
            peak_frame, peak_raw_value = max(velocity_window, key=lambda item: item[1])
            peak_value = abs(float(peak_raw_value))
            peaks.append(peak_value)
            take_metrics.setdefault(take_id, {})["Max"] = {
                "value": peak_value,
                "source_frame": int(peak_frame),
            }

    return {
        "curves": curves,
        "events": {
            "fp_event_frames": [fp_frames[take_id] - br_frames[take_id] for take_id in fp_frames if take_id in br_frames],
            "mer_event_frames": [mer_frames[take_id] - br_frames[take_id] for take_id in mer_frames if take_id in br_frames],
        },
        "metrics": {
            "Max": {
                "mean": float(np.mean(peaks)) if peaks else None,
                "std": float(np.std(peaks, ddof=1 if len(peaks) > 1 else 0)) if peaks else None,
            }
        },
        "take_metrics": take_metrics,
    }


def normalize_pelvis_velocity_value(value, component, handedness):
    if value is None:
        return value
    if component == "x":
        # Match position convention: positive is forward tilt velocity.
        return -value
    if component == "y" and handedness == "R":
        # Match position convention: positive is glove-side lateral tilt velocity.
        return -value
    if component == "z" and handedness == "L":
        # Match rotation convention: RHP z + 90, LHP 90 - z.
        return -value
    return value


def normalize_torso_pelvis_rotation_value(value, handedness):
    if value is None:
        return value
    # Positive torso-pelvis rotation means hip-shoulder separation for both handedness.
    return -value if handedness == "R" else value


def build_report_pelvis_velocity_window_data(report_rows, loader_fn, peak_selector, window_frames=40, component=None, peak_before_br_sign=None, normalizer_fn=None, peak_window=None):
    if not report_rows:
        return {"curves": {}, "events": {}, "metrics": {}}

    take_ids = [row[0] for row in report_rows]
    take_handedness = {row[0]: row[4] for row in report_rows}
    take_ids_by_handedness = {
        hand: [take_id for take_id in take_ids if take_handedness.get(take_id) == hand]
        for hand in ("R", "L")
    }

    def load_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    pelvis_velocity_data = load_by_handedness(loader_fn)

    event_maps = get_report_event_frame_maps(take_ids)
    br_frames = event_maps["BR"]
    mer_frames = event_maps["MER"]
    fp_frames = event_maps["FP"]
    pkh_frames = event_maps["PKH"]

    if any(take_id not in br_frames or take_id not in mer_frames or take_id not in fp_frames for take_id in take_ids):
        cg_data = load_by_handedness(get_hand_cg_velocity)
        shoulder_er_data = load_by_handedness(get_shoulder_er_angles)
        br_frames = {}
        for take_id, curve in cg_data.items():
            valid = [(idx, value) for idx, value in enumerate(curve["x"]) if value is not None]
            if valid:
                idx, _ = max(valid, key=lambda item: item[1])
                br_frames[take_id] = curve["frame"][idx] + 4

        mer_frames = {}
        for take_id, curve in shoulder_er_data.items():
            valid = [(frame, value) for frame, value in zip(curve["frame"], curve["z"]) if value is not None]
            if valid:
                mer_frames[take_id] = (
                    min(valid, key=lambda item: item[1])[0]
                    if take_handedness.get(take_id) == "R"
                    else max(valid, key=lambda item: item[1])[0]
                )

        fp_frames = {}
        for hand, ids in take_ids_by_handedness.items():
            if not ids:
                continue
            ankle_peak_frames = get_peak_ankle_prox_x_velocity(ids, hand)
            ankle_min_frames = get_ankle_min_frame(ids, hand, ankle_peak_frames, mer_frames)
            ankle_zero_frames = get_foot_plant_frame_zero_cross(ids, hand, ankle_min_frames, mer_frames)
            heel_anchor_frames = {
                take_id: ankle_zero_frames.get(take_id, ankle_min_frames.get(take_id))
                for take_id in ids
            }
            heel_frames = get_lead_heel_contact_frame(ids, hand, ankle_peak_frames, mer_frames, heel_anchor_frames)
            for take_id in ids:
                candidates = [
                    value
                    for value in [
                        ankle_zero_frames.get(take_id),
                        heel_frames.get(take_id),
                        ankle_min_frames.get(take_id),
                        ankle_peak_frames.get(take_id),
                    ]
                    if value is not None
                ]
                if candidates:
                    fp_frames[take_id] = int(max(candidates[:2]) if len(candidates[:2]) == 2 else candidates[0])
        for take_id, fp_frame in list(fp_frames.items()):
            if take_id in mer_frames and fp_frame > mer_frames[take_id]:
                fp_frames[take_id] = mer_frames[take_id]

        pkh_frames = {}
        if peak_window == "pkh_to_br":
            for hand, ids in take_ids_by_handedness.items():
                if ids:
                    pkh_frames.update(get_peak_glove_knee_pre_br(ids, hand, br_frames))

    curves = {}
    peak_values = []
    take_metrics = {}
    for take_id, curve in pelvis_velocity_data.items():
        br_frame = br_frames.get(take_id)
        if br_frame is None:
            continue

        valid_samples = [
            (
                frame,
                float(normalizer_fn(value, take_handedness.get(take_id)))
                if normalizer_fn else
                float(normalize_pelvis_velocity_value(value, component, take_handedness.get(take_id)))
                if component else
                float(value)
            )
            for frame, value in zip(curve.get("frame", []), curve.get("value", []))
            if value is not None and np.isfinite(value)
        ]
        if not valid_samples:
            continue

        curves[take_id] = {
            "frame": [rel_frame_to_ms(frame - br_frame) for frame, _ in valid_samples],
            "value": [value for _, value in valid_samples],
        }

        fp_frame = fp_frames.get(take_id)
        if peak_before_br_sign:
            if peak_window == "pkh_to_br":
                window_start = pkh_frames.get(take_id)
                if window_start is None:
                    window_start = fp_frame if fp_frame is not None else br_frame - (window_frames * 2)
            else:
                window_start = fp_frame - window_frames if fp_frame is not None else br_frame - (window_frames * 2)
            windowed_samples = [
                (frame, value)
                for frame, value in valid_samples
                if window_start <= frame <= br_frame
                and ((peak_before_br_sign == "positive" and value > 0) or (peak_before_br_sign == "negative" and value < 0))
            ]
        else:
            windowed_samples = [
                (frame, value)
                for frame, value in valid_samples
                if fp_frame is not None and fp_frame - window_frames <= frame <= fp_frame + window_frames
            ]
        if not windowed_samples:
            windowed_samples = [
                (frame, value)
                for frame, value in valid_samples
                if frame <= br_frame
                and (
                    not peak_before_br_sign
                    or (peak_before_br_sign == "positive" and value > 0)
                    or (peak_before_br_sign == "negative" and value < 0)
                )
            ]
        if not windowed_samples:
            continue
        windowed_values = [value for _frame, value in windowed_samples]
        peak_value = abs(float(peak_selector(np.asarray(windowed_values, dtype=float), take_handedness.get(take_id))))
        peak_frame, _raw_value = min(
            windowed_samples,
            key=lambda item: abs(abs(float(item[1])) - peak_value),
        )
        peak_values.append(peak_value)
        take_metrics.setdefault(take_id, {})["Max"] = {
            "value": peak_value,
            "source_frame": int(peak_frame),
        }

    return {
        "curves": curves,
        "events": {
            "fp_event_frames": [fp_frames[take_id] - br_frames[take_id] for take_id in fp_frames if take_id in br_frames],
            "mer_event_frames": [mer_frames[take_id] - br_frames[take_id] for take_id in mer_frames if take_id in br_frames],
        },
        "metrics": {
            "Max": {
                "mean": float(np.mean(peak_values)) if peak_values else None,
                "std": float(np.std(peak_values, ddof=1 if len(peak_values) > 1 else 0)) if peak_values else None,
            }
        },
        "take_metrics": take_metrics,
    }


def build_report_pelvis_forward_tilt_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_pelvis_angular_velocity_x,
        lambda values, handedness: np.nanmax(values),
        component="x",
        peak_before_br_sign="positive",
    )


def build_report_pelvis_lateral_tilt_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_pelvis_angular_velocity_y,
        lambda values, handedness: np.nanmax(values),
        component="y",
        peak_before_br_sign="positive",
    )


def build_report_pelvis_rotation_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_pelvis_angular_velocity_z,
        lambda values, handedness: np.nanmax(values),
        component="z",
        peak_before_br_sign="positive",
    )


def build_report_torso_forward_tilt_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_torso_angular_velocity_x,
        lambda values, handedness: np.nanmax(values),
        window_frames=50,
        component="x",
        peak_before_br_sign="positive",
    )


def build_report_torso_backward_tilt_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_torso_angular_velocity_x,
        lambda values, handedness: np.nanmin(values),
        window_frames=50,
        component="x",
        peak_before_br_sign="negative",
    )


def build_report_torso_lateral_tilt_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_torso_angular_velocity_y,
        lambda values, handedness: np.nanmax(values),
        window_frames=50,
        component="y",
        peak_before_br_sign="positive",
    )


def build_report_torso_rotation_velocity_data(report_rows):
    return build_report_pelvis_velocity_window_data(
        report_rows,
        get_torso_angular_velocity_z,
        lambda values, handedness: np.nanmax(values),
        window_frames=50,
        component="z",
        peak_before_br_sign="positive",
    )


def build_report_torso_pelvis_velocity_data(report_rows, component, peak_direction=None):
    if component == "x":
        if peak_direction == "extension":
            return build_report_pelvis_velocity_window_data(
                report_rows,
                lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
                lambda values, handedness: np.nanmin(values),
                window_frames=50,
                component="x",
                peak_before_br_sign="negative",
            )
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
            lambda values, handedness: np.nanmax(values),
            window_frames=50,
            component="x",
            peak_before_br_sign="positive",
        )
    if component == "y":
        if peak_direction == "arm_side":
            return build_report_pelvis_velocity_window_data(
                report_rows,
                lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
                lambda values, handedness: np.nanmin(values),
                window_frames=50,
                component="y",
                peak_before_br_sign="negative",
            )
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
            lambda values, handedness: np.nanmax(values),
            window_frames=50,
            component="y",
            peak_before_br_sign="positive",
        )
    if component == "z":
        return build_report_pelvis_velocity_window_data(
            report_rows,
            lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
            lambda values, handedness: np.nanmax(values),
            window_frames=50,
            component="z",
            peak_before_br_sign="positive",
            normalizer_fn=normalize_torso_pelvis_rotation_value,
        )
    selectors = {}
    return build_report_pelvis_velocity_window_data(
        report_rows,
        lambda take_ids, handedness: get_torso_pelvis_angular_velocity_component(take_ids, component, handedness),
        selectors[component],
    )


def report_metric_spec(metric_key, label, group, unit, builder_fn, athlete_key=None, reference_key=None, source_category=None, source_segment=None, source_axis=None):
    base_key = metric_key
    return {
        "metric_key": metric_key,
        "label": label,
        "group": group,
        "unit": unit,
        "builder": builder_fn,
        "athlete_key": athlete_key or f"athlete_{base_key}",
        "reference_key": reference_key or f"reference_{base_key}",
        "source_category": source_category,
        "source_segment": source_segment,
        "source_axis": source_axis,
    }


def get_report_metric_specs():
    return [
        report_metric_spec("elbow_flexion", "Elbow Flexion", "Throwing Arm Kinematics", "deg", build_report_elbow_flexion_data, athlete_key="athlete", reference_key="reference"),
        report_metric_spec("shoulder_abduction", "Shoulder Abduction", "Throwing Arm Kinematics", "deg", build_report_shoulder_abduction_data),
        report_metric_spec("shoulder_horizontal_abduction", "Shoulder Horizontal Abduction", "Throwing Arm Kinematics", "deg", build_report_shoulder_horizontal_abduction_data),
        report_metric_spec("shoulder_rotation", "Shoulder Rotation", "Throwing Arm Kinematics", "deg", build_report_shoulder_rotation_data),
        report_metric_spec("elbow_extension_velocity", "Elbow Extension Angular Velocity", "Throwing Arm Kinematics", "deg/s", lambda rows: build_report_arm_velocity_data(rows, get_elbow_angular_velocity, value_key="x", invert_for_all=True)),
        report_metric_spec("shoulder_external_rotation_velocity", "Shoulder External Rotation Angular Velocity", "Throwing Arm Kinematics", "deg/s", lambda rows: build_report_arm_velocity_data(rows, get_shoulder_ir_velocity, value_key="x", invert_left=True, peak_mode="external")),
        report_metric_spec("shoulder_internal_rotation_velocity", "Shoulder Internal Rotation Angular Velocity", "Throwing Arm Kinematics", "deg/s", lambda rows: build_report_arm_velocity_data(rows, get_shoulder_ir_velocity, value_key="x", invert_left=True)),
        report_metric_spec("shoulder_horizontal_abduction_velocity", "Shoulder Horizontal Abduction Angular Velocity", "Throwing Arm Kinematics", "deg/s", build_report_shoulder_horizontal_abduction_velocity_data),
        report_metric_spec("pelvis_forward_tilt", "Pelvis Forward Tilt", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_pelvis_component_data(rows, "x")),
        report_metric_spec("pelvis_lateral_tilt", "Pelvis Lateral Tilt", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_pelvis_component_data(rows, "y")),
        report_metric_spec("pelvis_rotation", "Pelvis Rotation", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_pelvis_component_data(rows, "z")),
        report_metric_spec("torso_forward_tilt", "Torso Forward Tilt", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_torso_component_data(rows, "x")),
        report_metric_spec("torso_lateral_tilt", "Torso Lateral Tilt", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_torso_component_data(rows, "y")),
        report_metric_spec("torso_rotation", "Torso Rotation", "Pelvis & Torso Kinematics", "deg", lambda rows: build_report_torso_component_data(rows, "z")),
        report_metric_spec("torso_pelvis_forward_tilt", "Torso-Pelvis Flexion/Extension", "Torso-Pelvis Kinematics", "deg", lambda rows: build_report_torso_pelvis_component_data(rows, "x")),
        report_metric_spec("torso_pelvis_lateral_tilt", "Torso-Pelvis Lateral Flexion", "Torso-Pelvis Kinematics", "deg", lambda rows: build_report_torso_pelvis_component_data(rows, "y")),
        report_metric_spec("torso_pelvis_rotation", "Torso-Pelvis Rotation", "Torso-Pelvis Kinematics", "deg", lambda rows: build_report_torso_pelvis_component_data(rows, "z")),
        report_metric_spec("torso_pelvis_flexion_velocity", "Torso-Pelvis Flexion Angular Velocity", "Torso-Pelvis Kinematics", "deg/s", lambda rows: build_report_torso_pelvis_velocity_data(rows, "x", peak_direction="flexion")),
        report_metric_spec("torso_pelvis_extension_velocity", "Torso-Pelvis Extension Angular Velocity", "Torso-Pelvis Kinematics", "deg/s", lambda rows: build_report_torso_pelvis_velocity_data(rows, "x", peak_direction="extension")),
        report_metric_spec("torso_pelvis_glove_side_lateral_flexion_velocity", "Torso-Pelvis Glove-Side Lateral Flexion Angular Velocity", "Torso-Pelvis Kinematics", "deg/s", lambda rows: build_report_torso_pelvis_velocity_data(rows, "y", peak_direction="glove_side")),
        report_metric_spec("torso_pelvis_arm_side_lateral_flexion_velocity", "Torso-Pelvis Arm-Side Lateral Flexion Angular Velocity", "Torso-Pelvis Kinematics", "deg/s", lambda rows: build_report_torso_pelvis_velocity_data(rows, "y", peak_direction="arm_side")),
        report_metric_spec("torso_pelvis_separation_velocity", "Torso-Pelvis Separation Angular Velocity", "Torso-Pelvis Kinematics", "deg/s", lambda rows: build_report_torso_pelvis_velocity_data(rows, "z")),
        report_metric_spec("back_hip_forward_tilt", "Back Hip Flexion/Extension", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "back", "x")),
        report_metric_spec("back_hip_lateral_tilt", "Back Hip Ab/Adduction", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "back", "y")),
        report_metric_spec("back_hip_rotation", "Back Hip Rotation", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "back", "z")),
        report_metric_spec("lead_hip_forward_tilt", "Lead Hip Flexion/Extension", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "lead", "x")),
        report_metric_spec("lead_hip_lateral_tilt", "Lead Hip Ab/Adduction", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "lead", "y")),
        report_metric_spec("lead_hip_rotation", "Lead Hip Rotation", "Hips Kinematics", "deg", lambda rows: build_report_hip_component_data(rows, "lead", "z")),
        report_metric_spec("cg_velocity_x", "COG Anterior/Posterior Velocity", "COG Velocity", "m/s", lambda rows: build_report_cg_velocity_data(rows, "x")),
        report_metric_spec("cg_velocity_y", "COG Medial/Lateral Velocity", "COG Velocity", "m/s", lambda rows: build_report_cg_velocity_data(rows, "y")),
        report_metric_spec("cg_velocity_z", "COG Vertical Velocity", "COG Velocity", "m/s", lambda rows: build_report_cg_velocity_data(rows, "z")),
        report_metric_spec("back_knee_flexion_extension", "Back Knee Flexion/Extension", "Lower Extremity Kinematics", "deg", lambda rows: build_report_lower_extremity_component_data(rows, "back", "KNEE", "x")),
        report_metric_spec("back_ankle_flexion_extension", "Back Ankle Flexion/Extension", "Lower Extremity Kinematics", "deg", lambda rows: build_report_lower_extremity_component_data(rows, "back", "ANKLE", "x")),
        report_metric_spec("front_knee_flexion_extension", "Front Knee Flexion/Extension", "Lower Extremity Kinematics", "deg", lambda rows: build_report_lower_extremity_component_data(rows, "front", "KNEE", "x")),
        report_metric_spec("front_ankle_flexion_extension", "Front Ankle Flexion/Extension", "Lower Extremity Kinematics", "deg", lambda rows: build_report_lower_extremity_component_data(rows, "front", "ANKLE", "x")),
        report_metric_spec("back_ankle_eversion_inversion", "Back Ankle Eversion/Inversion", "Lower Extremity Kinematics", "deg", lambda rows: build_report_lower_extremity_component_data(rows, "back", "ANKLE", "y")),
        report_metric_spec("back_knee_flexion_extension_velocity", "Back Knee Extension Angular Velocity", "Lower Extremity Kinematics", "deg/s", lambda rows: build_report_lower_extremity_velocity_data(rows, "back", "KNEE", "x")),
        report_metric_spec("back_ankle_flexion_extension_velocity", "Back Ankle Plantarflexion Angular Velocity", "Lower Extremity Kinematics", "deg/s", lambda rows: build_report_lower_extremity_velocity_data(rows, "back", "ANKLE", "x")),
        report_metric_spec("front_knee_flexion_extension_velocity", "Front Knee Extension Angular Velocity", "Lower Extremity Kinematics", "deg/s", lambda rows: build_report_lower_extremity_velocity_data(rows, "front", "KNEE", "x")),
        report_metric_spec("front_ankle_flexion_extension_velocity", "Front Ankle Plantarflexion Angular Velocity", "Lower Extremity Kinematics", "deg/s", lambda rows: build_report_lower_extremity_velocity_data(rows, "front", "ANKLE", "x")),
        report_metric_spec("back_ankle_eversion_inversion_velocity", "Back Ankle Ev/Inv Angular Velocity", "Lower Extremity Kinematics", "deg/s", lambda rows: build_report_lower_extremity_velocity_data(rows, "back", "ANKLE", "y")),
        report_metric_spec("back_hip_flexion_velocity", "Back Hip Flexion Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "back", "x", peak_direction="flexion")),
        report_metric_spec("back_hip_extension_velocity", "Back Hip Extension Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "back", "x", peak_direction="extension")),
        report_metric_spec("back_hip_lateral_tilt_velocity", "Back Hip Adduction Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "back", "y")),
        report_metric_spec("back_hip_external_rotation_velocity", "Back Hip External Rotation Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "back", "z", peak_direction="external")),
        report_metric_spec("back_hip_internal_rotation_velocity", "Back Hip Internal Rotation Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "back", "z", peak_direction="internal")),
        report_metric_spec("lead_hip_extension_velocity", "Lead Hip Extension Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "lead", "x", peak_direction="extension")),
        report_metric_spec("lead_hip_lateral_tilt_velocity", "Lead Hip Adduction Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "lead", "y")),
        report_metric_spec("lead_hip_internal_rotation_velocity", "Lead Hip Internal Rotation Angular Velocity", "Hips Kinematics", "deg/s", lambda rows: build_report_hip_velocity_data(rows, "lead", "z", peak_direction="internal")),
        report_metric_spec("pelvis_forward_tilt_velocity", "Pelvis Forward Tilt Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_pelvis_forward_tilt_velocity_data),
        report_metric_spec("pelvis_lateral_tilt_velocity", "Pelvis Lateral Tilt Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_pelvis_lateral_tilt_velocity_data),
        report_metric_spec("pelvis_rotation_velocity", "Pelvis Rotation Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_pelvis_rotation_velocity_data),
        report_metric_spec("torso_forward_tilt_velocity", "Torso Forward Tilt Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_torso_forward_tilt_velocity_data),
        report_metric_spec("torso_backward_tilt_velocity", "Torso Backward Tilt Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_torso_backward_tilt_velocity_data),
        report_metric_spec("torso_lateral_tilt_velocity", "Torso Lateral Tilt Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_torso_lateral_tilt_velocity_data),
        report_metric_spec("torso_rotation_velocity", "Torso Rotation Angular Velocity", "Pelvis & Torso Kinematics", "deg/s", build_report_torso_rotation_velocity_data),
    ]


def load_or_build_report_metric(report_rows, spec):
    take_ids = [row[0] for row in report_rows]
    cached_data = get_cached_report_metric_data(take_ids, spec["metric_key"])
    if cached_data is not None:
        return cached_data

    metric_data = spec["builder"](report_rows)
    cache_report_metric_data(
        report_rows,
        spec["metric_key"],
        spec["label"],
        spec["group"],
        spec["unit"],
        spec.get("source_category"),
        spec.get("source_segment"),
        spec.get("source_axis"),
        metric_data,
    )
    return metric_data


def build_report_arm_kinematics(report_rows, reference_report_rows):
    arm_kinematics = {}
    specs = get_report_metric_specs()
    metric_keys = [spec["metric_key"] for spec in specs]
    athlete_take_ids = [row[0] for row in report_rows]
    reference_take_ids = [row[0] for row in reference_report_rows]
    athlete_bundle = get_cached_report_metric_bundle(athlete_take_ids, metric_keys)
    reference_bundle = get_cached_report_metric_bundle(reference_take_ids, metric_keys)
    for spec in specs:
        metric_key = spec["metric_key"]
        arm_kinematics[spec["athlete_key"]] = (
            athlete_bundle.get(metric_key)
            or load_or_build_report_metric(report_rows, spec)
        )
        arm_kinematics[spec["reference_key"]] = (
            reference_bundle.get(metric_key)
            or load_or_build_report_metric(reference_report_rows, spec)
        )
    arm_kinematics["athlete_stride"] = build_report_stride_data(report_rows)
    arm_kinematics["reference_stride"] = build_report_stride_data(reference_report_rows)
    return arm_kinematics


def build_report_pdf(athlete_name, session_date, summary_rows, curves_by_segment, report_events, arm_kinematics=None, report_context=None, selected_sections=None):
    page_width, page_height = 792, 612
    margin = 36
    ops = []
    report_context = report_context or {}
    logo_image = None
    terra_logo = None

    try:
        import xml.etree.ElementTree as ET

        def svg_dimension(value, fallback):
            match = re.search(r"[-+]?\d*\.?\d+", str(value or ""))
            return float(match.group(0)) if match else fallback

        logo_root = ET.parse(LOGO_PATH).getroot()
        terra_logo = {
            "width": svg_dimension(logo_root.get("width"), 1050.0),
            "height": svg_dimension(logo_root.get("height"), 450.0),
            "paths": [],
        }
        for elem in logo_root.iter():
            if not elem.tag.endswith("path"):
                continue
            transform = elem.get("transform", "")
            translate_match = re.search(
                r"translate\(\s*([-+]?\d*\.?\d+)\s*[, ]\s*([-+]?\d*\.?\d+)\s*\)",
                transform,
            )
            translate_x = float(translate_match.group(1)) if translate_match else 0.0
            translate_y = float(translate_match.group(2)) if translate_match else 0.0
            terra_logo["paths"].append({
                "d": elem.get("d", ""),
                "fill": elem.get("fill", "#111111"),
                "translate_x": translate_x,
                "translate_y": translate_y,
            })
    except Exception:
        terra_logo = None

    try:
        from PIL import Image

        logo_path = ASSETS_DIR / "favicon.png"
        if not terra_logo and logo_path.exists():
            with Image.open(logo_path) as img:
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.alpha_composite(img)
                rgb_img = background.convert("RGB")
                logo_image = {
                    "width": rgb_img.width,
                    "height": rgb_img.height,
                    "data": zlib.compress(rgb_img.tobytes()),
                }
    except Exception:
        logo_image = None

    def rgb(hex_value):
        hex_value = hex_value.lstrip("#")
        return tuple(int(hex_value[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def esc(value):
        return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(x, y, value, size=10, bold=False, color="#111827"):
        r, g, b = rgb(color)
        font = "F2" if bold else "F1"
        ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        ops.append(f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({esc(value)}) Tj ET")

    def text_centered(x, y, width, value, size=10, bold=False, color="#111827"):
        estimated_width = len(str(value)) * size * (0.56 if bold else 0.50)
        text(x + (width - estimated_width) / 2, y, value, size=size, bold=bold, color=color)

    def text_rotated(x, y, value, size=10, bold=False, color="#111827", angle=90):
        r, g, b = rgb(color)
        font = "F2" if bold else "F1"
        radians = np.deg2rad(angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        ops.append(
            f"q {cos_a:.4f} {sin_a:.4f} {-sin_a:.4f} {cos_a:.4f} {x:.2f} {y:.2f} cm "
            f"BT /{font} {size} Tf 0 0 Td ({esc(value)}) Tj ET Q"
        )

    def image(name, x, y, width, height):
        ops.append(f"q {width:.2f} 0 0 {height:.2f} {x:.2f} {y:.2f} cm /{name} Do Q")

    def svg_logo(logo, x, y, max_width, max_height):
        if not logo or not logo.get("paths"):
            return False
        scale = min(max_width / logo["width"], max_height / logo["height"])
        draw_width = logo["width"] * scale
        draw_height = logo["height"] * scale
        origin_x = x + max_width - draw_width
        origin_y = y + max_height - draw_height
        token_re = re.compile(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")

        def is_command(token):
            return len(token) == 1 and token.isalpha()

        def map_point(px, py, translate_x, translate_y):
            svg_x = px + translate_x
            svg_y = py + translate_y
            return origin_x + svg_x * scale, origin_y + (logo["height"] - svg_y) * scale

        for path in logo["paths"]:
            tokens = token_re.findall(path["d"])
            if not tokens:
                continue
            r, g, b = rgb(path["fill"])
            parts = [f"{r:.4f} {g:.4f} {b:.4f} rg"]
            command = None
            index = 0
            while index < len(tokens):
                if is_command(tokens[index]):
                    command = tokens[index]
                    index += 1
                if command == "M":
                    first_point = True
                    while index + 1 < len(tokens) and not is_command(tokens[index]):
                        px, py = float(tokens[index]), float(tokens[index + 1])
                        mapped_x, mapped_y = map_point(px, py, path["translate_x"], path["translate_y"])
                        parts.append(f"{mapped_x:.2f} {mapped_y:.2f} {'m' if first_point else 'l'}")
                        first_point = False
                        index += 2
                elif command == "C":
                    while index + 5 < len(tokens) and not is_command(tokens[index]):
                        x1, y1 = map_point(float(tokens[index]), float(tokens[index + 1]), path["translate_x"], path["translate_y"])
                        x2, y2 = map_point(float(tokens[index + 2]), float(tokens[index + 3]), path["translate_x"], path["translate_y"])
                        x3, y3 = map_point(float(tokens[index + 4]), float(tokens[index + 5]), path["translate_x"], path["translate_y"])
                        parts.append(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} c")
                        index += 6
                elif command in ("Z", "z"):
                    parts.append("h")
                    command = None
                else:
                    break
            parts.append("f")
            ops.append("\n".join(parts))
        return True

    def line(x1, y1, x2, y2, color="#111827", width=0.75):
        r, g, b = rgb(color)
        ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        ops.append(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def dashed_line(x1, y1, x2, y2, color="#111827", width=0.75, pattern="3 3"):
        r, g, b = rgb(color)
        ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        ops.append(f"[{pattern}] 0 d {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S [] 0 d")

    def rect(x, y, width, height, stroke="#D1D5DB", fill=None, line_width=0.75):
        if fill:
            r, g, b = rgb(fill)
            ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
            ops.append(f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re f")
        r, g, b = rgb(stroke)
        ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        ops.append(f"{line_width:.2f} w {x:.2f} {y:.2f} {width:.2f} {height:.2f} re S")

    def circle(cx, cy, radius, stroke="#D1D5DB", fill=None, line_width=0.75):
        kappa = 0.5522847498
        c = radius * kappa
        parts = []
        if fill:
            r, g, b = rgb(fill)
            parts.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        r, g, b = rgb(stroke)
        parts.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        parts.append(f"{line_width:.2f} w")
        parts.append(f"{cx + radius:.2f} {cy:.2f} m")
        parts.append(f"{cx + radius:.2f} {cy + c:.2f} {cx + c:.2f} {cy + radius:.2f} {cx:.2f} {cy + radius:.2f} c")
        parts.append(f"{cx - c:.2f} {cy + radius:.2f} {cx - radius:.2f} {cy + c:.2f} {cx - radius:.2f} {cy:.2f} c")
        parts.append(f"{cx - radius:.2f} {cy - c:.2f} {cx - c:.2f} {cy - radius:.2f} {cx:.2f} {cy - radius:.2f} c")
        parts.append(f"{cx + c:.2f} {cy - radius:.2f} {cx + radius:.2f} {cy - c:.2f} {cx + radius:.2f} {cy:.2f} c")
        parts.append("B" if fill else "S")
        ops.append("\n".join(parts))

    def poly(points, stroke=None, fill=None, line_width=0.75):
        if len(points) < 3:
            return
        parts = []
        if fill:
            r, g, b = rgb(fill)
            parts.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        if stroke:
            r, g, b = rgb(stroke)
            parts.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
            parts.append(f"{line_width:.2f} w")
        parts.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
        for x, y in points[1:]:
            parts.append(f"{x:.2f} {y:.2f} l")
        parts.append("h")
        parts.append("B" if stroke and fill else "f" if fill else "S")
        ops.append("\n".join(parts))

    def path(points, color="#111827", width=1.5):
        if len(points) < 2:
            return
        r, g, b = rgb(color)
        parts = [f"{r:.4f} {g:.4f} {b:.4f} RG", f"{width:.2f} w"]
        parts.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
        for x, y in points[1:]:
            parts.append(f"{x:.2f} {y:.2f} l")
        parts.append("S")
        ops.append("\n".join(parts))

    def arrow(x1, y1, x2, y2, color="#111827", width=1.5, head_size=6.0):
        line(x1, y1, x2, y2, color=color, width=width)
        angle = np.arctan2(y2 - y1, x2 - x1)
        for delta in (np.pi * 0.78, -np.pi * 0.78):
            hx = x2 + head_size * np.cos(angle + delta)
            hy = y2 + head_size * np.sin(angle + delta)
            line(x2, y2, hx, hy, color=color, width=width)

    def draw_detail(label, value, x, y, width):
        text(x, y + 16, label.upper(), 7.5, bold=True, color="#64748B")
        text(x, y, value or "N/A", 13, bold=True, color="#111827")
        line(x, y - 10, x + width, y - 10, color="#E5E7EB", width=0.6)

    pdf_palette = {
        "arm_primary": ("#6D5DF6", "#E9E7FF"),
        "arm_secondary": ("#A855F7", "#F3E8FF"),
        "arm_tertiary": ("#0E9FAD", "#CCFBF1"),
        "arm_rotation": ("#D9468C", "#FCE7F3"),
        "pelvis_primary": ("#2563EB", "#DBEAFE"),
        "pelvis_secondary": ("#3B82F6", "#DBEAFE"),
        "pelvis_rotation": ("#0F766E", "#CCFBF1"),
        "torso_primary": ("#EA580C", "#FED7AA"),
        "torso_secondary": ("#F97316", "#FFEDD5"),
        "torso_rotation": ("#B45309", "#FEF3C7"),
        "torso_pelvis_primary": ("#7C3AED", "#DDD6FE"),
        "torso_pelvis_secondary": ("#C2410C", "#FFEDD5"),
        "torso_pelvis_rotation": ("#0891B2", "#CFFAFE"),
        "hip_back": ("#15803D", "#DCFCE7"),
        "hip_lead": ("#0D9488", "#CCFBF1"),
        "hip_rotation": ("#2563EB", "#DBEAFE"),
        "lower_back": ("#64748B", "#E2E8F0"),
        "lower_front": ("#475569", "#E2E8F0"),
        "lower_ankle": ("#0F766E", "#CCFBF1"),
        "cog": ("#334155", "#E2E8F0"),
        "cog_secondary": ("#64748B", "#E2E8F0"),
        "cog_vertical": ("#0891B2", "#CFFAFE"),
        "stride_primary": ("#334155", "#E2E8F0"),
        "stride_secondary": ("#0F766E", "#CCFBF1"),
        "stride_tertiary": ("#0891B2", "#CFFAFE"),
    }

    def palette_color(name):
        return pdf_palette[name][0]

    def palette_fill(name):
        return pdf_palette[name][1]

    logo_max_width = 190
    logo_max_height = 82
    logo_x = page_width - margin - logo_max_width
    logo_y = page_height - margin - logo_max_height - 4
    if not svg_logo(terra_logo, logo_x, logo_y, logo_max_width, logo_max_height) and logo_image:
        logo_aspect = logo_image["width"] / logo_image["height"]
        logo_width = min(logo_max_width, logo_max_height * logo_aspect)
        logo_height = logo_width / logo_aspect
        image("ImLogo", page_width - margin - logo_width, logo_y + logo_max_height - logo_height, logo_width, logo_height)
    text(margin, page_height - 78, athlete_name, 34, bold=True, color="#111827")
    text(margin, page_height - 114, "Pitching Motion Capture Report", 22, bold=True, color="#111827")
    line(margin, page_height - 140, page_width - margin, page_height - 140, color="#111827", width=2.0)

    left_x = margin
    right_x = page_width / 2 + 18
    detail_width = page_width / 2 - margin - 42
    text(left_x, page_height - 194, "Report Selection", 17, bold=True, color="#111827")
    draw_detail("Session Date", session_date, left_x, page_height - 238, detail_width)
    draw_detail("Velocity Range", report_context.get("velocity_range_label", "All"), left_x, page_height - 302, detail_width)

    text(right_x, page_height - 194, "Reference Group", 17, bold=True, color="#111827")
    draw_detail("Athletes", report_context.get("reference_athletes_label", "All"), right_x, page_height - 238, detail_width)
    draw_detail("Handedness", report_context.get("reference_handedness_label", "All"), right_x, page_height - 302, detail_width)
    draw_detail("Velocity Range", report_context.get("reference_velocity_range_label", "All"), right_x, page_height - 366, detail_width)
    draw_detail("Arm Slot", report_context.get("reference_arm_slot_label", "All"), right_x, page_height - 430, detail_width)
    cover_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Kinematic Sequence", 28, bold=True, color="#111827")

    chart_left = margin
    chart_bottom = 230
    chart_width = page_width - (margin * 2)
    chart_height = 285
    plot_left = chart_left + 52
    plot_right = chart_left + chart_width - 22
    plot_bottom = chart_bottom + 39
    plot_top = chart_bottom + chart_height - 28

    fp_event_frames = report_events.get("fp_event_frames", [])
    mer_event_frames = report_events.get("mer_event_frames", [])
    take_count = report_events.get("take_count", 0)
    window_start_frame = int(np.median(fp_event_frames)) - ms_to_rel_frame(100) if fp_event_frames else -ms_to_rel_frame(100)
    x_min = rel_frame_to_ms(window_start_frame)
    x_max = rel_frame_to_ms(ms_to_rel_frame(150))

    color_map = {
        "Pelvis Rotation": "#1F77B4",
        "Torso Rotation": "#FF7F0E",
        "Elbow Extension": "#2CA02C",
        "Shoulder Internal Rotation": "#D62728",
    }
    band_color_map = {
        "Pelvis Rotation": "#DBEAFE",
        "Torso Rotation": "#FED7AA",
        "Elbow Extension": "#DCFCE7",
        "Shoulder Internal Rotation": "#FEE2E2",
    }

    grouped_by_segment = {}
    for segment, curves in curves_by_segment.items():
        if curves:
            x_vals, y_vals, q1_vals, q3_vals = aggregate_curves(curves, "Mean")
            grouped_by_segment[segment] = {
                "frame": x_vals,
                "value": y_vals,
                "q1": q1_vals,
                "q3": q3_vals,
            }
    visible_values = [
        value
        for grouped in grouped_by_segment.values()
        for time_ms, value in zip(grouped["frame"], grouped["value"])
        if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
    ]
    y_min = min(visible_values) if visible_values else 0
    y_max = max(visible_values) if visible_values else 1
    y_span = max(y_max - y_min, 1)
    y_min -= 0.10 * y_span
    y_max += 0.35 * y_span

    def sx(value):
        return plot_left + ((value - x_min) / (x_max - x_min)) * (plot_right - plot_left)

    def sy(value):
        return plot_bottom + ((value - y_min) / (y_max - y_min)) * (plot_top - plot_bottom)

    median_fp_ms = rel_frame_to_ms(int(np.median(fp_event_frames))) if fp_event_frames else None

    def peak_index_for_pdf(segment, grouped):
        valid_idxs = [
            i
            for i, (time_ms, value) in enumerate(zip(grouped["frame"], grouped["value"]))
            if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
        ]
        if segment in {"Pelvis Rotation", "Torso Rotation"} and median_fp_ms is not None:
            valid_idxs = [
                i
                for i in valid_idxs
                if median_fp_ms <= grouped["frame"][i] <= 0
            ]
        if not valid_idxs:
            return None
        return max(valid_idxs, key=lambda i: grouped["value"][i])

    tick_start = int(np.ceil(x_min / 100) * 100)
    tick_end = int(np.floor(x_max / 100) * 100)
    for tick in range(tick_start, tick_end + 1, 100):
        x = sx(tick)
        line(x, plot_bottom, x, plot_top, color="#F3F4F6", width=0.35)
        text(x - 10, plot_bottom - 11, tick, 7, color="#4B5563")

    y_tick_step = 1000
    y_tick_start = np.ceil(y_min / y_tick_step) * y_tick_step
    y_tick_end = np.floor(y_max / y_tick_step) * y_tick_step
    y_ticks = np.arange(y_tick_start, y_tick_end + (y_tick_step * 0.5), y_tick_step)
    for tick in y_ticks:
        y = sy(tick)
        if abs(tick) > 1e-9 and plot_bottom < y < plot_top:
            line(plot_left, y, plot_right, y, color="#F3F4F6", width=0.35)
        text(plot_left - 30, y - 2, f"{tick:.0f}°/s", 7, color="#4B5563")

    def draw_event(label, event_frames, color, band_fill):
        if not event_frames:
            return
        q1 = rel_frame_to_ms(int(np.percentile(event_frames, 25)))
        q3 = rel_frame_to_ms(int(np.percentile(event_frames, 75)))
        median_value = rel_frame_to_ms(int(np.median(event_frames)))
        if q1 != q3:
            left = max(x_min, q1)
            right = min(x_max, q3)
            if left < right:
                rect(sx(left), plot_bottom, sx(right) - sx(left), plot_top - plot_bottom, stroke=band_fill, fill=band_fill, line_width=0.1)
        if x_min <= median_value <= x_max:
            x = sx(median_value)
            line(x, plot_bottom, x, plot_top, color=color, width=1.2)
            text(x + 3, plot_top - 9, label, 7, bold=True, color=color)

    draw_event("FP", fp_event_frames, "#16A34A", "#DCFCE7")
    draw_event("MER", mer_event_frames, "#DC2626", "#FEE2E2")
    draw_event("BR", [0] * max(take_count, 1), "#2563EB", "#DBEAFE")

    peak_arrows = []
    for segment in ["Pelvis Rotation", "Torso Rotation", "Elbow Extension", "Shoulder Internal Rotation"]:
        grouped = grouped_by_segment.get(segment)
        if not grouped:
            continue
        upper = []
        lower = []
        line_points = []
        for time_ms, mean_value, q1, q3 in zip(grouped["frame"], grouped["value"], grouped["q1"], grouped["q3"]):
            if time_ms < x_min or time_ms > x_max or mean_value is None:
                continue
            line_points.append((sx(time_ms), sy(mean_value)))
            if q1 is not None and q3 is not None and np.isfinite(q1) and np.isfinite(q3):
                upper.append((sx(time_ms), sy(q3)))
                lower.append((sx(time_ms), sy(q1)))
        if upper and lower:
            poly(upper + lower[::-1], fill=band_color_map[segment])
        path(line_points, color=color_map[segment], width=1.8)
        peak_idx = peak_index_for_pdf(segment, grouped)
        if peak_idx is not None:
            peak_x = grouped["frame"][peak_idx]
            peak_y = grouped["value"][peak_idx]
            px = sx(peak_x)
            py = sy(peak_y)
            arrow_start_y = min(plot_top - 4, py + 22)
            if arrow_start_y - py < 12:
                arrow_start_y = min(plot_top - 2, py + 14)
            peak_arrows.append((px, arrow_start_y, px, py + 2, color_map[segment]))

    if y_min < 0 < y_max:
        zero_y = sy(0)
        dashed_line(plot_left, zero_y, plot_right, zero_y, color="#111111", width=0.55, pattern="2 2")

    for x1, y1, x2, y2, color in peak_arrows:
        arrow(x1, y1, x2, y2, color=color, width=1.4, head_size=5.5)

    rect(plot_left, plot_bottom, plot_right - plot_left, plot_top - plot_bottom, stroke="#111111", fill=None, line_width=1.0)

    legend_label_map = {
        "Pelvis Rotation": "Pelvis Rotation",
        "Torso Rotation": "Torso Rotation",
        "Elbow Extension": "Elbow Extension",
        "Shoulder Internal Rotation": "Shoulder Rotation",
    }
    legend_width = 122
    legend_height = 54
    legend_x = plot_right - legend_width - 8
    legend_y = plot_top - legend_height - 7
    rect(legend_x, legend_y, legend_width, legend_height, stroke="#D1D5DB", fill="#FFFFFF", line_width=0.75)
    for index, segment in enumerate(["Pelvis Rotation", "Torso Rotation", "Elbow Extension", "Shoulder Internal Rotation"]):
        y = legend_y + legend_height - 12 - (index * 11)
        line(legend_x + 8, y + 4, legend_x + 25, y + 4, color=color_map[segment], width=2)
        text(legend_x + 32, y, legend_label_map[segment], 7)
    rect(plot_left + 6, plot_bottom + 6, 135, 12, stroke="#FFFFFF", fill="#FFFFFF", line_width=0.1)
    text(plot_left + 10, plot_bottom + 9, "Bands show interquartile range across throws", 6.2, color="#64748B")
    text((plot_left + plot_right) / 2 - 65, plot_bottom - 27, "Time Relative to Ball Release (ms)", 8, bold=True, color="#111111")
    text_rotated(chart_left + 6, (plot_bottom + plot_top) / 2 - 42, "Angular Velocity (deg/s)", 8, bold=True, color="#111111", angle=90)

    summary_by_segment = {row["Segment"]: row for row in summary_rows}

    table_x = margin
    table_y = 90
    table_height = 112
    table_width = page_width - (margin * 2)
    gap = 8
    segment_order = ["Pelvis Rotation", "Torso Rotation", "Elbow Extension", "Shoulder Internal Rotation"]
    segment_width = (table_width - (gap * (len(segment_order) - 1))) / len(segment_order)
    reference_labels = {
        "Pelvis Rotation": "Timing from FP",
        "Torso Rotation": "Timing from Pelvis",
        "Elbow Extension": "Timing from Torso",
        "Shoulder Internal Rotation": "Timing from Elbow",
    }
    segment_title_labels = {
        "Pelvis Rotation": "Pelvis Rotation",
        "Torso Rotation": "Torso Rotation",
        "Elbow Extension": "Elbow Extension",
        "Shoulder Internal Rotation": "Shoulder Rotation",
    }

    def segment_peak_std(segment):
        peak_values = []
        curves = curves_by_segment.get(segment, {})
        for curve in curves.values():
            valid_idxs = [
                i
                for i, (time_ms, value) in enumerate(zip(curve.get("frame", []), curve.get("value", [])))
                if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
            ]
            if segment in {"Pelvis Rotation", "Torso Rotation"} and median_fp_ms is not None:
                valid_idxs = [
                    i
                    for i in valid_idxs
                    if median_fp_ms <= curve["frame"][i] <= 0
                ]
            if valid_idxs:
                peak_values.append(max(curve["value"][i] for i in valid_idxs))
        if not peak_values:
            return None
        return float(np.std(peak_values, ddof=1 if len(peak_values) > 1 else 0))

    cursor_x = table_x
    for segment in segment_order:
        row = summary_by_segment.get(segment, {})
        peak = row.get("Peak (deg/s)")
        reference_time = row.get("Peak Time from Reference (ms)")
        if reference_time is None or (isinstance(reference_time, float) and pd.isna(reference_time)):
            reference_time = row.get("Peak Time (ms rel BR)")
        peak_value = "" if peak is None or pd.isna(peak) else f"{peak:.0f}°/s"
        if reference_time is None or pd.isna(reference_time):
            timing_value = ""
        else:
            timing_ms = int(round(reference_time))
            timing_value = f"{timing_ms} ms" if timing_ms < 0 else f"{abs(timing_ms)} ms"
        std_dev = segment_peak_std(segment)
        std_value = "" if std_dev is None or pd.isna(std_dev) else f"± {std_dev:.0f}°/s"
        segment_color = color_map[segment]
        segment_fill = band_color_map[segment]

        rect(cursor_x, table_y, segment_width, table_height, stroke="#E5E7EB", fill="#FFFFFF", line_width=0.75)
        rect(cursor_x, table_y + table_height - 7, segment_width, 7, stroke=segment_color, fill=segment_color, line_width=0.1)
        rect(cursor_x, table_y + table_height - 34, segment_width, 27, stroke=segment_fill, fill=segment_fill, line_width=0.1)
        text_centered(cursor_x, table_y + table_height - 25, segment_width, segment_title_labels[segment], 12, bold=True, color=segment_color)

        peak_label_y = table_y + 60
        peak_value_y = table_y + 40
        detail_label_y = table_y + 21
        detail_value_y = table_y + 7
        detail_width = (segment_width - 12) / 2
        timing_x = cursor_x + 6
        std_x = timing_x + detail_width + 12

        line(cursor_x + 12, table_y + 35, cursor_x + segment_width - 12, table_y + 35, color="#E5E7EB", width=0.6)
        text_centered(cursor_x, peak_label_y, segment_width, "Peak", 8.5, bold=True, color="#64748B")
        text_centered(cursor_x, peak_value_y, segment_width, peak_value, 16, bold=True, color="#111827")
        text_centered(timing_x, detail_label_y, detail_width, reference_labels[segment], 7.8, bold=True, color="#64748B")
        text_centered(std_x, detail_label_y, detail_width, "Std Dev", 7.8, bold=True, color="#64748B")
        text_centered(timing_x, detail_value_y, detail_width, timing_value, 11.5, bold=True, color="#111827")
        text_centered(std_x, detail_value_y, detail_width, std_value, 11.5, bold=True, color="#111827")

        cursor_x += segment_width + gap

    first_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Throwing Arm Kinematics", 28, bold=True, color="#111827")
    arm_kinematics = arm_kinematics or {}
    arm_metric_labels = [
        ("Athlete", "mean"),
        ("Ref Group", "ref_mean"),
        ("SD ±", "std"),
        ("Ref SD ±", "ref_std"),
    ]

    def draw_arm_metric_row(title, athlete_data, reference_data, row_y, color):
        rect(margin, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(margin + 10, row_y + 14, title, 10, bold=True, color=color)
        table_x = margin + 132
        table_width = page_width - margin - table_x
        event_width = table_width / 4
        athlete_metrics = athlete_data.get("metrics", {})
        reference_metrics = reference_data.get("metrics", {})
        for card_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
            event_x = table_x + (card_index * event_width)
            athlete_event = athlete_metrics.get(event_label, {})
            reference_event = reference_metrics.get(event_label, {})
            values = {
                "mean": athlete_event.get("mean"),
                "ref_mean": reference_event.get("mean"),
                "std": athlete_event.get("std"),
                "ref_std": reference_event.get("std"),
            }
            cell_width = event_width / 4
            for metric_index, (_, value_key) in enumerate(arm_metric_labels):
                cell_x = event_x + (metric_index * cell_width)
                metric_value = values[value_key]
                value_text = "" if metric_value is None or pd.isna(metric_value) else f"{metric_value:.1f}°"
                text_centered(cell_x, row_y + 14, cell_width, value_text, 7.7, bold=True, color="#111827")
        line(margin + 10, row_y, page_width - margin - 10, row_y, color="#E5E7EB", width=0.55)

    def draw_arm_metric_plot(title, athlete_data, reference_data, chart_x, chart_y, color, fill, unit="deg", chart_width=346, chart_height=140, title_size=12, peak_marker=None):
        plot_left = chart_x + 43
        plot_right = chart_x + chart_width - 12
        plot_bottom = chart_y + 36
        plot_top = chart_y + chart_height - 25
        text(chart_x, chart_y + chart_height + 8, title, title_size, bold=True, color="#111827")
        athlete_curves = athlete_data.get("curves", {})
        reference_curves = reference_data.get("curves", {})
        athlete_x, athlete_y, _, _ = aggregate_curves(athlete_curves, "Mean") if athlete_curves else ([], [], [], [])
        ref_x, _, ref_q1, ref_q3 = aggregate_curves(reference_curves, "Mean") if reference_curves else ([], [], [], [])
        fp_events = athlete_data.get("events", {}).get("fp_event_frames", []) or report_events.get("fp_event_frames", [])
        mer_events = athlete_data.get("events", {}).get("mer_event_frames", []) or report_events.get("mer_event_frames", [])
        x_min = rel_frame_to_ms(int(np.median(fp_events)) - ms_to_rel_frame(100)) if fp_events else -200
        x_max = rel_frame_to_ms(ms_to_rel_frame(150))
        visible_values = [
            value
            for frames, values in [(athlete_x, athlete_y), (ref_x, ref_q1), (ref_x, ref_q3)]
            for time_ms, value in zip(frames, values)
            if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
        ]
        y_min = min(visible_values) if visible_values else 0
        y_max = max(visible_values) if visible_values else 1
        y_span = max(y_max - y_min, 1)
        y_min -= 0.12 * y_span
        y_max += 0.18 * y_span

        def px(value):
            return plot_left + ((value - x_min) / (x_max - x_min)) * (plot_right - plot_left)

        def py(value):
            return plot_bottom + ((value - y_min) / (y_max - y_min)) * (plot_top - plot_bottom)

        for tick in range(int(np.ceil(x_min / 100) * 100), int(np.floor(x_max / 100) * 100) + 1, 100):
            x = px(tick)
            line(x, plot_bottom, x, plot_top, color="#F3F4F6", width=0.35)
            text(x - 9, plot_bottom - 11, tick, 6.2, color="#4B5563")
        for tick in np.linspace(y_min, y_max, 5):
            y = py(tick)
            line(plot_left, y, plot_right, y, color="#F3F4F6", width=0.35)
            tick_suffix = "°/s" if unit == "deg/s" else "°"
            text(plot_left - 30, y - 2, f"{tick:.0f}{tick_suffix}", 6.2, color="#4B5563")
        ref_upper = [(px(t), py(q3)) for t, q3 in zip(ref_x, ref_q3) if q3 is not None and x_min <= t <= x_max]
        ref_lower = [(px(t), py(q1)) for t, q1 in zip(ref_x, ref_q1) if q1 is not None and x_min <= t <= x_max]
        if ref_upper and ref_lower:
            poly(ref_upper + ref_lower[::-1], fill=fill)
        athlete_line = [(px(t), py(v)) for t, v in zip(athlete_x, athlete_y) if v is not None and x_min <= t <= x_max]
        path(athlete_line, color=color, width=1.8)
        marker_target = None
        if peak_marker and athlete_x and athlete_y:
            mer_ms = rel_frame_to_ms(int(np.median(mer_events))) if mer_events else None
            marker_candidates = [
                (time_ms, value)
                for time_ms, value in zip(athlete_x, athlete_y)
                if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
            ]
            if peak_marker == "min_before_mer" and mer_ms is not None:
                marker_candidates = [(time_ms, value) for time_ms, value in marker_candidates if time_ms <= mer_ms]
                if marker_candidates:
                    marker_target = min(marker_candidates, key=lambda item: item[1])
            elif peak_marker == "max":
                if marker_candidates:
                    marker_target = max(marker_candidates, key=lambda item: item[1])
        for label, frames, event_color in [("FP", fp_events, "#16A34A"), ("MER", mer_events, "#DC2626"), ("BR", [0], "#2563EB")]:
            if frames:
                event_ms = rel_frame_to_ms(int(np.median(frames)))
                if x_min <= event_ms <= x_max:
                    event_x = px(event_ms)
                    line(event_x, plot_bottom, event_x, plot_top, color=event_color, width=1.0)
                    text(event_x + 2, plot_top - 8, label, 6.2, bold=True, color=event_color)
        rect(plot_left, plot_bottom, plot_right - plot_left, plot_top - plot_bottom, stroke="#111111", fill=None, line_width=0.9)
        if marker_target:
            marker_x = px(marker_target[0])
            marker_y = py(marker_target[1])
            arrow_start_y = min(plot_top - 5, marker_y + 24)
            if arrow_start_y <= marker_y + 6:
                arrow_start_y = max(plot_bottom + 5, marker_y - 24)
            arrow(marker_x, arrow_start_y, marker_x, marker_y, color=color, width=1.25, head_size=4.5)
        text((plot_left + plot_right) / 2 - 46, plot_bottom - 23, "Time Relative to Ball Release (ms)", 6, bold=True, color="#111111")
        y_axis_label = "Angular Velocity (deg/s)" if unit == "deg/s" else "Velocity (m/s)" if unit == "m/s" else "Position (deg)"
        text_rotated(chart_x + 7, (plot_bottom + plot_top) / 2 - 24, y_axis_label, 6, bold=True, color="#111111", angle=90)

    athlete_elbow = arm_kinematics.get("athlete", {})
    reference_elbow = arm_kinematics.get("reference", {})
    athlete_shoulder = arm_kinematics.get("athlete_shoulder_abduction", {})
    reference_shoulder = arm_kinematics.get("reference_shoulder_abduction", {})
    athlete_horizontal = arm_kinematics.get("athlete_shoulder_horizontal_abduction", {})
    reference_horizontal = arm_kinematics.get("reference_shoulder_horizontal_abduction", {})
    athlete_rotation = arm_kinematics.get("athlete_shoulder_rotation", {})
    reference_rotation = arm_kinematics.get("reference_shoulder_rotation", {})
    shared_table_x = margin + 132
    shared_table_width = page_width - margin - shared_table_x
    shared_event_width = shared_table_width / 4
    header_y = 502
    header_height = 36
    rect(margin, header_y, page_width - (margin * 2), header_height, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = shared_table_x + (event_index * shared_event_width)
        if event_index:
            line(event_x, 362, event_x, header_y + header_height, color="#E5E7EB", width=0.6)
        text_centered(event_x, header_y + 24, shared_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = shared_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), header_y + 8, sub_width, metric_label, 6.2, bold=True, color="#64748B")
    draw_arm_metric_row("Elbow Flexion", athlete_elbow, reference_elbow, 467, palette_color("arm_primary"))
    draw_arm_metric_row("Shoulder Abduction", athlete_shoulder, reference_shoulder, 432, palette_color("arm_secondary"))
    draw_arm_metric_row("Shoulder Horizontal Abd.", athlete_horizontal, reference_horizontal, 397, palette_color("arm_tertiary"))
    draw_arm_metric_row("Shoulder Rotation", athlete_rotation, reference_rotation, 362, palette_color("arm_rotation"))
    draw_arm_metric_plot("Elbow Flexion", athlete_elbow, reference_elbow, margin, 180, palette_color("arm_primary"), palette_fill("arm_primary"))
    draw_arm_metric_plot("Shoulder Abduction", athlete_shoulder, reference_shoulder, 410, 180, palette_color("arm_secondary"), palette_fill("arm_secondary"))
    draw_arm_metric_plot("Shoulder Horizontal Abd.", athlete_horizontal, reference_horizontal, margin, 26, palette_color("arm_tertiary"), palette_fill("arm_tertiary"))
    draw_arm_metric_plot("Shoulder Rotation", athlete_rotation, reference_rotation, 410, 26, palette_color("arm_rotation"), palette_fill("arm_rotation"))

    second_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Throwing Arm Kinematics", 28, bold=True, color="#111827")
    velocity_rows = [
        ("Elbow Extension Angular Velocity", "athlete_elbow_extension_velocity", "reference_elbow_extension_velocity", palette_color("arm_primary")),
        ("Shoulder External Rotation Angular Velocity", "athlete_shoulder_external_rotation_velocity", "reference_shoulder_external_rotation_velocity", palette_color("arm_secondary")),
        ("Shoulder Internal Rotation Angular Velocity", "athlete_shoulder_internal_rotation_velocity", "reference_shoulder_internal_rotation_velocity", palette_color("arm_rotation")),
        ("Shoulder Horizontal Abduction Angular Velocity", "athlete_shoulder_horizontal_abduction_velocity", "reference_shoulder_horizontal_abduction_velocity", palette_color("arm_tertiary")),
    ]
    velocity_table_x = margin
    velocity_table_y = 360
    velocity_table_width = page_width - (margin * 2)
    velocity_name_width = 220
    velocity_value_width = (velocity_table_width - velocity_name_width) / 4
    rect(velocity_table_x, velocity_table_y + 140, velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(velocity_table_x + velocity_name_width + (index * velocity_value_width), velocity_table_y + 151, velocity_value_width, header, 8.8, bold=True, color="#64748B")
    for row_index, (label, athlete_key, reference_key, color) in enumerate(velocity_rows):
        row_y = velocity_table_y + 105 - (row_index * 35)
        athlete_data = arm_kinematics.get(athlete_key, {})
        reference_data = arm_kinematics.get(reference_key, {})
        athlete_max = athlete_data.get("metrics", {}).get("Max", {})
        reference_max = reference_data.get("metrics", {}).get("Max", {})
        values = [athlete_max.get("mean"), athlete_max.get("std"), reference_max.get("mean"), reference_max.get("std")]
        rect(velocity_table_x, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(velocity_table_x + 10, row_y + 14, label, 9.3, bold=True, color=color)
        for value_index, value in enumerate(values):
            value_text = "" if value is None or pd.isna(value) else f"{value:.0f}°/s"
            text_centered(velocity_table_x + velocity_name_width + (value_index * velocity_value_width), row_y + 14, velocity_value_width, value_text, 9, bold=True, color="#111827")
        line(velocity_table_x + 10, row_y, velocity_table_x + velocity_table_width - 10, row_y, color="#E5E7EB", width=0.55)
    velocity_plot_specs = [
        ("Elbow Ext. Angular Velocity", "athlete_elbow_extension_velocity", "reference_elbow_extension_velocity", margin, 188, palette_color("arm_primary"), palette_fill("arm_primary"), None),
        ("Shoulder ER Angular Velocity", "athlete_shoulder_external_rotation_velocity", "reference_shoulder_external_rotation_velocity", 410, 188, palette_color("arm_secondary"), palette_fill("arm_secondary"), "min_before_mer"),
        ("Shoulder IR Angular Velocity", "athlete_shoulder_internal_rotation_velocity", "reference_shoulder_internal_rotation_velocity", margin, 34, palette_color("arm_rotation"), palette_fill("arm_rotation"), "max"),
        ("Shoulder Horiz. Abd. Angular Velocity", "athlete_shoulder_horizontal_abduction_velocity", "reference_shoulder_horizontal_abduction_velocity", 410, 34, palette_color("arm_tertiary"), palette_fill("arm_tertiary"), None),
    ]
    for title, athlete_key, reference_key, chart_x, chart_y, color, fill, peak_marker in velocity_plot_specs:
        draw_arm_metric_plot(title, arm_kinematics.get(athlete_key, {}), arm_kinematics.get(reference_key, {}), chart_x, chart_y, color, fill, unit="deg/s", peak_marker=peak_marker)
    third_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Pelvis & Torso Kinematics", 28, bold=True, color="#111827")
    pelvis_rows = [
        ("Pelvis Forward Tilt", "athlete_pelvis_forward_tilt", "reference_pelvis_forward_tilt", palette_color("pelvis_primary")),
        ("Pelvis Lateral Tilt", "athlete_pelvis_lateral_tilt", "reference_pelvis_lateral_tilt", palette_color("pelvis_secondary")),
        ("Pelvis Rotation", "athlete_pelvis_rotation", "reference_pelvis_rotation", palette_color("pelvis_rotation")),
    ]
    pelvis_table_x = margin
    pelvis_table_y = 408
    pelvis_table_width = page_width - (margin * 2)
    pelvis_name_width = 150
    pelvis_event_width = (pelvis_table_width - pelvis_name_width) / 4
    rect(pelvis_table_x, pelvis_table_y + 105, pelvis_table_width, 36, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = pelvis_table_x + pelvis_name_width + (event_index * pelvis_event_width)
        text_centered(event_x, pelvis_table_y + 129, pelvis_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = pelvis_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), pelvis_table_y + 112, sub_width, metric_label, 6, bold=True, color="#64748B")
    def draw_pelvis_torso_rows(rows, first_row_y, unit="°"):
      for row_index, (label, athlete_key, reference_key, color) in enumerate(rows):
        row_y = first_row_y - (row_index * 32)
        athlete_data = arm_kinematics.get(athlete_key, {})
        reference_data = arm_kinematics.get(reference_key, {})
        rect(pelvis_table_x, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(pelvis_table_x + 10, row_y + 14, label, 9.5, bold=True, color=color)
        for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
            event_x = pelvis_table_x + pelvis_name_width + (event_index * pelvis_event_width)
            athlete_event = athlete_data.get("metrics", {}).get(event_label, {})
            reference_event = reference_data.get("metrics", {}).get(event_label, {})
            values = [athlete_event.get("mean"), reference_event.get("mean"), athlete_event.get("std"), reference_event.get("std")]
            cell_width = pelvis_event_width / 4
            for value_index, value in enumerate(values):
                value_text = "" if value is None or pd.isna(value) else f"{value:.1f}{unit}"
                text_centered(event_x + (value_index * cell_width), row_y + 14, cell_width, value_text, 7.6, bold=True, color="#111827")
        line(pelvis_table_x + 10, row_y, pelvis_table_x + pelvis_table_width - 10, row_y, color="#E5E7EB", width=0.55)

    draw_pelvis_torso_rows(pelvis_rows, pelvis_table_y + 70)

    def draw_pelvis_plot(title, athlete_key, reference_key, chart_x, chart_y, color, fill, unit="deg", chart_width=238, chart_height=130, title_size=10.5):
        plot_left = chart_x + 36
        plot_right = chart_x + chart_width - 9
        plot_bottom = chart_y + 32
        plot_top = chart_y + chart_height - 23
        text(chart_x, chart_y + chart_height + 7, title, title_size, bold=True, color="#111827")
        athlete_data = arm_kinematics.get(athlete_key, {})
        reference_data = arm_kinematics.get(reference_key, {})
        athlete_curves = athlete_data.get("curves", {})
        reference_curves = reference_data.get("curves", {})
        athlete_x, athlete_y, _, _ = aggregate_curves(athlete_curves, "Mean") if athlete_curves else ([], [], [], [])
        ref_x, _, ref_q1, ref_q3 = aggregate_curves(reference_curves, "Mean") if reference_curves else ([], [], [], [])
        fp_events = athlete_data.get("events", {}).get("fp_event_frames", []) or report_events.get("fp_event_frames", [])
        mer_events = athlete_data.get("events", {}).get("mer_event_frames", []) or report_events.get("mer_event_frames", [])
        x_min = rel_frame_to_ms(int(np.median(fp_events)) - ms_to_rel_frame(100)) if fp_events else -200
        x_max = rel_frame_to_ms(ms_to_rel_frame(150))
        visible_values = [
            value for frames, values in [(athlete_x, athlete_y), (ref_x, ref_q1), (ref_x, ref_q3)]
            for time_ms, value in zip(frames, values)
            if value is not None and np.isfinite(value) and x_min <= time_ms <= x_max
        ]
        y_min = min(visible_values) if visible_values else 0
        y_max = max(visible_values) if visible_values else 1
        y_span = max(y_max - y_min, 1)
        y_min -= 0.12 * y_span
        y_max += 0.18 * y_span
        px = lambda value: plot_left + ((value - x_min) / (x_max - x_min)) * (plot_right - plot_left)
        py = lambda value: plot_bottom + ((value - y_min) / (y_max - y_min)) * (plot_top - plot_bottom)
        for tick in range(int(np.ceil(x_min / 100) * 100), int(np.floor(x_max / 100) * 100) + 1, 100):
            x = px(tick)
            line(x, plot_bottom, x, plot_top, color="#F3F4F6", width=0.35)
            text(x - 8, plot_bottom - 10, tick, 5.8, color="#4B5563")
        for tick in np.linspace(y_min, y_max, 5):
            y = py(tick)
            line(plot_left, y, plot_right, y, color="#F3F4F6", width=0.35)
            tick_text = f"{tick:.1f}" if unit == "m/s" else f"{tick:.0f}°"
            text(plot_left - 25, y - 2, tick_text, 5.8, color="#4B5563")
        ref_upper = [(px(t), py(v)) for t, v in zip(ref_x, ref_q3) if v is not None and x_min <= t <= x_max]
        ref_lower = [(px(t), py(v)) for t, v in zip(ref_x, ref_q1) if v is not None and x_min <= t <= x_max]
        if ref_upper and ref_lower:
            poly(ref_upper + ref_lower[::-1], fill=fill)
        path([(px(t), py(v)) for t, v in zip(athlete_x, athlete_y) if v is not None and x_min <= t <= x_max], color=color, width=1.8)
        for label, frames, event_color in [("FP", fp_events, "#16A34A"), ("MER", mer_events, "#DC2626"), ("BR", [0], "#2563EB")]:
            if frames:
                event_ms = rel_frame_to_ms(int(np.median(frames)))
                if x_min <= event_ms <= x_max:
                    event_x = px(event_ms)
                    line(event_x, plot_bottom, event_x, plot_top, color=event_color, width=0.9)
                    text(event_x + 2, plot_top - 8, label, 5.8, bold=True, color=event_color)
        rect(plot_left, plot_bottom, plot_right - plot_left, plot_top - plot_bottom, stroke="#111111", fill=None, line_width=0.85)
        text((plot_left + plot_right) / 2 - 39, plot_bottom - 22, "Time Relative to BR (ms)", 5.8, bold=True, color="#111111")
        y_axis_label = "Velocity (m/s)" if unit == "m/s" else "Position (deg)"
        text_rotated(chart_x + 6, (plot_bottom + plot_top) / 2 - 20, y_axis_label, 5.8, bold=True, color="#111111", angle=90)

    torso_rows = [
        ("Torso Forward Tilt", "athlete_torso_forward_tilt", "reference_torso_forward_tilt", palette_color("torso_primary")),
        ("Torso Lateral Tilt", "athlete_torso_lateral_tilt", "reference_torso_lateral_tilt", palette_color("torso_secondary")),
        ("Torso Rotation", "athlete_torso_rotation", "reference_torso_rotation", palette_color("torso_rotation")),
    ]
    draw_pelvis_torso_rows(torso_rows, 366)
    draw_pelvis_plot("Pelvis Forward Tilt", "athlete_pelvis_forward_tilt", "reference_pelvis_forward_tilt", margin, 142, palette_color("pelvis_primary"), palette_fill("pelvis_primary"))
    draw_pelvis_plot("Pelvis Lateral Tilt", "athlete_pelvis_lateral_tilt", "reference_pelvis_lateral_tilt", 278, 142, palette_color("pelvis_secondary"), palette_fill("pelvis_secondary"))
    draw_pelvis_plot("Pelvis Rotation", "athlete_pelvis_rotation", "reference_pelvis_rotation", 520, 142, palette_color("pelvis_rotation"), palette_fill("pelvis_rotation"))
    draw_pelvis_plot("Torso Forward Tilt", "athlete_torso_forward_tilt", "reference_torso_forward_tilt", margin, 3, palette_color("torso_primary"), palette_fill("torso_primary"))
    draw_pelvis_plot("Torso Lateral Tilt", "athlete_torso_lateral_tilt", "reference_torso_lateral_tilt", 278, 3, palette_color("torso_secondary"), palette_fill("torso_secondary"))
    draw_pelvis_plot("Torso Rotation", "athlete_torso_rotation", "reference_torso_rotation", 520, 3, palette_color("torso_rotation"), palette_fill("torso_rotation"))
    fourth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Torso-Pelvis Kinematics", 28, bold=True, color="#111827")
    torso_pelvis_rows = [
        ("Torso-Pelvis Flex/Ext", "athlete_torso_pelvis_forward_tilt", "reference_torso_pelvis_forward_tilt", palette_color("torso_pelvis_primary")),
        ("Torso-Pelvis Lateral Flexion", "athlete_torso_pelvis_lateral_tilt", "reference_torso_pelvis_lateral_tilt", palette_color("torso_pelvis_secondary")),
        ("Torso-Pelvis Rotation", "athlete_torso_pelvis_rotation", "reference_torso_pelvis_rotation", palette_color("torso_pelvis_rotation")),
    ]
    torso_pelvis_table_y = 408
    rect(margin, torso_pelvis_table_y + 105, page_width - (margin * 2), 36, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = margin + pelvis_name_width + (event_index * pelvis_event_width)
        text_centered(event_x, torso_pelvis_table_y + 129, pelvis_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = pelvis_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), torso_pelvis_table_y + 112, sub_width, metric_label, 6, bold=True, color="#64748B")
    draw_pelvis_torso_rows(torso_pelvis_rows, torso_pelvis_table_y + 70)
    wide_three_plot_args = {"chart_width": 346, "chart_height": 125, "title_size": 9.6}
    centered_wide_plot_x = (page_width - 346) / 2
    draw_pelvis_plot("Flex/Ext", "athlete_torso_pelvis_forward_tilt", "reference_torso_pelvis_forward_tilt", margin, 215, palette_color("torso_pelvis_primary"), palette_fill("torso_pelvis_primary"), **wide_three_plot_args)
    draw_pelvis_plot("Lateral Flexion", "athlete_torso_pelvis_lateral_tilt", "reference_torso_pelvis_lateral_tilt", 410, 215, palette_color("torso_pelvis_secondary"), palette_fill("torso_pelvis_secondary"), **wide_three_plot_args)
    draw_pelvis_plot("Rotation", "athlete_torso_pelvis_rotation", "reference_torso_pelvis_rotation", centered_wide_plot_x, 55, palette_color("torso_pelvis_rotation"), palette_fill("torso_pelvis_rotation"), **wide_three_plot_args)
    fifth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Torso-Pelvis Kinematics", 28, bold=True, color="#111827")
    torso_pelvis_velocity_rows = [
        ("Torso-Pelvis Flex/Ext Angular Velocity", "athlete_torso_pelvis_flexion_velocity", "reference_torso_pelvis_flexion_velocity", palette_color("torso_pelvis_primary")),
        ("Torso-Pelvis Lateral Flexion Angular Velocity", "athlete_torso_pelvis_glove_side_lateral_flexion_velocity", "reference_torso_pelvis_glove_side_lateral_flexion_velocity", palette_color("torso_pelvis_secondary")),
        ("Torso-Pelvis Rotation Angular Velocity", "athlete_torso_pelvis_separation_velocity", "reference_torso_pelvis_separation_velocity", palette_color("torso_pelvis_rotation")),
    ]
    tpv_table_y = 360
    tpv_name_width = 245
    tpv_value_width = (page_width - (margin * 2) - tpv_name_width) / 4
    rect(margin, tpv_table_y + 140, page_width - (margin * 2), 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(margin + tpv_name_width + (index * tpv_value_width), tpv_table_y + 151, tpv_value_width, header, 8.4, bold=True, color="#64748B")
    for row_index, (label, athlete_key, reference_key, color) in enumerate(torso_pelvis_velocity_rows):
        row_y = tpv_table_y + 105 - (row_index * 35)
        athlete_max = arm_kinematics.get(athlete_key, {}).get("metrics", {}).get("Max", {})
        reference_max = arm_kinematics.get(reference_key, {}).get("metrics", {}).get("Max", {})
        values = [athlete_max.get("mean"), athlete_max.get("std"), reference_max.get("mean"), reference_max.get("std")]
        rect(margin, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(margin + 10, row_y + 14, label, 9, bold=True, color=color)
        for value_index, value in enumerate(values):
            value_text = "" if value is None or pd.isna(value) else f"{value:.0f}°/s"
            text_centered(margin + tpv_name_width + (value_index * tpv_value_width), row_y + 14, tpv_value_width, value_text, 9, bold=True, color="#111827")
        line(margin + 10, row_y, page_width - margin - 10, row_y, color="#E5E7EB", width=0.55)
    torso_pelvis_velocity_plot_args = {"unit": "deg/s", "chart_width": 346, "chart_height": 125, "title_size": 9.6}
    centered_wide_plot_x = (page_width - 346) / 2
    draw_arm_metric_plot("Torso-Pelvis Flex/Ext Angular Velocity", arm_kinematics.get("athlete_torso_pelvis_flexion_velocity", {}), arm_kinematics.get("reference_torso_pelvis_flexion_velocity", {}), margin, 215, palette_color("torso_pelvis_primary"), palette_fill("torso_pelvis_primary"), **torso_pelvis_velocity_plot_args)
    draw_arm_metric_plot("Lateral Flexion Angular Velocity", arm_kinematics.get("athlete_torso_pelvis_glove_side_lateral_flexion_velocity", {}), arm_kinematics.get("reference_torso_pelvis_glove_side_lateral_flexion_velocity", {}), 410, 215, palette_color("torso_pelvis_secondary"), palette_fill("torso_pelvis_secondary"), **torso_pelvis_velocity_plot_args)
    draw_arm_metric_plot("Rotation Angular Velocity", arm_kinematics.get("athlete_torso_pelvis_separation_velocity", {}), arm_kinematics.get("reference_torso_pelvis_separation_velocity", {}), centered_wide_plot_x, 55, palette_color("torso_pelvis_rotation"), palette_fill("torso_pelvis_rotation"), **torso_pelvis_velocity_plot_args)
    torso_pelvis_velocity_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Hips Kinematics", 28, bold=True, color="#111827")
    hip_rows = [
        ("Back Hip Flexion/Extension", "athlete_back_hip_forward_tilt", "reference_back_hip_forward_tilt", palette_color("hip_back")),
        ("Back Hip Ab/Adduction", "athlete_back_hip_lateral_tilt", "reference_back_hip_lateral_tilt", palette_color("hip_lead")),
        ("Back Hip Rotation", "athlete_back_hip_rotation", "reference_back_hip_rotation", palette_color("hip_rotation")),
        ("Lead Hip Flexion/Extension", "athlete_lead_hip_forward_tilt", "reference_lead_hip_forward_tilt", palette_color("hip_back")),
        ("Lead Hip Ab/Adduction", "athlete_lead_hip_lateral_tilt", "reference_lead_hip_lateral_tilt", palette_color("hip_lead")),
        ("Lead Hip Rotation", "athlete_lead_hip_rotation", "reference_lead_hip_rotation", palette_color("hip_rotation")),
    ]
    hip_table_y = 408
    rect(margin, hip_table_y + 105, page_width - (margin * 2), 36, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = margin + pelvis_name_width + (event_index * pelvis_event_width)
        text_centered(event_x, hip_table_y + 129, pelvis_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = pelvis_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), hip_table_y + 112, sub_width, metric_label, 6, bold=True, color="#64748B")
    draw_pelvis_torso_rows(hip_rows, hip_table_y + 70)
    draw_pelvis_plot("Back Hip Flex/Ext", "athlete_back_hip_forward_tilt", "reference_back_hip_forward_tilt", margin, 142, palette_color("hip_back"), palette_fill("hip_back"))
    draw_pelvis_plot("Back Hip Ab/Adduction", "athlete_back_hip_lateral_tilt", "reference_back_hip_lateral_tilt", 278, 142, palette_color("hip_lead"), palette_fill("hip_lead"))
    draw_pelvis_plot("Back Hip Rotation", "athlete_back_hip_rotation", "reference_back_hip_rotation", 520, 142, palette_color("hip_rotation"), palette_fill("hip_rotation"))
    draw_pelvis_plot("Lead Hip Flex/Ext", "athlete_lead_hip_forward_tilt", "reference_lead_hip_forward_tilt", margin, 3, palette_color("hip_back"), palette_fill("hip_back"))
    draw_pelvis_plot("Lead Hip Ab/Adduction", "athlete_lead_hip_lateral_tilt", "reference_lead_hip_lateral_tilt", 278, 3, palette_color("hip_lead"), palette_fill("hip_lead"))
    draw_pelvis_plot("Lead Hip Rotation", "athlete_lead_hip_rotation", "reference_lead_hip_rotation", 520, 3, palette_color("hip_rotation"), palette_fill("hip_rotation"))
    sixth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Center of Gravity", 28, bold=True, color="#111827")
    cg_velocity_rows = [
        ("Anterior/Posterior", "athlete_cg_velocity_x", "reference_cg_velocity_x", palette_color("cog")),
        ("Medial/Lateral", "athlete_cg_velocity_y", "reference_cg_velocity_y", palette_color("cog_secondary")),
        ("Vertical", "athlete_cg_velocity_z", "reference_cg_velocity_z", palette_color("cog_vertical")),
    ]
    cg_table_y = 408
    rect(margin, cg_table_y + 105, page_width - (margin * 2), 36, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = margin + pelvis_name_width + (event_index * pelvis_event_width)
        text_centered(event_x, cg_table_y + 129, pelvis_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = pelvis_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), cg_table_y + 112, sub_width, metric_label, 6, bold=True, color="#64748B")
    draw_pelvis_torso_rows(cg_velocity_rows, cg_table_y + 70, unit=" m/s")
    cog_plot_args = {"unit": "m/s", "chart_width": 346, "chart_height": 125, "title_size": 9.6}
    centered_cog_plot_x = (page_width - 346) / 2
    draw_pelvis_plot("Anterior/Posterior", "athlete_cg_velocity_x", "reference_cg_velocity_x", margin, 215, palette_color("cog"), palette_fill("cog"), **cog_plot_args)
    draw_pelvis_plot("Medial/Lateral", "athlete_cg_velocity_y", "reference_cg_velocity_y", 410, 215, palette_color("cog_secondary"), palette_fill("cog_secondary"), **cog_plot_args)
    draw_pelvis_plot("Vertical", "athlete_cg_velocity_z", "reference_cg_velocity_z", centered_cog_plot_x, 55, palette_color("cog_vertical"), palette_fill("cog_vertical"), **cog_plot_args)
    seventh_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Stride", 28, bold=True, color="#111827")
    stride_rows = [
        ("Stride Angle", "stride_angle_deg", "°", palette_color("stride_primary")),
        ("Stride Length", "stride_length_in", " in", palette_color("stride_secondary")),
        ("Stride Length / Height", "stride_length_pct_height", "%", palette_color("stride_tertiary")),
    ]
    stride_athlete = arm_kinematics.get("athlete_stride", {})
    stride_reference = arm_kinematics.get("reference_stride", {})
    stride_table_x = margin
    stride_table_y = 330
    stride_table_width = page_width - (margin * 2)
    stride_name_width = 230
    stride_value_width = (stride_table_width - stride_name_width) / 4
    rect(stride_table_x, stride_table_y + 140, stride_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average", "SD ±", "Ref Group Average", "Ref Group SD ±"]):
        text_centered(stride_table_x + stride_name_width + (index * stride_value_width), stride_table_y + 151, stride_value_width, header, 8.8, bold=True, color="#64748B")
    for row_index, (label, key, unit, color) in enumerate(stride_rows):
        row_y = stride_table_y + 105 - (row_index * 34)
        athlete_metric = stride_athlete.get("metrics", {}).get(key, {})
        reference_metric = stride_reference.get("metrics", {}).get(key, {})
        values = [
            athlete_metric.get("mean"),
            athlete_metric.get("std"),
            reference_metric.get("mean"),
            reference_metric.get("std"),
        ]
        rect(stride_table_x, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(stride_table_x + 10, row_y + 14, label, 9.8, bold=True, color=color)
        for value_index, value in enumerate(values):
            value_text = "" if value is None or pd.isna(value) else f"{value:.1f}{unit}"
            text_centered(stride_table_x + stride_name_width + (value_index * stride_value_width), row_y + 14, stride_value_width, value_text, 9.5, bold=True, color="#111827")
        line(stride_table_x + 10, row_y, stride_table_x + stride_table_width - 10, row_y, color="#E5E7EB", width=0.55)
    stride_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Lower Extremity Kinematics", 28, bold=True, color="#111827")
    lower_extremity_rows = [
        ("Back Knee Flexion/Extension", "athlete_back_knee_flexion_extension", "reference_back_knee_flexion_extension", palette_color("lower_back")),
        ("Back Ankle Dorsi/Plantarflexion", "athlete_back_ankle_flexion_extension", "reference_back_ankle_flexion_extension", palette_color("lower_ankle")),
        ("Front Knee Flexion/Extension", "athlete_front_knee_flexion_extension", "reference_front_knee_flexion_extension", palette_color("lower_front")),
        ("Front Ankle Dorsi/Plantarflexion", "athlete_front_ankle_flexion_extension", "reference_front_ankle_flexion_extension", palette_color("arm_rotation")),
        ("Back Ankle Eversion/Inversion", "athlete_back_ankle_eversion_inversion", "reference_back_ankle_eversion_inversion", palette_color("hip_lead")),
    ]
    lower_table_y = 408
    rect(margin, lower_table_y + 105, page_width - (margin * 2), 36, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for event_index, event_label in enumerate(["FP", "MER", "BR", "Max"]):
        event_x = margin + pelvis_name_width + (event_index * pelvis_event_width)
        text_centered(event_x, lower_table_y + 129, pelvis_event_width, event_label, 10, bold=True, color="#111827")
        sub_width = pelvis_event_width / 4
        for metric_index, (metric_label, _) in enumerate(arm_metric_labels):
            text_centered(event_x + (metric_index * sub_width), lower_table_y + 112, sub_width, metric_label, 6, bold=True, color="#64748B")
    draw_pelvis_torso_rows(lower_extremity_rows, lower_table_y + 70)
    draw_pelvis_plot("Back Knee Flex/Ext", "athlete_back_knee_flexion_extension", "reference_back_knee_flexion_extension", margin, 190, palette_color("lower_back"), palette_fill("lower_back"))
    draw_pelvis_plot("Back Ankle Dorsi/Plantar", "athlete_back_ankle_flexion_extension", "reference_back_ankle_flexion_extension", 278, 190, palette_color("lower_ankle"), palette_fill("lower_ankle"))
    draw_pelvis_plot("Back Ankle Ev/Inv", "athlete_back_ankle_eversion_inversion", "reference_back_ankle_eversion_inversion", 520, 190, palette_color("hip_lead"), palette_fill("hip_lead"))
    draw_pelvis_plot("Front Knee Flex/Ext", "athlete_front_knee_flexion_extension", "reference_front_knee_flexion_extension", 155, 55, palette_color("lower_front"), palette_fill("lower_front"))
    draw_pelvis_plot("Front Ankle Dorsi/Plantar", "athlete_front_ankle_flexion_extension", "reference_front_ankle_flexion_extension", 398, 55, palette_color("arm_rotation"), palette_fill("arm_rotation"))
    eighth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Lower Extremity Kinematics", 28, bold=True, color="#111827")
    lower_velocity_rows = [
        ("Back Knee Extension Angular Velocity", "athlete_back_knee_flexion_extension_velocity", "reference_back_knee_flexion_extension_velocity", palette_color("lower_back")),
        ("Back Ankle Plantarflexion Angular Velocity", "athlete_back_ankle_flexion_extension_velocity", "reference_back_ankle_flexion_extension_velocity", palette_color("lower_ankle")),
        ("Back Ankle Eversion/Inversion Angular Velocity", "athlete_back_ankle_eversion_inversion_velocity", "reference_back_ankle_eversion_inversion_velocity", palette_color("hip_lead")),
        ("Front Knee Extension Angular Velocity", "athlete_front_knee_flexion_extension_velocity", "reference_front_knee_flexion_extension_velocity", palette_color("lower_front")),
        ("Front Ankle Plantarflexion Angular Velocity", "athlete_front_ankle_flexion_extension_velocity", "reference_front_ankle_flexion_extension_velocity", palette_color("arm_rotation")),
    ]
    lower_velocity_table_x = margin
    lower_velocity_table_y = 370
    lower_velocity_table_width = page_width - (margin * 2)
    lower_velocity_name_width = 245
    lower_velocity_value_width = (lower_velocity_table_width - lower_velocity_name_width) / 4
    rect(lower_velocity_table_x, lower_velocity_table_y + 140, lower_velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(lower_velocity_table_x + lower_velocity_name_width + (index * lower_velocity_value_width), lower_velocity_table_y + 151, lower_velocity_value_width, header, 8.4, bold=True, color="#64748B")
    for row_index, (label, athlete_key, reference_key, color) in enumerate(lower_velocity_rows):
        row_y = lower_velocity_table_y + 105 - (row_index * 28)
        athlete_data = arm_kinematics.get(athlete_key, {})
        reference_data = arm_kinematics.get(reference_key, {})
        athlete_max = athlete_data.get("metrics", {}).get("Max", {})
        reference_max = reference_data.get("metrics", {}).get("Max", {})
        values = [athlete_max.get("mean"), athlete_max.get("std"), reference_max.get("mean"), reference_max.get("std")]
        rect(lower_velocity_table_x, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
        text(lower_velocity_table_x + 10, row_y + 14, label, 8.8, bold=True, color=color)
        for value_index, value in enumerate(values):
            value_text = "" if value is None or pd.isna(value) else f"{value:.0f}°/s"
            text_centered(lower_velocity_table_x + lower_velocity_name_width + (value_index * lower_velocity_value_width), row_y + 14, lower_velocity_value_width, value_text, 8.8, bold=True, color="#111827")
        line(lower_velocity_table_x + 10, row_y, lower_velocity_table_x + lower_velocity_table_width - 10, row_y, color="#E5E7EB", width=0.55)
    lower_velocity_plot_args = {"unit": "deg/s", "chart_width": 230, "chart_height": 105, "title_size": 8.4}
    lower_velocity_plot_x_positions = [margin, 278, 520]
    draw_arm_metric_plot("Back Knee Extension Angular Velocity", arm_kinematics.get("athlete_back_knee_flexion_extension_velocity", {}), arm_kinematics.get("reference_back_knee_flexion_extension_velocity", {}), lower_velocity_plot_x_positions[0], 220, palette_color("lower_back"), palette_fill("lower_back"), **lower_velocity_plot_args)
    draw_arm_metric_plot("Back Ankle Plantarflexion Angular Velocity", arm_kinematics.get("athlete_back_ankle_flexion_extension_velocity", {}), arm_kinematics.get("reference_back_ankle_flexion_extension_velocity", {}), lower_velocity_plot_x_positions[1], 220, palette_color("lower_ankle"), palette_fill("lower_ankle"), **lower_velocity_plot_args)
    draw_arm_metric_plot("Back Ankle Ev/Inv Angular Velocity", arm_kinematics.get("athlete_back_ankle_eversion_inversion_velocity", {}), arm_kinematics.get("reference_back_ankle_eversion_inversion_velocity", {}), lower_velocity_plot_x_positions[2], 220, palette_color("hip_lead"), palette_fill("hip_lead"), **lower_velocity_plot_args)
    draw_arm_metric_plot("Front Knee Extension Angular Velocity", arm_kinematics.get("athlete_front_knee_flexion_extension_velocity", {}), arm_kinematics.get("reference_front_knee_flexion_extension_velocity", {}), 155, 88, palette_color("lower_front"), palette_fill("lower_front"), **lower_velocity_plot_args)
    draw_arm_metric_plot("Front Ankle Plantarflexion Angular Velocity", arm_kinematics.get("athlete_front_ankle_flexion_extension_velocity", {}), arm_kinematics.get("reference_front_ankle_flexion_extension_velocity", {}), 398, 88, palette_color("arm_rotation"), palette_fill("arm_rotation"), **lower_velocity_plot_args)
    thirteenth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Back Hip Kinematics", 28, bold=True, color="#111827")
    back_hip_velocity_rows = [
        ("Back Hip Flexion Angular Velocity", "athlete_back_hip_flexion_velocity", "reference_back_hip_flexion_velocity", palette_color("hip_back")),
        ("Back Hip Extension Angular Velocity", "athlete_back_hip_extension_velocity", "reference_back_hip_extension_velocity", palette_color("arm_secondary")),
        ("Back Hip Adduction Angular Velocity", "athlete_back_hip_lateral_tilt_velocity", "reference_back_hip_lateral_tilt_velocity", palette_color("hip_lead")),
        ("Back Hip External Rotation Angular Velocity", "athlete_back_hip_external_rotation_velocity", "reference_back_hip_external_rotation_velocity", palette_color("pelvis_rotation")),
        ("Back Hip Internal Rotation Angular Velocity", "athlete_back_hip_internal_rotation_velocity", "reference_back_hip_internal_rotation_velocity", palette_color("torso_pelvis_rotation")),
    ]
    hip_velocity_table_x = margin
    hip_velocity_table_y = 370
    hip_velocity_table_width = page_width - (margin * 2)
    hip_velocity_name_width = 230
    hip_velocity_value_width = (hip_velocity_table_width - hip_velocity_name_width) / 4
    rect(hip_velocity_table_x, hip_velocity_table_y + 140, hip_velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(hip_velocity_table_x + hip_velocity_name_width + (index * hip_velocity_value_width), hip_velocity_table_y + 151, hip_velocity_value_width, header, 8.8, bold=True, color="#64748B")

    def draw_velocity_rows(rows, table_x, table_y, name_width, value_width, row_step=28):
        for row_index, (label, athlete_key, reference_key, color) in enumerate(rows):
            row_y = table_y + 105 - (row_index * row_step)
            athlete_data = arm_kinematics.get(athlete_key, {})
            reference_data = arm_kinematics.get(reference_key, {})
            athlete_max = athlete_data.get("metrics", {}).get("Max", {})
            reference_max = reference_data.get("metrics", {}).get("Max", {})
            values = [athlete_max.get("mean"), athlete_max.get("std"), reference_max.get("mean"), reference_max.get("std")]
            rect(table_x, row_y + 7, 4, 20, stroke=color, fill=color, line_width=0.1)
            text(table_x + 10, row_y + 14, label, 8.8, bold=True, color=color)
            for value_index, value in enumerate(values):
                value_text = "" if value is None or pd.isna(value) else f"{value:.0f}°/s"
                text_centered(table_x + name_width + (value_index * value_width), row_y + 14, value_width, value_text, 9, bold=True, color="#111827")
            line(table_x + 10, row_y, table_x + hip_velocity_table_width - 10, row_y, color="#E5E7EB", width=0.55)

    draw_velocity_rows(back_hip_velocity_rows, hip_velocity_table_x, hip_velocity_table_y, hip_velocity_name_width, hip_velocity_value_width, row_step=28)
    hip_plot_args = {"unit": "deg/s", "chart_width": 230, "chart_height": 105, "title_size": 7.8}
    hip_plot_x_positions = [margin, 278, 520]
    draw_arm_metric_plot("Back Hip Flexion Angular Velocity", arm_kinematics.get("athlete_back_hip_flexion_velocity", {}), arm_kinematics.get("reference_back_hip_flexion_velocity", {}), hip_plot_x_positions[0], 210, palette_color("hip_back"), palette_fill("hip_back"), **hip_plot_args)
    draw_arm_metric_plot("Back Hip Extension Angular Velocity", arm_kinematics.get("athlete_back_hip_extension_velocity", {}), arm_kinematics.get("reference_back_hip_extension_velocity", {}), hip_plot_x_positions[1], 210, palette_color("arm_secondary"), palette_fill("arm_secondary"), **hip_plot_args)
    draw_arm_metric_plot("Back Hip Adduction Angular Velocity", arm_kinematics.get("athlete_back_hip_lateral_tilt_velocity", {}), arm_kinematics.get("reference_back_hip_lateral_tilt_velocity", {}), hip_plot_x_positions[2], 210, palette_color("hip_lead"), palette_fill("hip_lead"), **hip_plot_args)
    draw_arm_metric_plot("Back Hip External Rotation Angular Velocity", arm_kinematics.get("athlete_back_hip_external_rotation_velocity", {}), arm_kinematics.get("reference_back_hip_external_rotation_velocity", {}), 155, 78, palette_color("pelvis_rotation"), palette_fill("pelvis_rotation"), **hip_plot_args)
    draw_arm_metric_plot("Back Hip Internal Rotation Angular Velocity", arm_kinematics.get("athlete_back_hip_internal_rotation_velocity", {}), arm_kinematics.get("reference_back_hip_internal_rotation_velocity", {}), 398, 78, palette_color("torso_pelvis_rotation"), palette_fill("torso_pelvis_rotation"), **hip_plot_args)
    back_hip_velocity_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Lead Hip Kinematics", 28, bold=True, color="#111827")
    lead_hip_velocity_rows = [
        ("Lead Hip Extension Angular Velocity", "athlete_lead_hip_extension_velocity", "reference_lead_hip_extension_velocity", palette_color("arm_rotation")),
        ("Lead Hip Adduction Angular Velocity", "athlete_lead_hip_lateral_tilt_velocity", "reference_lead_hip_lateral_tilt_velocity", palette_color("hip_lead")),
        ("Lead Hip Internal Rotation Angular Velocity", "athlete_lead_hip_internal_rotation_velocity", "reference_lead_hip_internal_rotation_velocity", palette_color("hip_rotation")),
    ]
    lead_table_y = 390
    rect(hip_velocity_table_x, lead_table_y + 120, hip_velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(hip_velocity_table_x + hip_velocity_name_width + (index * hip_velocity_value_width), lead_table_y + 131, hip_velocity_value_width, header, 8.8, bold=True, color="#64748B")
    draw_velocity_rows(lead_hip_velocity_rows, hip_velocity_table_x, lead_table_y - 20, hip_velocity_name_width, hip_velocity_value_width, row_step=35)
    lead_hip_plot_args = {"unit": "deg/s", "chart_width": 346, "chart_height": 125, "title_size": 9.6}
    lead_centered_plot_x = (page_width - 346) / 2
    draw_arm_metric_plot("Lead Hip Extension Angular Velocity", arm_kinematics.get("athlete_lead_hip_extension_velocity", {}), arm_kinematics.get("reference_lead_hip_extension_velocity", {}), margin, 215, palette_color("arm_rotation"), palette_fill("arm_rotation"), **lead_hip_plot_args)
    draw_arm_metric_plot("Lead Hip Adduction Angular Velocity", arm_kinematics.get("athlete_lead_hip_lateral_tilt_velocity", {}), arm_kinematics.get("reference_lead_hip_lateral_tilt_velocity", {}), 410, 215, palette_color("hip_lead"), palette_fill("hip_lead"), **lead_hip_plot_args)
    draw_arm_metric_plot("Lead Hip Internal Rotation Angular Velocity", arm_kinematics.get("athlete_lead_hip_internal_rotation_velocity", {}), arm_kinematics.get("reference_lead_hip_internal_rotation_velocity", {}), lead_centered_plot_x, 55, palette_color("hip_rotation"), palette_fill("hip_rotation"), **lead_hip_plot_args)
    lead_hip_velocity_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Pelvis Kinematics", 28, bold=True, color="#111827")
    pelvis_velocity_table_x = margin
    pelvis_velocity_table_y = 360
    pelvis_velocity_table_width = page_width - (margin * 2)
    pelvis_velocity_name_width = 220
    pelvis_velocity_value_width = (pelvis_velocity_table_width - pelvis_velocity_name_width) / 4
    rect(pelvis_velocity_table_x, pelvis_velocity_table_y + 140, pelvis_velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(pelvis_velocity_table_x + pelvis_velocity_name_width + (index * pelvis_velocity_value_width), pelvis_velocity_table_y + 151, pelvis_velocity_value_width, header, 8.8, bold=True, color="#64748B")
    pelvis_forward_velocity_athlete = arm_kinematics.get("athlete_pelvis_forward_tilt_velocity", {})
    pelvis_forward_velocity_reference = arm_kinematics.get("reference_pelvis_forward_tilt_velocity", {})
    pelvis_lateral_velocity_athlete = arm_kinematics.get("athlete_pelvis_lateral_tilt_velocity", {})
    pelvis_lateral_velocity_reference = arm_kinematics.get("reference_pelvis_lateral_tilt_velocity", {})
    pelvis_rotation_velocity_athlete = arm_kinematics.get("athlete_pelvis_rotation_velocity", {})
    pelvis_rotation_velocity_reference = arm_kinematics.get("reference_pelvis_rotation_velocity", {})
    torso_forward_velocity_athlete = arm_kinematics.get("athlete_torso_forward_tilt_velocity", {})
    torso_forward_velocity_reference = arm_kinematics.get("reference_torso_forward_tilt_velocity", {})
    torso_backward_velocity_athlete = arm_kinematics.get("athlete_torso_backward_tilt_velocity", {})
    torso_backward_velocity_reference = arm_kinematics.get("reference_torso_backward_tilt_velocity", {})
    torso_lateral_velocity_athlete = arm_kinematics.get("athlete_torso_lateral_tilt_velocity", {})
    torso_lateral_velocity_reference = arm_kinematics.get("reference_torso_lateral_tilt_velocity", {})
    torso_rotation_velocity_athlete = arm_kinematics.get("athlete_torso_rotation_velocity", {})
    torso_rotation_velocity_reference = arm_kinematics.get("reference_torso_rotation_velocity", {})

    def draw_pelvis_velocity_row(title, athlete_data, reference_data, row_y, row_color):
        athlete_max = athlete_data.get("metrics", {}).get("Max", {})
        reference_max = reference_data.get("metrics", {}).get("Max", {})
        rect(pelvis_velocity_table_x, row_y + 7, 4, 20, stroke=row_color, fill=row_color, line_width=0.1)
        text(pelvis_velocity_table_x + 10, row_y + 14, title, 9.3, bold=True, color=row_color)
        for value_index, value in enumerate([
            athlete_max.get("mean"),
            athlete_max.get("std"),
            reference_max.get("mean"),
            reference_max.get("std"),
        ]):
            value_text = "" if value is None or pd.isna(value) else f"{value:.0f}°/s"
            text_centered(pelvis_velocity_table_x + pelvis_velocity_name_width + (value_index * pelvis_velocity_value_width), row_y + 14, pelvis_velocity_value_width, value_text, 9, bold=True, color="#111827")
        line(pelvis_velocity_table_x + 10, row_y, pelvis_velocity_table_x + pelvis_velocity_table_width - 10, row_y, color="#E5E7EB", width=0.55)

    draw_pelvis_velocity_row("Pelvis Forward Tilt Angular Velocity", pelvis_forward_velocity_athlete, pelvis_forward_velocity_reference, pelvis_velocity_table_y + 105, palette_color("pelvis_primary"))
    draw_pelvis_velocity_row("Pelvis Lateral Tilt Angular Velocity", pelvis_lateral_velocity_athlete, pelvis_lateral_velocity_reference, pelvis_velocity_table_y + 70, palette_color("pelvis_secondary"))
    draw_pelvis_velocity_row("Pelvis Rotation Angular Velocity", pelvis_rotation_velocity_athlete, pelvis_rotation_velocity_reference, pelvis_velocity_table_y + 35, palette_color("pelvis_rotation"))

    compact_width = 346
    compact_height = 125
    plot_x_positions = [margin, 410]
    centered_plot_x = (page_width - compact_width) / 2
    plot_args = {"unit": "deg/s", "chart_width": compact_width, "chart_height": compact_height, "title_size": 9.6}
    draw_arm_metric_plot("Pelvis Forward Tilt Angular Velocity", pelvis_forward_velocity_athlete, pelvis_forward_velocity_reference, plot_x_positions[0], 215, palette_color("pelvis_primary"), palette_fill("pelvis_primary"), **plot_args)
    draw_arm_metric_plot("Pelvis Lateral Tilt Angular Velocity", pelvis_lateral_velocity_athlete, pelvis_lateral_velocity_reference, plot_x_positions[1], 215, palette_color("pelvis_secondary"), palette_fill("pelvis_secondary"), **plot_args)
    draw_arm_metric_plot("Pelvis Rotation Angular Velocity", pelvis_rotation_velocity_athlete, pelvis_rotation_velocity_reference, centered_plot_x, 55, palette_color("pelvis_rotation"), palette_fill("pelvis_rotation"), **plot_args)
    tenth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    ops = []
    text(margin, page_height - 45, "Torso Kinematics", 28, bold=True, color="#111827")
    torso_velocity_table_y = 360
    rect(pelvis_velocity_table_x, torso_velocity_table_y + 140, pelvis_velocity_table_width, 30, stroke="#F8FAFC", fill="#F8FAFC", line_width=0.1)
    for index, header in enumerate(["Average Max", "SD ±", "Ref Group Average Max", "Ref Group SD ±"]):
        text_centered(pelvis_velocity_table_x + pelvis_velocity_name_width + (index * pelvis_velocity_value_width), torso_velocity_table_y + 151, pelvis_velocity_value_width, header, 8.8, bold=True, color="#64748B")
    draw_pelvis_velocity_row("Torso Forward Tilt Angular Velocity", torso_forward_velocity_athlete, torso_forward_velocity_reference, torso_velocity_table_y + 105, palette_color("torso_primary"))
    draw_pelvis_velocity_row("Torso Backward Tilt Angular Velocity", torso_backward_velocity_athlete, torso_backward_velocity_reference, torso_velocity_table_y + 70, palette_color("torso_secondary"))
    draw_pelvis_velocity_row("Torso Lateral Tilt Angular Velocity", torso_lateral_velocity_athlete, torso_lateral_velocity_reference, torso_velocity_table_y + 35, palette_color("torso_pelvis_secondary"))
    draw_pelvis_velocity_row("Torso Rotation Angular Velocity", torso_rotation_velocity_athlete, torso_rotation_velocity_reference, torso_velocity_table_y, palette_color("torso_rotation"))
    torso_plot_args = {"unit": "deg/s", "chart_width": 346, "chart_height": 125, "title_size": 10}
    draw_arm_metric_plot("Torso Forward Tilt Angular Velocity", torso_forward_velocity_athlete, torso_forward_velocity_reference, margin, 195, palette_color("torso_primary"), palette_fill("torso_primary"), **torso_plot_args)
    draw_arm_metric_plot("Torso Backward Tilt Angular Velocity", torso_backward_velocity_athlete, torso_backward_velocity_reference, 410, 195, palette_color("torso_secondary"), palette_fill("torso_secondary"), **torso_plot_args)
    draw_arm_metric_plot("Torso Lateral Tilt Angular Velocity", torso_lateral_velocity_athlete, torso_lateral_velocity_reference, margin, 55, palette_color("torso_pelvis_secondary"), palette_fill("torso_pelvis_secondary"), **torso_plot_args)
    draw_arm_metric_plot("Torso Rotation Angular Velocity", torso_rotation_velocity_athlete, torso_rotation_velocity_reference, 410, 55, palette_color("torso_rotation"), palette_fill("torso_rotation"), **torso_plot_args)
    twelfth_page_content = "\n".join(ops).encode("latin-1", errors="replace")

    selected_sections = set(selected_sections or [])
    page_contents = [cover_page_content]

    def add_section_pages(section_key, pages):
        if section_key not in selected_sections:
            return
        for page in pages:
            if page not in page_contents:
                page_contents.append(page)

    add_section_pages("kinematic_sequence", [first_page_content])
    add_section_pages("throwing_arm", [second_page_content, third_page_content])
    if "pelvis" in selected_sections or "torso" in selected_sections:
        if fourth_page_content not in page_contents:
            page_contents.append(fourth_page_content)
    add_section_pages("pelvis", [tenth_page_content])
    add_section_pages("torso", [twelfth_page_content])
    add_section_pages("torso_pelvis", [fifth_page_content, torso_pelvis_velocity_page_content])
    add_section_pages("hips", [sixth_page_content])
    add_section_pages("back_hip", [back_hip_velocity_page_content])
    add_section_pages("lead_hip", [lead_hip_velocity_page_content])
    add_section_pages("lower_extremity", [eighth_page_content, thirteenth_page_content])
    add_section_pages("cog", [seventh_page_content])
    add_section_pages("stride", [stride_page_content])
    page_count = len(page_contents)

    def page_number_footer(page_number, total_pages):
        label = f"{page_number}"
        estimated_width = len(label) * 7 * 0.50
        x = page_width - margin - estimated_width
        y = 18
        r, g, b = rgb("#64748B")
        return (
            f"\n{r:.4f} {g:.4f} {b:.4f} rg\n"
            f"BT /F1 7 Tf {x:.2f} {y:.2f} Td ({esc(label)}) Tj ET\n"
        ).encode("latin-1", errors="replace")

    page_contents = [
        content + page_number_footer(page_index + 1, page_count)
        for page_index, content in enumerate(page_contents)
    ]

    font_regular_obj = 3 + page_count
    font_bold_obj = font_regular_obj + 1
    logo_obj = font_bold_obj + 1 if logo_image else None
    first_content_obj = font_bold_obj + 1 + (1 if logo_image else 0)
    page_objects = []
    for page_index in range(page_count):
        content_obj = first_content_obj + page_index
        resources = f"/Resources << /Font << /F1 {font_regular_obj} 0 R /F2 {font_bold_obj} 0 R >>"
        if logo_obj:
            resources += f" /XObject << /ImLogo {logo_obj} 0 R >>"
        resources += " >>"
        page_objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"{resources} "
            f"/Contents {content_obj} 0 R >>".encode("ascii")
        )
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
        *page_objects,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    if logo_image:
        objects.append(
            (
                f"<< /Type /XObject /Subtype /Image /Width {logo_image['width']} "
                f"/Height {logo_image['height']} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(logo_image['data'])} >>\nstream\n"
            ).encode("ascii")
            + logo_image["data"]
            + b"\nendstream"
        )
    objects.extend(
        [
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
            for content in page_contents
        ]
    )
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


with tab_report:
    st.subheader("PDF Reports")
    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] > button {
            background-color: #FF4B4B;
            border-color: #FF4B4B;
            color: #FFFFFF;
            font-weight: 800;
            min-height: 2.65rem;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #FF2B2B;
            border-color: #FF2B2B;
            color: #FFFFFF;
        }
        div[data-testid="stDownloadButton"] > button:focus:not(:active) {
            border-color: #FF4B4B;
            color: #FFFFFF;
        }
        .report-section-copy {
            margin-top: 1.1rem;
            margin-bottom: 0.35rem;
            color: #64748B;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    report_athletes = get_all_pitchers()
    if not report_athletes:
        st.info("No athletes found in the database.")
    else:
        reference_metadata = get_report_reference_take_metadata()
        report_col, date_col = st.columns([1.3, 2.7])
        with report_col:
            report_athlete = st.selectbox("Athlete", options=report_athletes, key="report_athlete")
        report_dates = metadata_session_dates(reference_metadata, report_athlete)
        with date_col:
            report_selected_dates = st.multiselect(
                "Session Date",
                options=report_dates,
                default=report_dates,
                key="report_session_dates",
            ) if report_dates else []

        report_throw_col, report_velocity_col = st.columns([1.3, 2.7])
        with report_throw_col:
            report_throw_types = st.multiselect(
                "Throw Type",
                options=["Mound", "Pulldown"],
                default=["Mound"],
                key="report_throw_types",
            )
            if not report_throw_types:
                report_throw_types = ["Mound"]
        report_base_rows = filter_reference_metadata(
            reference_metadata,
            pitchers=[report_athlete],
            session_dates=report_selected_dates,
            throw_types=report_throw_types,
        )
        report_velocity_bounds = metadata_velocity_bounds(report_base_rows)
        with report_velocity_col:
            if report_velocity_bounds[0] is not None and report_velocity_bounds[1] is not None:
                report_velocity_min = float(report_velocity_bounds[0])
                report_velocity_max = float(report_velocity_bounds[1])
                if report_velocity_min == report_velocity_max:
                    report_velocity_range = (report_velocity_min, report_velocity_max)
                    st.caption(f"Velocity Range (mph): {report_velocity_min:.1f}")
                else:
                    report_velocity_range = st.slider(
                        "Velocity Range (mph)",
                        min_value=report_velocity_min,
                        max_value=report_velocity_max,
                        value=(report_velocity_min, report_velocity_max),
                        step=0.1,
                        key="report_velocity_range",
                    )
            else:
                report_velocity_range = (None, None)
                st.caption("Velocity data is not available for this athlete.")

        report_take_rows = filter_reference_metadata(
            reference_metadata,
            pitchers=[report_athlete],
            session_dates=report_selected_dates,
            throw_types=report_throw_types,
            velocity_range=report_velocity_range,
        )
        report_take_options, report_label_to_take_id = build_take_options_from_metadata(report_take_rows)
        report_excluded_labels = st.multiselect(
            "Exclude Takes",
            options=report_take_options,
            key="report_excluded_takes",
        )
        report_excluded_take_ids = [
            report_label_to_take_id[label]
            for label in report_excluded_labels
            if label in report_label_to_take_id
        ]

        st.markdown("### Reference Group")
        reference_pitcher_col, reference_hand_col = st.columns([2.8, 1.2])

        def reset_report_reference_pitchers():
            st.session_state["report_reference_pitchers"] = ["All"]

        if "report_reference_pitchers" not in st.session_state:
            st.session_state["report_reference_pitchers"] = ["All"]
        reference_handedness_label = st.session_state.get("report_reference_handedness", "All")
        reference_handedness = {"All": None, "RHP": "R", "LHP": "L"}[reference_handedness_label]
        reference_throw_types_state = st.session_state.get("report_reference_throw_types", ["Mound"]) or ["Mound"]
        reference_velocity_state = st.session_state.get("report_reference_velocity_range", (None, None))
        reference_arm_slots_state = st.session_state.get("report_reference_arm_slots", ["All"]) or ["All"]
        arm_slot_metadata_available_state = any(row.get("arm_slot_deg") is not None for row in reference_metadata)
        reference_arm_slot_ranges_state = reference_arm_slot_ranges_from_labels(
            reference_arm_slots_state,
            arm_slot_metadata_available_state,
        )
        reference_pitcher_option_rows = filter_reference_metadata(
            reference_metadata,
            handedness=reference_handedness,
            throw_types=reference_throw_types_state,
            velocity_range=reference_velocity_state,
            arm_slot_ranges=reference_arm_slot_ranges_state,
        )
        reference_pitcher_options = metadata_pitchers(reference_pitcher_option_rows)
        if not reference_pitcher_options:
            reference_pitcher_options = metadata_pitchers(filter_reference_metadata(reference_metadata, handedness=reference_handedness))
        valid_reference_pitcher_values = set(["All"] + reference_pitcher_options)
        current_reference_pitchers = st.session_state.get("report_reference_pitchers", ["All"])
        if current_reference_pitchers and "All" not in current_reference_pitchers:
            cleaned_reference_pitchers = [
                pitcher for pitcher in current_reference_pitchers
                if pitcher in valid_reference_pitcher_values
            ]
            if cleaned_reference_pitchers != current_reference_pitchers:
                st.session_state["report_reference_pitchers"] = cleaned_reference_pitchers or ["All"]
        with reference_pitcher_col:
            reference_selected_pitchers = st.multiselect(
                "Select Pitcher(s)",
                options=["All"] + reference_pitcher_options,
                key="report_reference_pitchers",
            )
        with reference_hand_col:
            reference_handedness_label = st.selectbox(
                "Handedness",
                options=["All", "RHP", "LHP"],
                index=0,
                key="report_reference_handedness",
                on_change=reset_report_reference_pitchers,
            )
        reference_handedness = {"All": None, "RHP": "R", "LHP": "L"}[reference_handedness_label]
        reference_uses_all_pitchers = not reference_selected_pitchers or "All" in reference_selected_pitchers
        active_reference_pitchers = (
            reference_pitcher_options
            if reference_uses_all_pitchers else
            [pitcher for pitcher in reference_selected_pitchers if pitcher in reference_pitcher_options]
        )

        active_reference_metadata = filter_reference_metadata(
            reference_metadata,
            pitchers=active_reference_pitchers,
            handedness=reference_handedness,
            throw_types=reference_throw_types_state,
        )
        arm_slot_metadata_available = any(row.get("arm_slot_deg") is not None for row in active_reference_metadata)
        reference_velocity_bounds = metadata_velocity_bounds(active_reference_metadata)
        reference_throw_col, reference_velocity_col, reference_arm_slot_col = st.columns([1.2, 1.7, 2.1])
        with reference_throw_col:
            reference_throw_types = st.multiselect(
                "Throw Type",
                options=["Mound", "Pulldown"],
                default=["Mound"],
                key="report_reference_throw_types",
            )
            if not reference_throw_types:
                reference_throw_types = ["Mound"]
        with reference_velocity_col:
            if reference_velocity_bounds[0] is not None and reference_velocity_bounds[1] is not None:
                reference_min_velocity = float(reference_velocity_bounds[0])
                reference_max_velocity = float(reference_velocity_bounds[1])
                if reference_min_velocity == reference_max_velocity:
                    reference_velocity_range = (reference_min_velocity, reference_max_velocity)
                    st.caption(f"Velocity Range (mph): {reference_min_velocity:.1f}")
                else:
                    velocity_slider_kwargs = {}
                    current_reference_velocity_range = st.session_state.get(
                        "report_reference_velocity_range",
                        (reference_min_velocity, reference_max_velocity),
                    )
                    clamped_reference_velocity_range = (
                        max(reference_min_velocity, float(current_reference_velocity_range[0])),
                        min(reference_max_velocity, float(current_reference_velocity_range[1])),
                    )
                    if clamped_reference_velocity_range[0] > clamped_reference_velocity_range[1]:
                        clamped_reference_velocity_range = (reference_min_velocity, reference_max_velocity)
                    if "report_reference_velocity_range" in st.session_state:
                        if tuple(st.session_state["report_reference_velocity_range"]) != clamped_reference_velocity_range:
                            st.session_state["report_reference_velocity_range"] = clamped_reference_velocity_range
                    else:
                        velocity_slider_kwargs["value"] = clamped_reference_velocity_range
                    reference_velocity_range = st.slider(
                        "Velocity Range (mph)",
                        min_value=reference_min_velocity,
                        max_value=reference_max_velocity,
                        step=0.1,
                        key="report_reference_velocity_range",
                        **velocity_slider_kwargs,
                    )
            else:
                reference_velocity_range = (None, None)
                st.caption("No velocity data found for this reference group.")
        reference_arm_slot_options = [
            f"{label} ({minimum}° to {maximum}°)"
            for label, minimum, maximum in control_group_arm_slot_categories
        ]
        with reference_arm_slot_col:
            reference_arm_slots = st.multiselect(
                "Arm Slot",
                options=["All"] + reference_arm_slot_options,
                default=["All"],
                key="report_reference_arm_slots",
            )
            if not arm_slot_metadata_available:
                st.caption("Run the arm slot metadata backfill to enable this filter.")

        selected_arm_slot_ranges = reference_arm_slot_ranges_from_labels(
            reference_arm_slots,
            arm_slot_metadata_available,
        )
        current_reference_rows = filter_reference_metadata(
            reference_metadata,
            pitchers=active_reference_pitchers,
            handedness=reference_handedness,
            throw_types=reference_throw_types,
            velocity_range=reference_velocity_range,
            arm_slot_ranges=selected_arm_slot_ranges,
        )
        current_reference_pitchers = metadata_pitchers(current_reference_rows)
        if reference_uses_all_pitchers:
            active_reference_pitchers = current_reference_pitchers
        st.caption(
            f"All currently includes {len(current_reference_pitchers)} eligible pitcher"
            f"{'' if len(current_reference_pitchers) == 1 else 's'} and {len(current_reference_rows)} throws."
        )

        reference_pitcher_filters = {}
        if not reference_uses_all_pitchers:
            for reference_index, pitcher in enumerate(active_reference_pitchers):
                with st.expander(f"{pitcher} Filters", expanded=True):
                    pitcher_dates = metadata_session_dates(reference_metadata, pitcher)
                    pitcher_selected_dates = st.multiselect(
                        "Session Date",
                        options=pitcher_dates,
                        default=pitcher_dates,
                        key=f"report_reference_dates_{reference_index}",
                    )
                    pitcher_throw_types = st.multiselect(
                        "Throw Type",
                        options=["Mound", "Pulldown"],
                        default=["Mound"],
                        key=f"report_reference_throw_types_{reference_index}",
                    )
                    if not pitcher_throw_types:
                        pitcher_throw_types = ["Mound"]
                    pitcher_base_rows = filter_reference_metadata(
                        reference_metadata,
                        pitchers=[pitcher],
                        handedness=reference_handedness,
                        session_dates=pitcher_selected_dates,
                        throw_types=pitcher_throw_types,
                    )
                    pitcher_min_velocity, pitcher_max_velocity = metadata_velocity_bounds(pitcher_base_rows)
                    if pitcher_min_velocity is not None and pitcher_max_velocity is not None:
                        pitcher_min_velocity = float(pitcher_min_velocity)
                        pitcher_max_velocity = float(pitcher_max_velocity)
                        if pitcher_min_velocity == pitcher_max_velocity:
                            pitcher_velocity_range = (pitcher_min_velocity, pitcher_max_velocity)
                            st.caption(f"Velocity Range (mph): {pitcher_min_velocity:.1f}")
                        else:
                            pitcher_velocity_range = st.slider(
                                "Velocity Range (mph)",
                                min_value=pitcher_min_velocity,
                                max_value=pitcher_max_velocity,
                                value=(pitcher_min_velocity, pitcher_max_velocity),
                                step=0.1,
                                key=f"report_reference_velocity_range_{reference_index}",
                            )
                    else:
                        pitcher_velocity_range = (None, None)
                    pitcher_cfg = {
                        "selected_dates": pitcher_selected_dates,
                        "throw_types": pitcher_throw_types,
                        "velocity_min": pitcher_velocity_range[0],
                        "velocity_max": pitcher_velocity_range[1],
                    }
                    pitcher_take_rows = filter_reference_metadata(
                        reference_metadata,
                        pitchers=[pitcher],
                        handedness=reference_handedness,
                        session_dates=pitcher_selected_dates,
                        throw_types=pitcher_throw_types,
                        velocity_range=pitcher_velocity_range,
                    )
                    pitcher_take_options, pitcher_label_to_take_id = build_take_options_from_metadata(pitcher_take_rows)
                    pitcher_excluded_labels = st.multiselect(
                        "Exclude Takes",
                        options=pitcher_take_options,
                        key=f"report_reference_excluded_takes_{reference_index}",
                    )
                    pitcher_cfg["excluded_take_ids"] = [
                        pitcher_label_to_take_id[label]
                        for label in pitcher_excluded_labels
                        if label in pitcher_label_to_take_id
                    ]
                    reference_pitcher_filters[pitcher] = pitcher_cfg

        apply_reference_group = st.button(
            "Apply Reference Group",
            key="apply_report_reference_group",
            type="primary",
        )
        if apply_reference_group:
            with st.spinner("Building reference group..."):
                selected_arm_slot_ranges = [
                    (minimum, maximum)
                    for label, minimum, maximum in control_group_arm_slot_categories
                    if f"{label} ({minimum}° to {maximum}°)" in reference_arm_slots
                ] if arm_slot_metadata_available and "All" not in reference_arm_slots else []
                specific_reference_take_ids = None
                if reference_pitcher_filters:
                    specific_reference_take_ids = set()
                    for pitcher, cfg in reference_pitcher_filters.items():
                        filtered_metadata = filter_reference_metadata(
                            reference_metadata,
                            pitchers=[pitcher],
                            handedness=reference_handedness,
                            session_dates=cfg.get("selected_dates", []),
                            throw_types=cfg.get("throw_types", []),
                            velocity_range=(cfg.get("velocity_min"), cfg.get("velocity_max")),
                            excluded_take_ids=cfg.get("excluded_take_ids", []),
                        )
                        specific_reference_take_ids.update(row["take_id"] for row in filtered_metadata)
                applied_reference_rows = filter_reference_metadata(
                    reference_metadata,
                    pitchers=active_reference_pitchers,
                    handedness=reference_handedness,
                    throw_types=reference_throw_types,
                    velocity_range=reference_velocity_range,
                    arm_slot_ranges=selected_arm_slot_ranges,
                )
                st.session_state["report_reference_take_ids"] = [
                    row["take_id"]
                    for row in applied_reference_rows
                    if specific_reference_take_ids is None or row["take_id"] in specific_reference_take_ids
                ]
        reference_take_ids = st.session_state.get("report_reference_take_ids", [])
        if reference_take_ids:
            st.caption(f"Applied Reference Group: {len(reference_take_ids)} throws")
        else:
            st.caption("Choose filters, then click Apply Reference Group.")

        st.markdown("### Report Sections")
        st.markdown(
            '<div class="report-section-copy">Choose the pages to include in this PDF. The cover page is always included.</div>',
            unsafe_allow_html=True,
        )
        section_defaults = {
            "kinematic_sequence": True,
            "throwing_arm": True,
            "pelvis": True,
            "torso": True,
            "torso_pelvis": True,
            "cog": True,
            "stride": True,
            "hips": False,
            "back_hip": False,
            "lead_hip": False,
            "lower_extremity": False,
        }
        section_labels = [
            ("kinematic_sequence", "Kinematic Sequence"),
            ("throwing_arm", "Throwing Arm Kinematics"),
            ("pelvis", "Pelvis Kinematics"),
            ("torso", "Torso Kinematics"),
            ("torso_pelvis", "Torso-Pelvis Kinematics"),
            ("hips", "Hips Kinematics"),
            ("back_hip", "Back Hip Kinematics"),
            ("lead_hip", "Lead Hip Kinematics"),
            ("lower_extremity", "Lower Extremity Kinematics"),
            ("cog", "COG Velocities"),
            ("stride", "Stride Kinematics"),
        ]
        selected_report_sections = []
        section_cols = st.columns([1.35, 1.25, 1.15, 1.15])
        for idx, (section_key, section_label) in enumerate(section_labels):
            with section_cols[idx % len(section_cols)]:
                if st.checkbox(
                    section_label,
                    value=section_defaults[section_key],
                    key=f"report_section_{section_key}",
                ):
                    selected_report_sections.append(section_key)

        if not report_selected_dates:
            st.info("No session dates found for this athlete.")
        else:
            report_rows = get_report_take_rows(
                report_athlete,
                report_selected_dates,
                report_throw_types,
                report_velocity_range,
                report_excluded_take_ids,
            )
            if not report_rows:
                st.info("No throws found for this athlete and filter selection.")
            else:
                summary_rows, curves_by_segment, report_events = build_report_kinematic_summary(report_rows)
                if not summary_rows:
                    st.info("No Kinematic Sequence data found for this report selection.")
                elif not selected_report_sections:
                    st.warning("Select at least one report section before generating the PDF.")
                else:
                    report_date_label = ", ".join(report_selected_dates)
                    reference_report_rows = get_report_take_rows_by_ids(reference_take_ids)
                    reference_arm_slot_label = (
                        "All"
                        if not reference_arm_slots or "All" in reference_arm_slots
                        else format_report_list_label([
                            slot.split(" (", 1)[0]
                            for slot in reference_arm_slots
                        ])
                    )
                    report_context = {
                        "velocity_range_label": format_report_velocity_range(report_velocity_range),
                        "reference_athletes_label": "All" if reference_uses_all_pitchers else format_report_list_label(active_reference_pitchers),
                        "reference_handedness_label": reference_handedness_label,
                        "reference_velocity_range_label": format_report_velocity_range(reference_velocity_range),
                        "reference_arm_slot_label": reference_arm_slot_label,
                        "reference_take_count": len(reference_take_ids),
                    }
                    arm_kinematics = build_report_arm_kinematics(report_rows, reference_report_rows)
                    pdf_bytes = build_report_pdf(
                        report_athlete,
                        report_date_label,
                        summary_rows,
                        curves_by_segment,
                        report_events,
                        arm_kinematics,
                        report_context,
                        selected_report_sections,
                    )
                    generate_col, _, _ = st.columns([0.95, 1.3, 1.3])
                    with generate_col:
                        st.download_button(
                            "Generate PDF Report",
                            data=pdf_bytes,
                            file_name=f"{report_athlete.lower().replace(' ', '_')}_mocap_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary",
                        )


with tab_energy:
    st.subheader("Energy Flow")
    render_group_selection_summary()

    st.markdown(
        """
        <style>
        .energy-controls-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }

        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] label,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] p {
            font-size: 1rem !important;
            font-weight: 400 !important;
        }

        .energy-toggle-label {
            margin-top: -0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    energy_display_col, energy_options_col, energy_spacer_col = st.columns([1.45, 1.75, 2.2])
    with energy_display_col:
        st.markdown('<div class="energy-controls-label">Display Mode</div>', unsafe_allow_html=True)
        display_mode = st.segmented_control(
            "Select Display Mode",
            ["Individual Throws", "Grouped"],
            default="Grouped",
            key="energy_display_mode",
            label_visibility="collapsed",
        )
    with energy_options_col:
        st.markdown('<div class="energy-controls-label energy-toggle-label">Options</div>', unsafe_allow_html=True)
        energy_event_col, energy_signal_col = st.columns(2)
        with energy_event_col:
            show_energy_fp_iqr_band = st.toggle(
                "Event Bands",
                value=False,
                key="energy_show_fp_iqr_band",
                help="Shows the middle 50% range for event timing across selected throws.",
            )
        with energy_signal_col:
            show_energy_signal_iqr_band = st.toggle(
                "Signal Bands",
                value=True,
                key="energy_show_signal_iqr_band",
                help="Shows the middle 50% range around each grouped mean line.",
            )
    with energy_spacer_col:
        st.markdown("")

    energy_window_col, energy_window_spacer = st.columns([2.35, 3.65])
    with energy_window_col:
        st.markdown('<div class="energy-controls-label">View Window</div>', unsafe_allow_html=True)
        energy_window_mode = st.segmented_control(
            "Energy Flow View",
            ["Peak Knee Height View", "Foot Plant to Ball Release View"],
            default="Peak Knee Height View",
            key="energy_window_mode",
            label_visibility="collapsed",
        )
    with energy_window_spacer:
        st.markdown("")

    energy_select_col, energy_select_spacer = st.columns([3, 3])
    with energy_select_col:
        energy_metrics = st.multiselect(
            "Select Energy Flow Metrics",
            [
                "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)",
                "Arm Energy Flow (LAR_PROX | RAR_PROX)",
                "Glove Side Trunk-Shoulder Energy Flow",
                "Glove Arm Energy Flow",
                "Trunk-Shoulder Rotational Energy Flow",
                "Trunk-Shoulder Elevation/Depression Energy Flow",
                "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                "Arm Rotational Energy Flow",
                "Arm Elevation/Depression Energy Flow",
                "Arm Horizontal Abd/Add Energy Flow",
                "Throwing Shoulder Rotational Torque (Relative to Trunk)",
                *NEW_TRUNK_PELVIS_ENERGY_METRICS,
            ],
            default=[]
        )
    with energy_select_spacer:
        st.markdown("")

    if not energy_metrics:
        energy_empty_col, energy_empty_spacer = st.columns([3, 3])
        with energy_empty_col:
            st.info("Select at least one energy flow metric.")
        with energy_empty_spacer:
            st.markdown("")
        st.stop()

    if not take_ids:
        st.info("No takes available for Energy Flow.")
        st.stop()

    # --- Fixed color map for Energy Flow metrics (high-contrast palette) ---
    energy_color_map = {
        "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)": "#4C1D95",  # deep indigo / purple
        "Arm Energy Flow (LAR_PROX | RAR_PROX)": "#7C2D12",  # dark brown
        "Glove Side Trunk-Shoulder Energy Flow": "#E11D48",
        "Glove Arm Energy Flow": "#14B8A6",
        "Trunk-Shoulder Rotational Energy Flow": "#DC2626",  # strong red
        "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",  # vivid blue
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",     # strong green
        "Arm Rotational Energy Flow": "#F59E0B",        # amber
        "Arm Elevation/Depression Energy Flow": "#06B6D4",  # cyan
        "Arm Horizontal Abd/Add Energy Flow": "#9333EA",     # violet
        "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00",
        **NEW_TRUNK_PELVIS_ENERGY_COLOR_MAP,
    }

    # --- Load all selected metrics ---
    energy_data_by_metric = {}

    def load_energy_by_handedness(loader_fn):
        merged = {}
        for hand, ids in take_ids_by_handedness.items():
            if ids:
                merged.update(loader_fn(ids, hand))
        return merged

    for metric in energy_metrics:
        if metric == "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_distal_arm_segment_power)
        elif metric == "Arm Energy Flow (LAR_PROX | RAR_PROX)":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_proximal_energy_transfer)
        elif metric == "Glove Side Trunk-Shoulder Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_glove_side_trunk_shoulder_energy_flow)
        elif metric == "Glove Arm Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_glove_arm_energy_flow)
        elif metric == "Trunk-Shoulder Rotational Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_rot_energy_flow)
        elif metric == "Trunk-Shoulder Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_elev_energy_flow)
        elif metric == "Trunk-Shoulder Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_trunk_shoulder_horizabd_energy_flow)
        elif metric == "Arm Rotational Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_rot_energy_flow)
        elif metric == "Arm Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_elev_energy_flow)
        elif metric == "Arm Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_horizabd_energy_flow)
        elif metric == "Throwing Shoulder Rotational Torque (Relative to Trunk)":
            mmt_data = {}
            if take_ids_by_handedness.get("R"):
                mmt_data.update(
                    get_energy_flow_from_segment(
                        take_ids_by_handedness["R"],
                        "RT_SHOULDER_RTA_MMT",
                        component="z"
                    )
                )
            if take_ids_by_handedness.get("L"):
                mmt_data.update(
                    get_energy_flow_from_segment(
                        take_ids_by_handedness["L"],
                        "LT_SHOULDER_RTA_MMT",
                        component="z"
                    )
                )
            energy_data_by_metric[metric] = mmt_data
        elif metric in NEW_TRUNK_PELVIS_ENERGY_METRICS:
            segment_name, category_name = NEW_TRUNK_PELVIS_ENERGY_METRIC_MAP[metric]
            energy_data_by_metric[metric] = get_energy_flow_from_category_segment(
                take_ids,
                category_name,
                segment_name,
                component="x",
            )

    energy_data_by_metric = {
        k: v for k, v in energy_data_by_metric.items() if v
    }

    if not energy_data_by_metric:
        st.warning("No energy flow data found for the selected metrics.")
        st.stop()

    fig = go.Figure()

    # --- Date dash styles (same as KS) ---
    unique_dates = sorted(set(take_date_map.values()))
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    date_dash_map = {
        d: dash_styles[i % len(dash_styles)]
        for i, d in enumerate(unique_dates)
    }

    legend_keys_added = set()
    energy_median_pkh_frame = None
    if mound_only_sidebar and knee_event_frames:
        energy_median_pkh_frame = int(np.median(knee_event_frames))

    if energy_window_mode == "Foot Plant to Ball Release View":
        energy_median_fp_frame = int(np.median(fp_event_frames)) if fp_event_frames else None
        energy_window_start = (
            energy_median_fp_frame - 25
            if energy_median_fp_frame is not None else
            window_start
        )
        energy_window_end = 25
    else:
        energy_window_start = window_start
        energy_window_end = 50
        if energy_median_pkh_frame is not None:
            energy_window_start = min(window_start, energy_median_pkh_frame - 20)

    energy_window_start_ms = rel_frame_to_ms(energy_window_start)
    energy_window_end_ms = rel_frame_to_ms(energy_window_end)
    use_group_colors_energy = (
        comparison_grouping_enabled
        and len(energy_metrics) == 1
        and len(group_color_map) >= 2
    )

    # -------------------------------
    # Normalize to Ball Release and Plot
    # -------------------------------
    for metric, energy_data in energy_data_by_metric.items():
        metric_color = energy_color_map.get(metric, "#444")

        grouped_power = {}
        grouped_by_date = {}

        for take_id, d in energy_data.items():
            if take_id not in br_frames:
                continue

            frames = d["frame"]
            values = d["value"]
            br = br_frames[take_id]

            norm_f, norm_v = [], []
            for f, v in zip(frames, values):
                rel = f - br
                if energy_window_start <= rel <= energy_window_end:
                    norm_f.append(rel_frame_to_ms(rel))
                    norm_v.append(v)

            grouped_power[take_id] = {"frame": norm_f, "value": norm_v}

            date = take_date_map[take_id]
            group_label = take_group_map.get(take_id, "Ungrouped")
            pitcher_name = take_pitcher_map.get(take_id, "")
            control_group_take = is_control_group_label(group_label)
            hover_pitcher_name = "" if control_group_take else pitcher_name
            if comparison_grouping_enabled and control_group_take:
                date_key = group_label
            elif comparison_grouping_enabled:
                date_key = group_label if group_mode_aggregate_across_pitchers else ((group_label, pitcher_name, date) if multi_pitcher_mode else (group_label, date))
            else:
                date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }
            trace_color = (
                group_color_map.get(group_label, metric_color)
                if use_group_colors_energy else
                metric_color
            )

            if display_mode == "Individual Throws":
                legendgroup = (
                    f"{group_label}_{metric}_{pitcher_name}_{date}"
                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                    f"{group_label}_{metric}_{date}"
                    if comparison_grouping_enabled else
                    f"{metric}_{pitcher_name}_{date}"
                    if show_group_pitcher_breakout else
                    f"{metric}_{date}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        line=dict(
                            color=trace_color,
                            dash=date_dash_map[date]
                        ),
                        customdata=[[metric, date, take_order[take_id], take_velocity[take_id], hover_pitcher_name]] * len(norm_f),
                        hovertemplate=(
                            ("%{customdata[4]} | %{customdata[1]}" if show_group_pitcher_breakout else "%{customdata[1]}")
                            + "<br>%{customdata[0]}: %{y:.1f}"
                            + "<br>Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        name=(
                            f"Control Group | {metric} – Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if (comparison_grouping_enabled and control_group_take) else
                            f"{group_label} | {metric} – {date} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                            f"{group_label} | {metric} – {date} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
                            if comparison_grouping_enabled else None
                        ),
                        showlegend=False,
                        legendgroup=legendgroup
                    )
                )
                legend_key = (metric, date_key)
                if control_group_take and legend_key not in legend_keys_added:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(
                                color=trace_color,
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"Control Group | {metric}"
                                if (comparison_grouping_enabled and control_group_take) else
                                f"{group_label} | {metric} | {date} | {pitcher_name}"
                                if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                                f"{group_label} | {metric} | {date}"
                                if comparison_grouping_enabled else
                            f"{metric} | {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{metric} | {date}"
                        ),
                            showlegend=True,
                            legendgroup=legendgroup
                        )
                    )
                    legend_keys_added.add(legend_key)

        # -------------------------------
        # Grouped (Mean + IQR per date)
        # -------------------------------
        if display_mode == "Grouped":
            for date_key, curves in grouped_by_date.items():
                if comparison_grouping_enabled and date_key == "Control Group":
                    group_label = "Control Group"
                    pitcher_name = ""
                    date = "Selected Takes"
                elif comparison_grouping_enabled and show_group_pitcher_breakout:
                    group_label, pitcher_name, date = date_key
                elif comparison_grouping_enabled:
                    group_label = date_key
                    date = "Selected Takes"
                    pitcher_name = ""
                elif multi_pitcher_mode and not comparison_grouping_enabled:
                    pitcher_name, date = date_key
                    group_label = ""
                else:
                    date = date_key
                    pitcher_name = ""
                    group_label = ""
                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                avg_velocity = np.mean([take_velocity[tid] for tid in curves.keys()])
                legendgroup = (
                    f"{group_label}_{metric}_{pitcher_name}_{date}"
                    if (comparison_grouping_enabled and show_group_pitcher_breakout) else
                    f"{group_label}_{metric}_{date}"
                    if comparison_grouping_enabled else
                    f"{metric}_{pitcher_name}_{date}"
                    if show_group_pitcher_breakout else
                    f"{metric}_{date}"
                )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(
                            width=4,
                            color=(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color
                            ),
                            dash=date_dash_map.get(date, "solid")
                        ),
                        customdata=[[metric, date, group_label, pitcher_name]] * len(x),
                        hovertemplate=(
                            (f"{group_label}<br>" if comparison_grouping_enabled else "")
                            + ("%{customdata[0]}" if comparison_grouping_enabled else "%{customdata[3]} | %{customdata[1]}" if show_group_pitcher_breakout else "%{customdata[1]}")
                            + (" | %{customdata[3]}" if show_group_pitcher_breakout and comparison_grouping_enabled else "")
                            + (f"<br>Avg Velocity: {avg_velocity:.1f} mph" if avg_velocity is not None else "")
                            + "<br>%{customdata[0]}: %{y:.1f}"
                            + "<br>Time: %{x:.0f} ms rel BR"
                            + "<extra></extra>"
                        ),
                        showlegend=False,
                        legendgroup=legendgroup
                    )
                )

                if show_energy_signal_iqr_band:
                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=q3 + q1[::-1],
                            fill="toself",
                            fillcolor=to_rgba(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color,
                                alpha=0.35
                            ),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                            legendgroup=legendgroup
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(
                            color=(
                                group_color_map.get(group_label, metric_color)
                                if use_group_colors_energy else
                                metric_color
                            ),
                            dash=date_dash_map.get(date, "solid"),
                            width=4
                        ),
                        name=(
                            f"{group_label} | {metric} | {date} | {pitcher_name}"
                            if (show_group_pitcher_breakout and comparison_grouping_enabled) else
                            f"{group_label} | {metric} | {date}"
                            if comparison_grouping_enabled else
                            f"{metric} | {date} | {pitcher_name}"
                            if show_group_pitcher_breakout else
                            f"{metric} | {date}"
                        ),
                        showlegend=True,
                        legendgroup=legendgroup
                    )
                )

    # -------------------------------
    # Event Lines (with text labels above)
    # -------------------------------
    if energy_median_pkh_frame is not None:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_energy_fp_iqr_band)
        median_pkh = rel_frame_to_ms(energy_median_pkh_frame)
        fig.add_vline(x=median_pkh, line_width=3, line_dash="dash", line_color="gold")
        fig.add_annotation(
            x=median_pkh,
            y=1.06,
            xref="x",
            yref="paper",
            text="PKH",
            showarrow=False,
            font=dict(color="gold", size=13, family="Arial"),
            align="center"
        )
    elif knee_event_frames:
        add_event_iqr_band(fig, knee_event_frames, "gold", show_energy_fp_iqr_band)
        median_knee = rel_frame_to_ms(int(np.median(knee_event_frames)))
        fig.add_vline(x=median_knee, line_width=3, line_dash="dash", line_color="gold")
        fig.add_annotation(
            x=median_knee,
            y=1.06,
            xref="x",
            yref="paper",
            text="Knee",
            showarrow=False,
            font=dict(color="gold", size=13, family="Arial"),
            align="center"
        )

    if fp_event_frames:
        add_event_iqr_band(fig, fp_event_frames, "green", show_energy_fp_iqr_band)
        median_fp = rel_frame_to_ms(int(np.median(fp_event_frames)))
        fig.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=median_fp,
            y=1.06,
            xref="x",
            yref="paper",
            text="FP",
            showarrow=False,
            font=dict(color="green", size=13, family="Arial"),
            align="center"
        )

    if mer_event_frames:
        add_event_iqr_band(fig, mer_event_frames, "red", show_energy_fp_iqr_band)
        median_mer = rel_frame_to_ms(int(np.median(mer_event_frames)))
        fig.add_vline(x=median_mer, line_width=3, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=median_mer,
            y=1.06,
            xref="x",
            yref="paper",
            text="MER",
            showarrow=False,
            font=dict(color="red", size=13, family="Arial"),
            align="center"
        )

    add_event_iqr_band(fig, [0] * max(len(take_ids), 1), "blue", show_energy_fp_iqr_band)
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")
    fig.add_annotation(
        x=0,
        y=1.06,
        xref="x",
        yref="paper",
        text="BR",
        showarrow=False,
        font=dict(color="blue", size=13, family="Arial"),
        align="center"
    )

    fig.update_layout(
        xaxis_title="Time Relative to Ball Release (ms)",
        yaxis_title=get_energy_yaxis_title(energy_data_by_metric.keys()),
        xaxis_range=[energy_window_start_ms, energy_window_end_ms],
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5,
            groupclick="togglegroup"
        ),
        hoverlabel=dict(
            namelength=-1,
            font_size=13
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="energy_plot_main_tab")

    defined_energy_metrics = [
        metric for metric in energy_metrics if metric in energy_definitions
    ]
    if defined_energy_metrics:
        st.markdown("### Energy Flow Definitions")
        for metric in defined_energy_metrics:
            metric_info = energy_definitions.get(metric, {})
            st.markdown(
                (
                    f"<div style='font-size:1.15rem; line-height:1.6; margin:0.35rem 0 0.9rem 0;'>"
                    f"<strong>{metric}:</strong> {metric_info.get('definition', '')}"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("© Terra Sports | Biomechanics Dashboard")
