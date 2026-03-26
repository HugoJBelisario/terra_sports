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

@st.cache_data(ttl=300)
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
if "control_group_handedness" not in st.session_state:
    st.session_state["control_group_handedness"] = "Both"
if "control_group_arm_slot_ids" not in st.session_state:
    st.session_state["control_group_arm_slot_ids"] = []
if "control_group_pitchers" not in st.session_state:
    st.session_state["control_group_pitchers"] = []
if "control_group_velocity_range" not in st.session_state:
    st.session_state["control_group_velocity_range"] = (50.0, 100.0)
if "control_group_arm_slot_range" not in st.session_state:
    st.session_state["control_group_arm_slot_range"] = (0.0, 120.0)
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
            velocity_range_i = st.sidebar.slider(
                velocity_label,
                min_value=float(vel_min_i),
                max_value=float(vel_max_i),
                value=(float(vel_min_i), float(vel_max_i)),
                step=0.5,
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
    st.session_state["control_group_arm_slot_ids"] = []
    st.session_state["control_group_pitchers"] = []
    st.session_state["control_group_handedness"] = "Both"
    st.session_state["control_group_status_message"] = ""
    for key in ["control_group_velocity_range", "control_group_arm_slot_range"]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

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
                        step=0.5,
                        key="control_group_velocity_range"
                    )
                    st.slider(
                        "Arm Slot",
                        min_value=0.0,
                        max_value=120.0,
                        value=st.session_state.get("control_group_arm_slot_range", (0.0, 120.0)),
                        step=0.5,
                        key="control_group_arm_slot_range"
                    )
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
                    selected_arm_slot_range = st.session_state.get("control_group_arm_slot_range", (0.0, 120.0))

                    final_candidate_control_take_ids = []
                    for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                        if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                            continue
                        if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                            continue
                        if arm_slot_deg is None or not (selected_arm_slot_range[0] <= float(arm_slot_deg) <= selected_arm_slot_range[1]):
                            continue
                        final_candidate_control_take_ids.append(take_id)

                    st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                    st.session_state["control_group_status_message"] = (
                        f"Current control group: {len(final_candidate_control_take_ids)} takes"
                        if final_candidate_control_take_ids else
                        "No control-group takes found for the selected filters."
                    )
                    st.rerun()

                if st.session_state.get("control_group_status_message"):
                    st.sidebar.caption(st.session_state["control_group_status_message"])
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
                    step=0.5,
                    key="control_group_velocity_range"
                )
                st.slider(
                    "Arm Slot",
                    min_value=0.0,
                    max_value=120.0,
                    value=st.session_state.get("control_group_arm_slot_range", (0.0, 120.0)),
                    step=0.5,
                    key="control_group_arm_slot_range"
                )
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
                selected_arm_slot_range = st.session_state.get("control_group_arm_slot_range", (0.0, 120.0))

                final_candidate_control_take_ids = []
                for take_id, pitch_velo, athlete_name, _, arm_slot_deg in all_control_group_pool:
                    if selected_pitcher_set and athlete_name not in selected_pitcher_set:
                        continue
                    if pitch_velo is None or not (selected_velocity_range[0] <= float(pitch_velo) <= selected_velocity_range[1]):
                        continue
                    if arm_slot_deg is None or not (selected_arm_slot_range[0] <= float(arm_slot_deg) <= selected_arm_slot_range[1]):
                        continue
                    final_candidate_control_take_ids.append(take_id)

                st.session_state["control_group_take_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_arm_slot_ids"] = list(final_candidate_control_take_ids)
                st.session_state["control_group_status_message"] = (
                    f"Current control group: {len(final_candidate_control_take_ids)} takes"
                    if final_candidate_control_take_ids else
                    "No control-group takes found for the selected filters."
                )
                st.rerun()

            if st.session_state.get("control_group_status_message"):
                st.sidebar.caption(st.session_state["control_group_status_message"])

            primary_take_ids = list(shared_take_ids)
            control_take_ids = [
                tid for tid in st.session_state.get("control_group_take_ids", [])
                if tid not in primary_take_ids
            ]
            combined_take_ids = primary_take_ids + control_take_ids

            if control_take_ids and combined_take_ids:
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





st.title("Terra Sports Biomechanics Dashboard")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_labels = ["Kinematic Sequence", "Kinematics", "Energy Flow"]
tab_kinematic, tab_joint, tab_energy = st.tabs(tab_labels)

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
                    legendgroup = f"Torso_{take_date_map[take_id]}"
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
                    legend_key = (
                        ("Torso", take_date_map[take_id], pitcher_name)
                        if multi_pitcher_mode else
                        ("Torso", take_date_map[take_id])
                    )
                    if legend_key not in legend_keys_added:
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
                    legendgroup = f"Elbow_{take_date_map[take_id]}"
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
                    legend_key = (
                        ("Elbow", take_date_map[take_id], pitcher_name)
                        if multi_pitcher_mode else
                        ("Elbow", take_date_map[take_id])
                    )
                    if legend_key not in legend_keys_added:
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
                    legendgroup = f"Shoulder IR_{take_date_map[take_id]}"
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
                    legend_key = (
                        ("Shoulder IR", take_date_map[take_id], pitcher_name)
                        if multi_pitcher_mode else
                        ("Shoulder IR", take_date_map[take_id])
                    )
                    if legend_key not in legend_keys_added:
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
                legendgroup = f"Pelvis_{take_date_map[take_id]}"
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
                legend_key = (
                    ("Pelvis", take_date_map[take_id], pitcher_name)
                    if multi_pitcher_mode else
                    ("Pelvis", take_date_map[take_id])
                )
                if legend_key not in legend_keys_added:
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
                    color = color_map[label]
                    # Smoothing
                    if len(y_date) >= 11:
                        y_date = savgol_filter(y_date, window_length=7, polyorder=3)
                    dash = date_dash_map.get(date, "solid")
                    legendgroup = f"{label}_{date}_{pitcher_name}" if show_group_pitcher_breakout else f"{label}_{date}"
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
                                ("%{customdata[2]} | " if comparison_grouping_enabled else "")
                                + "%{customdata[0]} | %{customdata[1]}"
                                + (" | %{customdata[3]}" if show_group_pitcher_breakout else "")
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
                        peak_marker_y = max_y + max(0.10 * local_y_span, 120)

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
                                y=[peak_marker_y],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-down",
                                    size=18,
                                    color=color,
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
                    # --- IQR band (with legendgroup for toggleitem) ---
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

            for peak_marker_trace in peak_marker_traces:
                fig.add_trace(peak_marker_trace)

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
        "Elbow Flexion",
        "Hand Speed",
        "Center of Mass Velocity (X)",
        "Shoulder ER",
        "Shoulder Internal Rotation Velocity",
        "Shoulder Abduction",
        "Shoulder Horizontal Abduction",
        "Front Knee Flexion",
        "Front Knee Extension Velocity",
        "Forward Trunk Tilt",
        "Lateral Trunk Tilt",
        "Trunk Angle",
        "Pelvis Angle",
        "Pelvic Lateral Tilt",
        "Hip-Shoulder Separation",
        "Pelvis Rotational Velocity",
        "Trunk Rotational Velocity",
        "Torso-Pelvis Rotational Velocity",
        "Forearm Pronation/Supination"
    ]

    kinematic_definitions = {
        "Elbow Flexion": {
            "definition": "How bent the throwing elbow is throughout the motion.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "Helps describe arm path and timing during arm acceleration.",
            "interpretation": "Higher values generally mean a more flexed elbow position."
        },
        "Hand Speed": {
            "definition": "Linear speed of the throwing hand over time.",
            "measured_as": "Speed in meters per second (m/s).",
            "why": "Relates to how quickly force is transferred to the hand.",
            "interpretation": "Higher peak hand speed is usually associated with faster throws."
        },
        "Center of Mass Velocity (X)": {
            "definition": "Forward/backward velocity of the body center of mass.",
            "measured_as": "Velocity along the X axis in m/s.",
            "why": "Captures how efficiently momentum moves toward the plate.",
            "interpretation": "More positive forward values indicate stronger forward drive."
        },
        "Shoulder ER": {
            "definition": "External rotation angle of the shoulder as the arm cocks back.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "A key marker of arm-cocking position and timing.",
            "interpretation": "Larger peaks indicate greater external rotation range."
        },
        "Shoulder Internal Rotation Velocity": {
            "definition": "Speed of inward shoulder rotation during acceleration.",
            "measured_as": "Angular velocity in degrees per second (deg/s).",
            "why": "Reflects rapid arm acceleration demands near ball release.",
            "interpretation": "Higher peaks indicate faster inward rotational acceleration."
        },
        "Shoulder Abduction": {
            "definition": "How far the upper arm is elevated away from the torso.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "Provides context on arm slot and shoulder loading profile.",
            "interpretation": "Higher values generally mean a more elevated arm position."
        },
        "Shoulder Horizontal Abduction": {
            "definition": "How far the upper arm moves backward/forward in the horizontal plane.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "Helps describe scapular and shoulder positioning through cocking.",
            "interpretation": "Larger positive magnitude usually means more horizontal layback."
        },
        "Front Knee Flexion": {
            "definition": "Bend angle of the lead knee during stride and bracing.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "Important for lead-leg stabilization and force transfer.",
            "interpretation": "Lower values typically indicate a straighter, more braced lead leg."
        },
        "Front Knee Extension Velocity": {
            "definition": "Speed at which the lead knee extends or straightens.",
            "measured_as": "Angular velocity in degrees per second (deg/s).",
            "why": "Represents how quickly the front side firms up.",
            "interpretation": "Higher extension speed often aligns with stronger front-leg blocking."
        },
        "Forward Trunk Tilt": {
            "definition": "Forward lean angle of the trunk relative to upright.",
            "measured_as": "Angle component in degrees (deg).",
            "why": "Influences release posture and ball direction.",
            "interpretation": "Higher values indicate more forward trunk lean."
        },
        "Lateral Trunk Tilt": {
            "definition": "Side-to-side trunk lean angle during the throw.",
            "measured_as": "Angle component in degrees (deg).",
            "why": "Helps characterize trunk positioning and arm-slot adaptation.",
            "interpretation": "Magnitude reflects how much the torso tilts laterally."
        },
        "Trunk Angle": {
            "definition": "Rotational orientation of the trunk segment.",
            "measured_as": "Angle component in degrees (deg).",
            "why": "Tracks torso orientation through rotation and release.",
            "interpretation": "Changes indicate how trunk orientation evolves over time."
        },
        "Pelvis Angle": {
            "definition": "Rotational orientation of the pelvis segment.",
            "measured_as": "Angle in degrees (deg).",
            "why": "Used to evaluate lower-body rotational contribution.",
            "interpretation": "Steeper progression indicates faster pelvis opening."
        },
        "Pelvic Lateral Tilt": {
            "definition": "Side-to-side tilt of the pelvis.",
            "measured_as": "Angle in degrees (deg).",
            "why": "Provides insight into balance and lower-body control.",
            "interpretation": "Greater magnitude indicates more pelvic obliquity."
        },
        "Hip-Shoulder Separation": {
            "definition": "Difference in rotational angle between pelvis and shoulders.",
            "measured_as": "Relative angle in degrees (deg).",
            "why": "Classic measure of rotational sequencing and stretch.",
            "interpretation": "Larger separation can indicate greater torso-pelvis dissociation."
        },
        "Pelvis Rotational Velocity": {
            "definition": "Speed of pelvis rotation around the vertical axis.",
            "measured_as": "Angular velocity in degrees per second (deg/s).",
            "why": "Describes rotational contribution from the lower body.",
            "interpretation": "Higher peak values indicate faster pelvis turn."
        },
        "Trunk Rotational Velocity": {
            "definition": "Speed of torso rotation around the vertical axis.",
            "measured_as": "Angular velocity in degrees per second (deg/s).",
            "why": "A major contributor to upper-body energy transfer.",
            "interpretation": "Higher peaks indicate faster trunk rotation toward release."
        },
        "Torso-Pelvis Rotational Velocity": {
            "definition": "Relative rotational speed between torso and pelvis.",
            "measured_as": "Differential angular velocity in degrees per second (deg/s).",
            "why": "Highlights how quickly the torso rotates relative to the hips.",
            "interpretation": "Higher values suggest faster torso-over-pelvis separation rate."
        },
        "Forearm Pronation/Supination": {
            "definition": "Rotation of the forearm around its long axis.",
            "measured_as": "Joint angle in degrees (deg).",
            "why": "Useful for arm action context before and after release.",
            "interpretation": "Curve direction indicates pronation versus supination dominance."
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
        "Arm Energy Response (LAR_PROX | RAR_PROX)": {
            "definition": (
                "Measures how the throwing arm receives and responds to energy from the trunk "
                "at the shoulder connection. Positive values -> the arm is loading "
                "(receiving energy). Negative values -> the arm is being accelerated by the trunk."
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
        if kinematic_name in {"Hand Speed", "Center of Mass Velocity (X)"}:
            return "m/s"
        return "°"

    compare_energy_metrics = []
    compare_energy_display_mode = "Grouped"

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
                    "Arm Energy Response (LAR_PROX | RAR_PROX)",
                    "Trunk-Shoulder Rotational Energy Flow",
                    "Trunk-Shoulder Elevation/Depression Energy Flow",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                    "Arm Rotational Energy Flow",
                    "Arm Elevation/Depression Energy Flow",
                    "Arm Horizontal Abd/Add Energy Flow",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)"
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
        "Center of Mass Velocity (X)": "cyan",
        "Shoulder ER": "teal",
        "Shoulder Internal Rotation Velocity": "magenta",
        "Shoulder Abduction": "orange",
        "Shoulder Horizontal Abduction": "brown",
        "Forearm Pronation/Supination": "crimson",
        "Pelvis Rotational Velocity": "navy",
        "Trunk Rotational Velocity": "darkorange",
        "Torso-Pelvis Rotational Velocity": "dodgerblue",
    }
    joint_color_map.update({
        "Forward Trunk Tilt": "blue",
        "Lateral Trunk Tilt": "green",
        "Trunk Angle": "#E9FF70"
    })
    joint_color_map.update({
        "Pelvis Angle": "darkblue",
        "Pelvic Lateral Tilt": "#FF4FA3",
        "Hip-Shoulder Separation": "darkred"
    })
    joint_color_map.update({
        "Front Knee Flexion": "darkgreen"
    })
    joint_color_map.update({
        "Front Knee Extension Velocity": "olive"
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

    if "Center of Mass Velocity (X)" in selected_kinematics:
        joint_data["Center of Mass Velocity (X)"] = get_center_of_mass_velocity_x(take_ids)

    if "Shoulder ER" in selected_kinematics:
        joint_data["Shoulder ER"] = load_joint_by_handedness(get_shoulder_er_angle)

    if "Shoulder Internal Rotation Velocity" in selected_kinematics:
        joint_data["Shoulder Internal Rotation Velocity"] = load_joint_by_handedness(get_shoulder_ir_velocity)

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

    if "Front Knee Flexion" in selected_kinematics:
        joint_data["Front Knee Flexion"] = load_joint_by_handedness(get_front_knee_flexion_angle)

    if "Front Knee Extension Velocity" in selected_kinematics:
        joint_data["Front Knee Extension Velocity"] = load_joint_by_handedness(
            get_front_knee_extension_velocity
        )

    # --- Load Torso Angle components conditionally ---
    needs_torso_angle_data = any(
        metric in selected_kinematics
        for metric in ["Forward Trunk Tilt", "Lateral Trunk Tilt", "Trunk Angle"]
    )
    torso_angle_data = get_torso_angle_components(take_ids) if needs_torso_angle_data else {}

    if "Forward Trunk Tilt" in selected_kinematics:
        joint_data["Forward Trunk Tilt"] = {
            k: {"frame": v["frame"], "value": v["x"]}
            for k, v in torso_angle_data.items()
        }

    if "Lateral Trunk Tilt" in selected_kinematics:
        joint_data["Lateral Trunk Tilt"] = {
            k: {"frame": v["frame"], "value": v["y"]}
            for k, v in torso_angle_data.items()
        }

    if "Trunk Angle" in selected_kinematics:
        joint_data["Trunk Angle"] = {
            k: {"frame": v["frame"], "value": v["z"]}
            for k, v in torso_angle_data.items()
        }

    if "Pelvis Angle" in selected_kinematics:
        joint_data["Pelvis Angle"] = get_pelvis_angle(take_ids)

    if "Pelvic Lateral Tilt" in selected_kinematics:
        joint_data["Pelvic Lateral Tilt"] = get_pelvic_lateral_tilt(take_ids)

    if "Hip-Shoulder Separation" in selected_kinematics:
        joint_data["Hip-Shoulder Separation"] = get_hip_shoulder_separation(take_ids)

    # --- Helper for extracting value at a specific time (ms) ---
    def value_at_time_ms(times_ms, values, target_time_ms):
        if target_time_ms in times_ms:
            return values[times_ms.index(target_time_ms)]
        return None

    import pandas as pd
    summary_rows = []

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
        "Shoulder Internal Rotation Velocity",
        "Trunk Rotational Velocity",
        "Torso-Pelvis Rotational Velocity",
        "Pelvis Rotational Velocity",
    }
    right_hand_mirror_kinematics = {
        "Shoulder Horizontal Abduction",
        "Shoulder ER",
    }
    left_hand_mirror_kinematics = {
        "Forward Trunk Tilt",
        "Lateral Trunk Tilt",
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
                if legend_key not in legend_keys_added:
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

                # Smooth grouped curve ONLY
                if len(y) >= 11:
                    y = savgol_filter(y, window_length=11, polyorder=3)

                color = (
                    group_color_map.get(group_label, joint_color_map.get(kinematic, "#444"))
                    if use_group_colors_joint else
                    joint_color_map.get(kinematic, "#444")
                )
                dash = date_dash_map.get(date, "solid")

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
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

                # IQR band (color-matched)
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
                    "Arm Energy Response (LAR_PROX | RAR_PROX)": "#7C2D12",
                    "Trunk-Shoulder Rotational Energy Flow": "#DC2626",
                    "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",
                    "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",
                    "Arm Rotational Energy Flow": "#F59E0B",
                    "Arm Elevation/Depression Energy Flow": "#06B6D4",
                    "Arm Horizontal Abd/Add Energy Flow": "#9333EA",
                    "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00"
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
                    elif metric == "Arm Energy Response (LAR_PROX | RAR_PROX)":
                        compare_energy_data_by_metric[metric] = load_compare_energy_by_handedness(get_arm_proximal_energy_transfer)
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

                compare_energy_data_by_metric = {
                    k: v for k, v in compare_energy_data_by_metric.items() if v
                }

                if not compare_energy_data_by_metric:
                    st.warning("No energy flow data found for the selected metrics.")
                else:
                    energy_fig = go.Figure()
                    unique_dates = sorted(set(take_date_map.values()))
                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    date_dash_map = {
                        d: dash_styles[i % len(dash_styles)]
                        for i, d in enumerate(unique_dates)
                    }

                    energy_legend_keys = set()
                    energy_window_end = 50
                    energy_window_start_ms = rel_frame_to_ms(window_start)
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
                                if window_start <= rel <= energy_window_end:
                                    norm_f.append(rel_frame_to_ms(rel))
                                    norm_v.append(v)

                            date = take_date_map[take_id]
                            pitcher_name = take_pitcher_map.get(take_id, "")
                            date_key = (pitcher_name, date) if multi_pitcher_mode else date
                            grouped_by_date.setdefault(date_key, {})[take_id] = {
                                "frame": norm_f,
                                "value": norm_v
                            }

                            if compare_energy_display_mode == "Individual Throws":
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
                                            "%{customdata[0]} – %{customdata[1]} | "
                                            "Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                                            + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
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

                        if compare_energy_display_mode == "Grouped":
                            for date_key, curves in grouped_by_date.items():
                                if multi_pitcher_mode:
                                    pitcher_name, date = date_key
                                else:
                                    date = date_key
                                    pitcher_name = ""
                                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                                legendgroup = f"{metric}_{pitcher_name}_{date}" if multi_pitcher_mode else f"{metric}_{date}"
                                energy_fig.add_trace(
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        mode="lines",
                                        line=dict(width=4, color=metric_color, dash=date_dash_map[date]),
                                        customdata=[[metric, date, pitcher_name]] * len(x),
                                        hovertemplate=(
                                            "%{customdata[0]} – %{customdata[1]}"
                                            + (" | %{customdata[2]}" if multi_pitcher_mode else "")
                                            + "<br>%{customdata[0]}: %{y:.1f}"
                                            + "<br>Time: %{x:.0f} ms rel BR"
                                            + "<extra></extra>"
                                        ),
                                        showlegend=False,
                                        legendgroup=legendgroup
                                    )
                                )
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
                                        x=[None],
                                        y=[None],
                                        mode="lines",
                                        line=dict(color=metric_color, dash=date_dash_map[date], width=4),
                                        name=(
                                            f"{metric} | {date} | {pitcher_name}"
                                            if multi_pitcher_mode else
                                            f"{metric} | {date}"
                                        ),
                                        showlegend=True,
                                        legendgroup=legendgroup
                                    )
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
                        yaxis_title="Energy Flow / Segment Power",
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
    if not show_single_kinematics_empty_state and summary_rows:
        st.markdown("### Kinematics Summary")
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
        )

    if not show_single_kinematics_empty_state:
        st.markdown("### Kinematic Definitions")
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

    energy_select_col, energy_select_spacer = st.columns([3, 3])
    with energy_select_col:
        energy_metrics = st.multiselect(
            "Select Energy Flow Metrics",
            [
                "Trunk-Shoulder Energy Flow (RTA_DIST_L | RTA_DIST_R)",
                "Arm Energy Response (LAR_PROX | RAR_PROX)",
                "Trunk-Shoulder Rotational Energy Flow",
                "Trunk-Shoulder Elevation/Depression Energy Flow",
                "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
                "Arm Rotational Energy Flow",
                "Arm Elevation/Depression Energy Flow",
                "Arm Horizontal Abd/Add Energy Flow",
                "Throwing Shoulder Rotational Torque (Relative to Trunk)"
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
        "Arm Energy Response (LAR_PROX | RAR_PROX)": "#7C2D12",  # dark brown
        "Trunk-Shoulder Rotational Energy Flow": "#DC2626",  # strong red
        "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",  # vivid blue
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",     # strong green
        "Arm Rotational Energy Flow": "#F59E0B",        # amber
        "Arm Elevation/Depression Energy Flow": "#06B6D4",  # cyan
        "Arm Horizontal Abd/Add Energy Flow": "#9333EA",     # violet
        "Throwing Shoulder Rotational Torque (Relative to Trunk)": "#FB8C00"
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
        elif metric == "Arm Energy Response (LAR_PROX | RAR_PROX)":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_arm_proximal_energy_transfer)
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
    energy_window_end = 50
    energy_window_start_ms = rel_frame_to_ms(window_start)
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
                if window_start <= rel <= energy_window_end:
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
                            "%{customdata[0]} – %{customdata[1]} | "
                            "Pitch %{customdata[2]} (%{customdata[3]:.1f} mph)"
                            + (" | %{customdata[4]}" if show_group_pitcher_breakout else "")
                            + "<br>%{customdata[0]}: %{y:.1f}"
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
                if legend_key not in legend_keys_added:
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
                        customdata=[[metric, date, pitcher_name]] * len(x),
                        hovertemplate=(
                            "%{customdata[0]} – %{customdata[1]}"
                            + (" | %{customdata[2]}" if show_group_pitcher_breakout else "")
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
        yaxis_title="Energy Flow / Segment Power",
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
