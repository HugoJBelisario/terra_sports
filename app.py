import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Terra Sports Dashboard",
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
    users = st.secrets["auth"]["users"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Login") # Note

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.user = username
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
def get_foot_plant_frame_zero_cross(
    take_ids,
    handedness,
    ankle_prox_x_peak_frames,
    shoulder_er_max_frames
):
    """
    Refined Foot Plant using zero-cross logic.

    Search window:
      peak lead ankle proximal X velocity → max shoulder ER

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
                px_frame = ankle_prox_x_peak_frames.get(take_id)
                er_frame = shoulder_er_max_frames.get(take_id)

                if px_frame is None or er_frame is None:
                    continue

                # refined biomechanical bounds
                if frame < px_frame or frame > er_frame:
                    continue

                # zero-cross detection
                if z >= -0.05 and take_id not in out:
                    out[take_id] = int(frame - 1)

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


# -------------------------------
# Initialize session state for excluded takes
# -------------------------------
if "excluded_take_ids" not in st.session_state:
    st.session_state["excluded_take_ids"] = []

pitcher_names = get_all_pitchers()

if not pitcher_names:
    st.sidebar.warning("No pitchers found in the database.")
    selected_pitchers = []
else:
    selected_pitchers = st.sidebar.multiselect(
        "Select Pitcher(s)",
        options=pitcher_names,
        default=[pitcher_names[0]] if pitcher_names else [],
        key="select_pitchers"
    )

# -------------------------------
# Per-Pitcher Filters
# -------------------------------
pitcher_filters = {}
multi_pitcher_mode = len(selected_pitchers) > 1
for i, pitcher in enumerate(selected_pitchers):
    label_suffix = f" - {pitcher}" if multi_pitcher_mode else ""
    if multi_pitcher_mode:
        st.sidebar.markdown(f"**{pitcher} Filters**")

    session_dates = get_session_dates_for_pitcher(pitcher)
    if session_dates:
        session_dates_with_all = ["All Dates"] + session_dates
        selected_dates_i = st.sidebar.multiselect(
            f"Session Dates{label_suffix}",
            options=session_dates_with_all,
            default=["All Dates"],
            key=f"select_session_dates_{i}"
        )
    else:
        st.sidebar.info(f"No session dates found for {pitcher}.")
        selected_dates_i = []

    throw_types_i = st.sidebar.multiselect(
        f"Throw Type{label_suffix}",
        options=["Mound", "Pulldown"],
        default=["Mound"],
        key=f"throw_types_{i}"
    )
    if not throw_types_i:
        throw_types_i = ["Mound"]

    vel_min_i, vel_max_i = get_velocity_bounds(pitcher, selected_dates_i)
    if vel_min_i is not None and vel_max_i is not None:
        velocity_range_i = st.sidebar.slider(
            f"Velocity Range{label_suffix} (mph)",
            min_value=float(vel_min_i),
            max_value=float(vel_max_i),
            value=(float(vel_min_i), float(vel_max_i)),
            step=0.5,
            key=f"velocity_range_{i}"
        )
        velocity_min_i, velocity_max_i = velocity_range_i
    else:
        velocity_min_i, velocity_max_i = None, None
        st.sidebar.info(f"Velocity data not available for {pitcher}.")

    pitcher_filters[pitcher] = {
        "selected_dates": selected_dates_i,
        "throw_types": throw_types_i,
        "velocity_min": velocity_min_i,
        "velocity_max": velocity_max_i,
    }

all_throw_types = sorted({
    t
    for cfg in pitcher_filters.values()
    for t in cfg["throw_types"]
})
mound_only_sidebar = bool(pitcher_filters) and all(
    set(cfg["throw_types"]) == {"Mound"}
    for cfg in pitcher_filters.values()
)





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


with tab_kinematic:
    st.subheader("Kinematic Sequence")
    display_mode = st.radio(
        "Select Display Mode",
        ["Individual Throws", "Grouped"],
        index=1,
        horizontal=True,
        key="ks_display_mode"
    )
    pitcher_handedness = {
        p: get_pitcher_handedness(p)
        for p in selected_pitchers
    }

    take_ids = []
    take_pitcher_map = {}

    # Resolve take_ids based on per-pitcher filters
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
                    if take_id not in take_pitcher_map:
                        take_pitcher_map[take_id] = pitcher
                        take_ids.append(take_id)
    finally:
        conn.close()

    take_handedness = {
        tid: pitcher_handedness.get(take_pitcher_map.get(tid))
        for tid in take_ids
    }
    take_ids = [tid for tid in take_ids if take_handedness.get(tid) in ("R", "L")]

    # --- Pitch order + velocity lookup ---
    if take_ids:
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

        date_groups = defaultdict(list)

        for tid, velo, date, pitcher in rows:
            date_groups[(pitcher, date)].append((tid, velo))

        take_order = {}
        take_velocity = {}
        take_date_map = {}
        take_pitcher_map = {}

        for (pitcher, date), items in date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                take_order[tid] = i
                take_velocity[tid] = velo
                take_date_map[tid] = date.strftime("%Y-%m-%d")
                take_pitcher_map[tid] = pitcher

        # -------------------------------
        # Sidebar: Exclude Takes
        # -------------------------------
        exclude_options = [
            (
                f"{take_pitcher_map[tid]} | {take_date_map[tid]} – "
                f"Pitch {take_order[tid]} ({take_velocity[tid]:.1f} mph)"
            )
            for tid in take_ids
        ]

        label_to_take_id = {
            (
                f"{take_pitcher_map[tid]} | {take_date_map[tid]} – "
                f"Pitch {take_order[tid]} ({take_velocity[tid]:.1f} mph)"
            ): tid
            for tid in take_ids
        }

        excluded_labels = st.sidebar.multiselect(
            "Exclude Takes",
            options=exclude_options,
            default=[
                label for label, tid in label_to_take_id.items()
                if tid in st.session_state["excluded_take_ids"]
            ],
            key="exclude_takes"
        )

        st.session_state["excluded_take_ids"] = [
            label_to_take_id[label] for label in excluded_labels
        ]

        # Filter take_ids to exclude selected takes
        take_ids = [
            tid for tid in take_ids
            if tid not in st.session_state["excluded_take_ids"]
        ]

    if not take_ids:
        st.info("No takes found for this selection.")
    else:
        from collections import defaultdict

        take_ids_by_handedness = defaultdict(list)
        for tid in take_ids:
            hand = take_handedness.get(tid)
            if hand in ("R", "L"):
                take_ids_by_handedness[hand].append(tid)

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
        # Cache Ball Release frames per take
        br_frames = {}

        for take_id in take_ids:
            if take_id in cg_data:
                cg_frames = cg_data[take_id]["frame"]
                cg_vals = cg_data[take_id]["x"]

                valid = [(i, v) for i, v in enumerate(cg_vals) if v is not None]
                if valid:
                    idx, _ = max(valid, key=lambda x: x[1])
                    br_frames[take_id] = cg_frames[idx]
        shoulder_data = load_by_handedness(get_shoulder_er_angles)
        # Cache max shoulder ER frames per take
        shoulder_er_max_frames = {}

        for take_id, d in shoulder_data.items():
            frames = d["frame"]
            values = d["z"]

            valid = [(f, v) for f, v in zip(frames, values) if v is not None]
            if not valid:
                continue

            hand = take_handedness.get(take_id)
            if hand == "R":
                er_frame, _ = min(valid, key=lambda x: x[1])  # most negative
            else:
                er_frame, _ = max(valid, key=lambda x: x[1])  # most positive

            shoulder_er_max_frames[take_id] = er_frame
        knee_peak_frames = {}
        foot_plant_frames = {}
        ankle_prox_x_peak_frames = {}
        foot_plant_zero_cross_frames = {}
        for hand, ids in take_ids_by_handedness.items():
            if not ids:
                continue
            knee_peak_frames.update(
                get_peak_glove_knee_pre_br(ids, hand, br_frames)
            )
            foot_plant_frames.update(
                get_foot_plant_frame(ids, hand, knee_peak_frames, br_frames)
            )
            ankle_prox_x_peak_frames.update(
                get_peak_ankle_prox_x_velocity(ids, hand)
            )
            foot_plant_zero_cross_frames.update(
                get_foot_plant_frame_zero_cross(
                    ids,
                    hand,
                    ankle_prox_x_peak_frames,
                    shoulder_er_max_frames
                )
            )

        # Collect normalized peak knee height frames for aggregation
        knee_event_frames = []

        for take_id, knee_frame in knee_peak_frames.items():
            if take_id in br_frames:
                knee_event_frames.append(knee_frame - br_frames[take_id])

        # Collect normalized refined Foot Plant (zero-cross) frames
        fp_event_frames = []
        for take_id, fp_frame in foot_plant_zero_cross_frames.items():
            if take_id in br_frames:
                fp_event_frames.append(fp_frame - br_frames[take_id])

        # Collect normalized max Shoulder ER frames
        mer_event_frames = []
        for take_id, er_frame in shoulder_er_max_frames.items():
            if take_id in br_frames:
                mer_event_frames.append(er_frame - br_frames[take_id])

        # Define plot window start from median Foot Plant (zero-cross)
        if fp_event_frames:
            window_start = int(np.median(fp_event_frames)) - 50
        else:
            # fallback: still ensure we have a reasonable window
            window_start = -100
        window_end = 50
        window_start_ms = rel_frame_to_ms(window_start)
        window_end_ms = rel_frame_to_ms(window_end)

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

            if take_id in shoulder_data:
                sh_frames = shoulder_data[take_id]["frame"]
                sh_values = shoulder_data[take_id]["z"]

                valid_sh = [
                    (f, v) for f, v in zip(sh_frames, sh_values)
                    if v is not None and f <= br_frame
                ]

                if valid_sh:
                    if take_hand == "R":
                        mer_frame, _ = min(valid_sh, key=lambda x: x[1])  # negative max
                    else:
                        mer_frame, _ = max(valid_sh, key=lambda x: x[1])  # positive max

                    mer_rel_frame = mer_frame - br_frame

            # -----------------------------
            # Normalize time to Ball Release
            # -----------------------------
            norm_frames = []
            norm_values = []

            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel_frame = f - br_frame

                # Keep frames from 50 before median knee height through +50 after BR
                if rel_frame >= window_start and rel_frame <= window_end:
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
                    if rel_frame >= window_start and rel_frame <= window_end:
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
                            customdata=[[ "Torso", take_date_map[take_id], take_order[take_id], take_velocity[take_id], pitcher_name ]] * len(norm_torso_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<extra></extra>"
                            ),
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
                    if rel_frame >= window_start and rel_frame <= window_end:
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
                            customdata=[[ "Elbow", take_date_map[take_id], take_order[take_id], take_velocity[take_id], pitcher_name ]] * len(norm_elbow_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<extra></extra>"
                            ),
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
                    if rel_frame >= window_start and rel_frame <= window_end:
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
                            customdata=[[ "Shoulder", take_date_map[take_id], take_order[take_id], take_velocity[take_id], pitcher_name ]] * len(norm_sh_frames),
                            hovertemplate=(
                                "%{customdata[0]} – %{customdata[1]} | "
                                "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                                + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                                + "<extra></extra>"
                            ),
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
                        customdata=[[ "Pelvis", take_date_map[take_id], take_order[take_id], take_velocity[take_id], pitcher_name ]] * len(norm_frames),
                        hovertemplate=(
                            "%{customdata[0]} – %{customdata[1]} | "
                            "Pitch %{customdata[2]} (%{customdata[3]:.1f} MPH)"
                            + (" | %{customdata[4]}" if multi_pitcher_mode else "")
                            + "<extra></extra>"
                        ),
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
            # --- Shared arrow offset based on global y-range (consistent across segments) ---
            all_grouped_vals = [
                v
                for curves in [grouped_pelvis, grouped_torso, grouped_elbow, grouped_shoulder_ir]
                for d in curves.values()
                for v in d["value"]
                if v is not None
            ]

            arrow_offset = (
                0.06 * (max(all_grouped_vals) - min(all_grouped_vals))
                if all_grouped_vals else 0
            )

            color_map = {
                "Pelvis": "blue",
                "Torso": "orange",
                "Elbow": "green",
                "Shoulder": "red"
            }

            # --- Condensed legend: track (Segment, Date) pairs ---
            legend_keys_added = set()

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
                    date_key = (pitcher_name, date) if multi_pitcher_mode else date
                    curves_by_date[date_key][take_id] = d
                for date_key, curves_date in curves_by_date.items():
                    if multi_pitcher_mode:
                        pitcher_name, date = date_key
                    else:
                        date = date_key
                        pitcher_name = ""
                    x_date, y_date, q1_date, q3_date = aggregate_curves(curves_date, "Mean")
                    color = color_map[label]
                    # Smoothing
                    if len(y_date) >= 11:
                        y_date = savgol_filter(y_date, window_length=7, polyorder=3)
                    dash = date_dash_map[date]
                    legendgroup = f"{label}_{date}_{pitcher_name}" if multi_pitcher_mode else f"{label}_{date}"
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
                            customdata=[[label, date, pitcher_name]] * len(x_date),
                            hovertemplate=(
                                "%{customdata[0]} | %{customdata[1]}"
                                + (" | %{customdata[2]}" if multi_pitcher_mode else "")
                                + "<extra></extra>"
                            ),
                            showlegend=False,
                            legendgroup=legendgroup
                        )
                    )
                    # --- Legend-only trace (once per Segment + Date, legendgroup set) ---
                    legend_key = (label, date, pitcher_name) if multi_pitcher_mode else (label, date)
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
                                    f"{label} AV | {date} | {pitcher_name}"
                                    if multi_pitcher_mode else
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
                        pelvis_time_ms_grouped = None
                        if label == "Pelvis" and fp_event_frames:
                            fp_rel = rel_frame_to_ms(int(np.median(fp_event_frames)))
                            pelvis_time_ms_grouped = max_x - fp_rel

                        kinematic_peak_rows.append({
                            **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
                            "Session Date": date,
                            "Segment": label,
                            "Peak Value (°/s)": max_y,
                            "Peak Time (ms rel BR)": max_x,
                            "Peak Time from FP (ms)": pelvis_time_ms_grouped if label == "Pelvis" else None
                        })
                        y_offset = arrow_offset * 0.6
                        fig.add_trace(
                            go.Scatter(
                                x=[max_x],
                                y=[max_y + y_offset],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-down",
                                    size=14,
                                    color=color
                                ),
                                showlegend=False,
                                hoverinfo="skip"
                            )
                        )
                    # --- IQR band (with legendgroup for toggleitem) ---
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

        # Median Peak Glove-Side Knee Height event
        if knee_event_frames:
            median_knee_frame = rel_frame_to_ms(int(np.median(knee_event_frames)))

            fig.add_vline(
                x=median_knee_frame,
                line_width=3,
                line_dash="dot",
                line_color="blue",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_knee_frame,
                y=1.055,
                xref="x",
                yref="paper",
                text="Knee",
                showarrow=False,
                font=dict(color="blue", size=14),
                align="center"
            )

        # Median Refined Foot Plant (zero-cross) event
        if fp_event_frames:
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
            yaxis_title="Angular Velocity",
            yaxis=dict(ticksuffix="°/s"),
            xaxis_range=[window_start_ms, window_end_ms],
            height=600,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.30,
                xanchor="center",
                x=0.5,
                groupclick="toggleitem"
            ),
            hoverlabel=dict(
                namelength=-1,
                font_size=13
            )
        )

        st.plotly_chart(fig, use_container_width=True)

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

                individual_rows.append({
                    **({"Pitcher": take_pitcher_map.get(take_id)} if multi_pitcher_mode else {}),
                    "Session Date": take_date_map[take_id],
                    "Pitch": take_order[take_id],
                    "Velocity (mph)": take_velocity[take_id],
                    "Pelvis Peak (°/s)": pelvis_peak,
                    "Pelvis Time from FP (ms)": pelvis_time_ms,
                    "Torso Peak (°/s)": torso_peak,
                    "Torso Time (ms rel BR)": torso_frame,
                    "Elbow Peak (°/s)": elbow_peak,
                    "Elbow Time (ms rel BR)": elbow_frame,
                    "Shoulder IR Peak (°/s)": shoulder_peak,
                    "Shoulder IR Time (ms rel BR)": shoulder_frame
                })

            if individual_rows:
                import pandas as pd

                st.markdown("### Kinematic Sequence - Individual Throws")

                df_individual = pd.DataFrame(individual_rows)

                # Sort logically: date → pitch order
                sort_cols = ["Session Date", "Pitch"]
                if multi_pitcher_mode and "Pitcher" in df_individual.columns:
                    sort_cols = ["Pitcher"] + sort_cols
                df_individual = df_individual.sort_values(sort_cols)

                def fmt(val, decimals=2):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    return f"{val:.{decimals}f}"

                styled_individual = (
                    df_individual
                    .style
                    .format({
                        "Velocity (mph)": lambda x: fmt(x, 1),
                        "Pelvis Peak (°/s)": lambda x: fmt(x, 1),
                        "Torso Peak (°/s)": lambda x: fmt(x, 1),
                        "Elbow Peak (°/s)": lambda x: fmt(x, 1),
                        "Shoulder IR Peak (°/s)": lambda x: fmt(x, 1),
                        "Pelvis Time from FP (ms)": lambda x: "" if x is None else f"{x:.0f}",
                        "Torso Time (ms rel BR)": lambda x: fmt(x, 0),
                        "Elbow Time (ms rel BR)": lambda x: fmt(x, 0),
                        "Shoulder IR Time (ms rel BR)": lambda x: fmt(x, 0),
                    })
                    .set_table_styles([
                        {"selector": "th", "props": [("text-align", "center")]}
                    ])
                    .set_properties(**{"text-align": "center"})
                )

                st.dataframe(styled_individual, use_container_width=True)

        # --- Kinematic Sequence Peak Summary Table (Segment-Grouped) ---
        if display_mode == "Grouped" and kinematic_peak_rows:
            import pandas as pd

            st.markdown("### Kinematic Sequence - Grouped")

            df = pd.DataFrame(kinematic_peak_rows)
            df_display = df.copy().rename(columns={
                "Peak Value (°/s)": "Peak (°/s)",
                "Peak Time (ms rel BR)": "Peak Time (ms rel BR)"
            })
            cols = [
                c for c in [
                    "Pitcher",
                    "Session Date",
                    "Segment",
                    "Peak (°/s)",
                    "Peak Time (ms rel BR)",
                    "Peak Time from FP (ms)"
                ] if c in df_display.columns
            ]
            df_display = df_display[cols]
            sort_cols = [c for c in ["Pitcher", "Session Date", "Segment"] if c in df_display.columns]
            if sort_cols:
                df_display = df_display.sort_values(sort_cols)

            styled = (
                df_display
                .style
                .format({
                    "Peak (°/s)": lambda x: "" if x is None else f"{x:.1f}",
                    "Peak Time (ms rel BR)": lambda x: "" if x is None else f"{x:.0f}",
                    "Peak Time from FP (ms)": lambda x: "" if x is None else f"{x:.0f}",
                })
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "center")]}
                ])
                .set_properties(**{"text-align": "center"})
            )
            st.dataframe(styled, use_container_width=True)


with tab_joint:
    st.subheader("Kinematics")
    display_mode = st.radio(
        "Select Display Mode",
        ["Individual Throws", "Grouped"],
        index=1,
        horizontal=True,
        key="joint_display_mode"
    )
    joint_mound_only_selected = mound_only_sidebar

    default_joint_selection = ["Elbow Flexion"]
    if joint_mound_only_selected:
        default_joint_selection = [
            "Elbow Flexion",
            "Center of Mass Velocity (X)",
            "Shoulder Internal Rotation Velocity",
            "Trunk Rotational Velocity",
            "Torso-Pelvis Rotational Velocity",
            "Pelvis Rotational Velocity",
        ]

    selected_kinematics = st.multiselect(
        "Select Kinematics",
        options=[
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
        ],
        default=default_joint_selection,
        key="joint_angles_select"
    )

    if not selected_kinematics:
        st.info("Select at least one kinematic.")
        st.stop()

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
    torso_angle_data = get_torso_angle_components(take_ids)

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

    # --- Per-take normalization and plotting ---
    grouped = {}
    grouped_by_date = {}
    mound_only_selected = mound_only_sidebar
    median_pkh_frame = None
    joint_window_start = window_start
    joint_window_end = 50
    joint_window_start_ms = rel_frame_to_ms(joint_window_start)
    joint_window_end_ms = rel_frame_to_ms(joint_window_end)

    # For mound throws, ensure the window includes PKH and 20 frames before it.
    if mound_only_selected and knee_event_frames:
        median_pkh_frame = int(np.median(knee_event_frames))
        joint_window_start = min(window_start, median_pkh_frame - 20)
        joint_window_start_ms = rel_frame_to_ms(joint_window_start)

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
                    # (Keep angles consistent, but DO NOT flip rotational velocities)
                    take_hand = take_handedness.get(take_id)
                    if "Velocity" not in kinematic and take_hand == "R" and kinematic in [
                        "Shoulder Horizontal Abduction",
                        "Shoulder ER"
                    ]:
                        norm_v.append(-v)
                    else:
                        norm_v.append(sign_flip * v)

            grouped[kinematic][take_id] = {"frame": norm_f, "value": norm_v}

            # --- Store by date for grouped plotting ---
            date = take_date_map[take_id]
            pitcher_name = take_pitcher_map.get(take_id, "")
            date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {}).setdefault(kinematic, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }

            if display_mode == "Individual Throws":
                # Use kinematic color and date-based dash for individual throws
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        line=dict(
                            color=joint_color_map[kinematic],
                            dash=date_dash_map[take_date_map[take_id]]
                        ),
                        name=(
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} "
                            f"({take_velocity[take_id]:.1f} mph) | {pitcher_name}"
                            if multi_pitcher_mode else
                            f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)"
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
                                color=joint_color_map[kinematic],
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"{kinematic} | {date} | {pitcher_name}"
                                if multi_pitcher_mode else
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
                    **({"Pitcher": take_pitcher_map.get(take_id)} if multi_pitcher_mode else {}),
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
            if multi_pitcher_mode:
                pitcher_name, date = date_key
            else:
                date = date_key
                pitcher_name = ""
            for kinematic, curves in kin_dict.items():
                if not curves:
                    continue

                x, y, q1, q3 = aggregate_curves(curves, "Mean")

                # Smooth grouped curve ONLY
                if len(y) >= 11:
                    y = savgol_filter(y, window_length=11, polyorder=3)

                color = joint_color_map.get(kinematic, "#444")
                dash = date_dash_map[date]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(width=4, color=color, dash=dash),
                        name=(
                            f"{kinematic} – {date} | {pitcher_name}"
                            if multi_pitcher_mode else
                            f"{kinematic} – {date}"
                        )
                    )
                )

                # IQR band (color-matched)
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
                    **({"Pitcher": pitcher_name} if multi_pitcher_mode else {}),
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

    st.plotly_chart(fig, use_container_width=True)

    # --- Kinematics Table ---
    if summary_rows:
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
        if multi_pitcher_mode:
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

        formatters = {
            "Average Velocity": lambda x: fmt(x, 1),
            "Max": lambda x: fmt(x, 2),
            "Peak Knee Height": lambda x: fmt(x, 2),
            "Foot Plant": lambda x: fmt(x, 2),
            "Ball Release": lambda x: fmt(x, 2),
            "Max External Rotation": lambda x: fmt(x, 2),
        }

        if display_mode == "Grouped":
            formatters["Standard Deviation"] = lambda x: f"±{fmt(x, 2)}" if x is not None else ""

        styled_summary = (
            df_summary
            .style
            .format(formatters)
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
        st.dataframe(styled_summary, use_container_width=True)

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

    energy_metrics = st.multiselect(
        "Select Energy Flow Metrics",
        [
            "Distal Arm Segment Power",
            "Arm Proximal Energy Transfer",
            "Trunk-Shoulder Rotational Energy Flow",
            "Trunk-Shoulder Elevation/Depression Energy Flow",
            "Trunk-Shoulder Horizontal Abd/Add Energy Flow",
            "Arm Rotational Energy Flow",
            "Arm Elevation/Depression Energy Flow",
            "Arm Horizontal Abd/Add Energy Flow"
        ],
        default=["Distal Arm Segment Power"]
    )

    if not energy_metrics:
        st.info("Select at least one energy flow metric.")
        st.stop()

    display_mode = st.radio(
        "Select Display Mode",
        ["Individual Throws", "Grouped"],
        index=1,
        horizontal=True,
        key="energy_display_mode"
    )

    if not take_ids:
        st.info("No takes available for Energy Flow.")
        st.stop()

    # --- Fixed color map for Energy Flow metrics (high-contrast palette) ---
    energy_color_map = {
        "Distal Arm Segment Power": "#4C1D95",            # deep indigo / purple
        "Arm Proximal Energy Transfer": "#7C2D12",        # dark brown
        "Trunk-Shoulder Rotational Energy Flow": "#DC2626",  # strong red
        "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",  # vivid blue
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",     # strong green
        "Arm Rotational Energy Flow": "#F59E0B",        # amber
        "Arm Elevation/Depression Energy Flow": "#06B6D4",  # cyan
        "Arm Horizontal Abd/Add Energy Flow": "#9333EA"     # violet
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
        if metric == "Distal Arm Segment Power":
            energy_data_by_metric[metric] = load_energy_by_handedness(get_distal_arm_segment_power)
        elif metric == "Arm Proximal Energy Transfer":
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
            pitcher_name = take_pitcher_map.get(take_id, "")
            date_key = (pitcher_name, date) if multi_pitcher_mode else date
            grouped_by_date.setdefault(date_key, {})[take_id] = {
                "frame": norm_f,
                "value": norm_v
            }

            if display_mode == "Individual Throws":
                fig.add_trace(
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
                            + "<extra></extra>"
                        ),
                        showlegend=False
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
                                color=metric_color,
                                dash=date_dash_map[date],
                                width=4
                            ),
                            name=(
                                f"{metric} | {date} | {pitcher_name}"
                                if multi_pitcher_mode else
                                f"{metric} | {date}"
                            ),
                            showlegend=True
                        )
                    )
                    legend_keys_added.add(legend_key)

        # -------------------------------
        # Grouped (Mean + IQR per date)
        # -------------------------------
        if display_mode == "Grouped":
            for date_key, curves in grouped_by_date.items():
                if multi_pitcher_mode:
                    pitcher_name, date = date_key
                else:
                    date = date_key
                    pitcher_name = ""
                x, y, q1, q3 = aggregate_curves(curves, "Mean")

                if len(y) >= 11:
                    y = savgol_filter(y, window_length=11, polyorder=3)

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(
                            width=4,
                            color=metric_color,
                            dash=date_dash_map[date]
                        ),
                        customdata=[[metric, date, pitcher_name]] * len(x),
                        hovertemplate=(
                            "%{customdata[0]} – %{customdata[1]}"
                            + (" | %{customdata[2]}" if multi_pitcher_mode else "")
                            + "<extra></extra>"
                        ),
                        showlegend=False
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=q3 + q1[::-1],
                        fill="toself",
                        fillcolor=to_rgba(metric_color, alpha=0.35),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

                fig.add_trace(
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
                        showlegend=True
                    )
                )

    # -------------------------------
    # Event Lines (with text labels above)
    # -------------------------------
    if fp_event_frames:
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
            x=0.5
        ),
        hoverlabel=dict(
            namelength=-1,
            font_size=13
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Energy Flow Table (Individual mode: per-pitch, long format with velocity)
    # --------------------------------------------------
    import pandas as pd
    import numpy as np

    if display_mode == "Individual Throws":
        rows = []

        for take_id in take_ids:
            pitch_num = take_order.get(take_id)
            velo = take_velocity.get(take_id)

            for metric in energy_metrics:
                metric_data = energy_data_by_metric.get(metric, {})
                take_data = metric_data.get(take_id)

                if not take_data:
                    continue

                br = br_frames.get(take_id)
                if br is None:
                    continue
                for frame, value in zip(take_data["frame"], take_data["value"]):
                    rel_time_ms = rel_frame_to_ms(frame - br)
                    rows.append({
                        **({"Pitcher": take_pitcher_map.get(take_id)} if multi_pitcher_mode else {}),
                        "Pitch": f"Pitch {pitch_num}",
                        "Time (ms rel BR)": rel_time_ms,
                        "Velocity (mph)": velo,
                        metric: value
                    })

        # Build dataframe
        energy_df = pd.DataFrame(rows)

        # Pivot so each metric becomes its own column
        energy_df = (
            energy_df
            .groupby(
                (["Pitcher"] if multi_pitcher_mode else [])
                + ["Pitch", "Time (ms rel BR)", "Velocity (mph)"]
            )
            .first()
            .reset_index()
        )

        # Ensure metric columns are ordered after metadata
        ordered_cols = (
            (["Pitcher"] if multi_pitcher_mode else [])
            + ["Pitch", "Time (ms rel BR)", "Velocity (mph)"] +
            [m for m in energy_metrics if m in energy_df.columns]
        )

        energy_df = energy_df[ordered_cols]

        st.dataframe(
            energy_df,
            use_container_width=True,
            height=420
        )
    else:
        if multi_pitcher_mode:
            grouped_rows = []
            pitchers_in_data = sorted({
                take_pitcher_map.get(take_id)
                for metric_data in energy_data_by_metric.values()
                for take_id in metric_data.keys()
                if take_id in br_frames and take_pitcher_map.get(take_id)
            })

            for pitcher_name in pitchers_in_data:
                pitcher_take_ids = {
                    tid for tid, p in take_pitcher_map.items()
                    if p == pitcher_name
                }

                common_times_ms = sorted(
                    set(
                        rel_frame_to_ms(frame - br_frames[take_id])
                        for metric_data in energy_data_by_metric.values()
                        for take_id, take_data in metric_data.items()
                        if take_id in pitcher_take_ids and take_id in br_frames
                        for frame in take_data["frame"]
                    )
                )

                for t_ms in common_times_ms:
                    row = {
                        "Pitcher": pitcher_name,
                        "Time (ms rel BR)": t_ms,
                    }
                    for metric in energy_metrics:
                        vals = []
                        for take_id, data in energy_data_by_metric[metric].items():
                            if take_id not in pitcher_take_ids or take_id not in br_frames:
                                continue
                            br = br_frames[take_id]
                            for f, v in zip(data["frame"], data["value"]):
                                if rel_frame_to_ms(f - br) == t_ms:
                                    vals.append(v)
                        row[metric] = np.nanmean(vals) if vals else np.nan
                    grouped_rows.append(row)

            energy_table = pd.DataFrame(grouped_rows)
            ordered_cols = ["Pitcher", "Time (ms rel BR)"] + [m for m in energy_metrics if m in energy_table.columns]
            energy_table = energy_table[ordered_cols] if not energy_table.empty else energy_table
        else:
            # 1) Build a common time axis (ms relative to BR)
            common_times_ms = sorted(
                set(
                    rel_frame_to_ms(frame - br_frames[take_id])
                    for metric_data in energy_data_by_metric.values()
                    for take_id, take_data in metric_data.items()
                    if take_id in br_frames
                    for frame in take_data["frame"]
                )
            )

            energy_table = pd.DataFrame({"Time (ms rel BR)": common_times_ms})

            # 2) Add one column per selected metric
            for metric in energy_metrics:
                metric_series = {}

                for take_id, data in energy_data_by_metric[metric].items():
                    if take_id not in br_frames:
                        continue
                    br = br_frames[take_id]
                    for f, v in zip(data["frame"], data["value"]):
                        t_ms = rel_frame_to_ms(f - br)
                        metric_series.setdefault(t_ms, []).append(v)

                # Average across takes if multiple exist
                metric_values = [
                    np.nanmean(metric_series.get(f, [np.nan]))
                    for f in common_times_ms
                ]

                energy_table[metric] = metric_values

        # 3) Display the table
        st.dataframe(
            energy_table,
            use_container_width=True,
            height=400
        )

    # -------------------------------
    # Energy Flow Summary Table (structure unchanged)
    # -------------------------------
    # (No change: compute_peak_segment_power, summary table logic, etc.)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("© Terra Sports | Biomechanics Dashboard")
