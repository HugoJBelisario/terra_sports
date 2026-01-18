import streamlit as st

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
from scipy.signal import savgol_filter
from dotenv import load_dotenv
from db.connection import get_connection

def login():
    users = st.secrets["auth"]["users"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Login")

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
        "black": (0, 0, 0),
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
    selected_pitcher = None
else:
    selected_pitcher = st.sidebar.selectbox(
        "Select Pitcher",
        options=pitcher_names,
        key="select_pitcher"
    )

# -------------------------------
# Select Session Dates
# -------------------------------
if selected_pitcher is None:
    selected_dates = []
else:
    session_dates = get_session_dates_for_pitcher(selected_pitcher)

    if session_dates:
        session_dates_with_all = ["All Dates"] + session_dates

        selected_dates = st.sidebar.multiselect(
            "Select Session Dates",
            options=session_dates_with_all,
            default=["All Dates"],
            key="select_session_dates"
        )
    else:
        st.sidebar.info("No session dates found for this pitcher.")
        selected_dates = []

# -------------------------------
# Velocity Filter
# -------------------------------
vel_min, vel_max = get_velocity_bounds(selected_pitcher, selected_dates)

if vel_min is not None and vel_max is not None:
    velocity_range = st.sidebar.slider(
        "Velocity Range (mph)",
        min_value=float(vel_min),
        max_value=float(vel_max),
        value=(float(vel_min), float(vel_max)),
        step=0.5
    )
    velocity_min, velocity_max = velocity_range
else:
    velocity_min, velocity_max = None, None
    st.sidebar.info("Velocity data not available for this selection.")





st.title("Terra Sports Biomechanics Dashboard")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_kinematic, tab_joint, tab_energy = st.tabs(
    ["Kinematic Sequence", "Joint Angles", "Energy Flow"]
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



    handedness = get_pitcher_handedness(selected_pitcher)

    # Resolve take_ids based on pitcher + dates + velocity filter
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if selected_pitcher is None:
                take_ids = []
            elif "All Dates" in selected_dates or not selected_dates:
                cur.execute("""
                    SELECT t.take_id
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.pitch_velo BETWEEN %s AND %s
                """, (selected_pitcher, velocity_min, velocity_max))
                take_ids = [r[0] for r in cur.fetchall()]
            else:
                placeholders = ",".join(["%s"] * len(selected_dates))
                cur.execute(f"""
                    SELECT t.take_id
                    FROM takes t
                    JOIN athletes a ON a.athlete_id = t.athlete_id
                    WHERE a.athlete_name = %s
                      AND t.take_date IN ({placeholders})
                      AND t.pitch_velo BETWEEN %s AND %s
                """, (selected_pitcher, *selected_dates, velocity_min, velocity_max))
                take_ids = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    # --- Pitch order + velocity lookup ---
    if take_ids:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(take_ids))
                cur.execute(f"""
                    SELECT take_id, pitch_velo, take_date
                    FROM takes
                    WHERE take_id IN ({placeholders})
                    ORDER BY take_date, take_id
                """, tuple(take_ids))
                rows = cur.fetchall()
        finally:
            conn.close()

        from collections import defaultdict

        date_groups = defaultdict(list)

        for tid, velo, date in rows:
            date_groups[date].append((tid, velo))

        take_order = {}
        take_velocity = {}
        take_date_map = {}

        for date, items in date_groups.items():
            for i, (tid, velo) in enumerate(items, start=1):
                take_order[tid] = i
                take_velocity[tid] = velo
                take_date_map[tid] = date.strftime("%Y-%m-%d")

        # -------------------------------
        # Sidebar: Exclude Takes
        # -------------------------------
        exclude_options = [
            f"{take_date_map[tid]} – Pitch {take_order[tid]} ({take_velocity[tid]:.1f} mph)"
            for tid in take_ids
        ]

        label_to_take_id = {
            f"{take_date_map[tid]} – Pitch {take_order[tid]} ({take_velocity[tid]:.1f} mph)": tid
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
        import plotly.graph_objects as go

        data = get_pelvis_angular_velocity(take_ids)
        cg_data = get_hand_cg_velocity(take_ids, handedness)
        torso_data = get_torso_angular_velocity(take_ids)
        elbow_data = get_elbow_angular_velocity(take_ids, handedness)
        shoulder_ir_data = get_shoulder_ir_velocity(take_ids, handedness)
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
        shoulder_data = get_shoulder_er_angles(take_ids, handedness)
        # Cache max shoulder ER frames per take
        shoulder_er_max_frames = {}

        for take_id, d in shoulder_data.items():
            frames = d["frame"]
            values = d["z"]

            valid = [(f, v) for f, v in zip(frames, values) if v is not None]
            if not valid:
                continue

            if handedness == "R":
                er_frame, _ = min(valid, key=lambda x: x[1])  # most negative
            else:
                er_frame, _ = max(valid, key=lambda x: x[1])  # most positive

            shoulder_er_max_frames[take_id] = er_frame
        knee_peak_frames = get_peak_glove_knee_pre_br(
            take_ids,
            handedness,
            br_frames
        )

        foot_plant_frames = get_foot_plant_frame(
            take_ids,
            handedness,
            knee_peak_frames,
            br_frames
        )
        ankle_prox_x_peak_frames = get_peak_ankle_prox_x_velocity(
            take_ids,
            handedness
        )

        foot_plant_zero_cross_frames = get_foot_plant_frame_zero_cross(
            take_ids,
            handedness,
            ankle_prox_x_peak_frames,
            shoulder_er_max_frames
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
                    if handedness == "R":
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
                if rel_frame >= window_start and rel_frame <= 50:
                    norm_frames.append(rel_frame)
                    # Handedness normalization for Pelvis AV (Kinematic Sequence only)
                    if handedness == "L":
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
                    if rel_frame >= window_start and rel_frame <= 50:
                        norm_torso_frames.append(rel_frame)
                        # Handedness normalization for Torso AV (Kinematic Sequence only)
                        if handedness == "L":
                            norm_torso_values.append(-v)
                        else:
                            norm_torso_values.append(v)

                grouped_torso[take_id] = {
                    "frame": norm_torso_frames,
                    "value": norm_torso_values
                }

                if norm_torso_frames and display_mode == "Individual Throws":
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
                            showlegend=False
                        )
                    )
                    # Legend-only trace (once per Torso + Date)
                    legend_key = ("Torso", take_date_map[take_id])
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
                                name=f"Torso AV | {take_date_map[take_id]}",
                                showlegend=True
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
                    if rel_frame >= window_start and rel_frame <= 50:
                        norm_elbow_frames.append(rel_frame)
                        # Flip sign so elbow extension is positive on the plot
                        norm_elbow_values.append(-v)

                grouped_elbow[take_id] = {
                    "frame": norm_elbow_frames,
                    "value": norm_elbow_values
                }

                if norm_elbow_frames and display_mode == "Individual Throws":
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
                            showlegend=False
                        )
                    )
                    # Legend-only trace (once per Elbow + Date)
                    legend_key = ("Elbow", take_date_map[take_id])
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
                                name=f"Elbow AV | {take_date_map[take_id]}",
                                showlegend=True
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
                    if rel_frame >= window_start and rel_frame <= 50:
                        norm_sh_frames.append(rel_frame)
                        # Normalize so IR velocity is positive for both handedness
                        if handedness == "L":
                            norm_sh_values.append(-v)
                        else:
                            norm_sh_values.append(v)

                grouped_shoulder_ir[take_id] = {
                    "frame": norm_sh_frames,
                    "value": norm_sh_values
                }

                if norm_sh_frames and display_mode == "Individual Throws":
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
                            showlegend=False
                        )
                    )
                    # Legend-only trace (once per Shoulder IR + Date)
                    legend_key = ("Shoulder IR", take_date_map[take_id])
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
                                name=f"Shoulder IR AV | {take_date_map[take_id]}",
                                showlegend=True
                            )
                        )
                        legend_keys_added.add(legend_key)
            if not norm_frames:
                continue

            if display_mode == "Individual Throws":
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
                        showlegend=False
                    )
                )
                # Legend-only trace (once per Pelvis + Date)
                legend_key = ("Pelvis", take_date_map[take_id])
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
                            name=f"Pelvis AV | {take_date_map[take_id]}",
                            showlegend=True
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

                x, y, q1, q3 = aggregate_curves(curves, "Mean")
                color = color_map[label]

                # --- Savitzky–Golay smoothing (grouped curves only) ---
                # Preserves peak timing and magnitude
                if len(y) >= 11:
                    y  = savgol_filter(y,  window_length=7, polyorder=3)
                    #q1 = savgol_filter(q1, window_length=11, polyorder=3)
                    #q3 = savgol_filter(q3, window_length=11, polyorder=3)

                # --- Grouped curve (NO legend entry, color = segment, dash = session date) ---
                # Determine date from the first take in this segment's group
                # (all takes in curves should be from the same date in grouped mode)
                # But in this code, curves is {take_id: ...} for all takes, possibly from multiple dates
                # For condensed legend, we want one entry per (segment x date)
                # So, for each unique date in this segment's curves, plot the curve with its dash and color
                # (But in the current grouped code, only one curve is plotted per segment, not per date)
                # To match individual condensed legend, we need to plot one grouped curve per (segment x date)
                # So, group curves by date:
                from collections import defaultdict
                curves_by_date = defaultdict(dict)
                for take_id, d in curves.items():
                    date = take_date_map[take_id]
                    curves_by_date[date][take_id] = d
                for date, curves_date in curves_by_date.items():
                    x_date, y_date, q1_date, q3_date = aggregate_curves(curves_date, "Mean")
                    # Smoothing
                    if len(y_date) >= 11:
                        y_date = savgol_filter(y_date, window_length=7, polyorder=3)
                    dash = date_dash_map[date]
                    # --- Grouped curve (no legend) ---
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
                            showlegend=False
                        )
                    )
                    # --- Legend-only trace (once per Segment + Date) ---
                    legend_key = (label, date)
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
                                name=f"{label} AV | {date}",
                                showlegend=True
                            )
                        )
                        legend_keys_added.add(legend_key)
                    # --- Peak arrow and marker for this grouped curve ---
                    if len(y_date) > 0:
                        # Restrict pelvis & torso peak search to FP → BR
                        if label in ["Pelvis", "Torso"] and fp_event_frames:
                            fp_rel = int(np.median(fp_event_frames))
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
                        # Store peak for summary table (only once per segment, keep old logic)
                        if date == list(curves_by_date.keys())[0]:
                            pelvis_time_ms_grouped = None
                            if label == "Pelvis" and fp_event_frames:
                                fp_rel = int(np.median(fp_event_frames))  # FP relative to BR (frames)
                                pelvis_time_ms_grouped = (max_x - fp_rel) * MS_PER_FRAME

                            kinematic_peak_rows.append({
                                "Segment": label,
                                "Peak Value (°/s)": max_y,
                                "Peak Frame (rel BR)": max_x,
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
                    # --- IQR band (always shown for grouped mode) ---
                    fig.add_trace(
                        go.Scatter(
                            x=x_date + x_date[::-1],
                            y=q3_date + q1_date[::-1],
                            fill="toself",
                            fillcolor=to_rgba(color, alpha=0.30),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip"
                        )
                    )

        # Median Peak Glove-Side Knee Height event
        if knee_event_frames:
            median_knee_frame = int(np.median(knee_event_frames))

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
            median_fp_frame = int(np.median(fp_event_frames))

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
            median_mer_frame = int(np.median(mer_event_frames))

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
            xaxis_title="Frames Relative to Ball Release",
            yaxis_title="Angular Velocity",
            yaxis=dict(ticksuffix="°/s"),
            xaxis_range=[window_start, 50],
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
                    pelvis_time_ms = (pelvis_frame - fp_rel) * MS_PER_FRAME
                torso_peak, torso_frame = peak_and_frame(grouped_torso)
                elbow_peak, elbow_frame = peak_and_frame(grouped_elbow)
                shoulder_peak, shoulder_frame = peak_and_frame(grouped_shoulder_ir)

                individual_rows.append({
                    "Session Date": take_date_map[take_id],
                    "Pitch": take_order[take_id],
                    "Velocity (mph)": take_velocity[take_id],
                    "Pelvis Peak (°/s)": pelvis_peak,
                    "Pelvis Time from FP (ms)": pelvis_time_ms,
                    "Torso Peak (°/s)": torso_peak,
                    "Torso Frame": torso_frame,
                    "Elbow Peak (°/s)": elbow_peak,
                    "Elbow Frame": elbow_frame,
                    "Shoulder IR Peak (°/s)": shoulder_peak,
                    "Shoulder IR Frame": shoulder_frame
                })

            if individual_rows:
                import pandas as pd

                st.markdown("### Kinematic Sequence - Individual Throws")

                df_individual = pd.DataFrame(individual_rows)

                # Sort logically: date → pitch order
                df_individual = df_individual.sort_values(
                    ["Session Date", "Pitch"]
                )

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
                        "Torso Frame": lambda x: fmt(x, 0),
                        "Elbow Frame": lambda x: fmt(x, 0),
                        "Shoulder IR Frame": lambda x: fmt(x, 0),
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

            # Pivot into segment-based column groups
            pivot = {}
            for _, row in df.iterrows():
                seg = row["Segment"]
                pivot[(seg, "Peak (°/s)")] = [row["Peak Value (°/s)"]]
                pivot[(seg, "Peak Frame")] = [row["Peak Frame (rel BR)"]]

                if seg == "Pelvis":
                    pivot[(seg, "Peak Time from FP (ms)")] = [row.get("Peak Time from FP (ms)")]

            df_pivot = pd.DataFrame(pivot)
            df_pivot.columns = pd.MultiIndex.from_tuples(df_pivot.columns)

            # --- Styling ---
            segment_colors = {
                "Pelvis":   "rgba(0, 128, 0, 0.12)",     # green
                "Torso":    "rgba(255, 215, 0, 0.12)",   # yellow
                "Elbow":    "rgba(128, 0, 128, 0.12)",   # purple
                "Shoulder": "rgba(255, 0, 0, 0.12)"      # red
            }

            def style_segments(col):
                seg = col[0]
                if seg in segment_colors:
                    return [f"background-color: {segment_colors[seg]}"] * len(df_pivot)
                return [""] * len(df_pivot)

            # --- Pre-format values safely (NO Styler value mutation) ---
            df_display = df_pivot.copy()

            for col in df_display.columns:
                if "Peak" in col[1] and "°/s" in col[1]:
                    df_display[col] = df_display[col].map(lambda x: f"{x:.1f}")
                elif "Time from FP" in col[1]:
                    df_display[col] = df_display[col].map(lambda x: "" if x is None else f"{x:.0f}")

            # --- Styling (CSS only) ---
            styled = (
                df_display
                .style
                .apply(style_segments, axis=0)
                .set_properties(**{
                    "text-align": "center",
                    "font-weight": "500"
                })
            )

            st.dataframe(styled, use_container_width=True)


with tab_joint:
    st.subheader("Joint Angles")
    display_mode = st.radio(
        "Select Display Mode",
        ["Individual Throws", "Grouped"],
        index=1,
        horizontal=True,
        key="joint_display_mode"
    )

    selected_kinematics = st.multiselect(
        "Select Kinematics",
        options=[
            "Elbow Flexion",
            "Shoulder ER",
            "Shoulder Abduction",
            "Shoulder Horizontal Abduction",
            "Front Knee Flexion",
            "Front Knee Extension Velocity",
            "Forward Trunk Tilt",
            "Lateral Trunk Tilt",
            "Trunk Angle",
            "Pelvis Angle",
            "Hip-Shoulder Separation",
            "Forearm Pronation/Supination"
        ],
        default=["Elbow Flexion"],
        key="joint_angles_select"
    )

    if not selected_kinematics:
        st.info("Select at least one kinematic.")
        st.stop()

    # --- Color map for joint types ---
    joint_color_map = {
        "Elbow Flexion": "purple",
        "Shoulder ER": "teal",
        "Shoulder Abduction": "orange",
        "Shoulder Horizontal Abduction": "brown",
        "Forearm Pronation/Supination": "crimson"
    }
    joint_color_map.update({
        "Forward Trunk Tilt": "blue",
        "Lateral Trunk Tilt": "green",
        "Trunk Angle": "black"
    })
    joint_color_map.update({
        "Pelvis Angle": "darkblue",
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

    if "Elbow Flexion" in selected_kinematics:
        joint_data["Elbow Flexion"] = get_elbow_flexion_angle(take_ids, handedness)

    if "Shoulder ER" in selected_kinematics:
        joint_data["Shoulder ER"] = get_shoulder_er_angle(take_ids, handedness)

    if "Shoulder Abduction" in selected_kinematics:
        joint_data["Shoulder Abduction"] = get_shoulder_abduction_angle(take_ids, handedness)

    if "Shoulder Horizontal Abduction" in selected_kinematics:
        joint_data["Shoulder Horizontal Abduction"] = get_shoulder_horizontal_abduction_angle(
            take_ids, handedness
        )

    if "Forearm Pronation/Supination" in selected_kinematics:
        joint_data["Forearm Pronation/Supination"] = get_forearm_pron_sup_angle(
            take_ids, handedness
        )

    if "Front Knee Flexion" in selected_kinematics:
        joint_data["Front Knee Flexion"] = get_front_knee_flexion_angle(
            take_ids, handedness
        )

    if "Front Knee Extension Velocity" in selected_kinematics:
        joint_data["Front Knee Extension Velocity"] = get_front_knee_extension_velocity(
            take_ids, handedness
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

    if "Hip-Shoulder Separation" in selected_kinematics:
        joint_data["Hip-Shoulder Separation"] = get_hip_shoulder_separation(take_ids)

    # --- Helper for extracting value at a specific frame ---
    def value_at_frame(frames, values, target_frame):
        if target_frame in frames:
            return values[frames.index(target_frame)]
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

    # For condensed legend: track which (kinematic, date) pairs have legend entries
    legend_keys_added = set()

    # Reuse take_order and take_velocity from Kinematic Sequence section if available
    for kinematic, data_dict in joint_data.items():
        grouped[kinematic] = {}

        for take_id in take_ids:
            if take_id not in data_dict or take_id not in br_frames:
                continue

            frames = data_dict[take_id]["frame"]
            values = data_dict[take_id]["value"]
            br = br_frames[take_id]

            norm_f, norm_v = [], []
            for f, v in zip(frames, values):
                if v is None:
                    continue

                rel = f - br
                if window_start <= rel <= 50:
                    norm_f.append(rel)

                    # --- Handedness normalization ---
                    if handedness == "R" and kinematic in [
                        "Shoulder Horizontal Abduction",
                        "Shoulder ER"
                    ]:
                        norm_v.append(-v)
                    else:
                        norm_v.append(v)

            grouped[kinematic][take_id] = {"frame": norm_f, "value": norm_v}

            # --- Store by date for grouped plotting ---
            date = take_date_map[take_id]
            grouped_by_date.setdefault(date, {}).setdefault(kinematic, {})[take_id] = {
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
                        name=f"{kinematic} – {take_date_map[take_id]} | Pitch {take_order[take_id]} ({take_velocity[take_id]:.1f} mph)",
                        showlegend=False
                    )
                )
                # Add one legend-only trace per (kinematic, date) (shows color + dash)
                legend_key = (kinematic, date)
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
                            name=f"{kinematic} | {date}",
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

                br_val = value_at_frame(frames, values, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = int(np.median(fp_event_frames))
                    fp_val = value_at_frame(frames, values, median_fp)

                # value at MER (same frame used in plot)
                mer_val = None
                if take_id in shoulder_er_max_frames:
                    mer_frame_rel = shoulder_er_max_frames[take_id] - br_frames[take_id]
                    mer_val = value_at_frame(frames, values, mer_frame_rel)

                summary_rows.append({
                    "Kinematic": kinematic,
                    "Session Date": take_date_map[take_id],
                    "Average Velocity": take_velocity[take_id],
                    "Max": max_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val
                })

    # --- Grouped plot (mean + IQR per date) ---
    if display_mode == "Grouped":
        for date, kin_dict in grouped_by_date.items():
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
                        name=f"{kinematic} – {date}"
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
                br_val = value_at_frame(x, y, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = int(np.median(fp_event_frames))
                    fp_val = value_at_frame(x, y, median_fp)

                max_vals = [np.max(d["value"]) for d in curves.values() if d["value"]]
                sd_val = np.std(max_vals)

                # value at MER from grouped mean curve
                mer_val = None
                if mer_event_frames:
                    median_mer = int(np.median(mer_event_frames))
                    mer_val = value_at_frame(x, y, median_mer)

                summary_rows.append({
                    "Kinematic": kinematic,
                    "Session Date": date,
                    "Average Velocity": np.mean([take_velocity[tid] for tid in curves.keys()]),
                    "Max": max_val,
                    "Foot Plant": fp_val,
                    "Ball Release": br_val,
                    "Max External Rotation": mer_val,
                    "Standard Deviation": sd_val
                })

    # --- Event lines and annotations (match Kinematic Sequence styling) ---
    if knee_event_frames:
        median_knee_frame = int(np.median(knee_event_frames))
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

    if fp_event_frames:
        median_fp_frame = int(np.median(fp_event_frames))
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
        median_mer_frame = int(np.median(mer_event_frames))
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
        xaxis_title="Frames Relative to Ball Release",
        yaxis_title="Joint Angle",
        yaxis=dict(ticksuffix="°"),
        xaxis_range=[window_start, 50],
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

    # --- Joint Angle Table ---
    if summary_rows:
        st.markdown("### Joint Angles Summary")
        df_summary = pd.DataFrame(summary_rows)
        # Reorder columns explicitly
        base_columns = [
            "Kinematic",
            "Session Date",
            "Average Velocity",
            "Max",
            "Foot Plant",
            "Ball Release",
            "Max External Rotation"
        ]

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
        "Trunk-Shoulder Rotational Energy Flow": "#DC2626",  # strong red
        "Trunk-Shoulder Elevation/Depression Energy Flow": "#2563EB",  # vivid blue
        "Trunk-Shoulder Horizontal Abd/Add Energy Flow": "#16A34A",     # strong green
        "Arm Rotational Energy Flow": "#F59E0B",        # amber
        "Arm Elevation/Depression Energy Flow": "#06B6D4",  # cyan
        "Arm Horizontal Abd/Add Energy Flow": "#9333EA"     # violet
    }

    # --- Load all selected metrics ---
    energy_data_by_metric = {}

    for metric in energy_metrics:
        if metric == "Distal Arm Segment Power":
            energy_data_by_metric[metric] = get_distal_arm_segment_power(take_ids, handedness)
        elif metric == "Trunk-Shoulder Rotational Energy Flow":
            energy_data_by_metric[metric] = get_trunk_shoulder_rot_energy_flow(take_ids, handedness)
        elif metric == "Trunk-Shoulder Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = get_trunk_shoulder_elev_energy_flow(take_ids, handedness)
        elif metric == "Trunk-Shoulder Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = get_trunk_shoulder_horizabd_energy_flow(take_ids, handedness)
        elif metric == "Arm Rotational Energy Flow":
            energy_data_by_metric[metric] = get_arm_rot_energy_flow(take_ids, handedness)
        elif metric == "Arm Elevation/Depression Energy Flow":
            energy_data_by_metric[metric] = get_arm_elev_energy_flow(take_ids, handedness)
        elif metric == "Arm Horizontal Abd/Add Energy Flow":
            energy_data_by_metric[metric] = get_arm_horizabd_energy_flow(take_ids, handedness)

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
                if window_start <= rel <= 50:
                    norm_f.append(rel)
                    norm_v.append(v)

            grouped_power[take_id] = {"frame": norm_f, "value": norm_v}

            date = take_date_map[take_id]
            grouped_by_date.setdefault(date, {})[take_id] = {
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
                        showlegend=False
                    )
                )
                # --- Label take with Pitch # and Velocity (with marker for hover) ---
                if norm_f and norm_v:
                    fig.add_trace(
                        go.Scatter(
                            x=[norm_f[-1]],
                            y=[norm_v[-1]],
                            mode="markers+text",
                            marker=dict(
                                size=6,
                                color=metric_color,
                                opacity=0.9
                            ),
                            text=[f"P{take_order[take_id]} | {take_velocity[take_id]:.1f} mph"],
                            textposition="top right",
                            textfont=dict(
                                size=11,
                                color=metric_color
                            ),
                            hovertext=[
                                f"""
                                Pitch: {take_order[take_id]}<br>
                                Velocity: {take_velocity[take_id]:.1f} mph<br>
                                Metric: {metric}<br>
                                Date: {take_date_map[take_id]}
                                """
                            ],
                            hoverinfo="text",
                            showlegend=False
                        )
                    )

                legend_key = (metric, date)
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
                            name=f"{metric} | {date}",
                            showlegend=True
                        )
                    )
                    legend_keys_added.add(legend_key)

        # -------------------------------
        # Grouped (Mean + IQR per date)
        # -------------------------------
        if display_mode == "Grouped":
            for date, curves in grouped_by_date.items():
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
                        showlegend=False
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=q3 + q1[::-1],
                        fill="toself",
                        fillcolor=f"{metric_color}55",
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
                        name=f"{metric} | {date}",
                        showlegend=True
                    )
                )

    # -------------------------------
    # Event Lines (REUSED — NO CHANGES)
    # -------------------------------
    if fp_event_frames:
        median_fp = int(np.median(fp_event_frames))
        fig.add_vline(x=median_fp, line_width=3, line_dash="dash", line_color="green")

    if mer_event_frames:
        median_mer = int(np.median(mer_event_frames))
        fig.add_vline(x=median_mer, line_width=3, line_dash="dash", line_color="red")

    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="blue")

    fig.update_layout(
        xaxis_title="Frames Relative to Ball Release",
        yaxis_title="Energy Flow / Segment Power",
        xaxis_range=[window_start, 50],
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
    # Energy Flow – Time-Series Data Table (Exact Plot Data)
    # --------------------------------------------------
    import pandas as pd

    table_rows = []

    if display_mode == "Individual Throws":
        for metric, energy_data in energy_data_by_metric.items():
            for take_id, d in energy_data.items():
                if take_id not in br_frames:
                    continue

                frames = d.get("frame", [])
                values = d.get("value", [])
                br = br_frames[take_id]

                for f, v in zip(frames, values):
                    rel = f - br
                    if window_start <= rel <= 50:
                        table_rows.append({
                            "Session Date": take_date_map.get(take_id),
                            "Pitch": take_order.get(take_id),
                            "Velocity (mph)": take_velocity.get(take_id),
                            "Metric": metric,
                            "Frame (rel BR)": rel,
                            "Value": v
                        })

    else:  # Grouped
        for metric, energy_data in energy_data_by_metric.items():
            grouped_by_date = {}
            for take_id, d in energy_data.items():
                if take_id not in br_frames:
                    continue

                date = take_date_map[take_id]
                grouped_by_date.setdefault(date, []).append((take_id, d))

            for date, items in grouped_by_date.items():
                curves = {
                    tid: {
                        "frame": [
                            f - br_frames[tid]
                            for f in d["frame"]
                            if window_start <= (f - br_frames[tid]) <= 50
                        ],
                        "value": [
                            v for f, v in zip(d["frame"], d["value"])
                            if window_start <= (f - br_frames[tid]) <= 50
                        ]
                    }
                    for tid, d in items
                }

                x, y, _, _ = aggregate_curves(curves, "Mean")

                for f, v in zip(x, y):
                    table_rows.append({
                        "Session Date": date,
                        "Pitch": "Grouped",
                        "Velocity (mph)": None,
                        "Metric": metric,
                        "Frame (rel BR)": f,
                        "Value": v
                    })

    if table_rows:
        df_energy_ts = pd.DataFrame(table_rows)

        with st.expander("Show Energy Flow Time-Series Data"):
            st.dataframe(
                df_energy_ts,
                use_container_width=True,
                hide_index=True
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