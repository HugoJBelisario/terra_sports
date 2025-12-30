import streamlit as st
import numpy as np
from scipy.signal import savgol_filter
from dotenv import load_dotenv
from db.connection import get_connection

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
    Returns shoulder external rotation angle (x_data) for the throwing shoulder.

    Category: ORIGINAL
    Segments:
      RHP → RT_SHOULDER_ANGLE
      LHP → LT_SHOULDER_ANGLE
    """
    if not take_ids or handedness not in ("R", "L"):
        return {}

    segment_name = (
        "RT_SHOULDER_ANGLE"
        if handedness == "R"
        else "LT_SHOULDER_ANGLE"
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
                JOIN segments s ON ts.segment_id = s.segment_id
                WHERE c.category_name = 'ORIGINAL'
                  AND s.segment_name = %s
                  AND ts.take_id IN ({placeholders})
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
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Terra Sports Dashboard",
    layout="wide",
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
# --- Sidebar Logo ---
st.sidebar.image(
    "assets/terra_sports.svg",
    use_container_width=True
)

st.sidebar.markdown("### Dashboard Controls")

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

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Secure Cloud Database")

if st.sidebar.button("Test Database Connection"):
    try:
        conn = get_connection()
        conn.close()
        st.sidebar.success("Database connection successful.")
    except Exception as e:
        st.sidebar.error("Database connection failed.")

# --------------------------------------------------
# Main header
# --------------------------------------------------
st.title("Terra Sports Performance Dashboard")

st.markdown(
    """
    This dashboard provides **performance, biomechanics, and training insights**
    powered by secure cloud infrastructure.
    """
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_overview, tab_metrics, tab_raw, tab_notes, tab_kinematic, tab_joint = st.tabs(
    ["Overview", "Metrics", "Raw Data", "Notes", "Kinematic Sequence", "Joint Angles"]
)

# --------------------------------------------------
# Overview Tab
# --------------------------------------------------
with tab_overview:
    st.subheader("Overview")

    st.info(
        "This section will contain high-level summaries, KPIs, and trends."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Athletes", value="—")

    with col2:
        st.metric(label="Sessions", value="—")

    with col3:
        st.metric(label="Avg Velocity", value="—")

# --------------------------------------------------
# Metrics Tab
# --------------------------------------------------
with tab_metrics:
    st.subheader("Performance Metrics")

    st.warning(
        "Detailed metrics and visualizations will appear here."
    )

    st.markdown(
        "- Velocity\n"
        "- Spin\n"
        "- Biomechanics\n"
        "- Force Plate Metrics"
    )

# --------------------------------------------------
# Raw Data Tab
# --------------------------------------------------
with tab_raw:
    st.subheader("Raw Data Explorer")

    st.warning(
        "Raw tables and filters will be added here."
    )

    st.markdown(
        "This tab is intended for deeper inspection and exports."
    )

# --------------------------------------------------
# Notes / Export Tab
# --------------------------------------------------
with tab_notes:
    st.subheader("Notes & Export")

    st.text_area(
        "Session / Athlete Notes",
        placeholder="Enter observations, coaching notes, or annotations here…",
        height=150,
    )

    st.button("Export Report (Coming Soon)")

# --------------------------------------------------
# Kinematic Sequence Tab
# --------------------------------------------------
with tab_kinematic:
    st.subheader("Kinematic Sequence")

    display_mode = st.radio(
        "Display Mode",
        options=["Individual Throws", "Grouped (by Date)"],
        horizontal=True
    )

    grouped_stat = st.selectbox(
        "Grouped Curve Statistic",
        options=["Median", "Mean"],
        disabled=(display_mode == "Individual Throws")
    )

    show_iqr = st.checkbox(
        "Show IQR (25–75%)",
        value=True,
        disabled=(display_mode == "Individual Throws")
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
                        norm_torso_values.append(v)

                grouped_torso[take_id] = {
                    "frame": norm_torso_frames,
                    "value": norm_torso_values
                }

                if norm_torso_frames and display_mode == "Individual Throws":
                    fig.add_trace(
                        go.Scatter(
                            x=norm_torso_frames,
                            y=norm_torso_values,
                            mode="lines",
                            name=f"Torso AV – Take {take_id}",
                            line=dict(color="orange"),
                            showlegend=False
                        )
                    )

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
                    fig.add_trace(
                        go.Scatter(
                            x=norm_elbow_frames,
                            y=norm_elbow_values,
                            mode="lines",
                            name=f"Elbow AV – Take {take_id}",
                            line=dict(color="green"),
                            showlegend=False
                        )
                    )

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
                    fig.add_trace(
                        go.Scatter(
                            x=norm_sh_frames,
                            y=norm_sh_values,
                            mode="lines",
                            name=f"Shoulder IR AV – Take {take_id}",
                            line=dict(color="red"),
                            showlegend=False
                        )
                    )
            if not norm_frames:
                continue

            if display_mode == "Individual Throws":
                # Plot normalized pelvis angular velocity
                fig.add_trace(
                    go.Scatter(
                        x=norm_frames,
                        y=norm_values,
                        mode="lines",
                        name=f"Pelvis AV – Take {take_id}",
                        line=dict(color="blue"),
                        showlegend=True
                    )
                )

        if display_mode == "Grouped (by Date)":
            color_map = {
                "Pelvis AV": "blue",
                "Torso AV": "orange",
                "Elbow AV": "green",
                "Shoulder IR AV": "red"
            }

            for label, curves in [
                ("Pelvis AV", grouped_pelvis),
                ("Torso AV", grouped_torso),
                ("Elbow AV", grouped_elbow),
                ("Shoulder IR AV", grouped_shoulder_ir)
            ]:
                if not curves:
                    continue

                x, y, q1, q3 = aggregate_curves(curves, grouped_stat)
                color = color_map[label]

                # --- Savitzky–Golay smoothing (grouped curves only) ---
                # Preserves peak timing and magnitude
                if len(y) >= 11:
                    y  = savgol_filter(y,  window_length=7, polyorder=3)
                    #q1 = savgol_filter(q1, window_length=11, polyorder=3)
                    #q3 = savgol_filter(q3, window_length=11, polyorder=3)

                # --- Grouped curve ---
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{label} ({grouped_stat})",
                        line=dict(width=4, color=color),
                    )
                )

                # --- Peak arrow (same color as curve) ---
                max_idx = int(np.argmax(y))
                max_x = x[max_idx]
                max_y = y[max_idx]

                # --- Peak marker (small downward arrow above max, matplotlib-style) ---
                y_offset = 0.04 * max(abs(v) for v in y)

                fig.add_trace(
                    go.Scatter(
                        x=[max_x],
                        y=[max_y + y_offset],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color=color
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

                # --- IQR band ---
                if show_iqr:
                    rgba_map = {
                        "blue": "0,0,255",
                        "orange": "255,165,0",
                        "green": "0,128,0",
                        "red": "255,0,0"
                    }

                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=q3 + q1[::-1],
                            fill="toself",
                            fillcolor=f"rgba({rgba_map[color]},0.35)",
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
                y=1.02,
                xref="x",
                yref="paper",
                text="Knee",
                showarrow=False,
                font=dict(color="blue", size=12),
                align="center"
            )

        # Median Refined Foot Plant (zero-cross) event
        if fp_event_frames:
            median_fp_frame = int(np.median(fp_event_frames))

            fig.add_vline(
                x=median_fp_frame,
                line_width=3,
                line_dash="dot",
                line_color="green",
                opacity=0.9
            )
            fig.add_annotation(
                x=median_fp_frame,
                y=1.02,
                xref="x",
                yref="paper",
                text="FP",
                showarrow=False,
                font=dict(color="green", size=12),
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
                y=1.02,
                xref="x",
                yref="paper",
                text="MER",
                showarrow=False,
                font=dict(color="red", size=12),
                align="center"
            )

        # Normalized Ball Release reference line
        fig.add_vline(
            x=0,
            line_width=4,
            line_dash="solid",
            line_color="black",
            opacity=0.8
        )
        fig.add_annotation(
            x=0,
            y=1.02,
            xref="x",
            yref="paper",
            text="BR",
            showarrow=False,
            font=dict(color="black", size=12, family="Arial Black"),
            align="center"
        )

        fig.update_layout(
            xaxis_title="Frames Relative to Ball Release",
            yaxis_title="Angular Velocity",
            xaxis_range=[window_start, 50],
            height=600,
            legend_title_text="Take ID"
        )

        st.plotly_chart(fig, use_container_width=True)

with tab_joint:
    st.subheader("Joint Angles")

    display_mode = st.radio(
        "Display Mode",
        ["Individual Throws", "Grouped (by Date)"],
        horizontal=True,
        key="joint_mode"
    )

    grouped_stat = st.selectbox(
        "Grouped Curve Statistic",
        ["Median", "Mean"],
        disabled=(display_mode == "Individual Throws"),
        key="joint_stat"
    )

    show_iqr = st.checkbox(
        "Show IQR (25–75%)",
        value=True,
        disabled=(display_mode == "Individual Throws"),
        key="joint_iqr"
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
            "Hip-Shoulder Separation"
        ],
        default=["Elbow Flexion"]
    )

    if not selected_kinematics:
        st.info("Select at least one kinematic.")
        st.stop()

    # --- Color map for joint types ---
    joint_color_map = {
        "Elbow Flexion": "purple",
        "Shoulder ER": "teal",
        "Shoulder Abduction": "orange",
        "Shoulder Horizontal Abduction": "brown"
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

    # --- Per-take normalization and plotting ---
    grouped = {}

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

            if display_mode == "Individual Throws":
                fig.add_trace(
                    go.Scatter(
                        x=norm_f,
                        y=norm_v,
                        mode="lines",
                        line=dict(color=joint_color_map[kinematic]),
                        name=f"{kinematic} – {take_id}"
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
                sd_val = np.std(values)

                br_val = value_at_frame(frames, values, 0)

                fp_val = None
                if fp_event_frames:
                    median_fp = int(np.median(fp_event_frames))
                    fp_val = value_at_frame(frames, values, median_fp)

                summary_rows.append({
                    "Kinematic": kinematic,
                    "Take": take_id,
                    "Max": max_val,
                    "At FP": fp_val,
                    "At BR": br_val,
                    "SD": sd_val
                })

    # --- Grouped plot (single curve + IQR) ---
    if display_mode == "Grouped (by Date)" and grouped:
        for kinematic, curves in grouped.items():
            if not curves:
                continue

            x, y, q1, q3 = aggregate_curves(curves, grouped_stat)

            # Smooth grouped curve ONLY
            if len(y) >= 11:
                y = savgol_filter(y, window_length=11, polyorder=3)

            color = joint_color_map[kinematic]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=4, color=color),
                    name=f"{kinematic} ({grouped_stat})"
                )
            )

            if show_iqr:
                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=q3 + q1[::-1],
                        fill="toself",
                        fillcolor=f"rgba(0,0,0,0.25)",
                        line=dict(width=0),
                        showlegend=False
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

            summary_rows.append({
                "Kinematic": kinematic,
                "Take": grouped_stat,
                "Max": max_val,
                "At FP": fp_val,
                "At BR": br_val,
                "SD": sd_val
            })

    # Event lines (remain after grouped plotting)
    if knee_event_frames:
        fig.add_vline(x=int(np.median(knee_event_frames)), line_dash="dot", line_color="blue")
    if fp_event_frames:
        fig.add_vline(x=int(np.median(fp_event_frames)), line_dash="dot", line_color="green")
    if mer_event_frames:
        fig.add_vline(x=int(np.median(mer_event_frames)), line_dash="dash", line_color="red")

    fig.add_vline(x=0, line_width=4, line_color="black")

    fig.update_layout(
        xaxis_title="Frames Relative to Ball Release",
        yaxis_title="Joint Angle (deg)",
        xaxis_range=[window_start, 50],
        height=600
    )

    # --- Display the summary table before the plot ---
    if summary_rows:
        st.markdown("### Joint Angle Summary")
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(
            df_summary.style.format({
                "Max": "{:.2f}",
                "At FP": "{:.2f}",
                "At BR": "{:.2f}",
                "SD": "{:.2f}"
            }),
            use_container_width=True
        )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("© Terra Sports | Internal Analytics Dashboard")