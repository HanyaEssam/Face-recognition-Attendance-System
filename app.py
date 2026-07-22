"""
app.py — Streamlit dashboard: live check-in/out + attendance analytics.

Run with:
    streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from db import init_db, get_all_employees, log_check_in, log_check_out, get_attendance_df
from face_pipeline import FacePipeline, LivenessChecker

st.set_page_config(page_title="Attendance Dashboard", layout="wide", page_icon="🪪")
init_db()

# ---------------- Styling ----------------
# Cards are built with plain HTML/CSS instead of st.metric, so the design
# doesn't depend on Streamlit's internal data-testid names (which change
# between versions and caused the previous styling to silently not apply).
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6D28D9, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle { color: #6B7280; margin-bottom: 1.5rem; }

    .card {
        border-radius: 14px;
        padding: 18px 20px;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.10);
    }
    .card-label { font-size: 0.85rem; opacity: 0.9; margin-bottom: 4px; }
    .card-value { font-size: 2.1rem; font-weight: 800; }

    .card-purple { background: linear-gradient(135deg, #7C3AED, #5B21B6); }
    .card-green  { background: linear-gradient(135deg, #10B981, #047857); }
    .card-red    { background: linear-gradient(135deg, #F43F5E, #BE123C); }
    .card-blue   { background: linear-gradient(135deg, #3B82F6, #1D4ED8); }
    .card-amber  { background: linear-gradient(135deg, #F59E0B, #B45309); }

    .alert-box {
        padding: 14px 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-top: 8px;
    }
    .alert-success { background: #DCFCE7; color: #166534; border-left: 5px solid #22C55E; }
    .alert-warning { background: #FEF9C3; color: #854D0E; border-left: 5px solid #EAB308; }
    .alert-danger  { background: #FEE2E2; color: #991B1B; border-left: 5px solid #EF4444; }
    .alert-info    { background: #DBEAFE; color: #1E40AF; border-left: 5px solid #3B82F6; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7C3AED, #2563EB) !important;
        color: white !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #7C3AED, #2563EB);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(col, label, value, style):
    col.markdown(f"""
        <div class="card {style}">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def alert(box, text, kind):
    box.markdown(f'<div class="alert-box alert-{kind}">{text}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return FacePipeline(), LivenessChecker()

pipeline, liveness = load_pipeline()

st.markdown('<div class="app-title">Face Recognition Attendance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Proof of concept — live check-in/out, employee directory, and attendance analytics</div>', unsafe_allow_html=True)

tab_checkin, tab_employees, tab_log, tab_summary = st.tabs(
    ["📸  Check In / Out", "👥  Employees", "📋  Attendance Log", "📊  Summary"]
)

# ---------------- Check In / Out ----------------
with tab_checkin:
    col1, col2 = st.columns([1, 1])
    with col1:
        img_file = st.camera_input("Look at the camera, then take a photo")
    with col2:
        action = st.radio("Action", ["Check In", "Check Out"], horizontal=True)
        result_box = st.empty()

    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        employees = get_all_employees()
        faces = pipeline.detect_faces(frame)

        if faces is None or len(faces) == 0:
            alert(result_box, "No face detected. Try again with better lighting and framing.", "warning")
        elif not employees:
            alert(result_box, "No employees enrolled yet. Run the enrollment script first.", "info")
        else:
            face_row = faces[0]
            is_live, live_score, live_details = liveness.is_live(frame, face_row, debug=True)

            # TEMPORARY: shows exactly what the liveness model is seeing so we
            # can calibrate it to your camera. Remove this st.json line once
            # the threshold is dialed in.
            st.json(live_details)

            if not is_live:
                alert(result_box,
                      f"Liveness check failed (score {live_score:.2f}). "
                      f"This looks like a photo or screen — please use your real face.",
                      "danger")
            else:
                embedding = pipeline.get_embedding(frame, face_row)
                match, score = pipeline.match(embedding, employees)

                if match is None:
                    alert(result_box, f"Face not recognized (best similarity {score:.2f}).", "danger")
                else:
                    if action == "Check In":
                        status = log_check_in(match["id"], match["shift_start"])
                        if status == "already_checked_in":
                            alert(result_box, f"{match['name']} is already checked in for today.", "info")
                        else:
                            clean_status = status.replace('_', ' ').title()
                            kind = "success" if status == "on_time" else "warning"
                            alert(result_box,
                                  f"{match['name']} checked in — {clean_status} "
                                  f"(similarity {score:.2f}, liveness {live_score:.2f})",
                                  kind)
                    else:
                        status = log_check_out(match["id"])
                        if status == "no_check_in_found":
                            alert(result_box, f"{match['name']} has no check-in recorded today.", "warning")
                        else:
                            alert(result_box, f"{match['name']} checked out successfully (similarity {score:.2f}).", "success")

# ---------------- Employees ----------------
with tab_employees:
    employees = get_all_employees()
    if employees:
        df_emp = pd.DataFrame([{"ID": e["id"], "Name": e["name"], "Department": e["department"],
                                 "Shift Start": e["shift_start"]} for e in employees])

        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("🔍 Search by employee name", placeholder="Type a name...")
        with col2:
            dept_filter = st.multiselect("Filter by department", options=sorted(df_emp["Department"].dropna().unique()), key="emp_dept_filter")

        filtered = df_emp.copy()
        if search_query:
            filtered = filtered[filtered["Name"].str.contains(search_query, case=False, na=False)]
        if dept_filter:
            filtered = filtered[filtered["Department"].isin(dept_filter)]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.info("No employees enrolled yet. Run `python enroll.py` from the terminal.")

# ---------------- Attendance Log ----------------
with tab_log:
    df = get_attendance_df()
    if df.empty:
        st.info("No attendance records generated yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            search_query = st.text_input("🔍 Search by employee name", key="log_search")
        with c2:
            dept_filter = st.multiselect("Filter by department", options=sorted(df["department"].dropna().unique()), key="log_dept_filter")
        with c3:
            status_filter = st.multiselect("Filter by status", options=sorted(df["status"].dropna().unique()), key="log_status_filter")

        filtered = df.copy()
        if search_query:
            filtered = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]
        if dept_filter:
            filtered = filtered[filtered["department"].isin(dept_filter)]
        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]

        st.dataframe(filtered, use_container_width=True, hide_index=True)

# ---------------- Summary ----------------
with tab_summary:
    df = get_attendance_df()
    employees = get_all_employees()

    if df.empty and not employees:
        st.info("No data available yet. Enroll employees to view analytics.")
    else:
        total_employees = len(employees)
        today_str = date.today().isoformat()
        if not df.empty:
            today_df = df[df["date"] == today_str]
            attended_today = today_df["name"].nunique()
        else:
            attended_today = 0
        absent_today = max(total_employees - attended_today, 0)

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "TOTAL EMPLOYEES", total_employees, "card-purple")
        metric_card(c2, "ATTENDED TODAY", attended_today, "card-green")
        metric_card(c3, "ABSENT TODAY", absent_today, "card-red")

        st.write("")

        if not df.empty:
            total = len(df)
            late_count = (df["status"] == "late").sum()
            on_time_count = (df["status"] == "on_time").sum()

            c4, c5 = st.columns(2)
            metric_card(c4, "ON-TIME % (ALL-TIME)", f"{on_time_count/total*100:.0f}%", "card-blue")
            metric_card(c5, "LATE % (ALL-TIME)", f"{late_count/total*100:.0f}%", "card-amber")

            st.write("")
            st.subheader("Attendance status by employee")
            pivot = df.groupby(["name", "status"]).size().unstack(fill_value=0)
            colors = ["#10B981", "#F59E0B"] if len(pivot.columns) > 1 else ["#3B82F6"]
            st.bar_chart(pivot, color=colors)
        else:
            st.write("No attendance history to chart yet.")