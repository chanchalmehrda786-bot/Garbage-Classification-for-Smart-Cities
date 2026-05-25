from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

try:
    import av
    import cv2
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
except Exception:
    av = None
    cv2 = None
    VideoProcessorBase = object
    webrtc_streamer = None

try:
    import tensorflow as tf
except Exception:
    tf = None


st.set_page_config(
    page_title="SmartWaste Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "waste_classification_model.h5",
    BASE_DIR / "Source Code" / "WebApp" / "waste_classification_model.h5",
]
DEMO_IMAGE_CANDIDATES = [
    BASE_DIR / "dataset-resized" / "plastic" / "plastic1.jpg",
    BASE_DIR / "dataset-resized" / "trash" / "trash1.jpg",
]

# ── Paths for persistent user registry and app state ──────────
USERS_PATH = BASE_DIR / "smartwaste_users.json"
APP_STATE_PATH = BASE_DIR / "smartwaste_state.json"
MODEL_CONFIG_PATH = BASE_DIR / "model_config.json"

IMG_SIZE = (150, 150)
NAV_ITEMS = [
    "Dashboard",
    "Live Detection",
    "Image Classification",
    "Waste Locations",
    "Collection Routes",
    "Analytics & Reports",
    "History",
    "Alerts",
    "Bin Status",
    "User Management",
    "Settings",
]

NAV_ICONS = {
    "Dashboard": "🏠",
    "Live Detection": "📹",
    "Image Classification": "🖼️",
    "Waste Locations": "📍",
    "Collection Routes": "🚛",
    "Analytics & Reports": "📊",
    "History": "🕐",
    "Alerts": "🔔",
    "Bin Status": "🗑️",
    "User Management": "👤",
    "Settings": "⚙️",
}

CATEGORY_META = {
    "Biodegradable": {
        "accent": "#1fb96d",
        "soft": "#ecfbf3",
        "icon": "🌿",
        "bin_label": "Wet waste bin",
        "examples": ["Banana Peel", "Leaf Waste", "Food Scraps", "Garden Waste"],
        "hint": "Send to composting or organic recovery.",
    },
    "Non-Biodegradable": {
        "accent": "#ef5d66",
        "soft": "#fff1f2",
        "icon": "🗑️",
        "bin_label": "Dry waste bin",
        "examples": ["Plastic Bottle", "Metal Can", "Glass Item", "Paper Cup"],
        "hint": "Keep it dry and route it for sorting or recycling.",
    },
}

CITY_OPTIONS = [
    "New Delhi, India", "Mumbai, India", "Bengaluru, India",
    "Chennai, India", "Hyderabad, India", "Kolkata, India",
    "Pune, India", "Ahmedabad, India",
]

ROLE_OPTIONS = ["Admin", "Operator", "Viewer"]


# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31,185,109,0.10), transparent 22%),
                radial-gradient(circle at top right, rgba(153,102,255,0.07), transparent 18%),
                linear-gradient(180deg, #fbfffd 0%, #f5fbf8 100%);
            color: #16312a;
        }

        /* ── Auth pages ── */
        .auth-outer {
            max-width: 480px;
            margin: 3rem auto 0 auto;
        }
        .login-shell {
            display: none;
        }
        .login-brand-row {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 2.2rem;
        }
        .login-brand-badge {
            width: 58px;
            height: 58px;
            border-radius: 15px;
            background: linear-gradient(135deg, #52d486, #16a05d);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.35rem;
            box-shadow: 0 12px 26px rgba(31, 185, 109, 0.22);
        }
        .login-brand-name {
            color: #102119;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .login-brand-subtitle {
            color: #6e7e76;
            font-size: 1rem;
            margin-top: 0.4rem;
        }
        .login-welcome-title {
            color: #102119;
            font-size: clamp(2.5rem, 5vw, 4rem);
            line-height: 1.05;
            font-weight: 800;
            margin: 0 0 1rem 0;
            letter-spacing: 0;
        }
        .login-welcome-copy {
            color: #697a72;
            font-size: 1.22rem;
            line-height: 1.65;
            max-width: 760px;
            margin: 0;
        }
        .login-chip-row {
            display: flex;
            gap: 0.9rem;
            flex-wrap: wrap;
            margin-top: 2rem;
        }
        .login-chip {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(31, 185, 109, 0.12);
            border-radius: 999px;
            color: #17824d;
            font-size: 0.95rem;
            font-weight: 800;
            padding: 0.78rem 1.15rem;
            box-shadow: 0 10px 26px rgba(26, 78, 52, 0.05);
        }
        .login-panel-title {
            color: #102119;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0 0 0.7rem 0;
        }
        .login-panel-subtitle {
            color: #5f7169;
            font-size: 1rem;
            margin: 0 0 1.4rem 0;
        }
        .login-demo-note {
            color: #65786f;
            font-size: 1rem;
            margin-top: 1.35rem;
        }
        .stApp:has(.login-shell) {
            background: #edf9f2;
            color: #102119;
        }
        .stApp:has(.login-shell) .block-container {
            padding: 2.2rem 2.4rem 1.2rem 2.4rem;
            max-width: 100%;
        }
        .stApp:has(.login-shell) section[data-testid="stSidebar"],
        .stApp:has(.login-shell) [data-testid="stHeader"],
        .stApp:has(.login-shell) [data-testid="stToolbar"],
        .stApp:has(.login-shell) [data-testid="stDecoration"],
        .stApp:has(.login-shell) header {
            display: none;
        }
        .stApp:has(.login-shell) div[data-testid="stForm"] {
            border: 0;
            padding: 0;
            background: transparent;
        }
        .stApp:has(.login-shell) label,
        .stApp:has(.login-shell) .stCheckbox p {
            color: #2b332f;
            font-size: 1rem;
        }
        .stApp:has(.login-shell) input {
            background: #f4f5f8;
            border: 1px solid transparent;
            border-radius: 4px;
            min-height: 3.35rem;
            color: #102119;
            font-size: 1rem;
        }
        .stApp:has(.login-shell) input:focus {
            border-color: rgba(22, 160, 93, 0.45);
            box-shadow: 0 0 0 1px rgba(22, 160, 93, 0.22);
        }
        .stApp:has(.login-shell) .stCheckbox svg {
            fill: #ef5d66;
        }
        .stApp:has(.login-shell) .stButton button,
        .stApp:has(.login-shell) .stForm button {
            min-height: 3.1rem;
            border-radius: 9px;
            background: linear-gradient(135deg, #1fa35d, #11884d) !important;
            border: 0 !important;
            color: #ffffff !important;
            font-size: 1rem;
            font-weight: 700;
            box-shadow: none;
        }
        .stApp:has(.login-shell) .stButton button:hover,
        .stApp:has(.login-shell) .stForm button:hover {
            background: linear-gradient(135deg, #169052, #0f7844) !important;
            color: #ffffff !important;
        }
        .stApp:has(.login-shell) button[aria-label="Show password text"],
        .stApp:has(.login-shell) button[title="Show password text"] {
            background: transparent !important;
            border: 0 !important;
            border-radius: 4px !important;
            box-shadow: none !important;
            color: #30343f !important;
            min-height: 2.5rem;
            width: 2.5rem;
        }
        .stApp:has(.login-shell) button[aria-label="Show password text"]:hover,
        .stApp:has(.login-shell) button[title="Show password text"]:hover {
            background: transparent !important;
            color: #30343f !important;
        }
        .stApp:has(.login-shell) button[aria-label="Show password text"] svg,
        .stApp:has(.login-shell) button[title="Show password text"] svg {
            fill: #30343f !important;
        }
        @media (max-width: 900px) {
            .login-brand-row { margin-top: 0.5rem; margin-bottom: 1.5rem; }
            .login-welcome-title { font-size: 2.55rem; }
            .login-welcome-copy { font-size: 1.05rem; }
            .login-chip-row { margin-bottom: 1.5rem; }
            .stApp:has(.login-shell) .block-container { padding: 1.4rem 1rem; }
        }
        .auth-brand {
            text-align: center;
            margin-bottom: 1.6rem;
        }
        .auth-brand-title {
            font-size: 2rem;
            font-weight: 800;
            color: #16312a;
            margin: 0;
        }
        .auth-brand-sub {
            color: #6e8b81;
            font-size: 0.9rem;
            margin-top: 0.25rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .auth-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(31,185,109,0.14);
            border-radius: 28px;
            padding: 2rem 2rem 1.6rem 2rem;
            box-shadow: 0 24px 60px rgba(26, 78, 52, 0.10);
        }
        .auth-card-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #16312a;
            margin: 0 0 0.3rem 0;
        }
        .auth-card-sub {
            color: #6e8b81;
            font-size: 0.9rem;
            margin-bottom: 1.4rem;
        }
        .auth-divider {
            text-align: center;
            margin: 1rem 0;
            color: #6e8b81;
            font-size: 0.85rem;
        }
        .auth-switch {
            text-align: center;
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #6e8b81;
        }
        .success-icon {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: #ecfbf3;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin: 0 auto 1rem auto;
            text-align: center;
        }

        /* ── Sidebar ── */
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4fbf8 100%);
            border-right: 1px solid rgba(31, 185, 109, 0.12);
        }
        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 0.5rem;
        }
        .sidebar-brand {
            padding: 1rem 0.6rem 1rem 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .sidebar-brand-badge {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #1fb96d, #17a05e);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 800;
            color: white;
            letter-spacing: -0.03em;
            flex-shrink: 0;
            box-shadow: 0 8px 18px rgba(31,185,109,0.28);
        }
        .brand-text-wrap { display: flex; flex-direction: column; }
        .brand-title { font-size: 1.25rem; font-weight: 800; color: #16312a; margin: 0; line-height: 1.15; }
        .brand-subtitle { color: #6e8b81; font-size: 0.78rem; margin-top: 0.1rem; font-weight: 500; }
        .sidebar-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(31,185,109,0.18), transparent);
            margin: 0.1rem 0.6rem 0.75rem 0.6rem;
        }
        .sidebar-nav-title {
            color: #9ab8ae; font-size: 0.73rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase;
            margin: 0 0 0.5rem 0.6rem;
        }
        .user-card, .info-surface, .action-card, .sidebar-note {
            background: rgba(255,255,255,0.90);
            border: 1px solid rgba(31,185,109,0.12);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 14px 35px rgba(26, 78, 52, 0.06);
        }
        .sidebar-user-card {
            background: linear-gradient(135deg, rgba(31,185,109,0.06), rgba(255,255,255,0.95));
            border: 1px solid rgba(31,185,109,0.14);
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            margin: 0 0.2rem;
        }
        .sidebar-note {
            background: linear-gradient(135deg, #ecfbf3, #f2fdf7);
            border: 1px solid rgba(31,185,109,0.16);
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            margin: 0 0.2rem;
        }

        /* ── Top hero ── */
        .hero-title { font-size: 1.9rem; font-weight: 800; color: #16312a; margin-bottom: 0.2rem; }
        .hero-subtitle { color: #6e8b81; font-size: 0.95rem; }
        .pill-row { display: flex; gap: 0.7rem; justify-content: flex-end; flex-wrap: wrap; }
        .top-pill {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(31,185,109,0.12);
            border-radius: 16px; padding: 0.65rem 0.9rem;
            font-weight: 600; color: #2c5044; font-size: 0.87rem;
            box-shadow: 0 8px 22px rgba(26, 78, 52, 0.05);
        }

        /* ── Metric cards ── */
        .metric-card {
            border-radius: 24px; padding: 1rem 1.05rem 0.8rem;
            border: 1px solid rgba(20,75,48,0.07);
            box-shadow: 0 16px 35px rgba(24,61,45,0.06);
            min-height: 160px;
        }
        .metric-badge {
            width: 42px; height: 42px; border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem; background: rgba(255,255,255,0.55);
        }
        .metric-title { font-size: 0.88rem; font-weight: 700; color: #32564a; margin-top: 0.7rem; }
        .metric-value { font-size: 1.85rem; font-weight: 800; margin-top: 0.2rem; line-height: 1.05; color: #16312a; }
        .metric-delta { font-size: 0.88rem; font-weight: 700; margin-top: 0.4rem; }

        /* ── Surface headings ── */
        .surface-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 0.15rem; color: #16312a; }
        .surface-subtitle, .muted { color: #6e8b81; font-size: 0.9rem; }

        /* ── Detection result chip ── */
        .result-chip {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.4rem 0.75rem; border-radius: 999px;
            font-weight: 700; font-size: 0.82rem;
        }

        /* ── Feature strip ── */
        .feature-strip {
            background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(238,251,244,0.94));
            border: 1px solid rgba(31,185,109,0.12); border-radius: 22px;
            padding: 1rem 1.1rem; box-shadow: 0 16px 36px rgba(24,61,45,0.05); height: 100%;
        }
        .feature-title { font-size: 0.95rem; font-weight: 800; margin-bottom: 0.15rem; color: #16312a; }
        .feature-copy { color: #6e8b81; font-size: 0.86rem; margin: 0; }

        /* ── Activity feed ── */
        .activity-item {
            display: flex; justify-content: space-between; gap: 1rem;
            padding: 0.75rem 0; border-bottom: 1px dashed rgba(110,139,129,0.22);
        }

        /* ── Buttons ── */
        .stButton button, .stDownloadButton button, .stForm button {
            border-radius: 14px;
            border: 1px solid rgba(31,185,109,0.18);
            background: linear-gradient(135deg, #1fb96d, #1aa25f);
            color: #ffffff; font-weight: 700;
            box-shadow: 0 12px 22px rgba(31,185,109,0.16);
        }
        .stButton button:hover, .stDownloadButton button:hover { color: #ffffff; }

        /* ── Sidebar nav buttons ── */
        div[data-testid="stSidebar"] .stButton button {
            background: transparent !important;
            border: none !important;
            border-radius: 12px !important;
            color: #33594e !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            text-align: left !important;
            padding: 0.6rem 0.85rem !important;
            box-shadow: none !important;
            transition: background 0.15s ease, color 0.15s ease;
        }
        div[data-testid="stSidebar"] .stButton button:hover {
            background: rgba(31,185,109,0.08) !important;
            color: #17824d !important;
        }
        div[data-testid="stSidebar"] .stButton button[kind="primary"] {
            background: linear-gradient(135deg, rgba(31,185,109,0.18), rgba(31,185,109,0.10)) !important;
            border: 1px solid rgba(31,185,109,0.22) !important;
            color: #17824d !important;
            font-weight: 700 !important;
            box-shadow: 0 6px 16px rgba(26,78,52,0.08) !important;
        }
        div[data-testid="stSidebar"] .stButton button[kind="secondary"] {
            background: transparent !important;
            border: none !important;
            color: #33594e !important;
            box-shadow: none !important;
        }
        /* Sign out button */
        div[data-testid="stSidebar"] .stButton:last-of-type button {
            background: rgba(239,93,102,0.07) !important;
            border: 1px solid rgba(239,93,102,0.15) !important;
            color: #b91c1c !important;
            font-weight: 700 !important;
        }
        div[data-testid="stSidebar"] .stButton:last-of-type button:hover {
            background: rgba(239,93,102,0.13) !important;
        }

        /* Premium SaaS sidebar */
        div[data-testid="stSidebar"] {
            width: 19rem !important;
            min-width: 19rem !important;
            max-width: 19rem !important;
            background:
                linear-gradient(140deg, rgba(7, 20, 17, 0.96), rgba(10, 42, 31, 0.92) 45%, rgba(7, 18, 16, 0.98)),
                radial-gradient(circle at 24% 8%, rgba(58, 255, 159, 0.22), transparent 30%),
                radial-gradient(circle at 90% 2%, rgba(121, 255, 196, 0.14), transparent 22%) !important;
            border-right: 1px solid rgba(100, 255, 177, 0.22) !important;
            box-shadow: 24px 0 60px rgba(4, 28, 18, 0.34);
            backdrop-filter: blur(26px) saturate(150%);
            transition:
                width 0.45s cubic-bezier(0.22, 1, 0.36, 1),
                min-width 0.45s cubic-bezier(0.22, 1, 0.36, 1),
                max-width 0.45s cubic-bezier(0.22, 1, 0.36, 1),
                transform 0.45s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.35s ease;
        }
        div[data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.075), transparent 24%),
                radial-gradient(circle at 15% 18%, rgba(44,255,146,0.15), transparent 26%),
                radial-gradient(circle at 80% 88%, rgba(14,165,106,0.20), transparent 30%);
            opacity: 0.9;
        }
        div[data-testid="stSidebar"]::after {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 1px;
            background: linear-gradient(180deg, transparent, rgba(109,255,188,0.7), transparent);
            box-shadow: 0 0 28px rgba(49, 255, 154, 0.55);
            pointer-events: none;
        }
        div[data-testid="stSidebar"] > div:first-child {
            padding: 0.85rem 0.9rem 0.9rem 0.9rem;
            overflow-y: auto;
            overflow-x: visible;
            scrollbar-width: thin;
            scrollbar-color: rgba(85,255,168,0.48) transparent;
        }
        div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { width: 7px; }
        div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track { background: transparent; }
        div[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(72,255,158,0.72), rgba(22,163,105,0.36));
            border-radius: 999px;
        }
        .sidebar-shell,
        .sidebar-brand,
        .sidebar-divider,
        .sidebar-nav-title,
        .sidebar-user-card,
        .sidebar-note,
        div[data-testid="stSidebar"] .stButton {
            position: relative;
            z-index: 1;
        }
        .sidebar-brand {
            background: linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045));
            border: 1px solid rgba(122,255,188,0.22);
            border-radius: 22px;
            padding: 0.82rem 0.78rem;
            box-shadow: 0 18px 42px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.13);
            transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.32s ease, border-color 0.32s ease;
        }
        .sidebar-brand:hover {
            transform: translateY(-2px);
            border-color: rgba(95,255,173,0.42);
            box-shadow: 0 22px 50px rgba(16,185,129,0.17), inset 0 1px 0 rgba(255,255,255,0.18);
        }
        .sidebar-brand-badge {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #5cffad, #13b875 52%, #075f43);
            border-radius: 16px;
            color: #062018;
            box-shadow: 0 16px 38px rgba(31,255,142,0.26), inset 0 1px 0 rgba(255,255,255,0.42);
            transition: transform 0.32s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.32s ease;
        }
        .sidebar-brand:hover .sidebar-brand-badge {
            transform: rotate(-4deg) scale(1.07);
            box-shadow: 0 20px 46px rgba(56,255,160,0.36), inset 0 1px 0 rgba(255,255,255,0.5);
        }
        .brand-title {
            color: #f4fff9;
            font-size: 1.19rem;
            letter-spacing: 0;
            text-shadow: 0 0 24px rgba(96,255,176,0.22);
        }
        .brand-subtitle {
            color: rgba(206,255,226,0.68);
            font-size: 0.8rem;
        }
        .sidebar-divider {
            height: 1px;
            margin: 0.85rem 0.25rem 0.95rem 0.25rem;
            background: linear-gradient(90deg, transparent, rgba(103,255,181,0.42), transparent);
        }
        .sidebar-nav-title {
            margin: 0.1rem 0.35rem 0.7rem 0.55rem;
            color: rgba(209,255,229,0.55);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.16em;
        }
        div[data-testid="stSidebar"] .stButton {
            margin-bottom: 0.45rem;
        }
        div[data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            position: relative;
            min-height: 3.05rem;
            isolation: isolate;
            overflow: hidden;
            border-radius: 999px !important;
            border: 1px solid rgba(119,255,190,0.12) !important;
            background:
                linear-gradient(90deg, rgba(255,255,255,0.09), rgba(255,255,255,0.035)) !important;
            color: rgba(232,255,241,0.82) !important;
            font-weight: 760 !important;
            font-size: 0.94rem !important;
            letter-spacing: 0 !important;
            text-align: left !important;
            padding: 0.72rem 0.95rem !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.09);
            transition:
                transform 0.34s cubic-bezier(0.22, 1, 0.36, 1),
                background 0.34s ease,
                border-color 0.34s ease,
                box-shadow 0.34s ease,
                color 0.34s ease;
        }
        div[data-testid="stSidebar"] .stButton button::before {
            content: "";
            position: absolute;
            left: 0.42rem;
            top: 50%;
            width: 4px;
            height: 0;
            border-radius: 999px;
            background: linear-gradient(180deg, #a7ffd0, #20e386);
            box-shadow: 0 0 22px rgba(50,255,150,0.92), 0 0 46px rgba(50,255,150,0.36);
            transform: translateY(-50%);
            transition: height 0.34s cubic-bezier(0.22, 1, 0.36, 1);
            z-index: 2;
        }
        div[data-testid="stSidebar"] .stButton button::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(110deg, transparent 0%, rgba(97,255,179,0.14) 32%, rgba(97,255,179,0.28) 48%, transparent 68%);
            opacity: 0;
            transform: translateX(-80%);
            transition: opacity 0.34s ease, transform 0.48s cubic-bezier(0.22, 1, 0.36, 1);
            z-index: -1;
        }
        div[data-testid="stSidebar"] .stButton button p {
            position: relative;
            z-index: 3;
            color: inherit !important;
            transition: transform 0.34s cubic-bezier(0.34, 1.56, 0.64, 1);
            white-space: normal;
        }
        div[data-testid="stSidebar"] .stButton button:hover,
        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
            transform: translateY(-3px) scale(1.018);
            background:
                linear-gradient(90deg, rgba(35,255,144,0.17), rgba(255,255,255,0.075)) !important;
            border-color: rgba(103,255,181,0.38) !important;
            color: #f7fff9 !important;
            box-shadow:
                0 16px 34px rgba(16,185,129,0.20),
                0 0 28px rgba(52,255,157,0.14),
                inset 0 1px 0 rgba(255,255,255,0.16) !important;
        }
        div[data-testid="stSidebar"] .stButton button:hover::after {
            opacity: 1;
            transform: translateX(72%);
        }
        div[data-testid="stSidebar"] .stButton button:hover p {
            transform: translateX(5px) scale(1.01);
        }
        div[data-testid="stSidebar"] .stButton button:active,
        [data-testid="stSidebar"] [data-testid="stButton"] > button:active {
            transform: translateY(0) scale(0.96);
            transition-duration: 0.08s;
        }
        div[data-testid="stSidebar"] .stButton button:focus-visible {
            outline: 2px solid rgba(96,255,176,0.5) !important;
            outline-offset: 2px;
        }
        div[data-testid="stSidebar"] .stButton button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
            background:
                linear-gradient(120deg, rgba(44,255,149,0.24), rgba(255,255,255,0.12) 48%, rgba(18,185,115,0.18)) !important;
            border-color: rgba(130,255,198,0.48) !important;
            color: #ffffff !important;
            box-shadow:
                0 18px 42px rgba(16,185,129,0.28),
                0 0 0 1px rgba(95,255,173,0.12),
                inset 0 1px 0 rgba(255,255,255,0.22) !important;
            animation: sidebarActivePulse 2.6s ease-in-out infinite;
        }
        div[data-testid="stSidebar"] .stButton button[kind="primary"]::before {
            height: 62%;
        }
        div[data-testid="stSidebar"] .stButton button[kind="primary"] p {
            text-shadow: 0 0 18px rgba(111,255,188,0.32);
        }
        @keyframes sidebarActivePulse {
            0%, 100% { box-shadow: 0 18px 42px rgba(16,185,129,0.24), 0 0 0 1px rgba(95,255,173,0.12), inset 0 1px 0 rgba(255,255,255,0.22); }
            50% { box-shadow: 0 20px 50px rgba(35,255,142,0.34), 0 0 0 1px rgba(95,255,173,0.22), inset 0 1px 0 rgba(255,255,255,0.28); }
        }
        .sidebar-user-card,
        .sidebar-note {
            background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
            border: 1px solid rgba(115,255,188,0.18);
            border-radius: 20px;
            box-shadow: 0 18px 38px rgba(0,0,0,0.20), inset 0 1px 0 rgba(255,255,255,0.08);
            backdrop-filter: blur(18px);
            transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.32s ease, border-color 0.32s ease;
        }
        .sidebar-user-card:hover,
        .sidebar-note:hover {
            transform: translateY(-2px);
            border-color: rgba(93,255,173,0.35);
            box-shadow: 0 24px 46px rgba(11,185,116,0.16), inset 0 1px 0 rgba(255,255,255,0.14);
        }
        .sidebar-user-card div,
        .sidebar-note div {
            color: rgba(237,255,244,0.82) !important;
        }
        .sidebar-note div:nth-child(2) {
            color: #66ffae !important;
        }
        div[data-testid="stSidebar"] .stButton:last-of-type button {
            background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,96,96,0.08)) !important;
            border-color: rgba(255,126,126,0.18) !important;
            color: #ffd2d2 !important;
        }
        div[data-testid="stSidebar"] .stButton:last-of-type button:hover {
            background: linear-gradient(90deg, rgba(255,118,118,0.20), rgba(255,255,255,0.07)) !important;
            border-color: rgba(255,150,150,0.36) !important;
            box-shadow: 0 16px 34px rgba(239,93,102,0.16) !important;
        }
        .sidebar-collapsed [data-testid="stSidebar"],
        [data-testid="stSidebar"]:has(.sidebar-collapsed) {
            width: 6.35rem !important;
            min-width: 6.35rem !important;
            max-width: 6.35rem !important;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .brand-text-wrap,
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .sidebar-nav-title,
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .sidebar-user-card,
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .sidebar-note {
            display: none;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .sidebar-brand {
            justify-content: center;
            padding: 0.72rem 0.2rem;
            border-radius: 20px;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .sidebar-brand-badge {
            width: 46px;
            height: 46px;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button {
            min-height: 3.15rem;
            padding: 0.65rem 0.2rem !important;
            text-align: center !important;
            font-size: 1.08rem !important;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button p {
            transform: none;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button:hover {
            transform: translateY(-3px) scale(1.04);
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button::before {
            left: 0.15rem;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button::after {
            display: none;
        }
        [data-testid="stSidebar"]:has(.sidebar-collapsed) .stButton button:hover::before {
            height: 58%;
        }
        @media (max-width: 768px) {
            div[data-testid="stSidebar"] {
                width: min(86vw, 21rem) !important;
                min-width: min(86vw, 21rem) !important;
                max-width: min(86vw, 21rem) !important;
                box-shadow: 28px 0 70px rgba(0,0,0,0.42);
            }
            [data-testid="stSidebar"]:has(.sidebar-collapsed) {
                width: 5.85rem !important;
                min-width: 5.85rem !important;
                max-width: 5.85rem !important;
            }
        }

        /* Readable glass fallback for Streamlit sidebar variants */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"] {
            width: 19rem !important;
            min-width: 19rem !important;
            max-width: 19rem !important;
            background:
                radial-gradient(circle at 18% 6%, rgba(55, 255, 151, 0.22), transparent 32%),
                radial-gradient(circle at 90% 0%, rgba(29, 185, 109, 0.14), transparent 26%),
                linear-gradient(160deg, rgba(246, 255, 251, 0.94), rgba(229, 248, 239, 0.88) 48%, rgba(238, 246, 243, 0.96)) !important;
            border-right: 1px solid rgba(31, 185, 109, 0.22) !important;
            box-shadow: 20px 0 48px rgba(17, 92, 58, 0.12) !important;
            backdrop-filter: blur(24px) saturate(145%);
        }
        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebar"] > div:first-child {
            padding: 0.85rem 0.9rem 0.9rem 0.9rem;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .brand-title {
            color: #102f25 !important;
            text-shadow: none !important;
        }
        .brand-subtitle,
        .sidebar-nav-title {
            color: #6f9588 !important;
        }
        .sidebar-brand,
        .sidebar-user-card,
        .sidebar-note {
            background: rgba(255, 255, 255, 0.62) !important;
            border-color: rgba(31, 185, 109, 0.18) !important;
            box-shadow: 0 18px 38px rgba(23, 109, 70, 0.10), inset 0 1px 0 rgba(255,255,255,0.72) !important;
        }
        section[data-testid="stSidebar"] .stButton button,
        div[data-testid="stSidebar"] .stButton button,
        section[data-testid="stSidebar"] [data-testid="stButton"] > button,
        div[data-testid="stSidebar"] [data-testid="stButton"] > button {
            background: rgba(255, 255, 255, 0.42) !important;
            border: 1px solid rgba(31, 185, 109, 0.16) !important;
            color: #214c3f !important;
            text-shadow: none !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.75) !important;
        }
        section[data-testid="stSidebar"] .stButton button p,
        div[data-testid="stSidebar"] .stButton button p {
            color: #214c3f !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton button:hover,
        div[data-testid="stSidebar"] .stButton button:hover {
            background: linear-gradient(90deg, rgba(198, 255, 225, 0.88), rgba(255,255,255,0.76)) !important;
            border-color: rgba(31, 185, 109, 0.36) !important;
            color: #075f43 !important;
            box-shadow: 0 16px 32px rgba(31,185,109,0.16), 0 0 26px rgba(50,255,150,0.16), inset 0 1px 0 rgba(255,255,255,0.85) !important;
        }
        section[data-testid="stSidebar"] .stButton button:hover p,
        div[data-testid="stSidebar"] .stButton button:hover p {
            color: #075f43 !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"],
        div[data-testid="stSidebar"] .stButton button[kind="primary"] {
            background: linear-gradient(120deg, rgba(31,185,109,0.26), rgba(255,255,255,0.78) 52%, rgba(84,255,168,0.20)) !important;
            border-color: rgba(31,185,109,0.42) !important;
            color: #055f3b !important;
            box-shadow: 0 18px 42px rgba(31,185,109,0.22), 0 0 0 1px rgba(31,185,109,0.10), inset 0 1px 0 rgba(255,255,255,0.9) !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="primary"] p,
        div[data-testid="stSidebar"] .stButton button[kind="primary"] p {
            color: #055f3b !important;
            font-weight: 800 !important;
        }
        section[data-testid="stSidebar"] .stButton:last-of-type button,
        div[data-testid="stSidebar"] .stButton:last-of-type button {
            color: #a83333 !important;
            background: rgba(255,255,255,0.58) !important;
            border-color: rgba(239,93,102,0.18) !important;
        }
        .sidebar-user-card div,
        .sidebar-note div {
            color: #22493d !important;
        }
        .sidebar-note div:nth-child(2) {
            color: #128352 !important;
        }
        section[data-testid="stSidebar"]:has(.sidebar-collapsed),
        div[data-testid="stSidebar"]:has(.sidebar-collapsed) {
            width: 6.35rem !important;
            min-width: 6.35rem !important;
            max-width: 6.35rem !important;
        }
        @media (max-width: 768px) {
            section[data-testid="stSidebar"],
            div[data-testid="stSidebar"] {
                width: min(86vw, 21rem) !important;
                min-width: min(86vw, 21rem) !important;
                max-width: min(86vw, 21rem) !important;
            }
        }

        /* ── Tabs ── */
        div[data-baseweb="tab-list"] { gap: 0.35rem; }
        button[data-baseweb="tab"] { border-radius: 11px; padding: 0.45rem 0.75rem; background: rgba(255,255,255,0.78); }
        button[data-baseweb="tab"][aria-selected="true"] { background: rgba(31,185,109,0.12); color: #17824d; }

        /* ── Alert / info boxes ── */
        .alert-error {
            background: #fff1f2; border: 1px solid #fca5a5; border-radius: 14px;
            padding: 0.75rem 1rem; color: #b91c1c; font-weight: 600; font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        .alert-success {
            background: #ecfbf3; border: 1px solid #86efac; border-radius: 14px;
            padding: 0.75rem 1rem; color: #15803d; font-weight: 600; font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        /* Dashboard-only alignment and spacing */
        .dashboard-page {
            display: none;
        }
        .stApp:has(.dashboard-page) .block-container {
            padding-top: 2.1rem;
            padding-left: clamp(1.4rem, 3vw, 3.2rem);
            padding-right: clamp(1.4rem, 3vw, 3.2rem);
            max-width: 1540px;
        }
        .stApp:has(.dashboard-page) .hero-title {
            margin-top: 0.25rem;
        }
        .stApp:has(.dashboard-page) .pill-row {
            justify-content: flex-end;
            align-items: center;
            gap: 0.7rem;
        }
        .stApp:has(.dashboard-page) .top-pill {
            min-height: 3.35rem;
            display: inline-flex;
            align-items: center;
        }
        .stApp:has(.dashboard-page) .metric-card {
            min-height: 154px;
            padding: 0.88rem 0.95rem 0.68rem;
            margin-bottom: 0.45rem;
            display: flex;
            flex-direction: column;
        }
        .stApp:has(.dashboard-page) .metric-badge {
            width: 38px;
            height: 38px;
            border-radius: 13px;
            font-size: 1.05rem;
        }
        .stApp:has(.dashboard-page) .metric-title {
            font-size: 0.8rem;
            margin-top: 0.58rem;
        }
        .stApp:has(.dashboard-page) .metric-value {
            font-size: 1.55rem;
        }
        .stApp:has(.dashboard-page) .metric-delta {
            font-size: 0.8rem;
            margin-top: 0.3rem;
        }
        .stApp:has(.dashboard-page) .metric-card svg {
            margin-top: auto;
            height: 42px;
        }
        .stApp:has(.dashboard-page) .info-surface {
            min-height: 76px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 0.8rem !important;
            padding: 0.9rem 1rem;
        }
        .stApp:has(.dashboard-page) .surface-title {
            line-height: 1.2;
        }
        .stApp:has(.dashboard-page) .surface-subtitle {
            line-height: 1.5;
            margin-top: 0.45rem;
        }
        .stApp:has(.dashboard-page) .action-card {
            height: 100%;
        }
        .stApp:has(.dashboard-page) .activity-item {
            min-height: 54px;
            align-items: flex-start;
        }
        .stApp:has(.dashboard-page) .feature-strip {
            min-height: 138px;
            margin-bottom: 0.85rem;
            padding: 1.05rem 1.1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .stApp:has(.dashboard-page) .feature-title {
            font-size: 0.92rem;
        }
        .stApp:has(.dashboard-page) .feature-copy {
            font-size: 0.85rem;
            line-height: 1.45;
        }
        .dashboard-capture-guide {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.2rem 0 0.85rem;
        }
        .dashboard-guide-card {
            background: rgba(255,255,255,0.66);
            border: 1px solid rgba(31,185,109,0.13);
            border-radius: 15px;
            padding: 0.72rem 0.78rem;
            box-shadow: 0 10px 24px rgba(24,61,45,0.05);
        }
        .dashboard-guide-title {
            color: #16312a;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 0.18rem;
        }
        .dashboard-guide-copy {
            color: #6e8b81;
            font-size: 0.76rem;
            line-height: 1.45;
            margin: 0;
        }
        .dashboard-map-stat {
            height: auto !important;
            min-height: 84px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 0.7rem 0.6rem;
            margin-top: 0.55rem;
        }
        .dashboard-map-stat-label {
            color: #16312a;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.25;
        }
        .stApp:has(.dashboard-page) div[data-testid="stVerticalBlock"] {
            gap: 0.48rem;
        }
        .stApp:has(.dashboard-page) div[data-testid="stHorizontalBlock"] {
            gap: 0.95rem;
        }
        .stApp:has(.dashboard-page) div[data-testid="stDataFrame"] {
            margin-top: 0.2rem;
        }
        .stApp:has(.dashboard-page) iframe[title="streamlit_folium.st_folium"],
        .stApp:has(.dashboard-page) iframe {
            border-radius: 18px;
        }
        @media (max-width: 1100px) {
            .stApp:has(.dashboard-page) .pill-row {
                justify-content: flex-start;
                margin-top: 1rem;
            }
            .dashboard-capture-guide {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Path, default_factory) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = default_factory()
    write_json(path, data)
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))


# ─────────────────────────────────────────────────────────────
# USER REGISTRY  (multi-user, no default)
# ─────────────────────────────────────────────────────────────
def load_users() -> dict[str, Any]:
    """Load users dict. Keys are lowercase emails."""
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_users(users: dict[str, Any]) -> None:
    write_json(USERS_PATH, users)


def register_user(name: str, email: str, password: str, city: str, role: str = "Admin") -> str | None:
    """Returns error string or None on success."""
    users = load_users()
    key = email.strip().lower()
    if key in users:
        return "An account with this email already exists."
    users[key] = {
        "display_name": name.strip(),
        "email": email.strip(),
        "city": city,
        "role": role,
        "password_hash": hash_password(password),
        "created_at": now_iso(),
    }
    save_users(users)
    return None


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """Returns user dict or None."""
    users = load_users()
    user = users.get(email.strip().lower())
    if user and user["password_hash"] == hash_password(password):
        return user
    return None


# ─────────────────────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────────────────────
def default_app_state() -> dict[str, Any]:
    now = datetime.now().replace(second=0, microsecond=0)
    return {
        "locations": [
            {"id": "LOC-001", "name": "Connaught Place", "area": "Central Delhi", "lat": 28.6315, "lon": 77.2167, "severity": "High", "detections": 11, "status": "Active", "last_updated": (now - timedelta(minutes=20)).isoformat()},
            {"id": "LOC-002", "name": "Mayur Vihar", "area": "East Delhi", "lat": 28.6080, "lon": 77.2950, "severity": "Medium", "detections": 7, "status": "Monitoring", "last_updated": (now - timedelta(minutes=28)).isoformat()},
            {"id": "LOC-003", "name": "Saket", "area": "South Delhi", "lat": 28.5245, "lon": 77.2066, "severity": "Low", "detections": 4, "status": "Stable", "last_updated": (now - timedelta(minutes=34)).isoformat()},
            {"id": "LOC-004", "name": "Rajouri Garden", "area": "West Delhi", "lat": 28.6422, "lon": 77.1167, "severity": "High", "detections": 9, "status": "Active", "last_updated": (now - timedelta(minutes=12)).isoformat()},
            {"id": "LOC-005", "name": "Dwarka Sector 15", "area": "South West Delhi", "lat": 28.5910, "lon": 77.0460, "severity": "Low", "detections": 3, "status": "Stable", "last_updated": (now - timedelta(minutes=41)).isoformat()},
        ],
        "history": [
            {"id": "DET-001", "time": (now - timedelta(minutes=16)).isoformat(), "item": "Plastic Bottle", "category": "Non-Biodegradable", "confidence": 0.986, "source": "Upload", "location": "Connaught Place", "bin": "Dry waste bin"},
            {"id": "DET-002", "time": (now - timedelta(minutes=12)).isoformat(), "item": "Banana Peel", "category": "Biodegradable", "confidence": 0.992, "source": "Camera", "location": "Mayur Vihar", "bin": "Wet waste bin"},
            {"id": "DET-003", "time": (now - timedelta(minutes=9)).isoformat(), "item": "Paper Cup", "category": "Non-Biodegradable", "confidence": 0.971, "source": "Upload", "location": "Saket", "bin": "Dry waste bin"},
            {"id": "DET-004", "time": (now - timedelta(minutes=5)).isoformat(), "item": "Leaf Waste", "category": "Biodegradable", "confidence": 0.963, "source": "Live", "location": "Rajouri Garden", "bin": "Wet waste bin"},
        ],
        "bins": [
            {"id": "BIN-001", "name": "Wet Bin A1", "type": "Biodegradable", "location": "Connaught Place", "fill_level": 62, "status": "Normal", "updated_at": (now - timedelta(minutes=18)).isoformat()},
            {"id": "BIN-002", "name": "Dry Bin B4", "type": "Non-Biodegradable", "location": "Connaught Place", "fill_level": 78, "status": "Attention", "updated_at": (now - timedelta(minutes=15)).isoformat()},
            {"id": "BIN-003", "name": "Wet Bin C2", "type": "Biodegradable", "location": "Mayur Vihar", "fill_level": 49, "status": "Normal", "updated_at": (now - timedelta(minutes=25)).isoformat()},
            {"id": "BIN-004", "name": "Dry Bin D7", "type": "Non-Biodegradable", "location": "Rajouri Garden", "fill_level": 88, "status": "Needs Pickup", "updated_at": (now - timedelta(minutes=11)).isoformat()},
        ],
        "routes": [
            {"id": "ROUTE-001", "route_name": "Route A", "zones": "Connaught Place -> Rajouri Garden", "priority": "High", "eta": "18 min", "truck": "Truck-12", "status": "Ready"},
            {"id": "ROUTE-002", "route_name": "Route B", "zones": "Mayur Vihar -> Saket", "priority": "Medium", "eta": "26 min", "truck": "Truck-07", "status": "In Progress"},
            {"id": "ROUTE-003", "route_name": "Route C", "zones": "Dwarka Sector 15", "priority": "Low", "eta": "12 min", "truck": "Truck-02", "status": "Ready"},
        ],
        "alerts": [
            {"id": "ALERT-001", "time": (now - timedelta(minutes=14)).isoformat(), "title": "High waste detected in Connaught Place", "level": "High", "location": "Connaught Place", "status": "Open", "action": "Dispatch dry-waste collection", "source_key": "LOC-001-high"},
            {"id": "ALERT-002", "time": (now - timedelta(minutes=10)).isoformat(), "title": "Dry Bin D7 nearing capacity", "level": "Medium", "location": "Rajouri Garden", "status": "Open", "action": "Schedule pickup route", "source_key": "BIN-004-capacity"},
        ],
    }


def load_app_state() -> dict[str, Any]:
    data = read_json(APP_STATE_PATH, default_app_state)
    for key, value in default_app_state().items():
        data.setdefault(key, value)
    return data


def save_app_state() -> None:
    write_json(APP_STATE_PATH, st.session_state.app_data)


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> tuple[Any, str]:
    for path in MODEL_CANDIDATES:
        if path.exists():
            if tf is None:
                return None, "TensorFlow not available — demo inference mode."
            try:
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                except Exception:
                    from keras.src.ops.numpy import Subtract, TrueDivide

                    with tf.keras.utils.custom_object_scope({"TrueDivide": TrueDivide, "Subtract": Subtract}):
                        model = tf.keras.models.load_model(path, compile=False)
                return model, f"✅ Model loaded: {path.name}"
            except Exception as exc:
                return None, f"⚠️ Model load failed ({exc}) — demo mode."
    return None, "ℹ️ Model file not found — demo inference mode."


def choose_demo_image() -> Image.Image:
    for path in DEMO_IMAGE_CANDIDATES:
        if path.exists():
            return Image.open(path).convert("RGB")
    return Image.new("RGB", (640, 420), "#d9f3e4")


def fallback_detection() -> dict[str, Any]:
    return {"id": "DET-DEFAULT", "time": now_iso(), "item": "No detections yet", "category": "Biodegradable", "confidence": 0.0, "source": "System", "location": "—", "bin": "Wet waste bin"}


def category_from_probability(probability: float) -> tuple[str, float]:
    return ("Non-Biodegradable", probability) if probability > 0.5 else ("Biodegradable", 1 - probability)


def category_from_class_name(class_name: str, confidence: float) -> tuple[str, float]:
    cfg = read_json(MODEL_CONFIG_PATH, {})
    biodegradable = set(cfg.get("biodegradable_classes", ["O"]))
    if class_name in biodegradable:
        return "Biodegradable", confidence
    return "Non-Biodegradable", confidence


def item_from_class_name(class_name: str, category: str, image: Image.Image) -> str:
    cfg = read_json(MODEL_CONFIG_PATH, {})
    display_names = cfg.get("display_names", {})
    return display_names.get(class_name) or pick_item(category, image)


def infer_fallback_probability(image: Image.Image) -> float:
    array = np.asarray(image.resize(IMG_SIZE).convert("RGB"), dtype=np.float32)
    probability = 0.38 + (array.mean() / 255.0) * 0.24 + (array.std() / 255.0) * 0.42
    return float(np.clip(probability, 0.05, 0.95))


def preprocess_image(image: Image.Image) -> np.ndarray:
    fitted = ImageOps.fit(image.convert("RGB"), IMG_SIZE, Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(fitted).astype(np.float32) / 255.0, axis=0)


def preprocess_model_image(image: Image.Image, model: Any) -> np.ndarray:
    input_shape = getattr(model, "input_shape", None) or (None, *IMG_SIZE, 3)
    height = input_shape[1] or IMG_SIZE[0]
    width = input_shape[2] or IMG_SIZE[1]
    fitted = ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    values = np.asarray(fitted).astype(np.float32)
    if getattr(model, "output_shape", (None, 1))[-1] == 1:
        values = values / 255.0
    return np.expand_dims(values, axis=0)


def decode_model_prediction(prediction: np.ndarray, image: Image.Image) -> tuple[str, float, str]:
    cfg = read_json(MODEL_CONFIG_PATH, {})
    class_names = list(cfg.get("class_names", ["O", "R"]))
    values = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if len(values) > 1:
        index = int(np.argmax(values))
        class_name = class_names[index] if index < len(class_names) else str(index)
        category, confidence = category_from_class_name(class_name, float(values[index]))
        return category, confidence, item_from_class_name(class_name, category, image)
    probability = float(values[0])
    class_name = "R" if probability >= 0.5 else "O"
    category, confidence = category_from_probability(probability)
    return category, confidence, item_from_class_name(class_name, category, image)


def classify_image(image: Image.Image) -> tuple[str, float, str]:
    model, _ = load_model()
    if model:
        prediction = model.predict(preprocess_model_image(image, model), verbose=0)[0]
        return decode_model_prediction(prediction, image)
    probability = infer_fallback_probability(image)
    category, confidence = category_from_probability(probability)
    return category, confidence, pick_item(category, image)


def pick_item(category: str, image: Image.Image) -> str:
    options = CATEGORY_META[category]["examples"]
    sample = np.asarray(image.resize((32, 32)).convert("RGB"), dtype=np.int32)
    return options[int(sample.sum()) % len(options)]


# ─────────────────────────────────────────────────────────────
# SESSION BOOTSTRAP
# ─────────────────────────────────────────────────────────────
def bootstrap_session() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "signin"   # "signin" | "signup" | "success"
    if "app_data" not in st.session_state:
        st.session_state.app_data = load_app_state()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "latest_image" not in st.session_state:
        st.session_state.latest_image = choose_demo_image()
    if "latest_detection" not in st.session_state:
        history = st.session_state.app_data["history"]
        st.session_state.latest_detection = history[-1] if history else fallback_detection()
    if "selected_location" not in st.session_state:
        st.session_state.selected_location = st.session_state.app_data["locations"][0]["name"]
    if "signup_success_name" not in st.session_state:
        st.session_state.signup_success_name = ""
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False


# ─────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────
def render_auth_brand() -> None:
    st.markdown(
        """
        <div class="auth-brand">
            <div class="auth-brand-title">♻️ SmartWaste</div>
            <div class="auth-brand-sub">For Smart Cities</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signup_page() -> None:
    render_auth_brand()
    st.markdown('<div class="auth-outer">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="auth-card">
            <div class="auth-card-title">Create your account</div>
            <div class="auth-card-sub">Join SmartWaste to manage your city's waste intelligently.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("signup_form", clear_on_submit=False):
        name = st.text_input("Full Name", placeholder="e.g. Priya Sharma")
        email = st.text_input("Email Address", placeholder="you@city.gov.in")
        city = st.selectbox("City", CITY_OPTIONS)
        role = st.selectbox("Role", ROLE_OPTIONS)
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        # Validation
        errors = []
        if not name.strip():
            errors.append("Full name is required.")
        if not is_valid_email(email):
            errors.append("Please enter a valid email address.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        if errors:
            for err in errors:
                st.markdown(f'<div class="alert-error">⚠️ {err}</div>', unsafe_allow_html=True)
        else:
            error = register_user(name, email, password, city, role)
            if error:
                st.markdown(f'<div class="alert-error">⚠️ {error}</div>', unsafe_allow_html=True)
            else:
                st.session_state.signup_success_name = name.strip()
                st.session_state.auth_page = "success"
                st.rerun()

    st.markdown('<div class="auth-switch">Already have an account? </div>', unsafe_allow_html=True)
    if st.button("← Back to Sign In", key="goto_signin_from_signup", use_container_width=True):
        st.session_state.auth_page = "signin"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_signup_success_page() -> None:
    render_auth_brand()
    st.markdown('<div class="auth-outer">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="auth-card" style="text-align:center;">
            <div class="success-icon">✅</div>
            <div class="auth-card-title">Account Created!</div>
            <div class="auth-card-sub" style="margin-bottom:1.5rem;">
                Welcome, <strong>{st.session_state.signup_success_name}</strong>!
                Your SmartWaste account is ready. You can now sign in.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Go to Sign In →", key="goto_signin_from_success", use_container_width=True):
        st.session_state.auth_page = "signin"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_legacy_signin_page() -> None:
    render_auth_brand()
    st.markdown('<div class="auth-outer">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="auth-card">
            <div class="auth-card-title">Welcome back 👋</div>
            <div class="auth-card-sub">Sign in to your SmartWaste dashboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("signin_form", clear_on_submit=False):
        email = st.text_input("Email Address", placeholder="you@city.gov.in")
        password = st.text_input("Password", type="password", placeholder="Your password")
        submitted = st.form_submit_button("Sign In →", use_container_width=True)

    if submitted:
        if not email.strip() or not password.strip():
            st.markdown('<div class="alert-error">⚠️ Please fill in all fields.</div>', unsafe_allow_html=True)
        else:
            user = authenticate_user(email, password)
            if user is None:
                st.markdown('<div class="alert-error">⚠️ Incorrect email or password. Please try again.</div>', unsafe_allow_html=True)
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = user
                st.session_state.current_page = "Dashboard"
                st.rerun()

    users = load_users()
    if not users:
        st.markdown(
            '<div class="alert-success">ℹ️ No accounts yet. Create one below!</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="auth-switch">Don\'t have an account?</div>', unsafe_allow_html=True)
    if st.button("Create Account →", key="goto_signup", use_container_width=True):
        st.session_state.auth_page = "signup"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_signin_page() -> None:
    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large", vertical_alignment="center")

    with left:
        st.markdown(
            """
            <div class="login-brand-row">
                <div class="login-brand-badge">SW</div>
                <div>
                    <div class="login-brand-name">SmartWaste</div>
                    <div class="login-brand-subtitle">Garbage Classification for Smart Cities</div>
                </div>
            </div>
            <h1 class="login-welcome-title">Welcome to SmartWaste</h1>
            <p class="login-welcome-copy">
                AI powered biodegradable and non-biodegradable waste detection
                for cleaner, faster and more sustainable city operations.
            </p>
            <div class="login-chip-row">
                <div class="login-chip">Dataset: Organic / Recyclable</div>
                <div class="login-chip">Live city weather</div>
                <div class="login-chip">Smart bins and routes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="login-panel-title">Login</div>
            <p class="login-panel-subtitle">Use your account to enter the dashboard.</p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("signin_form", clear_on_submit=False):
            email = st.text_input("Email or username", placeholder="admin or you@city.gov.in")
            password = st.text_input("Password", type="password")
            st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email.strip() or not password.strip():
                st.markdown('<div class="alert-error">Please fill in all fields.</div>', unsafe_allow_html=True)
            else:
                user = authenticate_user(email, password)
                if user is None:
                    st.markdown('<div class="alert-error">Incorrect email or password. Please try again.</div>', unsafe_allow_html=True)
                else:
                    st.session_state.authenticated = True
                    st.session_state.current_user = user
                    st.session_state.current_page = "Dashboard"
                    st.rerun()

        if st.button("Don't have an Account? Register here", key="goto_signup", use_container_width=True):
            st.session_state.auth_page = "signup"
            st.rerun()

        st.markdown(
            '<div class="login-demo-note">Demo login: <strong>admin</strong> / <strong>admin123</strong></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_auth_flow() -> None:
    page = st.session_state.auth_page
    if page == "signup":
        render_signup_page()
    elif page == "success":
        render_signup_success_page()
    else:
        render_signin_page()


# ─────────────────────────────────────────────────────────────
# OPERATIONAL LOGIC
# ─────────────────────────────────────────────────────────────
def severity_rank(level: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(level, 0)


def format_time(value: str, fmt: str = "%I:%M %p") -> str:
    return pd.to_datetime(value).strftime(fmt)


def detection_count_series(history: list[dict[str, Any]]) -> list[int]:
    if not history:
        return [0] * 7
    recent = history[-7:]
    counts = list(range(max(1, len(history) - len(recent) + 1), len(history) + 1))
    while len(counts) < 7:
        counts.insert(0, counts[0] if counts else 0)
    return counts[-7:]


def confidence_series(history: list[dict[str, Any]]) -> list[float]:
    values = [row["confidence"] * 100 for row in history[-7:]]
    while len(values) < 7:
        values.insert(0, values[0] if values else 0)
    return values[-7:]


def build_dashboard_stats(data: dict[str, Any]) -> dict[str, Any]:
    history = data["history"]
    locations = data["locations"]
    routes = data["routes"]
    total = len(history)
    bio = sum(1 for row in history if row["category"] == "Biodegradable")
    non_bio = total - bio
    avg_conf = sum(row["confidence"] for row in history) / max(total, 1)
    high_areas = sum(1 for row in locations if row["severity"] == "High")
    active_trucks = sum(1 for row in routes if row["status"] != "Completed")
    return {"total": total, "bio": bio, "non_bio": non_bio, "accuracy": avg_conf,
            "waste_ton": round(total * 0.18, 1), "high_areas": high_areas, "active_trucks": active_trucks}


def ensure_alert(data: dict[str, Any], title: str, level: str, location: str, action: str, source_key: str) -> None:
    for alert in data["alerts"]:
        if alert["source_key"] == source_key:
            if alert["status"] == "Open":
                alert.update({"time": now_iso(), "level": level, "title": title, "action": action, "location": location})
            return
    data["alerts"].append({"id": f"ALERT-{uuid4().hex[:6].upper()}", "time": now_iso(), "title": title,
                            "level": level, "location": location, "status": "Open", "action": action, "source_key": source_key})


def refresh_bin_statuses(data: dict[str, Any]) -> None:
    for b in data["bins"]:
        fl = int(b["fill_level"])
        b["status"] = "Needs Pickup" if fl >= 90 else "Attention" if fl >= 75 else "Normal"
        if fl >= 85:
            ensure_alert(data, f"{b['name']} nearing capacity", "Medium", b["location"], "Schedule pickup route", f"{b['id']}-capacity")


def refresh_location_statuses(data: dict[str, Any]) -> None:
    for loc in data["locations"]:
        det = int(loc["detections"])
        if det >= 10:
            loc["severity"] = "High"; loc["status"] = "Active"
            ensure_alert(data, f"High waste detected in {loc['name']}", "High", loc["name"], "Dispatch dry-waste collection", f"{loc['id']}-high")
        elif det >= 6:
            loc["severity"] = "Medium"; loc["status"] = "Monitoring"
        else:
            loc["severity"] = "Low"; loc["status"] = "Stable"


def refresh_routes(data: dict[str, Any]) -> None:
    existing = {r["route_name"]: r for r in data["routes"]}
    ordered = sorted(data["locations"], key=lambda r: (severity_rank(r["severity"]), int(r["detections"])), reverse=True)
    groups = [("Route A", ordered[:2]), ("Route B", ordered[2:4]), ("Route C", ordered[4:5])]
    rebuilt = []
    for route_name, locs in groups:
        if not locs:
            continue
        prev = existing.get(route_name, {})
        rebuilt.append({"id": prev.get("id", f"ROUTE-{uuid4().hex[:6].upper()}"), "route_name": route_name,
                         "zones": " -> ".join(l["name"] for l in locs), "priority": locs[0]["severity"],
                         "eta": f"{12 + len(locs) * 6} min", "truck": prev.get("truck", f"Truck-{len(rebuilt)+1:02d}"),
                         "status": prev.get("status", "Ready")})
    data["routes"] = rebuilt


def refresh_operational_state(data: dict[str, Any]) -> None:
    refresh_location_statuses(data)
    refresh_bin_statuses(data)
    refresh_routes(data)


def update_detection_effects(data: dict[str, Any], category: str, location_name: str) -> str:
    loc = next((r for r in data["locations"] if r["name"] == location_name), None)
    if loc:
        loc["detections"] = int(loc["detections"]) + 1
        loc["last_updated"] = now_iso()
    matching_bin = next((b for b in data["bins"] if b["location"] == location_name and b["type"] == category), None)
    if matching_bin is None:
        matching_bin = next((b for b in data["bins"] if b["type"] == category), None)
    if matching_bin is None:
        matching_bin = {"id": f"BIN-{uuid4().hex[:6].upper()}", "name": f"{CATEGORY_META[category]['bin_label']} {location_name}",
                        "type": category, "location": location_name, "fill_level": 20, "status": "Normal", "updated_at": now_iso()}
        data["bins"].append(matching_bin)
    matching_bin["fill_level"] = min(100, int(matching_bin["fill_level"]) + (12 if category == "Non-Biodegradable" else 9))
    matching_bin["updated_at"] = now_iso()
    refresh_operational_state(data)
    return matching_bin["name"]


def run_prediction(image: Image.Image, source: str, location_name: str) -> dict[str, Any]:
    category, confidence, item = classify_image(image)
    data = st.session_state.app_data
    bin_name = update_detection_effects(data, category, location_name)
    record = {"id": f"DET-{uuid4().hex[:6].upper()}", "time": now_iso(), "item": item, "category": category,
              "confidence": float(confidence), "source": source, "location": location_name, "bin": bin_name}
    data["history"].append(record)
    save_app_state()
    st.session_state.latest_detection = record
    st.session_state.latest_image = image
    return record


def run_upload_prediction(image: Image.Image, location_name: str) -> dict[str, Any]:
    category, confidence, item = classify_image(image)
    data = st.session_state.app_data
    bin_name = update_detection_effects(data, category, location_name)
    record = {"id": f"DET-{uuid4().hex[:6].upper()}", "time": now_iso(), "item": item, "category": category,
              "confidence": float(confidence), "source": "Upload", "location": location_name, "bin": bin_name}
    data["history"].append(record)
    save_app_state()
    st.session_state.latest_detection = record
    st.session_state.latest_image = image
    return record


def run_live_prediction(image: Image.Image) -> dict[str, Any]:
    category, confidence, _ = classify_image(image)
    return {"category": category, "confidence": confidence}


def location_options() -> list[str]:
    return [row["name"] for row in st.session_state.app_data["locations"]]


def recent_table(history: list[dict[str, Any]], limit: int = 6) -> pd.DataFrame:
    rows = [{"Time": format_time(r["time"]), "Item": r["item"], "Category": r["category"],
             "Confidence": f"{r['confidence']*100:.1f}%", "Source": r["source"], "Location": r["location"]}
            for r in sorted(history, key=lambda x: x["time"], reverse=True)[:limit]]
    return pd.DataFrame(rows)


def activity_feed(data: dict[str, Any]) -> list[tuple[str, str]]:
    entries = [(a["title"], format_time(a["time"])) for a in sorted(data["alerts"], key=lambda r: r["time"], reverse=True)[:3]]
    entries += [(f"{r['item']} classified in {r['location']}", format_time(r["time"]))
                for r in sorted(data["history"], key=lambda r: r["time"], reverse=True)[:3]]
    return entries[:4]


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def classification_figure(history: list[dict[str, Any]]) -> go.Figure:
    total = len(history)
    if total == 0:
        values, labels, colors = [1], ["No data"], ["#dceee6"]
        annotation_text = "<b>0</b><br>Total"
    else:
        bio = sum(1 for r in history if r["category"] == "Biodegradable")
        values = [bio, total - bio]; labels = ["Biodegradable", "Non-Biodegradable"]
        colors = ["#1fb96d", "#ef5d66"]; annotation_text = f"<b>{total}</b><br>Total"
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.68, sort=False,
                                  marker=dict(colors=colors), textinfo="none")])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      annotations=[dict(text=annotation_text, showarrow=False, font=dict(size=18, color="#16312a"))],
                      showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02))
    return fig


def confidence_trend_figure(history: list[dict[str, Any]]) -> go.Figure:
    if not history:
        x, y = ["No data"], [0]
    else:
        recent = sorted(history, key=lambda r: r["time"])[-8:]
        x = [format_time(r["time"]) for r in recent]; y = [round(r["confidence"] * 100, 1) for r in recent]
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers",
                                      line=dict(color="#4d89ff", width=3), marker=dict(size=8))])
    fig.update_layout(height=300, margin=dict(l=20, r=10, t=10, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Confidence %")
    return fig


def severity_figure(locations: list[dict[str, Any]]) -> go.Figure:
    ordered = sorted(locations, key=lambda r: (severity_rank(r["severity"]), int(r["detections"])), reverse=True)
    fig = go.Figure(data=[go.Bar(x=[r["name"] for r in ordered], y=[int(r["detections"]) for r in ordered],
                                  marker_color=["#ef5d66" if r["severity"] == "High" else "#f1a73b" if r["severity"] == "Medium" else "#1fb96d" for r in ordered])])
    fig.update_layout(height=300, margin=dict(l=20, r=10, t=10, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Detections")
    return fig


# ─────────────────────────────────────────────────────────────
# SHARED RENDER COMPONENTS
# ─────────────────────────────────────────────────────────────
def normalize_sparkline(values: list[float], width: int = 160, height: int = 48) -> tuple[str, str]:
    mn, mx = min(values), max(values)
    spread = mx - mn or 1
    pts = [f"{i*(width/max(len(values)-1,1)):.1f},{height-((v-mn)/spread)*(height-8)-4:.1f}" for i, v in enumerate(values)]
    fill = " ".join(pts + [f"{width},{height}", f"0,{height}"])
    return " ".join(pts), fill


def metric_card(title: str, value: str, delta: str, accent: str, tint: str, icon: str, values: list[float]) -> str:
    line_pts, fill_pts = normalize_sparkline(values)
    return f"""
    <div class="metric-card" style="background:{tint};">
        <div class="metric-badge">{icon}</div>
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta" style="color:{accent};">{delta}</div>
        <svg width="100%" height="52" viewBox="0 0 160 48" fill="none">
            <polygon points="{fill_pts}" fill="{accent}" opacity="0.10"/>
            <polyline points="{line_pts}" stroke="{accent}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
    </div>"""


def render_section_heading(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="info-surface" style="margin-bottom:1rem;">
        <div class="surface-title">{title}</div>
        <div class="surface-subtitle">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(model_status: str) -> None:
    user = st.session_state.current_user
    collapsed = st.session_state.sidebar_collapsed
    shell_state = "sidebar-collapsed" if collapsed else "sidebar-expanded"
    with st.sidebar:
        st.markdown(f'<div class="sidebar-shell {shell_state}"></div>', unsafe_allow_html=True)
        # ── Brand ──────────────────────────────────────
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-badge">SW</div>
            <div class="brand-text-wrap">
                <div class="brand-title">SmartWaste</div>
                <div class="brand-subtitle">For Smart Cities</div>
            </div>
        </div>
        <div class="sidebar-divider"></div>""", unsafe_allow_html=True)
        collapse_label = "☰" if collapsed else "‹ Collapse"
        if st.button(collapse_label, key="sidebar_collapse_btn", use_container_width=True, help="Expand sidebar" if collapsed else "Collapse sidebar"):
            st.session_state.sidebar_collapsed = not collapsed
            st.rerun()

        # ── Navigation ─────────────────────────────────
        if not collapsed:
            st.markdown("<div class='sidebar-nav-title'>Navigation</div>", unsafe_allow_html=True)

        nav_items = NAV_ITEMS if user.get("role") == "Admin" else [n for n in NAV_ITEMS if n != "User Management"]
        for item in nav_items:
            icon = NAV_ICONS.get(item, "•")
            label = icon if collapsed else f"{icon}  {item}"
            btn_type = "primary" if item == st.session_state.current_page else "secondary"
            if st.button(label, key=f"nav_{item}", use_container_width=True, type=btn_type, help=item if collapsed else None):
                st.session_state.current_page = item
                st.rerun()

        st.write("")
        # ── User card ──────────────────────────────────
        initials = "".join(w[0].upper() for w in user.get("display_name", "U").split()[:2])
        if not collapsed:
            st.markdown(f"""
        <div class="sidebar-user-card">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#1fb96d,#17a05e);
                    display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:0.82rem;flex-shrink:0;">
                    {initials}
                </div>
                <div>
                    <div style="font-weight:800;color:#16312a;font-size:0.9rem;line-height:1.2;">{user.get('display_name','User')}</div>
                    <div style="color:#6e8b81;font-size:0.78rem;">{user.get('email','')}</div>
                </div>
            </div>
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
                <span style="background:rgba(31,185,109,0.12);color:#17824d;font-size:0.74rem;font-weight:700;
                    padding:0.2rem 0.55rem;border-radius:99px;">{user.get('role','Operator')}</span>
                <span style="background:rgba(77,137,255,0.10);color:#3563c5;font-size:0.74rem;font-weight:600;
                    padding:0.2rem 0.55rem;border-radius:99px;">📍 {user.get('city','').split(',')[0]}</span>
            </div>
            <div style="margin-top:0.5rem;color:#9ab8ae;font-size:0.75rem;">{model_status}</div>
            </div>""", unsafe_allow_html=True)

        st.write("")
        # ── Keep Our City note ──────────────────────────
        if not collapsed:
            st.markdown("""
        <div class="sidebar-note">
            <div style="font-weight:800;color:#16312a;font-size:0.85rem;">Keep Our City</div>
            <div style="font-weight:800;color:#1fb96d;font-size:0.85rem;">Clean & Green 🌱</div>
            <div style="color:#6e8b81;font-size:0.78rem;margin-top:0.3rem;">Together for a sustainable tomorrow.</div>
            </div>""", unsafe_allow_html=True)

        st.write("")
        signout_label = "🚪" if collapsed else "🚪  Sign Out"
        if st.button(signout_label, key="signout_btn", use_container_width=True, help="Sign Out" if collapsed else None):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.auth_page = "signin"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────
def render_top_area() -> None:
    user = st.session_state.current_user
    data = st.session_state.app_data
    stats = build_dashboard_stats(data)
    left, right = st.columns([1.6, 1], vertical_alignment="center")
    with left:
        st.markdown(f"""
        <div class="hero-title">Welcome back, {user.get('display_name','User')}! 👋</div>
        <p class="hero-subtitle">Here's what's happening in your city today.</p>""", unsafe_allow_html=True)
    with right:
        open_alerts = sum(1 for r in data["alerts"] if r["status"] == "Open")
        st.markdown(f"""
        <div class="pill-row">
            <div class="top-pill">📍 {user.get('city','')}</div>
            <div class="top-pill">🚨 {open_alerts} Open Alerts</div>
            <div class="top-pill">🚛 {stats['active_trucks']} Active Trucks</div>
        </div>""", unsafe_allow_html=True)


def render_metric_row(data: dict[str, Any]) -> None:
    stats = build_dashboard_stats(data)
    history = data["history"]
    bio_hist = [r for r in history if r["category"] == "Biodegradable"]
    non_hist = [r for r in history if r["category"] == "Non-Biodegradable"]
    cards = [
        ("Total Detections", f"{stats['total']:,}", f"↑ Live count", "#1fb96d", "#eefbf4", "📊", detection_count_series(history)),
        ("Biodegradable", f"{stats['bio']:,}", f"{stats['bio']/max(stats['total'],1)*100:.1f}% of total", "#1fb96d", "#f2fff6", "🌿", detection_count_series(bio_hist or history)),
        ("Non-Biodegradable", f"{stats['non_bio']:,}", f"{stats['non_bio']/max(stats['total'],1)*100:.1f}% of total", "#ef5d66", "#fff2f2", "🗑️", detection_count_series(non_hist or history)),
        ("Accuracy", f"{stats['accuracy']*100:.1f}%", "Avg confidence", "#4d89ff", "#f2f7ff", "🎯", confidence_series(history)),
        ("Waste Collected", f"{stats['waste_ton']:.1f} Ton", "Estimated today", "#9966ff", "#f7f2ff", "🚛", detection_count_series(history)),
    ]
    cols = st.columns(5, gap="medium")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(metric_card(*card), unsafe_allow_html=True)


def render_detection_controls(section_key: str) -> None:
    data = st.session_state.app_data
    loc_options = location_options()
    idx = loc_options.index(st.session_state.selected_location) if st.session_state.selected_location in loc_options else 0
    location_name = st.selectbox("Detection location", loc_options, index=idx, key=f"{section_key}_location")
    st.session_state.selected_location = location_name

    latest = st.session_state.latest_detection or fallback_detection()
    meta = CATEGORY_META.get(latest["category"], CATEGORY_META["Biodegradable"])

    if section_key == "classify":
        tab_upload = st.container()
    else:
        tab_upload, tab_camera, tab_live = st.tabs(["📤 Upload Image", "📷 Use Camera", "🔴 Live Preview"])

    if section_key == "dashboard":
        st.markdown("""
        <div class="dashboard-capture-guide">
            <div class="dashboard-guide-card">
                <div class="dashboard-guide-title">Upload Image</div>
                <p class="dashboard-guide-copy">Best for saved JPG or PNG waste photos.</p>
            </div>
            <div class="dashboard-guide-card">
                <div class="dashboard-guide-title">Use Camera</div>
                <p class="dashboard-guide-copy">Capture one clear item on a plain background.</p>
            </div>
            <div class="dashboard-guide-card">
                <div class="dashboard-guide-title">Live Preview</div>
                <p class="dashboard-guide-copy">Use for quick scanning when camera tools are installed.</p>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    with tab_upload:
        uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"], key=f"{section_key}_upload")
        if uploaded:
            preview = Image.open(uploaded).convert("RGB")
            if section_key == "classify":
                upload_id = f"{uploaded.name}:{uploaded.size}"
                if st.session_state.get("classify_upload_id") != upload_id:
                    st.session_state.classify_result_ready = False
                    st.session_state.classify_upload_id = upload_id
                st.session_state.classify_preview_image = preview
            st.image(preview, use_container_width=True)
            if section_key == "live":
                upload_id = f"{uploaded.name}:{uploaded.size}:{location_name}"
                if st.session_state.get("live_upload_id") != upload_id:
                    with st.spinner("Analyzing..."):
                        result = run_upload_prediction(preview, location_name)
                    st.session_state.live_upload_id = upload_id
                    st.session_state.live_upload_result = result
                    st.session_state.live_result_ready = True
                result = st.session_state.get("live_upload_result")
                if result and section_key != "live":
                    st.success(f"✅ {result['item']} detected as {result['category']} ({result['confidence']*100:.1f}% confidence)")
            if st.button("🔍 Analyze Uploaded Image", key=f"{section_key}_analyze_upload"):
                with st.spinner("Analyzing..."):
                    result = run_upload_prediction(preview, location_name)
                if section_key == "classify":
                    st.session_state.classify_result_ready = True
                if section_key == "live":
                    st.session_state.live_result_ready = True
                else:
                    st.success(f"✅ {result['item']} detected as {result['category']} ({result['confidence']*100:.1f}% confidence)")
                if section_key == "classify":
                    st.rerun()

    if section_key != "classify":
        with tab_camera:
            snapped = st.camera_input("Capture from your camera", key=f"{section_key}_camera")
            if snapped:
                preview = Image.open(snapped).convert("RGB")
                if st.button("🔍 Analyze Camera Image", key=f"{section_key}_analyze_camera"):
                    with st.spinner("Analyzing..."):
                        result = run_prediction(preview, "Camera", location_name)
                    if section_key == "live":
                        st.session_state.live_result_ready = True
                    else:
                        st.success(f"✅ {result['item']} detected as {result['category']} ({result['confidence']*100:.1f}% confidence)")
                    if section_key == "classify":
                        st.rerun()

        with tab_live:
            if webrtc_streamer is None or av is None or cv2 is None:
                st.info("Live preview needs `streamlit-webrtc`, `opencv-python`, and `av` installed.")
            else:
                st.caption("Start the live preview and point the camera at a waste item.")

                class WasteProcessor(VideoProcessorBase):
                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pred = run_live_prediction(Image.fromarray(rgb))
                        accent = CATEGORY_META[pred["category"]]["accent"]
                        bgr = tuple(int(accent[i:i+2], 16) for i in (5, 3, 1))
                        overlay = img.copy()
                        cv2.rectangle(overlay, (14, 14), (420, 110), (255,255,255), -1)
                        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
                        cv2.putText(img, pred["category"], (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                        cv2.putText(img, f"{pred['confidence']*100:.1f}% confidence", (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_streamer(key=f"{section_key}_live", video_processor_factory=WasteProcessor,
                                media_stream_constraints={"video": True, "audio": False},
                                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    if section_key == "dashboard":
        return

    if section_key == "live" and not st.session_state.get("live_result_ready"):
        return

    if section_key == "classify" and not st.session_state.get("classify_result_ready"):
        return

    latest = st.session_state.latest_detection or fallback_detection()
    meta = CATEGORY_META.get(latest["category"], CATEGORY_META["Biodegradable"])

    st.write("")
    preview_col, result_col = st.columns([1.6, 1], vertical_alignment="top")
    with preview_col:
        preview_image = st.session_state.get("classify_preview_image") if section_key == "classify" else st.session_state.latest_image
        st.image(preview_image, use_container_width=True, caption="Uploaded image" if section_key == "classify" else "Latest analyzed image")
    with result_col:
        conf_bar = int(latest["confidence"] * 100)
        st.markdown(f"""
        <div class="action-card">
            <div class="surface-title" style="font-size:1.05rem;">Detection Result</div>
            <div class="result-chip" style="background:{meta['soft']};color:{meta['accent']};">
                {meta['icon']} {latest['category']}
            </div>
            <div style="margin-top:1rem;color:#6e8b81;font-size:0.88rem;">Item</div>
            <div style="font-weight:800;font-size:1.05rem;color:#16312a;">{latest['item']}</div>
            <div style="margin-top:0.75rem;color:#6e8b81;font-size:0.88rem;">Confidence</div>
            <div style="font-weight:800;font-size:1.05rem;color:#16312a;">{conf_bar:.1f}%</div>
            <div style="background:#e5e7eb;border-radius:99px;height:6px;margin-top:0.35rem;">
                <div style="background:{meta['accent']};width:{conf_bar}%;height:6px;border-radius:99px;"></div>
            </div>
            <div style="margin-top:0.8rem;color:#6e8b81;font-size:0.88rem;">Location</div>
            <div style="font-weight:700;color:#16312a;">{latest['location']}</div>
            <div style="margin-top:0.75rem;color:#6e8b81;font-size:0.88rem;">Recommended Bin</div>
            <div style="font-weight:700;color:#16312a;">🗑️ {latest['bin']}</div>
            <div style="margin-top:0.75rem;color:#6e8b81;font-size:0.88rem;">Recommendation</div>
            <div style="font-weight:700;color:#16312a;">{meta['hint']}</div>
        </div>""", unsafe_allow_html=True)


def render_map_panel(compact: bool = False) -> None:
    locs = st.session_state.app_data["locations"]
    render_section_heading("Waste Locations", "Priority hotspots across the city.")
    df = pd.DataFrame(locs).rename(columns={"lat": "LAT", "lon": "LON"})
    if compact:
        st.map(df, latitude="LAT", longitude="LON", size="detections", height=320)
    else:
        st.map(df, latitude="LAT", longitude="LON", size="detections")
    sev = pd.DataFrame(locs)["severity"].value_counts()
    stats = build_dashboard_stats(st.session_state.app_data)
    cols = st.columns(4, gap="small")
    for col, (label, val, color) in zip(cols, [
        ("🟢 Low Areas", int(sev.get("Low", 0)), "#1fb96d"),
        ("🟡 Medium Areas", int(sev.get("Medium", 0)), "#f1a73b"),
        ("🔴 High Areas", int(sev.get("High", 0)), "#ef5d66"),
        ("🚛 Active Trucks", stats["active_trucks"], "#4d89ff"),
    ]):
        with col:
            st.markdown(f"""
            <div class="action-card dashboard-map-stat">
                <div style="color:{color};font-weight:800;font-size:1.5rem;">{val}</div>
                <div class="dashboard-map-stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)


def render_lower_grid() -> None:
    data = st.session_state.app_data
    col1, col2 = st.columns([1.0, 1.35], gap="large", vertical_alignment="top")

    with col1:
        render_section_heading("Classification Overview", "Biodegradable vs Non-Biodegradable split.")
        fig = classification_figure(data["history"])
        fig.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        render_section_heading("Recent Detections", "Latest classification events.")
        df = recent_table(data["history"])
        st.dataframe(df, use_container_width=True, hide_index=True, height=320)

    st.write("")
    col3, col4 = st.columns([0.9, 1.35], gap="large", vertical_alignment="top")

    with col3:
        render_section_heading("Quick Actions", "Fast operational controls.")
        if st.button("▶ Start Live Detection", key="qa_live"):
            st.session_state.current_page = "Live Detection"; st.rerun()
        if st.button("🗺 View Routes", key="qa_routes"):
            st.session_state.current_page = "Collection Routes"; st.rerun()
        csv = recent_table(data["history"]).to_csv(index=False)
        st.download_button("📄 Generate Report", csv, file_name="smartwaste-report.csv", mime="text/csv")
        if st.button("📍 Add Location", key="qa_loc"):
            st.session_state.current_page = "Waste Locations"; st.rerun()

    with col4:
        render_section_heading("Activity Feed", "Latest alerts and updates.")
        for msg, ts in activity_feed(data):
            st.markdown(f"""
            <div class="activity-item">
                <div style="font-weight:700;color:#16312a;font-size:0.88rem;">{msg}</div>
                <div style="color:#6e8b81;white-space:nowrap;font-size:0.82rem;">{ts}</div>
            </div>""", unsafe_allow_html=True)


def render_feature_strip() -> None:
    features = [
        ("🤖 AI Powered Detection", "Upload, camera, and live preview feed the same classifier."),
        ("📡 Real-time Monitoring", "Hotspots, bins, alerts, and routes stay in sync."),
        ("🗺 Smart Routing", "Routes are rebuilt from current location severity."),
        ("📊 Data Analytics", "History, charts, and exports available anytime."),
        ("🌱 Sustainable City", "Building a cleaner and greener tomorrow."),
    ]
    cols = st.columns(5, gap="medium")
    for col, (title, copy) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-strip">
                <div class="feature-title">{title}</div>
                <p class="feature-copy">{copy}</p>
            </div>""", unsafe_allow_html=True)


def render_dashboard_page() -> None:
    st.markdown('<div class="dashboard-page"></div>', unsafe_allow_html=True)
    render_top_area()
    st.write("")
    render_metric_row(st.session_state.app_data)
    st.write("")
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="top")
    with left:
        render_section_heading("Detection Workspace", "Analyze waste items from upload, camera, or live preview.")
        render_detection_controls("dashboard")
    with right:
        render_map_panel(compact=True)
    st.write("")
    render_lower_grid()
    st.write("")
    render_feature_strip()


# ─────────────────────────────────────────────────────────────
# OTHER PAGES
# ─────────────────────────────────────────────────────────────
def render_live_detection_page() -> None:
    render_section_heading("Live Detection", "Analyze waste items — upload, camera, or live stream.")
    render_detection_controls("live")


def render_image_classification_page() -> None:
    render_section_heading("Image Classification", "Upload and classify waste images.")
    render_detection_controls("classify")


def render_locations_page() -> None:
    data = st.session_state.app_data
    render_section_heading("Waste Locations", "Add locations, review severity, manage area activity.")
    left, right = st.columns([1.2, 1], vertical_alignment="top")
    with left:
        render_map_panel()
        df = pd.DataFrame(data["locations"])[["name", "area", "severity", "detections", "status", "last_updated"]].copy()
        df["last_updated"] = df["last_updated"].apply(lambda v: format_time(v, "%Y-%m-%d %I:%M %p"))
        df.columns = ["Location", "Area", "Severity", "Detections", "Status", "Last Updated"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    with right:
        with st.form("add_loc_form", clear_on_submit=True):
            st.subheader("Add New Location")
            name = st.text_input("Location name")
            area = st.text_input("Area / Zone")
            lat = st.number_input("Latitude", value=28.6139, format="%.6f")
            lon = st.number_input("Longitude", value=77.2090, format="%.6f")
            if st.form_submit_button("Save Location"):
                if not name.strip():
                    st.error("Location name is required.")
                elif any(r["name"].lower() == name.strip().lower() for r in data["locations"]):
                    st.error("That location already exists.")
                else:
                    data["locations"].append({"id": f"LOC-{uuid4().hex[:6].upper()}", "name": name.strip(),
                                               "area": area.strip() or "Unknown", "lat": float(lat), "lon": float(lon),
                                               "severity": "Low", "detections": 0, "status": "Stable", "last_updated": now_iso()})
                    refresh_operational_state(data); save_app_state()
                    st.success("Location added."); st.rerun()
        st.write("")
        selected = st.selectbox("Update existing location", location_options(), key="edit_loc")
        sel_row = next(r for r in data["locations"] if r["name"] == selected)
        with st.form("edit_loc_form"):
            status = st.selectbox("Status", ["Stable", "Monitoring", "Active"],
                                  index=["Stable", "Monitoring", "Active"].index(sel_row["status"]))
            dets = st.number_input("Detections", min_value=0, value=int(sel_row["detections"]), step=1)
            if st.form_submit_button("Update Location"):
                sel_row["status"] = status; sel_row["detections"] = int(dets); sel_row["last_updated"] = now_iso()
                refresh_operational_state(data); save_app_state()
                st.success("Location updated."); st.rerun()


def render_routes_page() -> None:
    data = st.session_state.app_data
    render_section_heading("Collection Routes", "Routes are rebuilt from current hotspots.")
    df = pd.DataFrame(data["routes"])
    if not df.empty:
        df.columns = ["ID", "Route", "Zones", "Priority", "ETA", "Truck", "Status"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not data["routes"]:
        st.info("No routes available yet."); return
    sel = st.selectbox("Select route", [r["route_name"] for r in data["routes"]])
    row = next(r for r in data["routes"] if r["route_name"] == sel)
    with st.form("route_update_form"):
        truck = st.text_input("Assigned truck", value=row["truck"])
        status = st.selectbox("Status", ["Ready", "In Progress", "Completed"],
                              index=["Ready", "In Progress", "Completed"].index(row["status"]))
        if st.form_submit_button("Update Route"):
            row["truck"] = truck.strip() or row["truck"]; row["status"] = status
            save_app_state(); st.success("Route updated."); st.rerun()


def render_analytics_page() -> None:
    data = st.session_state.app_data
    render_section_heading("Analytics & Reports", "Monitor accuracy, trends, and hotspot distribution.")
    left, right = st.columns(2)
    with left:
        st.subheader("Confidence Trend")
        st.plotly_chart(confidence_trend_figure(data["history"]), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Location Severity")
        st.plotly_chart(severity_figure(data["locations"]), use_container_width=True, config={"displayModeBar": False})
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(recent_table(data["history"], limit=10), use_container_width=True, hide_index=True)
    with c2:
        snapshot = {"generated_at": now_iso(), "stats": build_dashboard_stats(data),
                    "open_alerts": [r for r in data["alerts"] if r["status"] == "Open"]}
        st.download_button("📥 Download Analytics JSON", data=json.dumps(snapshot, indent=2),
                           file_name="smartwaste-analytics.json", mime="application/json")


def render_history_page() -> None:
    data = st.session_state.app_data
    history = data["history"]

    # ── Page heading ───────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.25rem;">
        <div class="hero-title" style="font-size:1.6rem;">Detection History</div>
        <p class="hero-subtitle">A full log of every waste classification event.</p>
    </div>""", unsafe_allow_html=True)

    # ── Summary stat cards ─────────────────────────────────────
    total = len(history)
    bio = sum(1 for r in history if r["category"] == "Biodegradable")
    non_bio = total - bio
    avg_conf = (sum(r["confidence"] for r in history) / max(total, 1)) * 100
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val, color, bg in [
        (s1, "Total Records", str(total), "#16312a", "#eefbf4"),
        (s2, "Biodegradable", str(bio), "#1fb96d", "#f0fff7"),
        (s3, "Non-Biodegradable", str(non_bio), "#ef5d66", "#fff2f2"),
        (s4, "Avg Confidence", f"{avg_conf:.1f}%", "#4d89ff", "#f2f7ff"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background:{bg};min-height:auto;padding:0.85rem 1rem;">
                <div class="metric-title">{label}</div>
                <div class="metric-value" style="font-size:1.55rem;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")

    # ── Filters ────────────────────────────────────────────────
    st.markdown('<div class="info-surface" style="margin-bottom:0.75rem;padding:0.9rem 1.1rem;">'
                '<div class="surface-title" style="font-size:1rem;margin-bottom:0.6rem;">🔍 Filter Records</div>',
                unsafe_allow_html=True)
    cats = ["All"] + sorted({r["category"] for r in history})
    srcs = ["All"] + sorted({r["source"] for r in history})
    locs = ["All"] + sorted({r["location"] for r in history})
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_filter = st.selectbox("Category", cats, key="hist_cat")
    with c2:
        src_filter = st.selectbox("Source", srcs, key="hist_src")
    with c3:
        loc_filter = st.selectbox("Location", locs, key="hist_loc")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Filter & build dataframe ───────────────────────────────
    filtered = [
        r for r in history
        if (cat_filter == "All" or r["category"] == cat_filter)
        and (src_filter == "All" or r["source"] == src_filter)
        and (loc_filter == "All" or r["location"] == loc_filter)
    ]
    df = (
        pd.DataFrame(filtered).sort_values("time", ascending=False)
        if filtered
        else pd.DataFrame(columns=["time", "item", "category", "confidence", "source", "location", "bin"])
    )
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %I:%M %p")
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        df = df[["time", "item", "category", "confidence", "source", "location", "bin"]]
        df.columns = ["Time", "Item", "Category", "Confidence", "Source", "Location", "Bin"]

    st.markdown('<div class="info-surface" style="padding:1rem;">'
                '<div class="surface-title" style="font-size:1rem;margin-bottom:0.6rem;">📋 Detection Records'
                f'<span style="font-size:0.82rem;font-weight:600;color:#6e8b81;margin-left:0.6rem;">({len(filtered)} records)</span></div>',
                unsafe_allow_html=True)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Confidence": st.column_config.TextColumn("Confidence"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    ca, cb = st.columns(2)
    with ca:
        csv = df.to_csv(index=False) if not df.empty else "Time,Item,Category,Confidence,Source,Location,Bin\n"
        st.download_button("📥 Export CSV", data=csv, file_name="smartwaste-history.csv", mime="text/csv", use_container_width=True)
    with cb:
        if st.button("🗑️ Clear All History", key="clear_history", use_container_width=True):
            data["history"] = []
            st.session_state.latest_detection = fallback_detection()
            st.session_state.latest_image = choose_demo_image()
            save_app_state()
            st.success("History cleared.")
            st.rerun()


def render_alerts_page() -> None:
    data = st.session_state.app_data
    render_section_heading("Alerts", "Acknowledge or create operational alerts.")
    df = pd.DataFrame(sorted(data["alerts"], key=lambda r: r["time"], reverse=True))
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %I:%M %p")
        df = df[["id", "time", "title", "level", "location", "status", "action"]]
        df.columns = ["ID", "Time", "Alert", "Level", "Location", "Status", "Action"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    open_alerts = [r for r in data["alerts"] if r["status"] == "Open"]
    c1, c2 = st.columns(2)
    with c1:
        if open_alerts:
            sel = st.selectbox("Open alerts", [f"{r['id']} — {r['title']}" for r in open_alerts])
            if st.button("✅ Mark Selected Resolved", key="resolve_alert"):
                aid = sel.split(" — ")[0]
                for a in data["alerts"]:
                    if a["id"] == aid: a["status"] = "Resolved"; break
                save_app_state(); st.success("Alert resolved."); st.rerun()
        else:
            st.success("✅ No open alerts right now.")
    with c2:
        with st.form("manual_alert_form", clear_on_submit=True):
            title = st.text_input("Alert title")
            loc = st.selectbox("Location", location_options(), key="manual_alert_loc")
            level = st.selectbox("Level", ["Low", "Medium", "High"])
            action = st.text_input("Recommended action", value="Review area")
            if st.form_submit_button("➕ Create Alert"):
                if not title.strip(): st.error("Title is required.")
                else:
                    ensure_alert(data, title.strip(), level, loc, action.strip() or "Review area", f"manual-{uuid4().hex[:6]}")
                    save_app_state(); st.success("Alert created."); st.rerun()


def render_bin_status_page() -> None:
    data = st.session_state.app_data
    render_section_heading("Bin Status", "Update fill levels and operational status for collection bins.")
    df = pd.DataFrame(data["bins"])
    df["updated_at"] = df["updated_at"].apply(lambda v: format_time(v, "%Y-%m-%d %I:%M %p"))
    df.columns = ["ID", "Bin", "Type", "Location", "Fill Level (%)", "Status", "Updated At"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    sel = st.selectbox("Select bin to update", [r["name"] for r in data["bins"]])
    row = next(r for r in data["bins"] if r["name"] == sel)
    with st.form("bin_update_form"):
        fill = st.slider("Fill level (%)", 0, 100, int(row["fill_level"]))
        status = st.selectbox("Status", ["Normal", "Attention", "Needs Pickup"],
                              index=["Normal", "Attention", "Needs Pickup"].index(row["status"]))
        if st.form_submit_button("Update Bin"):
            row["fill_level"] = fill; row["status"] = status; row["updated_at"] = now_iso()
            refresh_bin_statuses(data); save_app_state(); st.success("Bin updated."); st.rerun()


def render_user_management_page() -> None:
    """Admin-only: view and manage all registered users."""
    render_section_heading("User Management", "View and manage all registered SmartWaste accounts.")
    users = load_users()
    if not users:
        st.info("No registered users yet.")
        return
    rows = [{"Email": k, "Name": v["display_name"], "Role": v["role"], "City": v["city"], "Created": format_time(v.get("created_at", now_iso()), "%Y-%m-%d %I:%M %p")}
            for k, v in users.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("Change User Role")
    emails = list(users.keys())
    sel_email = st.selectbox("Select user", emails, format_func=lambda e: f"{users[e]['display_name']} ({e})")
    new_role = st.selectbox("New role", ROLE_OPTIONS, index=ROLE_OPTIONS.index(users[sel_email]["role"]))
    if st.button("Update Role"):
        users[sel_email]["role"] = new_role
        save_users(users)
        st.success(f"Role updated for {users[sel_email]['display_name']}.")

    st.write("")
    st.subheader("Delete User")
    del_email = st.selectbox("Select user to delete", [e for e in emails if e != st.session_state.current_user["email"]],
                             format_func=lambda e: f"{users[e]['display_name']} ({e})")
    if st.button("🗑️ Delete User", key="delete_user_btn"):
        del users[del_email]
        save_users(users)
        st.success("User deleted."); st.rerun()


def render_settings_page(model_status: str) -> None:
    user = st.session_state.current_user
    render_section_heading("Settings", "Manage your profile and password.")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("profile_form"):
            display_name = st.text_input("Display name", value=user["display_name"])
            city = st.selectbox("City", CITY_OPTIONS, index=CITY_OPTIONS.index(user["city"]) if user["city"] in CITY_OPTIONS else 0)
            if st.form_submit_button("Save Profile"):
                users = load_users()
                key = user["email"].lower()
                if key in users:
                    users[key]["display_name"] = display_name.strip() or user["display_name"]
                    users[key]["city"] = city
                    save_users(users)
                    st.session_state.current_user["display_name"] = users[key]["display_name"]
                    st.session_state.current_user["city"] = city
                st.success("Profile updated."); st.rerun()
    with c2:
        with st.form("password_form"):
            current_pw = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password (min 8 chars)", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Change Password"):
                users = load_users()
                key = user["email"].lower()
                if users.get(key, {}).get("password_hash") != hash_password(current_pw):
                    st.error("Current password is incorrect.")
                elif len(new_pw) < 8:
                    st.error("New password must be at least 8 characters.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    users[key]["password_hash"] = hash_password(new_pw)
                    save_users(users)
                    st.success("Password updated.")

    st.write("")
    st.markdown(f"""
    <div class="action-card">
        <div style="font-weight:800;margin-bottom:0.4rem;color:#16312a;">System Status</div>
        <div style="color:#497063;">{model_status}</div>
        <div style="margin-top:0.6rem;color:#6e8b81;font-size:0.88rem;">
            Logged in as <strong>{user['display_name']}</strong> · Role: <strong>{user['role']}</strong>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def render_main_app() -> None:
    _, model_status = load_model()
    render_sidebar(model_status)
    page = st.session_state.current_page

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Live Detection":
        render_live_detection_page()
    elif page == "Image Classification":
        render_image_classification_page()
    elif page == "Waste Locations":
        render_locations_page()
    elif page == "Collection Routes":
        render_routes_page()
    elif page == "Analytics & Reports":
        render_analytics_page()
    elif page == "History":
        render_history_page()
    elif page == "Alerts":
        render_alerts_page()
    elif page == "Bin Status":
        render_bin_status_page()
    elif page == "User Management":
        render_user_management_page()
    elif page == "Settings":
        render_settings_page(model_status)


def main() -> None:
    inject_styles()
    bootstrap_session()
    refresh_operational_state(st.session_state.app_data)
    save_app_state()

    if not st.session_state.authenticated:
        render_auth_flow()
        return

    render_main_app()


if __name__ == "__main__":
    main()
    
