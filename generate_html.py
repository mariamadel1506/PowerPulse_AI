# %%
import base64
import os
from pathlib import Path
from flask import Flask, render_template_string

# PowerPulse AI - Flask UI
# Backend expected at: http://127.0.0.1:8000
# Run this file in VS Code. Cells are separated with # %%.

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)


BACKEND_API_URL = os.getenv("POWERPULSE_API_URL", "")

# %%
def find_and_convert_image(target_names):
    if isinstance(target_names, str):
        target_names = [target_names]

    target_names_lower = {name.lower() for name in target_names}

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower() in target_names_lower:
                img_path = Path(root) / file
                ext = file.rsplit(".", 1)[-1].lower()
                mime_type = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "webp": "image/webp",
                }.get(ext, "image/jpeg")

                try:
                    encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    return f"data:{mime_type};base64,{encoded}"
                except Exception:
                    continue

    return ""


bg_base64 = find_and_convert_image(
    ["welcome.jpeg", "welcome.jpg", "welcome.png", "welcome.webp", "bg.jpg"]
)

logo_base64 = find_and_convert_image(
    ["shield-logo.jpeg", "shield-logo.jpg", "shield-logo.png", "logo.png", "logo.jpg"]
)


# %%
html_template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PowerPulse AI</title>

    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --primary-cyan: #00d2ff;
            --primary-blue: #0072ff;
            --bg-dark: #070b14;
            --bg-main: #050811;
            --sidebar-bg: #090e1a;
            --card-bg: rgba(13, 22, 40, 0.65);
            --card-solid: #0d1628;
            --card-border: rgba(0, 210, 255, 0.15);
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --text-muted: #64748b;
            --input-bg: rgba(15, 25, 45, 0.8);
            --success: #00e676;
            --warning: #ffb020;
            --danger: #ff4d6d;
            --critical: #ff0055;
            --shadow: rgba(0, 0, 0, 0.3);
        }

        body.light-mode {
            --bg-dark: #eef3f8;
            --bg-main: #f6f9fc;
            --sidebar-bg: #ffffff;
            --card-bg: rgba(255, 255, 255, 0.92);
            --card-solid: #ffffff;
            --card-border: rgba(15, 23, 42, 0.11);
            --text-main: #0f172a;
            --text-sub: #334155;
            --text-muted: #475569;
            --input-bg: #f3f6fa;
            --shadow: rgba(15, 23, 42, 0.08);
        }

        body.light-mode .card-text,
        body.light-mode .step-desc,
        body.light-mode .form-hint,
        body.light-mode .settings-description,
        body.light-mode .report-section p,
        body.light-mode .report-section li,
        body.light-mode .report-key,
        body.light-mode .metric-label,
        body.light-mode .stat-label,
        body.light-mode .weather-stat-label,
        body.light-mode .menu-label {
            color: #334155;
        }

        body.light-mode .sidebar {
            border-right-color: rgba(15, 23, 42, 0.08);
            box-shadow: 4px 0 22px rgba(15, 23, 42, 0.06);
        }

        body.light-mode .nav-item,
        body.light-mode .report-code {
            color: #334155;
        }

        body.light-mode .nav-item svg {
            stroke: #334155;
        }


        body.light-mode .sidebar-menu {
            scrollbar-color: rgba(71, 85, 105, 0.32) transparent;
        }

        body.light-mode .sidebar-menu::-webkit-scrollbar-thumb {
            background: rgba(71, 85, 105, 0.32);
        }

        body.light-mode .hero-card {
            background: linear-gradient(135deg, #e8f3fa 0%, #dce9f7 100%);
            border-color: rgba(0, 114, 255, 0.16);
        }

        body.light-mode .hero-headline {
            color: #0f172a;
        }

        body.light-mode .hero-card .card-text {
            color: #334155;
        }

        body.light-mode .hero-visual-flow {
            background: rgba(255, 255, 255, 0.72);
            border-color: rgba(0, 114, 255, 0.18);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }

        body.light-mode .flow-node {
            background: rgba(15, 23, 42, 0.055);
            color: #334155;
        }

        body.light-mode .flow-node.active-node {
            color: #06233d;
        }

        body.light-mode .progress-bar-bg {
            background: rgba(15, 23, 42, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
        }

        body {
            background: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            transition: background 0.25s ease, color 0.25s ease;
        }

        button, input, select {
            font: inherit;
        }

        .hidden {
            display: none !important;
        }

        /* ==================== WELCOME ==================== */

        #welcome-section {
            width: 100%;
            min-height: 100vh;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 40px 60px;
            background-image:
                linear-gradient(135deg, rgba(7, 11, 20, 0.35), rgba(4, 8, 15, 0.55)),
                url("__BG_BASE64__");
            background-position: center;
            background-size: cover;
            background-repeat: no-repeat;
            box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.4);
        }

        .welcome-header {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .welcome-logo-img {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background-image: url("__LOGO_BASE64__");
            background-position: center;
            background-size: cover;
            border: 2px solid var(--primary-cyan);
            box-shadow: 0 0 12px rgba(0, 210, 255, 0.5);
        }

        .welcome-brand-name {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
        }

        .welcome-brand-name span {
            color: var(--primary-cyan);
        }

        .welcome-hero {
            max-width: 750px;
            margin-top: -30px;
        }

        .welcome-subtitle {
            color: var(--primary-cyan);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }

        .welcome-title {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 20px;
            color: #ffffff;
        }

        .highlight {
            background: linear-gradient(90deg, #00d2ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .welcome-description {
            font-size: 1.05rem;
            color: #c2cbd6;
            line-height: 1.6;
            margin-bottom: 35px;
        }

        .btn-enter-platform,
        .btn-primary-action {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 14px 28px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(90deg, #0072ff, #00d2ff);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
            transition: all 0.25s ease;
        }

        .btn-enter-platform {
            padding: 16px 32px;
            font-size: 1rem;
        }

        .btn-enter-platform:hover,
        .btn-primary-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 210, 255, 0.55);
        }

        .welcome-cards-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 40px;
        }

        .feature-card {
            background: rgba(13, 22, 40, 0.65);
            border: 1px solid rgba(0, 210, 255, 0.15);
            backdrop-filter: blur(12px);
            border-radius: 10px;
            padding: 20px;
        }

        .feature-card-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--primary-cyan);
            margin-bottom: 6px;
        }

        .feature-card-desc {
            font-size: 0.85rem;
            color: #9ca9b8;
            line-height: 1.4;
        }

        .welcome-footer-ticker {
            display: flex;
            justify-content: space-around;
            align-items: center;
            background: rgba(7, 11, 20, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px 20px;
            margin-top: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: #8b949e;
        }

        .ticker-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .ticker-dot {
            width: 6px;
            height: 6px;
            background-color: var(--primary-cyan);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--primary-cyan);
            flex-shrink: 0;
        }

        /* ==================== DASHBOARD ==================== */

        .dashboard-container {
            display: flex;
            width: 100vw;
            height: 100vh;
            background: var(--bg-main);
            color: var(--text-main);
            overflow: hidden;
        }

        /* ==================== SIDEBAR / BRAND ==================== */

        .sidebar {
            width: 270px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            padding: 18px 16px 20px;
            flex-shrink: 0;
            box-shadow: 5px 0 25px var(--shadow);
            z-index: 10;
            position: relative;
            overflow: hidden;
            transition:
                width .34s cubic-bezier(.22,1,.36,1),
                padding .34s cubic-bezier(.22,1,.36,1),
                background-color .25s ease,
                border-color .25s ease;
        }

        .sidebar-toggle {
            position: absolute;
            top: 14px;
            left: calc(100% - 29px);
            right: auto;
            transform: translateX(-50%);
            width: 30px;
            height: 30px;
            margin: 0;
            padding: 0;
            border: 0;
            border-radius: 9px;
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            z-index: 5;
            transition:
                left .34s cubic-bezier(.22,1,.36,1),
                color .2s ease,
                background-color .2s ease,
                transform .34s cubic-bezier(.22,1,.36,1);
        }

        .sidebar-toggle:hover {
            background: rgba(0, 210, 255, 0.07);
            color: var(--primary-cyan);
        }

        .sidebar-toggle:focus-visible {
            outline: 2px solid rgba(0, 210, 255, .45);
            outline-offset: 2px;
        }

        .sidebar-toggle svg {
            width: 18px;
            height: 18px;
            transition: transform .34s cubic-bezier(.22,1,.36,1);
        }

        .sidebar-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            min-height: 94px;
            margin: 46px 0 28px;
            padding: 0;
            text-align: center;
            position: relative;
            flex-shrink: 0;
            transition:
                margin .34s cubic-bezier(.22,1,.36,1),
                min-height .34s cubic-bezier(.22,1,.36,1);
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            margin-inline: auto;
            border-radius: 50%;
            background-image: url("__LOGO_BASE64__");
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            border: 2px solid var(--primary-cyan);
            box-shadow: 0 0 10px rgba(0, 210, 255, 0.34);
            flex: 0 0 48px;
            transition:
                width .34s cubic-bezier(.22,1,.36,1),
                height .34s cubic-bezier(.22,1,.36,1),
                flex-basis .34s cubic-bezier(.22,1,.36,1),
                box-shadow .25s ease;
        }

        .logo-text {
            display: block;
            width: 100%;
            text-align: center;
            font-size: 1.05rem;
            line-height: 1.2;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: .45px;
            white-space: nowrap;
            opacity: 1;
            transform: translateY(0);
            transition:
                opacity .18s ease,
                transform .34s cubic-bezier(.22,1,.36,1);
        }

        .logo-text span {
            color: var(--primary-cyan);
        }

        .sidebar-menu {
            display: flex;
            flex-direction: column;
            gap: 22px;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0 2px 2px 0;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }

        .sidebar-menu::-webkit-scrollbar {
            width: 0;
            height: 0;
            display: none;
        }

        .dashboard-container.sidebar-collapsed .sidebar {
            width: 82px;
            padding-left: 10px;
            padding-right: 10px;
        }

        .dashboard-container.sidebar-collapsed .sidebar-toggle {
            left: 50%;
            right: auto;
            transform: translateX(-50%);
        }

        .dashboard-container.sidebar-collapsed .sidebar-header {
            width: 100%;
            min-height: 58px;
            margin: 50px 0 30px;
            padding: 0;
            align-items: center;
            justify-content: center;
        }

        .dashboard-container.sidebar-collapsed .logo-icon {
            width: 44px;
            height: 44px;
            flex-basis: 44px;
            box-shadow: 0 0 9px rgba(0, 210, 255, 0.28);
        }

        .dashboard-container.sidebar-collapsed .logo-text,
        .dashboard-container.sidebar-collapsed .menu-label,
        .dashboard-container.sidebar-collapsed .nav-item span {
            display: none;
        }

        .dashboard-container.sidebar-collapsed .sidebar-menu {
            gap: 18px;
            align-items: center;
        }

        .dashboard-container.sidebar-collapsed .menu-group {
            align-items: center;
            width: 100%;
        }

        .dashboard-container.sidebar-collapsed .nav-item {
            justify-content: center;
            width: 52px;
            padding: 13px;
            gap: 0;
        }

        .dashboard-container.sidebar-collapsed .nav-item.active {
            border-left: none;
            border-right: 3px solid var(--primary-cyan);
        }

        .menu-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .menu-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 1.5px;
            margin-bottom: 4px;
            padding-left: 10px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 12px 14px;
            border-radius: 10px;
            color: var(--text-sub);
            text-decoration: none;
            font-size: 0.92rem;
            font-weight: 500;
            transition: all 0.25s ease;
            cursor: pointer;
            border: none;
            background: transparent;
            width: 100%;
            text-align: left;
        }

        .nav-item svg {
            width: 20px;
            height: 20px;
            stroke: var(--text-sub);
            transition: stroke 0.25s ease;
            flex-shrink: 0;
        }

        .nav-item:hover {
            background: rgba(0, 210, 255, 0.08);
            color: var(--text-main);
        }

        .nav-item.active {
            background: linear-gradient(
                90deg,
                rgba(0, 210, 255, 0.18),
                rgba(0, 114, 255, 0.05)
            );
            color: var(--primary-cyan);
            font-weight: 700;
            border-left: 3px solid var(--primary-cyan);
        }

        .nav-item.active svg {
            stroke: var(--primary-cyan);
        }

        .main-content {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            background: var(--bg-main);
        }

        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 22px 36px;
            border-bottom: 1px solid var(--card-border);
            background: color-mix(in srgb, var(--sidebar-bg) 88%, transparent);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 5;
        }

        .command-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .top-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.82rem;
            color: var(--text-sub);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
        }

        .theme-toggle {
            width: 38px;
            height: 38px;
            border-radius: 9px;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            color: var(--text-main);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .view-wrapper {
            padding: 32px 36px;
            display: flex;
            flex-direction: column;
            gap: 28px;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
        }

        .dashboard-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 28px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px var(--shadow);
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 12px;
        }

        .card-text {
            font-size: 0.92rem;
            color: var(--text-sub);
            line-height: 1.6;
        }

        .hero-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 30px;
            background: linear-gradient(
                135deg,
                rgba(13, 22, 40, 0.95),
                rgba(0, 114, 255, 0.2)
            );
        }

        .hero-headline {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 12px;
        }

        .hero-visual-flow {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(7, 11, 20, 0.7);
            padding: 18px 24px;
            border-radius: 12px;
            border: 1px solid rgba(0, 210, 255, 0.25);
            flex-shrink: 0;
        }

        .flow-node {
            padding: 10px 18px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
            font-weight: 600;
            color: #8b949e;
        }

        .flow-node.active-node {
            background: var(--primary-cyan);
            color: #000;
            box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
        }

        .flow-arrow {
            color: var(--primary-cyan);
            font-weight: bold;
        }

        .dashboard-grid-2,
        .results-grid-2,
        .climate-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        .feature-bullets {
            list-style: none;
            margin-top: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .feature-bullets li {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9rem;
            color: var(--primary-cyan);
            font-weight: 600;
        }

        .steps-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 18px;
        }

        .step-card,
        .metric-card,
        .stat-card {
            background: rgba(7, 11, 20, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
        }

        body.light-mode .step-card,
        body.light-mode .metric-card,
        body.light-mode .stat-card {
            background: rgba(255, 255, 255, 0.72);
            border-color: rgba(15, 23, 42, 0.08);
        }

        .step-num {
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--primary-cyan);
            letter-spacing: 1px;
        }

        .step-title {
            font-size: 1rem;
            font-weight: 700;
            margin: 8px 0;
        }

        .step-desc {
            font-size: 0.85rem;
            color: var(--text-sub);
        }

        .cta-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 25px;
            border-color: rgba(0, 210, 255, 0.3);
        }

        /* ==================== FORMS ==================== */

        .form-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
            margin-top: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .form-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .form-control {
            width: 100%;
            padding: 14px 18px;
            background-color: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .form-control:focus {
            border-color: var(--primary-cyan);
            box-shadow: 0 0 12px rgba(0, 210, 255, 0.25);
        }

        select.form-control option {
            background: #070b14;
            color: #fff;
        }

        .form-hint {
            font-size: 0.8rem;
            color: var(--text-sub);
        }

        .api-state {
            margin-top: 15px;
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            display: none;
        }

        .api-state.error {
            display: block;
            background: rgba(255, 0, 85, 0.08);
            border: 1px solid rgba(255, 0, 85, 0.25);
            color: #ff8da8;
        }

        .api-state.loading {
            display: block;
            background: rgba(0, 210, 255, 0.08);
            border: 1px solid rgba(0, 210, 255, 0.2);
            color: var(--primary-cyan);
        }

        /* ==================== RESULTS ==================== */

        .summary-banner {
            background: rgba(0, 210, 255, 0.08);
            border: 1px solid var(--primary-cyan);
            border-radius: 10px;
            padding: 18px 24px;
            font-size: 1.05rem;
            font-weight: 600;
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .metric-header,
        .report-row,
        .section-heading-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--text-sub);
            font-weight: 600;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary-cyan);
        }

        .progress-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #0072ff, #00d2ff);
            width: 0%;
            transition: width 0.6s ease;
        }

        .report-box {
            background: rgba(9, 14, 26, 0.8);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        body.light-mode .report-box {
            background: rgba(255,255,255,0.7);
        }

        .report-row {
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
        }

        .report-key {
            color: var(--text-sub);
            font-weight: 500;
        }

        .report-val {
            color: var(--text-main);
            font-weight: 700;
            text-align: right;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 18px;
        }

        .chart-container.small {
            height: 240px;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .stat-card .stat-label {
            font-size: 0.78rem;
            color: var(--text-sub);
            margin-bottom: 8px;
        }

        .stat-card .stat-value {
            font-size: 1.35rem;
            font-weight: 800;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: .4px;
        }

        .badge-low {
            background: rgba(0, 230, 118, 0.12);
            color: #00e676;
            border: 1px solid rgba(0, 230, 118, 0.25);
        }

        .badge-medium {
            background: rgba(255, 176, 32, 0.12);
            color: #ffb020;
            border: 1px solid rgba(255, 176, 32, 0.25);
        }

        .badge-high {
            background: rgba(255, 77, 109, 0.12);
            color: #ff4d6d;
            border: 1px solid rgba(255, 77, 109, 0.25);
        }

        .badge-critical {
            background: rgba(255, 0, 85, 0.12);
            color: #ff0055;
            border: 1px solid rgba(255, 0, 85, 0.25);
        }

        /* ==================== CLIMATE ==================== */

        .climate-hero {
            background:
                radial-gradient(circle at top right, rgba(0,210,255,.12), transparent 35%),
                var(--card-bg);
        }

        .weather-stat {
            text-align: center;
            padding: 22px 12px;
        }

        .weather-stat-icon {
            margin-bottom: 8px;
            color: var(--primary-cyan);
        }

        .weather-stat-icon svg {
            width: 28px;
            height: 28px;
        }

        .weather-stat-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--primary-cyan);
        }

        .weather-stat-label {
            font-size: .78rem;
            color: var(--text-sub);
            margin-top: 5px;
        }

        /* ==================== REPORT ==================== */

        .report-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            margin-bottom: 18px;
        }

        .btn-print {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 11px 18px;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            background: var(--card-bg);
            color: var(--text-main);
            font-weight: 700;
            cursor: pointer;
        }

        .official-report {
            background: var(--card-solid);
            border: 1px solid var(--card-border);
        }

        .report-section {
            padding: 22px 0;
            border-bottom: 1px solid var(--card-border);
        }

        .report-section:last-child {
            border-bottom: none;
        }

        .report-section h4 {
            font-size: 1rem;
            margin-bottom: 12px;
            color: var(--primary-cyan);
        }

        .report-section p,
        .report-section li {
            color: var(--text-sub);
            font-size: .9rem;
            line-height: 1.7;
        }

        .report-section ul {
            padding-left: 20px;
        }

        .risk-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            font-size: .82rem;
        }

        .risk-table th,
        .risk-table td {
            padding: 11px;
            border: 1px solid var(--card-border);
            text-align: left;
        }

        .risk-table th {
            color: var(--text-main);
            background: rgba(0,210,255,.06);
        }

        .risk-table td {
            color: var(--text-sub);
        }

        .risk-chart-caption {
            margin-top: 4px;
            font-size: .78rem;
            color: var(--text-sub);
        }

        .report-next-action {
            display: flex;
            justify-content: flex-end;
            margin-top: 24px;
        }

        .report-header-official {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: flex-start;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--primary-cyan);
        }

        .report-title-official {
            font-size: 1.45rem;
            font-weight: 800;
        }

        .report-code {
            color: var(--text-sub);
            font-size: .78rem;
            text-align: right;
        }

        /* ==================== SETTINGS ==================== */

        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 0;
            border-bottom: 1px solid var(--card-border);
        }

        .settings-row:last-child {
            border-bottom: none;
        }

        .settings-title {
            font-weight: 700;
            margin-bottom: 5px;
        }

        .settings-description {
            color: var(--text-sub);
            font-size: .82rem;
        }

        .theme-buttons {
            display: flex;
            gap: 8px;
        }

        .theme-option {
            padding: 9px 15px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            color: var(--text-main);
            cursor: pointer;
            font-weight: 700;
        }

        .theme-option.active {
            border-color: var(--primary-cyan);
            color: var(--primary-cyan);
        }

        .empty-state {
            padding: 30px;
            text-align: center;
            color: var(--text-sub);
        }

        /* ==================== PRINT ==================== */

        @media print {
            body {
                background: white !important;
                color: black !important;
            }

            #welcome-section,
            .sidebar,
            .top-bar,
            #home-view,
            #analyze-view,
            #results-view,
            #climate-view,
            #settings-view,
            .report-toolbar {
                display: none !important;
            }

            .dashboard-container,
            .main-content {
                display: block !important;
                width: 100% !important;
                height: auto !important;
                overflow: visible !important;
                background: white !important;
            }

            #reports-view {
                display: block !important;
            }

            #reports-view .view-wrapper {
                max-width: none;
                padding: 0;
            }

            .official-report,
            .dashboard-card {
                box-shadow: none !important;
                background: white !important;
                color: black !important;
                border: none !important;
            }

            .report-section p,
            .report-section li,
            .report-key,
            .report-val,
            .report-section h4,
            .report-title-official,
            .report-code {
                color: black !important;
            }

            .risk-table th,
            .risk-table td {
                color: black !important;
                border-color: #bbb !important;
            }
        }

        /* ==================== RESPONSIVE ==================== */

        @media (max-width: 1050px) {
            .welcome-cards-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .hero-card {
                flex-direction: column;
                align-items: flex-start;
            }

            .dashboard-grid-2,
            .results-grid-2,
            .climate-grid {
                grid-template-columns: 1fr;
            }

            .stat-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 760px) {
            #welcome-section {
                padding: 28px 22px;
            }

            .welcome-title {
                font-size: 2.3rem;
            }

            .welcome-cards-grid {
                grid-template-columns: 1fr;
            }

            .welcome-footer-ticker {
                flex-direction: column;
                gap: 10px;
            }

            .sidebar {
                width: 220px;
            }

            .view-wrapper {
                padding: 22px 18px;
            }

            .top-bar {
                padding: 18px;
            }

            .status-badge {
                display: none;
            }

            .steps-grid {
                grid-template-columns: 1fr;
            }

            .stat-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

<!-- ==================== WELCOME ==================== -->

<div id="welcome-section">
    <header class="welcome-header">
        <div class="welcome-logo-img"></div>
        <h1 class="welcome-brand-name">PowerPulse <span>AI</span></h1>
    </header>

    <main class="welcome-hero">
        <p class="welcome-subtitle">Smart. Fair. Sustainable.</p>

        <h2 class="welcome-title">
            Climate-Aware Electricity <br>
            <span class="highlight">Anomaly Detection</span>
        </h2>

        <p class="welcome-description">
            PowerPulse AI combines advanced artificial intelligence with environmental
            intelligence to detect abnormal electricity consumption while evaluating
            climate impacts on the grid.
        </p>

        <button class="btn-enter-platform" onclick="navigateToDashboard()">
            Enter PowerPulse Platform
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
                 stroke="currentColor" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
        </button>
    </main>

    <footer>
        <div class="welcome-cards-grid">
            <div class="feature-card">
                <h3 class="feature-card-title">Detect</h3>
                <p class="feature-card-desc">Unusual patterns & anomalies</p>
            </div>

            <div class="feature-card">
                <h3 class="feature-card-title">Understand</h3>
                <p class="feature-card-desc">Climate impact on load</p>
            </div>

            <div class="feature-card">
                <h3 class="feature-card-title">Assess Risk</h3>
                <p class="feature-card-desc">Explainable AI risk scoring</p>
            </div>

            <div class="feature-card">
                <h3 class="feature-card-title">Take Action</h3>
                <p class="feature-card-desc">Prioritize & act securely</p>
            </div>
        </div>

        <div class="welcome-footer-ticker">
            <div class="ticker-item">
                <span class="ticker-dot"></span>
                AI-POWERED SMART DETECTION
            </div>
            <div class="ticker-item">
                <span class="ticker-dot"></span>
                CLIMATE-AWARE INTELLIGENCE
            </div>
            <div class="ticker-item">
                <span class="ticker-dot"></span>
                REAL-TIME GRID MONITORING
            </div>
        </div>
    </footer>
</div>


<!-- ==================== DASHBOARD ==================== -->

<div id="dashboard-section" class="dashboard-container hidden">

    <aside class="sidebar">
        <button class="sidebar-toggle" type="button" onclick="toggleSidebar()" title="Collapse sidebar" aria-label="Collapse sidebar">
            <svg id="sidebar-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
        </button>
        <div class="sidebar-header">
            <div class="logo-icon"></div>
            <span class="logo-text">PowerPulse <span>AI</span></span>
        </div>

        <nav class="sidebar-menu">

            <div class="menu-group">
                <span class="menu-label">GENERAL</span>

                <button class="nav-item active" onclick="switchView('home', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                    <span>Home</span>
                </button>
            </div>

            <div class="menu-group">
                <span class="menu-label">ANALYSIS</span>

                <button class="nav-item" onclick="switchView('analyze', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    <span>Analyze</span>
                </button>

                <button class="nav-item" onclick="switchView('results', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                    </svg>
                    <span>Analysis Result</span>
                </button>
            </div>

            <div class="menu-group">
                <span class="menu-label">INTELLIGENCE</span>

                <button class="nav-item" onclick="switchView('climate', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M18.36 5.64l1.41-1.41"></path>
                    </svg>
                    <span>Climate & Environment</span>
                </button>
            </div>

            <div class="menu-group">
                <span class="menu-label">REPORTING</span>

                <button class="nav-item" onclick="switchView('reports', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                    <span>Reports & Analytics</span>
                </button>
            </div>

            <div class="menu-group">
                <span class="menu-label">SYSTEM</span>

                <button class="nav-item" onclick="switchView('settings', this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                    <span>Settings</span>
                </button>
            </div>

        </nav>
    </aside>


    <!-- Main Content -->
    <main class="main-content">

        <header class="top-bar">
            <h2 class="command-title">PowerPulse Command Center</h2>

            <div class="top-actions">
                <div class="status-badge">
                    <span>Session Analyses: <strong id="session-count">00</strong></span>
                    <span class="status-dot"></span>
                    <span id="system-status">Operational</span>
                </div>

                <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme" aria-label="Toggle theme">
                    <svg id="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3 6.7 6.7 0 0 0 21 12.8Z"></path>
                    </svg>
                </button>
            </div>
        </header>


        <!-- ==================== HOME ==================== -->

        <div id="home-view" class="view-wrapper">

            <section class="dashboard-card hero-card">
                <div class="hero-text-content">
                    <h1 class="hero-headline">
                        Understand Weather. <br>
                        <span class="highlight">Predict Energy.</span>
                    </h1>

                    <p class="card-text">
                        Analyze the relationship between weather conditions and electricity
                        consumption to make smarter energy decisions.
                    </p>
                </div>

                <div class="hero-visual-flow">
                    <div class="flow-node">Weather</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-node active-node">Analysis</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-node">Energy</div>
                </div>
            </section>

            <div class="dashboard-grid-2">
                <section class="dashboard-card">
                    <h3 class="card-title">What Does Our Platform Do?</h3>
                    <p class="card-text">
                        Our platform combines weather and electricity data to identify
                        patterns, analyze their relationship, and provide insights into
                        energy consumption.
                    </p>

                    <ul class="feature-bullets">
                        <li><span class="ticker-dot"></span> Weather Analysis</li>
                        <li><span class="ticker-dot"></span> Electricity Analysis</li>
                        <li><span class="ticker-dot"></span> Energy Prediction</li>
                    </ul>
                </section>

                <section class="dashboard-card">
                    <h3 class="card-title">Why It Matters</h3>
                    <p class="card-text">
                        Changes in temperature and weather conditions can significantly
                        affect electricity demand. Understanding this relationship can
                        help improve energy planning, reduce waste, and support more
                        efficient energy use.
                    </p>
                </section>
            </div>

            <section class="dashboard-card">
                <h3 class="card-title">How It Works</h3>

                <div class="steps-grid">
                    <div class="step-card">
                        <span class="step-num">01</span>
                        <h4 class="step-title">Collect Data</h4>
                        <p class="step-desc">Weather & electricity data</p>
                    </div>

                    <div class="step-card">
                        <span class="step-num">02</span>
                        <h4 class="step-title">Analyze</h4>
                        <p class="step-desc">Find patterns and correlations</p>
                    </div>

                    <div class="step-card">
                        <span class="step-num">03</span>
                        <h4 class="step-title">Predict & Understand</h4>
                        <p class="step-desc">Generate useful energy insights</p>
                    </div>
                </div>
            </section>

            <section class="dashboard-card cta-card">
                <div>
                    <h3 class="card-title">Turn Data Into Energy Insights</h3>
                    <p class="card-text">
                        Explore the connection between weather and electricity consumption.
                    </p>
                </div>

                <button class="btn-primary-action"
                        onclick="goToView('analyze')">
                    Explore Now →
                </button>
            </section>

        </div>


        <!-- ==================== ANALYZE ==================== -->

        <div id="analyze-view" class="view-wrapper hidden">

            <section class="dashboard-card">

                <h3 class="card-title">
                    Electricity & Climate Anomaly Analysis
                </h3>

                <p class="card-text">
                    Select a US state and provide consumption metrics to perform
                    an AI-driven climate-aware risk assessment.
                </p>

                <form onsubmit="runAnalysis(event)" class="form-grid">

                    <div class="form-group">
                        <label class="form-label" for="input-state">
                            Select US State
                        </label>

                        <select id="input-state" class="form-control" required>
                            <option value="" disabled selected>-- Select a State --</option>
                            <option>Alabama</option>
                            <option>Alaska</option>
                            <option>Arizona</option>
                            <option>Arkansas</option>
                            <option>California</option>
                            <option>Colorado</option>
                            <option>Connecticut</option>
                            <option>Delaware</option>
                            <option>Florida</option>
                            <option>Georgia</option>
                            <option>Hawaii</option>
                            <option>Idaho</option>
                            <option>Illinois</option>
                            <option>Indiana</option>
                            <option>Iowa</option>
                            <option>Kansas</option>
                            <option>Kentucky</option>
                            <option>Louisiana</option>
                            <option>Maine</option>
                            <option>Maryland</option>
                            <option>Massachusetts</option>
                            <option>Michigan</option>
                            <option>Minnesota</option>
                            <option>Mississippi</option>
                            <option>Missouri</option>
                            <option>Montana</option>
                            <option>Nebraska</option>
                            <option>Nevada</option>
                            <option>New Hampshire</option>
                            <option>New Jersey</option>
                            <option>New Mexico</option>
                            <option>New York</option>
                            <option>North Carolina</option>
                            <option>North Dakota</option>
                            <option>Ohio</option>
                            <option>Oklahoma</option>
                            <option>Oregon</option>
                            <option>Pennsylvania</option>
                            <option>Rhode Island</option>
                            <option>South Carolina</option>
                            <option>South Dakota</option>
                            <option>Tennessee</option>
                            <option>Texas</option>
                            <option>Utah</option>
                            <option>Vermont</option>
                            <option>Virginia</option>
                            <option>Washington</option>
                            <option>West Virginia</option>
                            <option>Wisconsin</option>
                            <option>Wyoming</option>
                        </select>
                    </div>

                    <div class="dashboard-grid-2">

                        <div class="form-group">
                            <label class="form-label" for="input-historical">
                                Historical Baseline (kWh)
                            </label>

                            <input type="number"
                                   id="input-historical"
                                   class="form-control"
                                   placeholder="e.g. 450.0"
                                   step="any"
                                   min="0.000001"
                                   required>

                            <span class="form-hint">
                                Average consumption benchmark for this period
                            </span>
                        </div>

                        <div class="form-group">
                            <label class="form-label" for="input-current">
                                Current Consumption (kWh)
                            </label>

                            <input type="number"
                                   id="input-current"
                                   class="form-control"
                                   placeholder="e.g. 820.5"
                                   step="any"
                                   min="0"
                                   required>

                            <span class="form-hint">
                                Current measured consumption value
                            </span>
                        </div>

                    </div>

                    <div>
                        <button id="analyze-btn"
                                type="submit"
                                class="btn-primary-action"
                                style="width:100%;">
                            Run Analysis ➔
                        </button>
                    </div>

                    <div id="api-state" class="api-state"></div>

                </form>
            </section>

        </div>


        <!-- ==================== RESULTS ==================== -->

        <div id="results-view" class="view-wrapper hidden">

            <div id="result-summary-text" class="summary-banner">
                No analysis has been completed yet.
            </div>

            <div class="stat-grid">

                <div class="stat-card">
                    <div class="stat-label">Current Consumption</div>
                    <div class="stat-value" id="stat-current">--</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Baseline</div>
                    <div class="stat-value" id="stat-baseline">--</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Consumption Change</div>
                    <div class="stat-value" id="stat-change">--</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Risk Score</div>
                    <div class="stat-value" id="stat-risk">--</div>
                </div>

            </div>

            <div class="results-grid-2">

                <section class="dashboard-card">
                    <h3 class="card-title">Consumption Comparison</h3>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Historical Baseline</span>
                            <span id="val-hist" class="metric-value">0 kWh</span>
                        </div>

                        <div class="progress-bar-bg">
                            <div id="bar-hist" class="progress-bar-fill"></div>
                        </div>
                    </div>

                    <div class="metric-card" style="margin-top:14px;">
                        <div class="metric-header">
                            <span class="metric-label">Current Measured</span>
                            <span id="val-curr" class="metric-value">0 kWh</span>
                        </div>

                        <div class="progress-bar-bg">
                            <div id="bar-curr" class="progress-bar-fill"></div>
                        </div>
                    </div>

                    <div class="chart-container">
                        <canvas id="consumptionChart"></canvas>
                    </div>
                </section>


                <section class="dashboard-card">
                    <h3 class="card-title">AI Model & Decision</h3>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Abnormal Probability</span>
                            <span id="val-prob" class="metric-value">0%</span>
                        </div>

                        <div class="progress-bar-bg">
                            <div id="bar-prob"
                                 class="progress-bar-fill"
                                 style="background:linear-gradient(90deg,#ff9900,#ff0055);">
                            </div>
                        </div>
                    </div>

                    <div class="chart-container small">
                        <canvas id="probabilityChart"></canvas>
                    </div>

                    <div class="report-box" style="margin-top:14px;padding:16px;">

                        <div class="report-row">
                            <span class="report-key">Prediction Status:</span>
                            <span id="res-pred" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Recommended Action:</span>
                            <span id="res-decision"
                                  class="report-val"
                                  style="color:var(--primary-cyan);">
                                --
                            </span>
                        </div>

                    </div>
                </section>

            </div>


            <div class="results-grid-2">

                <section class="dashboard-card">
                    <h3 class="card-title">Weather Conditions</h3>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Temperature:</span>
                            <span id="weather-temp" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Humidity:</span>
                            <span id="weather-humidity" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Wind Speed:</span>
                            <span id="weather-wind" class="report-val">--</span>
                        </div>

                    </div>
                </section>


                <section class="dashboard-card">
                    <h3 class="card-title">FortyGuard Heat Intelligence</h3>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Heat Index:</span>
                            <span id="fg-heat" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Apparent Temp:</span>
                            <span id="fg-apparent" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Wet Bulb Temp:</span>
                            <span id="fg-wetbulb" class="report-val">--</span>
                        </div>

                    </div>
                </section>

            </div>


            <section class="dashboard-card">
                <div class="section-heading-row">
                    <div>
                        <h3 class="card-title" style="margin-bottom:6px;">Risk Assessment Profile</h3>
                        <p class="risk-chart-caption">Likelihood, impact, and model evidence contributing to the calculated case risk.</p>
                    </div>
                    <span id="results-risk-badge" class="badge badge-low">LOW · 0/25</span>
                </div>

                <div class="chart-container">
                    <canvas id="resultsRiskProfileChart"></canvas>
                </div>
            </section>

        </div>


        <!-- ==================== CLIMATE & ENVIRONMENT ==================== -->

        <div id="climate-view" class="view-wrapper hidden">

            <section class="dashboard-card climate-hero">
                <div class="section-heading-row">
                    <div>
                        <h3 class="card-title">Climate & Environment Intelligence</h3>
                        <p class="card-text">
                            Environmental conditions retrieved from the analysis APIs,
                            including real-time weather and FortyGuard thermal indicators.
                        </p>
                    </div>

                    <span id="climate-location-badge" class="badge badge-low">
                        Awaiting analysis
                    </span>
                </div>
            </section>


            <div class="climate-grid">

                <section class="dashboard-card">
                    <h3 class="card-title">Current Weather Profile</h3>

                    <div class="stat-grid">

                        <div class="weather-stat">
                            <div class="weather-stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 14.76V5a2 2 0 0 0-4 0v9.76a4 4 0 1 0 4 0Z"></path><line x1="12" y1="8" x2="12" y2="17"></line></svg></div>
                            <div class="weather-stat-value" id="climate-temp">--</div>
                            <div class="weather-stat-label">Temperature</div>
                        </div>

                        <div class="weather-stat">
                            <div class="weather-stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3s6 6.2 6 11a6 6 0 0 1-12 0c0-4.8 6-11 6-11Z"></path></svg></div>
                            <div class="weather-stat-value" id="climate-humidity">--</div>
                            <div class="weather-stat-label">Humidity</div>
                        </div>

                        <div class="weather-stat">
                            <div class="weather-stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 8h11a3 3 0 1 0-3-3"></path><path d="M3 12h15a3 3 0 1 1-3 3"></path><path d="M3 16h8"></path></svg></div>
                            <div class="weather-stat-value" id="climate-wind">--</div>
                            <div class="weather-stat-label">Wind Speed</div>
                        </div>

                        <div class="weather-stat">
                            <div class="weather-stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"></path><circle cx="12" cy="10" r="2.5"></circle></svg></div>
                            <div class="weather-stat-value" id="climate-state">--</div>
                            <div class="weather-stat-label">State</div>
                        </div>

                    </div>

                    <div class="chart-container">
                        <canvas id="weatherChart"></canvas>
                    </div>
                </section>


                <section class="dashboard-card">
                    <h3 class="card-title">Thermal Intelligence Profile</h3>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Heat Index</span>
                            <span id="climate-heat" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Apparent Temperature</span>
                            <span id="climate-apparent" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Wet Bulb Temperature</span>
                            <span id="climate-wetbulb" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">FortyGuard Status</span>
                            <span id="climate-fg-status" class="report-val">--</span>
                        </div>

                    </div>

                    <div class="chart-container">
                        <canvas id="thermalChart"></canvas>
                    </div>
                </section>

            </div>


            <section class="dashboard-card">
                <h3 class="card-title">Environmental Interpretation</h3>

                <div id="climate-interpretation" class="report-box">
                    <div class="empty-state">
                        Run an analysis to generate climate-aware interpretation.
                    </div>
                </div>
            </section>

        </div>


        <!-- ==================== REPORTS ==================== -->

        <div id="reports-view" class="view-wrapper hidden">

            <div class="report-toolbar">
                <div>
                    <h3 class="card-title" style="margin-bottom:4px;">
                        Reports & Analytics
                    </h3>
                    <p class="card-text">
                        Formal risk assessment and decision-support report.
                    </p>
                </div>

                <button class="btn-print" onclick="printReport()">
                    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M6 9V3h12v6"></path><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><path d="M6 14h12v7H6z"></path>
                    </svg>
                    Print Report
                </button>
            </div>


            <section class="dashboard-card official-report">

                <div class="report-header-official">

                    <div>
                        <div class="report-title-official">
                            PowerPulse AI
                        </div>

                        <div style="color:var(--text-sub);margin-top:5px;">
                            Climate-Aware Electricity Anomaly Assessment
                        </div>
                    </div>

                    <div class="report-code">
                        <div>OFFICIAL ANALYTICAL REPORT</div>
                        <div id="official-report-date">--</div>
                    </div>

                </div>


                <!-- 1 Executive Summary -->
                <div class="report-section">
                    <h4>1. Executive Summary</h4>

                    <p id="report-executive-summary">
                        No completed assessment is currently available.
                    </p>
                </div>


                <!-- 2 Case Profile -->
                <div class="report-section">

                    <h4>2. Case Profile</h4>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Case Location</span>
                            <span id="report-case-location" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Relevant Factors</span>
                            <span id="report-factors" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Observed Indicators</span>
                            <span id="report-indicators" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Available Data</span>
                            <span id="report-available-data" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Missing Data</span>
                            <span id="report-missing-data" class="report-val">--</span>
                        </div>

                    </div>
                </div>


                <!-- 3 Methodology -->
                <div class="report-section">

                    <h4>3. Risk Assessment Methodology</h4>

                    <p>
                        The PowerPulse AI risk assessment combines the model-derived
                        anomaly probability with electricity consumption deviation and
                        climate-related thermal indicators. The resulting assessment
                        is translated into a five-level risk framework.
                    </p>

                    <div class="report-box" style="margin-top:15px;">

                        <div class="report-row">
                            <span class="report-key">Likelihood</span>
                            <span id="report-likelihood" class="report-val">-- / 5</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Impact / Severity</span>
                            <span id="report-impact" class="report-val">-- / 5</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Risk Score</span>
                            <span id="report-risk-score" class="report-val">-- / 25</span>
                        </div>

                    </div>

                    <table class="risk-table">
                        <thead>
                            <tr>
                                <th>Score</th>
                                <th>Risk Level</th>
                                <th>Required Action</th>
                            </tr>
                        </thead>

                        <tbody>
                            <tr>
                                <td>1–4</td>
                                <td>Low</td>
                                <td>Routine monitoring</td>
                            </tr>
                            <tr>
                                <td>5–9</td>
                                <td>Medium</td>
                                <td>Preventive intervention and closer monitoring</td>
                            </tr>
                            <tr>
                                <td>10–15</td>
                                <td>High</td>
                                <td>Urgent intervention and intensive follow-up</td>
                            </tr>
                            <tr>
                                <td>16–25</td>
                                <td>Critical</td>
                                <td>Immediate escalation and specialized intervention</td>
                            </tr>
                        </tbody>
                    </table>

                </div>


                <!-- 4 Risk Findings -->
                <div class="report-section">

                    <h4>4. Risk Findings</h4>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Risk Level</span>
                            <span id="report-risk-level" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Primary Classification Drivers</span>
                            <span id="report-risk-reasons" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Key Risk Factors</span>
                            <span id="report-risk-factors" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Protective Factors</span>
                            <span id="report-protective-factors" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Assessment Confidence</span>
                            <span id="report-confidence" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Potential Consequences Without Intervention</span>
                            <span id="report-consequences" class="report-val">--</span>
                        </div>

                    </div>
                </div>


                <!-- 5 Recommendations -->
                <div class="report-section">

                    <h4>5. Recommendations</h4>

                    <ul id="report-recommendations">
                        <li>Complete an analysis to generate case-specific recommendations.</li>
                    </ul>

                </div>


                <!-- 6 Monitoring -->
                <div class="report-section">

                    <h4>6. Monitoring & Reassessment</h4>

                    <div class="report-box">

                        <div class="report-row">
                            <span class="report-key">Recommended Reassessment</span>
                            <span id="report-reassessment" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Indicators That Increase Risk</span>
                            <span id="report-increase-risk" class="report-val">--</span>
                        </div>

                        <div class="report-row">
                            <span class="report-key">Indicators That Reduce Risk</span>
                            <span id="report-reduce-risk" class="report-val">--</span>
                        </div>

                        <div class="report-row" style="border:none;padding:0;">
                            <span class="report-key">Risk-Level Transition Trigger</span>
                            <span id="report-transition" class="report-val">--</span>
                        </div>

                    </div>

                    <p style="margin-top:15px;">
                        Risk status should be treated as dynamic and reassessed when
                        material changes occur in consumption behavior, environmental
                        conditions, or model evidence.
                    </p>

                    <div class="report-next-action">
                        <button class="btn-primary-action" type="button" onclick="startNewAnalysis()">
                            Analyse New Data
                            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <line x1="5" y1="12" x2="19" y2="12"></line>
                                <polyline points="12 5 19 12 12 19"></polyline>
                            </svg>
                        </button>
                    </div>

                </div>

            </section>

        </div>


        <!-- ==================== SETTINGS ==================== -->

        <div id="settings-view" class="view-wrapper hidden">

            <section class="dashboard-card">

                <h3 class="card-title">System Settings</h3>

                <p class="card-text">
                    Configure the PowerPulse AI interface preferences.
                </p>


                <div class="settings-row">

                    <div>
                        <div class="settings-title">Appearance</div>
                        <div class="settings-description">
                            Select the preferred interface theme.
                        </div>
                    </div>

                    <div class="theme-buttons">

                        <button id="light-theme-btn"
                                class="theme-option"
                                onclick="setTheme('light')">
                            Light
                        </button>

                        <button id="dark-theme-btn"
                                class="theme-option"
                                onclick="setTheme('dark')">
                            Dark
                        </button>

                    </div>

                </div>


                <div class="settings-row">

                    <div>
                        <div class="settings-title">Backend API</div>
                        <div class="settings-description">
                            Active analysis service endpoint.
                        </div>
                    </div>

                    <strong style="font-size:.8rem;color:var(--primary-cyan);">
                        /api/v1/analyze
                    </strong>

                </div>

            </section>

        </div>

    </main>
</div>


<script>
    // ============================================================
    // PowerPulse AI Frontend
    // ============================================================

    const BACKEND_URL = "__BACKEND_URL__";

    let latestAnalysis = null;
    let sessionCount = 0;

    let consumptionChart = null;
    let probabilityChart = null;
    let weatherChart = null;
    let thermalChart = null;
    let resultsRiskProfileChart = null;
    let riskProfileChart = null;


    // ============================================================
    // NAVIGATION
    // ============================================================

    function navigateToDashboard() {
        document.getElementById("welcome-section").classList.add("hidden");
        document.getElementById("dashboard-section").classList.remove("hidden");
    }


    function toggleSidebar() {
        const dashboard = document.getElementById("dashboard-section");
        const collapsed = dashboard.classList.toggle("sidebar-collapsed");
        localStorage.setItem("powerpulse-sidebar-collapsed", collapsed ? "1" : "0");
        const button = document.querySelector(".sidebar-toggle");
        const icon = document.getElementById("sidebar-toggle-icon");
        if (button) {
            button.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
            button.setAttribute("aria-label", button.title);
        }
        if (icon) {
            icon.innerHTML = collapsed
                ? '<polyline points="9 18 15 12 9 6"></polyline>'
                : '<polyline points="15 18 9 12 15 6"></polyline>';
        }
    }


    function loadSidebarState() {
        const collapsed = localStorage.getItem("powerpulse-sidebar-collapsed") === "1";
        const dashboard = document.getElementById("dashboard-section");
        if (!dashboard) return;
        dashboard.classList.toggle("sidebar-collapsed", collapsed);
        const button = document.querySelector(".sidebar-toggle");
        const icon = document.getElementById("sidebar-toggle-icon");
        if (button) {
            button.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
            button.setAttribute("aria-label", button.title);
        }
        if (icon) {
            icon.innerHTML = collapsed
                ? '<polyline points="9 18 15 12 9 6"></polyline>'
                : '<polyline points="15 18 9 12 15 6"></polyline>';
        }
    }


    function switchView(viewName, element) {
        const views = [
            "home",
            "analyze",
            "results",
            "climate",
            "reports",
            "settings"
        ];

        views.forEach(view => {
            const node = document.getElementById(view + "-view");
            if (node) {
                node.classList.add("hidden");
            }
        });

        const target = document.getElementById(viewName + "-view");

        if (target) {
            target.classList.remove("hidden");
        }

        if (element) {
            document.querySelectorAll(".nav-item").forEach(item => {
                item.classList.remove("active");
            });

            element.classList.add("active");
        }
    }


    function goToView(viewName) {
        const mapping = {
            home: 0,
            analyze: 1,
            results: 2,
            climate: 3,
            reports: 4,
            settings: 5
        };

        const items = document.querySelectorAll(".nav-item");
        switchView(viewName, items[mapping[viewName]]);
    }


    // ============================================================
    // THEME
    // ============================================================

    function setTheme(theme) {
        const isLight = theme === "light";

        document.body.classList.toggle("light-mode", isLight);
        localStorage.setItem("powerpulse-theme", theme);

        document.getElementById("light-theme-btn").classList.toggle(
            "active",
            isLight
        );

        document.getElementById("dark-theme-btn").classList.toggle(
            "active",
            !isLight
        );

        const themeIcon = document.getElementById("theme-icon");
        if (themeIcon) {
            themeIcon.innerHTML = isLight
                ? '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M18.36 5.64l1.41-1.41"></path>'
                : '<path d="M21 12.8A8.5 8.5 0 1 1 11.2 3 6.7 6.7 0 0 0 21 12.8Z"></path>';
        }

        refreshCharts();
    }


    function toggleTheme() {
        const isLight = document.body.classList.contains("light-mode");
        setTheme(isLight ? "dark" : "light");
    }


    function loadTheme() {
        const saved = localStorage.getItem("powerpulse-theme") || "dark";
        setTheme(saved);
    }


    // ============================================================
    // API STATE
    // ============================================================

    function showApiState(message, type) {
        const box = document.getElementById("api-state");

        box.className = "api-state " + type;
        box.textContent = message;
    }


    function clearApiState() {
        const box = document.getElementById("api-state");
        box.className = "api-state";
        box.textContent = "";
    }


    // ============================================================
    // ANALYSIS API
    // ============================================================

    async function runAnalysis(event) {
        event.preventDefault();

        const state = document.getElementById("input-state").value;
        const historical = Number(
            document.getElementById("input-historical").value
        );
        const current = Number(
            document.getElementById("input-current").value
        );

        if (!state || !Number.isFinite(historical) || !Number.isFinite(current)) {
            showApiState("Please provide valid analysis inputs.", "error");
            return;
        }

        if (historical <= 0 || current < 0) {
            showApiState(
                "Historical baseline must be greater than zero and current consumption cannot be negative.",
                "error"
            );
            return;
        }

        const button = document.getElementById("analyze-btn");
        const oldText = button.textContent;

        button.disabled = true;
        button.textContent = "Running Analysis...";

        showApiState(
            "Connecting to PowerPulse AI analysis services...",
            "loading"
        );

        try {
            const response = await fetch(
                `${BACKEND_URL}/api/v1/analyze`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        state: state,
                        current_consumption_kwh: current,
                        historical_baseline_kwh: historical
                    })
                }
            );

            let payload;

            try {
                payload = await response.json();
            } catch {
                throw new Error(
                    "The backend returned an invalid JSON response."
                );
            }

            if (!response.ok) {
                const detail =
                    payload.detail ||
                    payload.message ||
                    "Backend analysis failed.";

                throw new Error(detail);
            }

            latestAnalysis = payload;

            sessionCount += 1;
            document.getElementById("session-count").textContent =
                String(sessionCount).padStart(2, "0");

            populateResults(payload);
            populateClimate(payload);
            populateReport(payload);

            clearApiState();

            goToView("results");

        } catch (error) {
            console.error(error);

            showApiState(
                `Analysis failed: ${error.message}`,
                "error"
            );
        } finally {
            button.disabled = false;
            button.textContent = oldText;
        }
    }


    // ============================================================
    // SAFE FORMATTERS
    // ============================================================

    function numberOrNull(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }


    function formatNumber(value, digits = 2) {
        const n = numberOrNull(value);

        if (n === null) {
            return "--";
        }

        return n.toLocaleString(undefined, {
            maximumFractionDigits: digits
        });
    }


    function formatPercent(value) {
        const n = numberOrNull(value);

        if (n === null) {
            return "--";
        }

        return `${(n * 100).toFixed(1)}%`;
    }


    function calculateChange(current, baseline) {
        if (!baseline) {
            return 0;
        }

        return ((current - baseline) / baseline) * 100;
    }


    // ============================================================
    // RISK ENGINE
    // ============================================================

    function calculateRisk(data) {
        const model = data.model || {};
        const fg = data.fortyguard || data.forty_guard || {};
        const prob = numberOrNull(model.abnormal_probability) || 0;
        const heatIndex = numberOrNull(fg.heat_index_c) || 30;

        const likelihood = Math.min(5, Math.max(1, Math.round(prob * 5)));
        const impact = heatIndex > 40 ? 5 : heatIndex > 35 ? 4 : 3;
        const score = likelihood * impact;

        let level = "Low";
        if (score >= 15) level = "Critical";
        else if (score >= 10) level = "High";
        else if (score >= 6) level = "Moderate";

        return {
            score,
            level,
            likelihood,
            impact,
            thermalSignal: heatIndex > 38
        };
    }



    // ============================================================
    // RESULTS
    // ============================================================

    function populateResults(data) {
        const consumption = data.consumption || {};
        const weather = data.weather || {};
        const fortyguard = data.fortyguard || {};
        const model = data.model || {};
        const location = data.location || {};

        const current = numberOrNull(consumption.current_kwh) ?? 0;
        const baseline =
            numberOrNull(consumption.historical_baseline_kwh) ?? 0;

        const probability =
            numberOrNull(model.abnormal_probability) ?? 0;

        const maxConsumption = Math.max(current, baseline, 1);

        document.getElementById("result-summary-text").textContent =
            `Analysis complete for ${location.state || "the selected state"}. ` +
            `The AI model classified the case as ${model.label || "Unknown"}.`;

        document.getElementById("stat-current").textContent =
            `${formatNumber(current)} kWh`;

        document.getElementById("stat-baseline").textContent =
            `${formatNumber(baseline)} kWh`;

        document.getElementById("stat-change").textContent =
            `${calculateChange(current, baseline).toFixed(1)}%`;

        const risk = calculateRisk(data);

        document.getElementById("stat-risk").textContent =
            `${risk.score}/25`;

        document.getElementById("val-hist").textContent =
            `${formatNumber(baseline)} kWh`;

        document.getElementById("val-curr").textContent =
            `${formatNumber(current)} kWh`;

        document.getElementById("bar-hist").style.width =
            `${Math.min(100, (baseline / maxConsumption) * 100)}%`;

        document.getElementById("bar-curr").style.width =
            `${Math.min(100, (current / maxConsumption) * 100)}%`;

        document.getElementById("val-prob").textContent =
            formatPercent(probability);

        document.getElementById("bar-prob").style.width =
            `${Math.min(100, probability * 100)}%`;

        document.getElementById("res-pred").textContent =
            model.label || "--";

        document.getElementById("res-decision").textContent =
            humanAction(data.decision_support?.action_code, risk);

        document.getElementById("weather-temp").textContent =
            `${formatNumber(weather.temperature_c)} °C`;

        document.getElementById("weather-humidity").textContent =
            `${formatNumber(weather.humidity_percent, 1)} %`;

        document.getElementById("weather-wind").textContent =
            `${formatNumber(weather.wind_speed_kmh, 1)} km/h`;

        document.getElementById("fg-heat").textContent =
            valueWithUnit(fortyguard.heat_index_c, "°C");

        document.getElementById("fg-apparent").textContent =
            valueWithUnit(fortyguard.apparent_temperature_c, "°C");

        document.getElementById("fg-wetbulb").textContent =
            valueWithUnit(fortyguard.wet_bulb_temperature_c, "°C");

        buildConsumptionChart(current, baseline);
        buildProbabilityChart(probability, model.label);
        buildResultsRiskProfileChart(data);
    }


    function valueWithUnit(value, unit) {
        const n = numberOrNull(value);

        return n === null ? "--" : `${n.toFixed(1)} ${unit}`;
    }


    function humanAction(actionCode, risk) {
        if (actionCode === "CRITICAL_INSPECT" || risk.score >= 15) {
            return "Immediate field inspection of electrical transformer and substation load required due to extreme thermal stress.";
        } else if (risk.score >= 10) {
            return "Schedule preventive maintenance and monitor real-time temperature telemetry closely.";
        }
        return "Maintain standard operational monitoring schedule.";
    }



    // ============================================================
    // CHART HELPERS
    // ============================================================

    function chartTextColor() {
        return document.body.classList.contains("light-mode")
            ? "#334155"
            : "#cbd5e1";
    }


    function chartGridColor() {
        return document.body.classList.contains("light-mode")
            ? "rgba(15,23,42,.10)"
            : "rgba(255,255,255,.08)";
    }


    function destroyChart(chart) {
        if (chart) {
            chart.destroy();
        }
    }


    function buildConsumptionChart(current, baseline) {
        destroyChart(consumptionChart);

        const ctx = document
            .getElementById("consumptionChart")
            .getContext("2d");

        consumptionChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Historical Baseline", "Current Consumption"],
                datasets: [{
                    label: "Energy Consumption (kWh)",
                    data: [baseline, current],
                    backgroundColor: [
                        "rgba(0,114,255,.55)",
                        "rgba(0,210,255,.65)"
                    ],
                    borderColor: [
                        "#0072ff",
                        "#00d2ff"
                    ],
                    borderWidth: 1,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartTextColor()
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: chartTextColor() },
                        grid: { color: chartGridColor() }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: chartTextColor() },
                        grid: { color: chartGridColor() }
                    }
                }
            }
        });
    }


    function buildProbabilityChart(probability, label) {
        destroyChart(probabilityChart);

        const ctx = document
            .getElementById("probabilityChart")
            .getContext("2d");

        probabilityChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Abnormal Probability", "Remaining Probability"],
                datasets: [{
                    data: [
                        probability * 100,
                        Math.max(0, 100 - probability * 100)
                    ],
                    backgroundColor: [
                        "#ff0055",
                        "rgba(255,255,255,.08)"
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "68%",
                plugins: {
                    legend: {
                        labels: {
                            color: chartTextColor()
                        }
                    },
                    title: {
                        display: true,
                        text: label || "Model Output",
                        color: chartTextColor()
                    }
                }
            }
        });
    }


    function buildResultsRiskProfileChart(data) {
        destroyChart(resultsRiskProfileChart);

        const risk = calculateRisk(data);
        const probability = numberOrNull(data?.model?.abnormal_probability) ?? 0;
        const canvas = document.getElementById("resultsRiskProfileChart");
        if (!canvas) return;

        resultsRiskProfileChart = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: ["Likelihood", "Impact", "AI Evidence"],
                datasets: [{
                    label: "Assessment scale (1–5)",
                    data: [risk.likelihood, risk.impact, Number((probability * 5).toFixed(2))],
                    backgroundColor: ["rgba(0,210,255,.72)", "rgba(255,176,32,.72)", "rgba(255,77,109,.72)"],
                    borderColor: ["#00d2ff", "#ffb020", "#ff4d6d"],
                    borderWidth: 1.5,
                    borderRadius: 7,
                    maxBarThickness: 58
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: chartTextColor() } },
                    tooltip: { callbacks: {
                        label: function(context) {
                            if (context.dataIndex === 2) return `AI Evidence: ${(probability * 100).toFixed(1)}%`;
                            return `${context.label}: ${context.raw}/5`;
                        },
                        afterBody: function() { return `Calculated risk: ${risk.score}/25 (${risk.level})`; }
                    }}
                },
                scales: {
                    x: { ticks: { color: chartTextColor() }, grid: { display: false } },
                    y: { beginAtZero: true, max: 5, ticks: { color: chartTextColor(), stepSize: 1 }, grid: { color: chartGridColor() } }
                }
            }
        });

        const badge = document.getElementById("results-risk-badge");
        if (badge) {
            badge.className = `badge badge-${risk.level.toLowerCase()}`;
            badge.textContent = `${risk.level} · ${risk.score}/25`;
        }
    }


    // ============================================================
    // CLIMATE
    // ============================================================

    function populateClimate(data) {
        const location = data.location || {};
        const weather = data.weather || {};
        const fortyguard = data.fortyguard || {};

        document.getElementById("climate-location-badge").textContent =
            location.state || "Unknown location";

        document.getElementById("climate-state").textContent =
            location.state || "--";

        document.getElementById("climate-temp").textContent =
            valueWithUnit(weather.temperature_c, "°C");

        document.getElementById("climate-humidity").textContent =
            valueWithUnit(weather.humidity_percent, "%");

        document.getElementById("climate-wind").textContent =
            valueWithUnit(weather.wind_speed_kmh, "km/h");

        document.getElementById("climate-heat").textContent =
            valueWithUnit(fortyguard.heat_index_c, "°C");

        document.getElementById("climate-apparent").textContent =
            valueWithUnit(fortyguard.apparent_temperature_c, "°C");

        document.getElementById("climate-wetbulb").textContent =
            valueWithUnit(fortyguard.wet_bulb_temperature_c, "°C");

        document.getElementById("climate-fg-status").textContent =
            fortyguard.status === "available"
                ? "Available"
                : "Fallback used";

        buildWeatherChart(weather);
        buildThermalChart(fortyguard);
        buildClimateInterpretation(data);
    }


    function buildWeatherChart(weather) {
        destroyChart(weatherChart);

        const temp = numberOrNull(weather.temperature_c) ?? 0;
        const humidity = numberOrNull(weather.humidity_percent) ?? 0;
        const wind = numberOrNull(weather.wind_speed_kmh) ?? 0;

        const canvas = document.getElementById("weatherChart");
        if (!canvas || typeof Chart === "undefined") return;

        const ctx = canvas.getContext("2d");

        weatherChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Temperature °C", "Humidity %", "Wind km/h"],
                datasets: [{
                    label: "Current Weather Indicators",
                    data: [temp, humidity, wind],
                    backgroundColor: [
                        "rgba(255,176,32,.60)",
                        "rgba(0,210,255,.60)",
                        "rgba(0,114,255,.60)"
                    ],
                    borderWidth: 0,
                    borderRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: chartTextColor() } }
                },
                scales: {
                    x: { ticks: { color: chartTextColor() }, grid: { color: chartGridColor() } },
                    y: { beginAtZero: true, ticks: { color: chartTextColor() }, grid: { color: chartGridColor() } }
                }
            }
        });
    }



    function buildRiskProfileChart(data) {
        destroyChart(riskProfileChart);

        const risk = calculateRisk(data);
        const probability =
            numberOrNull(data?.model?.abnormal_probability) ?? 0;

        const ctx = document
            .getElementById("riskProfileChart")
            .getContext("2d");

        riskProfileChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: [
                    "Likelihood",
                    "Impact",
                    "AI Probability"
                ],
                datasets: [{
                    label: "Risk Assessment",
                    data: [
                        risk.likelihood,
                        risk.impact,
                        Number((probability * 5).toFixed(2))
                    ],
                    backgroundColor: [
                        "rgba(0,210,255,.72)",
                        "rgba(255,176,32,.72)",
                        "rgba(255,77,109,.72)"
                    ],
                    borderColor: [
                        "#00d2ff",
                        "#ffb020",
                        "#ff4d6d"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartTextColor()
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                if (context.dataIndex === 2) {
                                    return `AI Probability: ${(probability * 100).toFixed(1)}%`;
                                }
                                return `${context.label}: ${value}/5`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: chartTextColor() },
                        grid: { display: false }
                    },
                    y: {
                        beginAtZero: true,
                        max: 5,
                        ticks: {
                            color: chartTextColor(),
                            stepSize: 1
                        },
                        grid: { color: chartGridColor() }
                    }
                }
            }
        });
    }

    function buildThermalChart(fortyguard) {
        destroyChart(thermalChart);
        const canvas = document.getElementById("thermalChart");
        if (!canvas || typeof Chart === "undefined") return;

        const heatIndex = numberOrNull(fortyguard.heat_index_c) ?? 0;
        const apparent = numberOrNull(fortyguard.apparent_temperature_c) ?? 0;
        const wetbulb = numberOrNull(fortyguard.wet_bulb_temperature_c) ?? 0;

        thermalChart = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: ["Heat Index", "Apparent Temp", "Wet Bulb"],
                datasets: [{
                    label: "FortyGuard Thermal Metrics (°C)",
                    data: [heatIndex, apparent, wetbulb],
                    backgroundColor: [
                        "rgba(255,0,85,.65)",
                        "rgba(255,176,32,.65)",
                        "rgba(0,210,255,.65)"
                    ],
                    borderColor: ["#ff0055", "#ffb020", "#00d2ff"],
                    borderWidth: 1,
                    borderRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: chartTextColor() } }
                },
                scales: {
                    x: { ticks: { color: chartTextColor() }, grid: { color: chartGridColor() } },
                    y: { beginAtZero: true, ticks: { color: chartTextColor() }, grid: { color: chartGridColor() } }
                }
            }
        });
    }



    function buildClimateInterpretation(data) {
        const container = document.getElementById("climate-interpretation");
        if (!container) return;

        const fg = data.fortyguard || data.forty_guard || {};
        const weather = data.weather || {};
        const risk = calculateRisk(data);

        const status = fg.status === "available"
            ? "FortyGuard API Active & Integrated"
            : "Using Climate Fallback Mode";

        const thermal = risk.thermalSignal
            ? "Elevated Heat Stress Detected"
            : "Normal Thermal Range";

        container.innerHTML = `
            <div class="report-row">
                <span class="report-key">Thermal Status:</span>
                <span class="report-val" style="color:var(--primary-cyan);">${escapeHtml(status)}</span>
            </div>
            <div class="report-row">
                <span class="report-key">Grid Thermal Stress:</span>
                <span class="report-val">${escapeHtml(thermal)}</span>
            </div>
            <div class="report-row" style="border:none;padding:0;">
                <span class="report-key">Environmental Impact Note:</span>
                <span class="report-val">Ambient temperature at ${escapeHtml(formatNumber(weather.temperature_c))}°C with heat index at ${escapeHtml(formatNumber(fg.heat_index_c))}°C directly influences regional power load deviations.</span>
            </div>
        `;
    }



    // ============================================================
    // REPORT ENGINE
    // ============================================================

    function populateReport(data) {
        const consumption = data.consumption || {};
        const weather = data.weather || {};
        const fg = data.fortyguard || data.forty_guard || {};
        const model = data.model || {};
        const location = data.location || {};
        const risk = calculateRisk(data);

        const dateElem = document.getElementById("official-report-date");
        if (dateElem) dateElem.textContent = new Date().toISOString().slice(0, 10);

        const summaryElem = document.getElementById("report-executive-summary");
        if (summaryElem) {
            summaryElem.textContent =
                `Assessment conducted for ${location.state || 'Selected State'}. The AI model classified consumption as ${model.label || 'Unknown'} ` +
                `with an abnormal probability of ${formatPercent(model.abnormal_probability)}. Risk score is calculated at ${risk.score}/25 (${risk.level}).`;
        }

        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setVal("report-case-location", location.state || "--");
        setVal("report-factors", `Temperature: ${formatNumber(weather.temperature_c)}°C, Heat Index: ${formatNumber(fg.heat_index_c)}°C`);
        setVal("report-indicators", `Current: ${formatNumber(consumption.current_kwh)} kWh vs Baseline: ${formatNumber(consumption.historical_baseline_kwh)} kWh`);
        setVal("report-available-data", "Consumption metrics, Weather API, FortyGuard Thermal Intelligence");
        setVal("report-missing-data", fg.fallback_used ? "FortyGuard thermal result unavailable for this assessment" : "None");

        setVal("report-likelihood", `${risk.likelihood} / 5`);
        setVal("report-impact", `${risk.impact} / 5`);
        setVal("report-risk-score", `${risk.score} / 25`);

        setVal("report-risk-level", `${risk.level} (${risk.score}/25)`);
        setVal("report-risk-reasons", model.label || "--");
        setVal("report-risk-factors", risk.thermalSignal ? "High ambient heat / FortyGuard thermal stress active" : "Standard deviation profile");
        setVal("report-protective-factors", risk.score < 10 ? "Stable grid environment & moderate baseline" : "None identified");
        setVal("report-confidence", formatPercent(model.abnormal_probability));
        setVal("report-consequences", risk.score >= 10 ? "Potential localized grid overload or unmetered anomaly escalation." : "Minor variance within operational tolerance.");

        const recs = document.getElementById("report-recommendations");
        if (recs) {
            recs.innerHTML = `
                <li><strong>Action Required:</strong> ${escapeHtml(humanAction(data.decision_support?.action_code, risk))}</li>
                <li><strong>Thermal Mitigation:</strong> Monitor infrastructure load in response to FortyGuard apparent temperature (${escapeHtml(formatNumber(fg.apparent_temperature_c))}°C).</li>
                <li><strong>Verification:</strong> Re-verify meter calibration if consumption deviation exceeds 30% under normal weather norms.</li>
            `;
        }

        setVal("report-reassessment", "Within 24-48 hours");
        setVal("report-increase-risk", "Sustained high temperatures and escalating consumption spikes");
        setVal("report-reduce-risk", "Cooling weather trends and consumption stabilization toward historical baseline");
        setVal("report-transition", "Triggered if risk score crosses threshold brackets");

        buildWeatherChart(weather);
        buildThermalChart(fg);
        buildClimateInterpretation(data);
    }



    function buildExecutiveSummary(data, risk) {
        const state = data.location?.state || "the assessed location";
        const label = data.model?.label || "Unknown";
        const probability =
            (numberOrNull(data.model?.abnormal_probability) ?? 0) * 100;

        return (
            `The assessment for ${state} resulted in a ${risk.level.toLowerCase()} `
            + `risk classification with a score of ${risk.score}/25. `
            + `The machine-learning model classified the electricity consumption as `
            + `${label.toLowerCase()}, with an abnormal probability of `
            + `${probability.toFixed(1)}%. `
            + `The assessment also considers the observed consumption deviation and `
            + `available climate and thermal indicators.`
        );
    }


    function buildRelevantFactors(data, risk) {
        const factors = [];

        if (Math.abs(risk.changePct) >= 15) {
            factors.push("material consumption deviation");
        }

        if ((data.model?.abnormal_probability ?? 0) >= .4) {
            factors.push("elevated anomaly probability");
        }

        if (risk.thermalSignal) {
            factors.push("thermal stress indicators");
        }

        if (!factors.length) {
            factors.push("current consumption and environmental conditions");
        }

        return factors.join("; ");
    }


    function buildRiskReasons(data, risk) {
        const reasons = [];

        if (Math.abs(risk.changePct) >= 15) {
            reasons.push(
                `consumption deviation of ${risk.changePct.toFixed(1)}%`
            );
        }

        if ((data.model?.abnormal_probability ?? 0) >= .4) {
            reasons.push("elevated AI anomaly probability");
        }

        if (risk.thermalSignal) {
            reasons.push("elevated thermal conditions");
        }

        return reasons.length
            ? reasons.join("; ")
            : "No major adverse signal exceeded the configured assessment thresholds";
    }


    function buildRiskFactors(data, risk) {
        const factors = [];

        if (risk.changePct > 30) {
            factors.push("substantial increase in electricity consumption");
        }

        if (risk.changePct < -30) {
            factors.push("substantial reduction in electricity consumption requiring contextual review");
        }

        if ((data.model?.abnormal_probability ?? 0) >= .6) {
            factors.push("high model-derived abnormal probability");
        }

        if (risk.thermalSignal) {
            factors.push("climate-related thermal stress");
        }

        return factors.length
            ? factors.join("; ")
            : "No high-severity risk factor identified from the available indicators";
    }


    function buildProtectiveFactors(data, risk) {
        const factors = [];

        if ((data.model?.abnormal_probability ?? 0) < .4) {
            factors.push("low-to-moderate model anomaly probability");
        }

        if (Math.abs(risk.changePct) < 15) {
            factors.push("consumption remains close to the historical baseline");
        }

        if (!risk.thermalSignal) {
            factors.push("no major thermal stress signal identified");
        }

        return factors.length
            ? factors.join("; ")
            : "No significant protective factor established by the current data";
    }


    function confidenceLabel(probability, label) {
        const distance = Math.abs(probability - .5);

        if (distance >= .35) {
            return "High";
        }

        if (distance >= .18) {
            return "Moderate";
        }

        return "Limited";
    }


    function buildConsequences(risk) {
        if (risk.level === "LOW") {
            return "Continued routine monitoring is expected to be sufficient unless new adverse evidence emerges.";
        }

        if (risk.level === "MEDIUM") {
            return "Failure to intervene may allow an emerging consumption anomaly or climate-related pressure to persist and develop.";
        }

        if (risk.level === "HIGH") {
            return "Delayed intervention may increase operational pressure, energy waste, and the likelihood of persistent or escalating abnormal consumption.";
        }

        return "Failure to act promptly may permit a severe operational anomaly or climate-related stress condition to escalate without adequate control.";
    }


    function buildRecommendations(risk) {
        if (risk.level === "LOW") {
            return [
                "Maintain routine operational monitoring.",
                "Continue periodic comparison against the historical consumption baseline.",
                "Reassess if material changes occur in consumption or environmental conditions."
            ];
        }

        if (risk.level === "MEDIUM") {
            return [
                "Initiate a targeted preventive review of the consumption deviation.",
                "Increase monitoring frequency for the affected period.",
                "Reassess within a shorter operational interval and investigate persistent deviation."
            ];
        }

        if (risk.level === "HIGH") {
            return [
                "Initiate urgent investigation of the abnormal consumption condition.",
                "Apply intensive follow-up monitoring until the deviation is explained or resolved.",
                "Escalate to the responsible technical or operational authority when required.",
                "Review climate-related demand pressure and available mitigation measures."
            ];
        }

        return [
            "Initiate immediate operational intervention.",
            "Escalate the case to the responsible specialist or operational authority.",
            "Maintain intensive monitoring and document all intervention actions.",
            "Do not rely on routine monitoring alone while the critical condition persists."
        ];
    }


    function reassessmentPeriod(risk) {
        if (risk.level === "LOW") {
            return "Routine periodic reassessment";
        }

        if (risk.level === "MEDIUM") {
            return "Within the next short monitoring cycle";
        }

        if (risk.level === "HIGH") {
            return "As soon as practical following urgent intervention";
        }

        return "Immediate reassessment following intervention and escalation";
    }


    // ============================================================
    // PRINT
    // ============================================================

    function startNewAnalysis() {
        latestAnalysis = null;
        const stateEl = document.getElementById("input-state");
        const historicalEl = document.getElementById("input-historical");
        const currentEl = document.getElementById("input-current");
        if (stateEl) stateEl.value = "";
        if (historicalEl) historicalEl.value = "";
        if (currentEl) currentEl.value = "";
        clearApiState();
        goToView("analyze");
    }



    function printReport() {
        if (!latestAnalysis) {
            alert("Please complete an analysis before printing the report.");
            return;
        }

        goToView("reports");

        setTimeout(() => {
            window.print();
        }, 100);
    }


    // ============================================================
    // CHART THEME REFRESH
    // ============================================================

    function refreshCharts() {
        if (!latestAnalysis) {
            return;
        }

        populateResults(latestAnalysis);
        populateClimate(latestAnalysis);
    }


    // ============================================================
    // SECURITY / TEXT ESCAPING
    // ============================================================

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }


    // ============================================================
    // STARTUP
    // ============================================================

    document.addEventListener("DOMContentLoaded", () => {
        loadTheme();
        loadSidebarState();
    });
</script>

</body>
</html>
"""


# %%
# Inject backend URL into the template.
# The API key is never exposed here.
rendered_template = (
    html_template
    .replace("__BG_BASE64__", bg_base64)
    .replace("__LOGO_BASE64__", logo_base64)
    .replace("__BACKEND_URL__", BACKEND_API_URL)
)


# %%
@app.route("/")
def index():
    return render_template_string(rendered_template)


# %%
if __name__ == "__main__":
    print("=" * 60)
    print("PowerPulse AI UI")
    print("Frontend : http://127.0.0.1:5000")
    print(f"Backend  : {BACKEND_API_URL}")
    print("Make sure FastAPI is running on port 8000.")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
