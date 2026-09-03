"""用藥提醒、緊急聯絡人與通知（免後台：開啟郵件/簡訊 App；進階可選自動代寄）。"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st

from i18n import get_lang, t

REMOTE_ROOT = Path(__file__).resolve().parent
CARE_SETTINGS_PATH = REMOTE_ROOT / "config" / "care_settings.local.json"


def _default_settings() -> dict[str, Any]:
    return {
        "emergency_contacts": [],
        "medications": [],
        "notify_channels": {"email": True, "sms": True},
        "delivery": {
            "auto_email": False,
            "auto_sms": False,
            "smtp_host": "",
            "smtp_port": "587",
            "smtp_user": "",
            "smtp_password": "",
            "smtp_from": "",
            "twilio_sid": "",
            "twilio_token": "",
            "twilio_from": "",
        },
    }


def load_care_settings() -> dict[str, Any]:
    if "care_settings" in st.session_state:
        data = st.session_state.care_settings
    elif CARE_SETTINGS_PATH.exists():
        try:
            data = json.loads(CARE_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = _default_settings()
    else:
        data = _default_settings()
    # merge defaults for new keys
    base = _default_settings()
    for k, v in base.items():
        if k not in data:
            data[k] = v
    if not isinstance(data.get("delivery"), dict):
        data["delivery"] = dict(base["delivery"])
    else:
        for dk, dv in base["delivery"].items():
            data["delivery"].setdefault(dk, dv)
    st.session_state.care_settings = data
    return data


def save_care_settings(settings: dict[str, Any]) -> None:
    st.session_state.care_settings = settings
    CARE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARE_SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _delivery(settings: dict[str, Any]) -> dict[str, Any]:
    return settings.get("delivery") or {}


def smtp_configured(settings: dict[str, Any] | None = None) -> bool:
    d = _delivery(settings or load_care_settings())
    if d.get("auto_email"):
        return bool(d.get("smtp_host") and d.get("smtp_user") and d.get("smtp_password"))
    return bool(
        os.getenv("SMTP_HOST", "").strip()
        and os.getenv("SMTP_USER", "").strip()
        and os.getenv("SMTP_PASSWORD", "").strip()
    )


def twilio_configured(settings: dict[str, Any] | None = None) -> bool:
    d = _delivery(settings or load_care_settings())
    if d.get("auto_sms"):
        return bool(d.get("twilio_sid") and d.get("twilio_token") and d.get("twilio_from"))
    return bool(
        os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        and os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        and os.getenv("TWILIO_FROM_NUMBER", "").strip()
    )


def _smtp_params(settings: dict[str, Any]) -> tuple[str, int, str, str, str]:
    d = _delivery(settings)
    if d.get("auto_email") and d.get("smtp_host"):
        host = str(d.get("smtp_host", "")).strip()
        port = int(str(d.get("smtp_port") or "587"))
        user = str(d.get("smtp_user", "")).strip()
        password = str(d.get("smtp_password", "")).strip()
        from_addr = str(d.get("smtp_from") or user).strip()
        return host, port, user, password, from_addr
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    from_addr = os.getenv("SMTP_FROM", user).strip()
    return host, port, user, password, from_addr


def _twilio_params(settings: dict[str, Any]) -> tuple[str, str, str]:
    d = _delivery(settings)
    if d.get("auto_sms") and d.get("twilio_sid"):
        return (
            str(d.get("twilio_sid", "")).strip(),
            str(d.get("twilio_token", "")).strip(),
            str(d.get("twilio_from", "")).strip(),
        )
    return (
        os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
        os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
        os.getenv("TWILIO_FROM_NUMBER", "").strip(),
    )


def _normalize_phone(p: str) -> str:
    return "".join(ch for ch in str(p or "") if ch.isdigit() or ch == "+")


def _contact_key(c: dict) -> tuple[str, str]:
    return (str(c.get("email") or "").strip().lower(), _normalize_phone(c.get("phone") or ""))


def mailto_link(email: str, subject: str, body: str) -> str:
    return f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"


def sms_link(phone: str, body: str) -> str:
    p = _normalize_phone(phone)
    return f"sms:{p}?&body={quote(body)}"


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = str(s).strip().split(":")
    h = int(parts[0]) if parts else 0
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


def check_medication_status(meds: list[dict]) -> list[dict[str, Any]]:
    now = datetime.now()
    out: list[dict[str, Any]] = []
    for med in meds:
        name = med.get("name") or ("Medicine" if get_lang() == "en" else "藥品")
        times = med.get("times") or []
        last_taken = med.get("last_taken")
        last_dt = None
        if last_taken:
            try:
                last_dt = datetime.fromisoformat(str(last_taken))
            except Exception:
                last_dt = None

        overdue_slots: list[str] = []
        due_soon: list[str] = []
        for slot in times:
            h, m = _parse_hhmm(slot)
            due = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now > due + timedelta(minutes=15):
                if last_dt is None or last_dt < due:
                    overdue_slots.append(slot)
            elif due - timedelta(minutes=10) <= now <= due + timedelta(minutes=15):
                if last_dt is None or last_dt.date() < now.date() or last_dt < due - timedelta(hours=1):
                    due_soon.append(slot)

        status = "ok"
        if overdue_slots:
            status = "overdue"
        elif due_soon:
            status = "due_soon"

        out.append({
            "name": name,
            "status": status,
            "overdue_slots": overdue_slots,
            "due_soon": due_soon,
            "last_taken": last_taken,
            "raw": med,
        })
    return out


def send_email_auto(settings: dict[str, Any], to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    host, port, user, password, from_addr = _smtp_params(settings)
    if not host or not user or not password or not to_addr:
        return False, t("care_auto_email_skip")

    mime = MIMEText(body, "plain", "utf-8")
    mime["Subject"] = subject
    mime["From"] = from_addr
    mime["To"] = to_addr

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, [to_addr], mime.as_string())
        return True, t("care_email_sent", addr=to_addr)
    except Exception as exc:
        return False, t("care_email_failed", err=str(exc))


def send_sms_auto(settings: dict[str, Any], phone: str, body: str) -> tuple[bool, str]:
    sid, token, from_num = _twilio_params(settings)
    if not (sid and token and from_num):
        return False, t("care_auto_sms_skip")

    try:
        import requests

        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        res = requests.post(
            url,
            auth=(sid, token),
            data={"From": from_num, "To": phone, "Body": body},
            timeout=30,
        )
        if res.ok:
            return True, t("care_sms_sent", phone=phone)
        return False, f"Twilio: {res.text[:120]}"
    except Exception as exc:
        return False, str(exc)


def notify_contacts(settings: dict[str, Any], subject: str, body: str) -> list[dict[str, Any]]:
    """通知聯絡人：優先提供一鍵開啟郵件/簡訊 App；若使用者有開自動代寄則一併嘗試。"""
    rows: list[dict[str, Any]] = []
    channels = settings.get("notify_channels") or {}
    contacts = settings.get("emergency_contacts") or []
    if not contacts:
        rows.append({"level": "warn", "text": t("care_no_contacts")})
        return rows

    rows.append({"level": "info", "text": t("care_notify_native_hint")})

    for c in contacts:
        name = c.get("name") or "Contact"
        email = (c.get("email") or "").strip()
        phone = (c.get("phone") or "").strip()

        if channels.get("email", True) and email:
            if smtp_configured(settings):
                ok, msg = send_email_auto(settings, email, subject, body)
                rows.append({
                    "level": "ok" if ok else "warn",
                    "text": f"{name} — {msg}",
                })
            link = mailto_link(email, subject, body)
            rows.append({
                "level": "action",
                "text": t("care_open_email", name=name),
                "href": link,
            })
        elif channels.get("email", True):
            rows.append({"level": "warn", "text": t("care_need_email", name=name)})

        if channels.get("sms", True) and phone:
            if twilio_configured(settings):
                ok, msg = send_sms_auto(settings, phone, f"{subject}\n{body}")
                rows.append({
                    "level": "ok" if ok else "warn",
                    "text": f"{name} — {msg}",
                })
            rows.append({
                "level": "action",
                "text": t("care_open_sms", name=name),
                "href": sms_link(phone, f"{subject}\n{body}"),
            })
        elif channels.get("sms", True):
            rows.append({"level": "warn", "text": t("care_need_phone", name=name)})

    return rows


def _show_notify_results(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        level = r.get("level", "info")
        text = r.get("text", "")
        href = r.get("href")
        if level == "action" and href:
            st.link_button(text, href, use_container_width=True)
        elif level == "ok":
            st.success(text)
        elif level == "err":
            st.error(text)
        elif level == "warn":
            st.warning(text)
        else:
            st.info(text)


def _render_delivery_settings(settings: dict[str, Any]) -> None:
    """進階：可選自動代寄（填一次即可，不必改 .env）。"""
    d = settings.setdefault("delivery", _default_settings()["delivery"])
    with st.expander(t("care_auto_send_optional"), expanded=False):
        st.caption(t("care_auto_send_hint"))
        d["auto_email"] = st.checkbox(t("care_auto_email_enable"), value=bool(d.get("auto_email")))
        if d.get("auto_email"):
            c1, c2 = st.columns(2)
            d["smtp_host"] = c1.text_input("SMTP Host", value=d.get("smtp_host", "smtp.gmail.com"))
            d["smtp_port"] = c2.text_input("Port", value=str(d.get("smtp_port", "587")))
            d["smtp_user"] = st.text_input("Email / User", value=d.get("smtp_user", ""))
            d["smtp_password"] = st.text_input(
                "Password / App password",
                value=d.get("smtp_password", ""),
                type="password",
            )
            d["smtp_from"] = st.text_input("From address", value=d.get("smtp_from", d.get("smtp_user", "")))

        d["auto_sms"] = st.checkbox(t("care_auto_sms_enable"), value=bool(d.get("auto_sms")))
        if d.get("auto_sms"):
            d["twilio_sid"] = st.text_input("Twilio Account SID", value=d.get("twilio_sid", ""))
            d["twilio_token"] = st.text_input("Twilio Auth Token", value=d.get("twilio_token", ""), type="password")
            d["twilio_from"] = st.text_input("Twilio From Number", value=d.get("twilio_from", ""))

        if st.button(t("care_save_delivery"), use_container_width=True):
            save_care_settings(settings)
            st.success(t("care_added"))


def render_care_panel(profile: dict[str, Any]) -> None:
    if not (profile.get("medication_panel") or profile.get("emergency_panel")):
        return

    settings = load_care_settings()
    st.markdown("---")
    st.subheader(f"💊 {t('care_title')}")

    if profile.get("emergency_panel"):
        st.info(t("care_add_and_notify_hint"))

    if profile.get("medication_panel"):
        st.markdown(f"**{t('care_meds')}**")
        med_status = check_medication_status(settings.get("medications") or [])
        for ms in med_status:
            name = ms["name"]
            if ms["status"] == "overdue":
                st.error(f"⚠️ {name}: {t('care_overdue')} ({', '.join(ms['overdue_slots'])})")
            elif ms["status"] == "due_soon":
                st.warning(f"🔔 {name}: {t('care_due_soon')} ({', '.join(ms['due_soon'])})")
            else:
                taken = ms.get("last_taken") or t("care_not_recorded")
                st.success(f"✅ {name}: {t('care_ok')} ({t('care_last')}: {taken})")

            if st.button(f"✔ {t('care_taken')} {name}", key=f"med_taken_{name}", use_container_width=True):
                for med in settings.get("medications") or []:
                    if med.get("name") == name:
                        med["last_taken"] = datetime.now().isoformat(timespec="minutes")
                save_care_settings(settings)
                st.rerun()

        overdue_any = any(m["status"] == "overdue" for m in med_status)
        if overdue_any and st.button(f"📨 {t('care_notify_missed')}", type="secondary", use_container_width=True):
            subj = "[Care] Missed medication" if get_lang() == "en" else "【照護提醒】尚未按時服藥"
            body = (
                "Medication time passed without confirmation. Please check in."
                if get_lang() == "en"
                else "系統偵測到用藥時間已過但尚未確認服用，請協助關心。"
            )
            _show_notify_results(notify_contacts(settings, subj, body))

    if profile.get("emergency_panel"):
        st.markdown(f"**{t('care_contacts')}**")
        if st.button(f"🆘 {t('care_emergency_btn')}", type="primary", use_container_width=True):
            subj = "[Emergency] Help requested" if get_lang() == "en" else "【緊急】使用者請求協助"
            body = (
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')} — User pressed emergency help. Please contact them."
                if get_lang() == "en"
                else f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} — 使用者按下緊急求助，請盡快聯繫。"
            )
            _show_notify_results(notify_contacts(settings, subj, body))

        with st.expander(t("care_manage"), expanded=len(settings.get("emergency_contacts") or []) == 0):
            st.caption(t("care_save_path"))
            contacts = list(settings.get("emergency_contacts") or [])

            c1, c2, c3 = st.columns(3)
            new_name = c1.text_input(t("care_name"), key="care_contact_name")
            new_phone = c2.text_input(t("care_phone"), key="care_contact_phone")
            new_email = c3.text_input(t("care_email"), key="care_contact_email")

            if st.button(f"➕ {t('care_add_contact')}"):
                name = new_name.strip()
                if not name:
                    st.warning(t("care_name"))
                elif not new_phone.strip() and not new_email.strip():
                    st.warning(t("care_need_phone_or_email"))
                else:
                    entry = {
                        "name": name,
                        "phone": new_phone.strip(),
                        "email": new_email.strip(),
                    }
                    key = _contact_key(entry)
                    if any(_contact_key(x) == key for x in contacts):
                        st.warning(t("care_dup_contact"))
                    else:
                        contacts.append(entry)
                        settings["emergency_contacts"] = contacts
                        save_care_settings(settings)
                        st.success(t("care_added"))
                        st.rerun()

            for i, c in enumerate(contacts):
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.write(f"**{c.get('name')}** — {c.get('phone') or '—'} — {c.get('email') or '—'}")
                with col_del:
                    if st.button(t("care_delete"), key=f"del_contact_{i}"):
                        contacts.pop(i)
                        settings["emergency_contacts"] = contacts
                        save_care_settings(settings)
                        st.success(t("care_deleted"))
                        st.rerun()

            st.markdown(f"**{t('care_meds')}**")
            meds = list(settings.get("medications") or [])
            med_name = st.text_input(t("care_med_name"), key="care_med_name")
            med_times = st.text_input(t("care_med_times"), key="care_med_times", placeholder="08:00, 20:00")

            if st.button(f"➕ {t('care_add_med')}"):
                if med_name.strip():
                    times = [x.strip() for x in med_times.split(",") if x.strip()]
                    meds.append({"name": med_name.strip(), "times": times or ["08:00"], "last_taken": None})
                    settings["medications"] = meds
                    save_care_settings(settings)
                    st.success(t("care_added"))
                    st.rerun()

            for i, m in enumerate(meds):
                col_m, col_d = st.columns([5, 1])
                with col_m:
                    st.write(f"- {m.get('name')} @ {', '.join(m.get('times') or [])}")
                with col_d:
                    if st.button(t("care_delete"), key=f"del_med_{i}"):
                        meds.pop(i)
                        settings["medications"] = meds
                        save_care_settings(settings)
                        st.success(t("care_deleted"))
                        st.rerun()

        _render_delivery_settings(settings)
