import base64
import io
import json
import os
import uuid
import imaplib
import smtplib
import sqlite3
import requests
from email import message_from_bytes
from email.message import EmailMessage
from datetime import datetime
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytz
import streamlit as st

# ---- OpenAI (chat + vision) ----
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Handled later


# ------------- CONFIG ------------- #
st.set_page_config(
    page_title="PowerDash Interview Scheduler",
    layout="wide"
)


# ------------- HELPERS ------------- #
def get_openai_client() -> Optional[Any]:
    """Create an OpenAI client using Streamlit secrets or environment variable."""
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
        return None

    if OpenAI is None:
        st.error("OpenAI Python SDK is not installed. Make sure 'openai' is in requirements.txt.")
        return None

    return OpenAI(api_key=api_key)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to a base64 PNG data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_slots_from_image(image: Image.Image) -> List[Dict[str, str]]:
    """
    Use GPT-4o Mini Vision to parse free/busy calendar images into slots.
    Expected JSON format:
    [
      {
        "date": "2025-12-03",
        "start": "09:00",
        "end": "09:30"
      },
      ...
    ]
    """
    client = get_openai_client()
    if not client:
        return []

    data_url = image_to_base64(image)

    system_prompt = (
        "You are an assistant that extracts interview availability slots from images of calendar free/busy views. "
        "Return ONLY valid JSON, no commentary, formatted as a list of objects with keys: "
        "\"date\" (YYYY-MM-DD), \"start\" (HH:MM in 24-hour format), \"end\" (HH:MM in 24-hour format). "
        "If you cannot find any slots, return an empty list []."
    )

    user_text = (
        "Extract all available interview slots from this image of a calendar free/busy view. "
        "Assume the local timezone is the same across all slots. "
        "Again: respond with ONLY JSON."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        # Ensure we capture just JSON if model adds backticks
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]

        slots = json.loads(content)
        # Basic validation
        valid_slots = []
        for s in slots:
            if all(k in s for k in ("date", "start", "end")):
                valid_slots.append(
                    {
                        "date": str(s["date"]),
                        "start": str(s["start"]),
                        "end": str(s["end"]),
                    }
                )
        return valid_slots
    except Exception as e:
        st.error(f"Error parsing slots with GPT-4o-mini vision: {e}")
        return []


def ensure_session_state():
    if "slots" not in st.session_state:
        st.session_state["slots"] = []
    if "email_log" not in st.session_state:
        st.session_state["email_log"] = []
    if "parsed_replies" not in st.session_state:
        st.session_state["parsed_replies"] = []
    if "selected_slot_index" not in st.session_state:
        st.session_state["selected_slot_index"] = None
    if "email_draft_subject" not in st.session_state:
        st.session_state["email_draft_subject"] = ""
    if "email_draft_body" not in st.session_state:
        st.session_state["email_draft_body"] = ""
    if "graph_event" not in st.session_state:
        # Stores the most recently created interview event metadata:
        # {"id": "...", "join_url": "...", "start_utc": "...", "end_utc": "...", "subject": "..."}
        st.session_state["graph_event"] = None
    if "audit_db_path" not in st.session_state:
        st.session_state["audit_db_path"] = os.environ.get("AUDIT_DB_PATH", "audit_log.db")



# -----------------------------
# Microsoft Graph + Audit Log
# -----------------------------

def _audit_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            action TEXT NOT NULL,
            details_json TEXT
        )"""
    )
    conn.commit()
    return conn


@st.cache_resource
def get_audit_conn(db_path: str) -> sqlite3.Connection:
    return _audit_conn(db_path)


def audit_log(action: str, details: Dict[str, Any]) -> None:
    """Append an audit entry to local SQLite. Safe to call frequently."""
    try:
        ts_utc = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        conn = get_audit_conn(st.session_state.get("audit_db_path", "audit_log.db"))
        conn.execute(
            "INSERT INTO audit_log (ts_utc, action, details_json) VALUES (?, ?, ?)",
            (ts_utc, action, json.dumps(details, ensure_ascii=False)),
        )
        conn.commit()
    except Exception:
        # Don't block UX on audit failures
        pass


def read_audit_log(limit: int = 200) -> List[Dict[str, Any]]:
    try:
        conn = get_audit_conn(st.session_state.get("audit_db_path", "audit_log.db"))
        cur = conn.execute(
            "SELECT ts_utc, action, details_json FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for ts_utc, action, details_json in rows:
            try:
                details = json.loads(details_json) if details_json else {}
            except Exception:
                details = {"raw": details_json}
            out.append({"ts_utc": ts_utc, "action": action, "details": details})
        return out
    except Exception:
        return []


def _graph_secrets() -> Dict[str, str]:
    """Reads Graph credentials from st.secrets (preferred) or env vars."""
    def _get(key: str, env_key: str) -> str:
        return str(st.secrets.get(key, os.environ.get(env_key, ""))).strip()

    return {
        "tenant_id": _get("graph_tenant_id", "GRAPH_TENANT_ID"),
        "client_id": _get("graph_client_id", "GRAPH_CLIENT_ID"),
        "client_secret": _get("graph_client_secret", "GRAPH_CLIENT_SECRET"),
        "scheduler_mailbox": _get("graph_scheduler_mailbox", "GRAPH_SCHEDULER_MAILBOX")
        or _get("smtp_from", "SMTP_FROM")
        or "scheduling@powerdashhr.com",
    }


@st.cache_resource
def _graph_token_cache() -> Dict[str, Any]:
    return {"token": None, "expires_at": 0}


def get_graph_access_token() -> str:
    s = _graph_secrets()
    if not (s["tenant_id"] and s["client_id"] and s["client_secret"]):
        raise RuntimeError(
            "Missing Graph credentials. Set graph_tenant_id, graph_client_id, graph_client_secret in Streamlit secrets."
        )

    cache = _graph_token_cache()
    now = int(datetime.utcnow().timestamp())
    if cache["token"] and cache["expires_at"] - 60 > now:
        return cache["token"]

    token_url = f"https://login.microsoftonline.com/{s['tenant_id']}/oauth2/v2.0/token"
    data = {
        "client_id": s["client_id"],
        "client_secret": s["client_secret"],
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(token_url, data=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed: {resp.status_code} {resp.text}")

    payload = resp.json()
    cache["token"] = payload["access_token"]
    cache["expires_at"] = now + int(payload.get("expires_in", 3599))
    return cache["token"]


def _graph_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {get_graph_access_token()}",
        "Content-Type": "application/json",
    }


def _to_utc_iso(dt_local_naive: datetime, tz_name: str) -> str:
    tz = pytz.timezone(tz_name)
    dt_local = tz.localize(dt_local_naive)
    dt_utc = dt_local.astimezone(pytz.UTC)
    return dt_utc.replace(microsecond=0).isoformat()


def _from_utc_iso(utc_iso: str, tz_name: str) -> str:
    tz = pytz.timezone(tz_name)
    dt_utc = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    dt_local = dt_utc.astimezone(tz)
    return dt_local.strftime("%Y-%m-%d %H:%M")


def create_graph_interview_event(
    subject: str,
    start_utc_iso: str,
    end_utc_iso: str,
    hiring_manager_email: str,
    candidate_email: str,
    recruiter_email: Optional[str],
    body_text: str,
    location: str,
    is_teams: bool,
) -> Dict[str, Any]:
    """Creates a calendar event in the scheduler mailbox and sends invites."""
    s = _graph_secrets()
    mailbox = s["scheduler_mailbox"]

    attendees = [
        {"emailAddress": {"address": candidate_email}, "type": "required"},
        {"emailAddress": {"address": hiring_manager_email}, "type": "required"},
    ]
    if recruiter_email:
        attendees.append({"emailAddress": {"address": recruiter_email}, "type": "optional"})

    payload: Dict[str, Any] = {
        "subject": subject,
        "body": {"contentType": "Text", "content": body_text},
        "start": {"dateTime": start_utc_iso, "timeZone": "UTC"},
        "end": {"dateTime": end_utc_iso, "timeZone": "UTC"},
        "attendees": attendees,
        "location": {"displayName": location},
        "allowNewTimeProposals": True,
    }
    if is_teams:
        payload["isOnlineMeeting"] = True
        payload["onlineMeetingProvider"] = "teamsForBusiness"

    url = f"https://graph.microsoft.com/v1.0/users/{mailbox}/events"
    resp = requests.post(url, headers=_graph_headers(), data=json.dumps(payload), timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Create event failed: {resp.status_code} {resp.text}")
    return resp.json()


def patch_graph_event_time(event_id: str, start_utc_iso: str, end_utc_iso: str) -> Dict[str, Any]:
    s = _graph_secrets()
    mailbox = s["scheduler_mailbox"]
    url = f"https://graph.microsoft.com/v1.0/users/{mailbox}/events/{event_id}?sendUpdates=all"
    payload = {
        "start": {"dateTime": start_utc_iso, "timeZone": "UTC"},
        "end": {"dateTime": end_utc_iso, "timeZone": "UTC"},
    }
    resp = requests.patch(url, headers=_graph_headers(), data=json.dumps(payload), timeout=30)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Patch event failed: {resp.status_code} {resp.text}")
    return resp.json() if resp.text else {"id": event_id}


def delete_graph_event(event_id: str) -> None:
    s = _graph_secrets()
    mailbox = s["scheduler_mailbox"]
    url = f"https://graph.microsoft.com/v1.0/users/{mailbox}/events/{event_id}?sendUpdates=all"
    resp = requests.delete(url, headers=_graph_headers(), timeout=30)
    if resp.status_code not in (202, 204):
        raise RuntimeError(f"Delete event failed: {resp.status_code} {resp.text}")


def validate_graph_setup() -> str:
    """Lightweight validation call; returns scheduler mailbox display name."""
    s = _graph_secrets()
    mailbox = s["scheduler_mailbox"]
    url = f"https://graph.microsoft.com/v1.0/users/{mailbox}?$select=displayName,mail"
    resp = requests.get(url, headers=_graph_headers(), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Graph validation failed: {resp.status_code} {resp.text}")
    j = resp.json()
    return f"{j.get('displayName','')} <{j.get('mail', mailbox)}>"


if "graph_event" not in st.session_state:
    # Stores the most recently created interview event metadata:
    # {"id": "...", "join_url": "...", "start_utc": "...", "end_utc": "...", "subject": "..."}
    st.session_state["graph_event"] = None
if "audit_db_path" not in st.session_state:
    st.session_state["audit_db_path"] = os.environ.get("AUDIT_DB_PATH", "audit_log.db")




def format_slot_label(slot: Dict[str, str], idx: int) -> str:
    return f"{idx + 1} – {slot['date']} {slot['start']}–{slot['end']}"


def build_scheduling_email(
    candidate_name: str,
    role: str,
    hiring_manager_name: str,
    recruiter_name: str,
    slots: List[Dict[str, str]],
) -> str:
    """Builds a warm, professional scheduling email offering numbered slots."""
    if not slots:
        return "No slots available. Please add availability first."

    slot_lines = []
    for i, s in enumerate(slots, start=1):
        slot_lines.append(f"{i}. {s['date']} at {s['start']}–{s['end']}")

    slot_text = "\n".join(slot_lines)

    email_body = f"""Hi {candidate_name},

Thank you again for your interest in the {role} opportunity with us.

We’d love to arrange your interview with {hiring_manager_name}. Please review the available time options below and reply to this email with the **number only** of your preferred option (for example: "2"):

{slot_text}

If none of these options work, simply reply with a note to let us know and we’ll be happy to suggest alternatives.

Best regards,
{recruiter_name}
Talent Acquisition
"""
    return email_body


def send_email_smtp(
    subject: str,
    body: str,
    to_emails: List[str],
    cc_emails: Optional[List[str]] = None,
    attachment: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send an email with optional ICS attachment using SMTP credentials from Streamlit secrets.

    Expected secrets:
    - smtp_server
    - smtp_port (optional, default 587)
    - smtp_username
    - smtp_password
    - smtp_from (optional, defaults to smtp_username)
    """
    required_keys = ["smtp_server", "smtp_username", "smtp_password"]
    for key in required_keys:
        if key not in st.secrets:
            st.error(f"Missing '{key}' in Streamlit secrets.")
            return False

    smtp_server = st.secrets["smtp_server"]
    smtp_port = int(st.secrets["smtp_port"]) if "smtp_port" in st.secrets else 587
    smtp_username = st.secrets["smtp_username"]
    smtp_password = st.secrets["smtp_password"]
    smtp_from = st.secrets.get("smtp_from", smtp_username)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join([e for e in to_emails if e])
    if cc_emails:
        msg["Cc"] = ", ".join([e for e in cc_emails if e])

    msg.set_content(body)

    if attachment:
        filename = attachment.get("filename", "attachment")
        content = attachment.get("content", "")
        maintype = attachment.get("maintype", "text")
        subtype = attachment.get("subtype", "plain")
        params = attachment.get("params", {})

        msg.add_attachment(
            content.encode("utf-8"),
            maintype=maintype,
            subtype=subtype,
            filename=filename,
            params=params,
        )

    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"SMTP send failed: {e}")
        return False


def fetch_unread_emails_imap() -> List[Dict[str, Any]]:
    """
    Fetch unread emails from IMAP inbox using credentials from Streamlit secrets.

    Expected secrets:
    - imap_server
    - imap_port (optional, default 993)
    - imap_username
    - imap_password
    """
    required_keys = ["imap_server", "imap_username", "imap_password"]
    for key in required_keys:
        if key not in st.secrets:
            st.error(f"Missing '{key}' in Streamlit secrets.")
            return []

    imap_server = st.secrets["imap_server"]
    imap_port = int(st.secrets["imap_port"]) if "imap_port" in st.secrets else 993
    imap_username = st.secrets["imap_username"]
    imap_password = st.secrets["imap_password"]

    emails = []
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(imap_username, imap_password)
        mail.select("INBOX")

        typ, data = mail.search(None, "UNSEEN")
        if typ != "OK":
            st.error("Failed to search IMAP mailbox.")
            return []

        for num in data[0].split():
            typ, msg_data = mail.fetch(num, "(RFC822)")
            if typ != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = message_from_bytes(raw_email)

            subject = msg["subject"] or "(no subject)"
            from_ = msg["from"] or "(unknown sender)"

            # Extract simple text body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            body += part.get_payload(decode=True).decode(charset, errors="ignore")
                        except Exception:
                            continue
            else:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                except Exception:
                    body = msg.get_payload()

            emails.append(
                {
                    "from": from_,
                    "subject": subject,
                    "body": body,
                }
            )

        mail.logout()
    except Exception as e:
        st.error(f"IMAP error: {e}")
        return []

    return emails


def detect_slot_choice_from_text(text: str) -> Dict[str, Any]:
    """
    Improved NLP for candidate replies.
    - Removes quoted email content.
    - Uses only the candidate's first response line.
    - Extracts the FIRST valid number only.
    """
    import re

    # 1. Remove quoted email replies ("On Wed", "From:", ">" etc.)
    quoted_patterns = [
        r"On\s.*wrote:",
        r"From:",
        r"Sent:",
        r"Subject:",
        r"^\s*>",  # quoted lines starting with '>'
        r"-----Original Message-----",
    ]

    cleaned = text
    for p in quoted_patterns:
        cleaned = re.split(p, cleaned, flags=re.IGNORECASE | re.MULTILINE)[0]

    # 2. Only use the first meaningful line
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""

    # 3. Find numbers in first line only
    numbers = re.findall(r"\b([1-9][0-9]?)\b", first_line)
    numbers = [int(n) for n in numbers]

    if not numbers:
        return {"status": "unclear", "reason": "no numbers detected", "choice": None}

    # Take only the FIRST number the candidate typed
    choice = numbers[0]
    return {"status": "ok", "reason": "", "choice": choice}


def generate_ics(
    start_dt_local: datetime,
    end_dt_local: datetime,
    timezone_str: str,
    subject: str,
    description: str,
    location: str,
    organizer_email: str,
    required_attendees: List[str],
    optional_attendees: Optional[List[str]] = None,
) -> str:
    """
    Generate an ICS string with METHOD:REQUEST.
    Candidate + Hiring Manager should be in required_attendees.
    Recruiter should be in optional_attendees.
    """
    tz = pytz.timezone(timezone_str)
    start_local = tz.localize(start_dt_local)
    end_local = tz.localize(end_dt_local)

    start_utc = start_local.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)

    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uuid.uuid4()}@powerdashhr.com"

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%SZ")

    attendee_lines = ""
    for a in required_attendees:
        if not a:
            continue
        attendee_lines += f"ATTENDEE;CN={a};ROLE=REQ-PARTICIPANT:MAILTO:{a}\n"

    if optional_attendees:
        for a in optional_attendees:
            if not a:
                continue
            attendee_lines += f"ATTENDEE;CN={a};ROLE=OPT-PARTICIPANT:MAILTO:{a}\n"

    ics = f"""BEGIN:VCALENDAR
PRODID:-//PowerDash HR//Interview Scheduler//EN
VERSION:2.0
CALSCALE:GREGORIAN
METHOD:REQUEST
BEGIN:VEVENT
DTSTAMP:{dtstamp}
DTSTART:{fmt(start_utc)}
DTEND:{fmt(end_utc)}
SUMMARY:{subject}
DESCRIPTION:{description}
UID:{uid}
ORGANIZER;CN=Recruiter:MAILTO:{organizer_email}
LOCATION:{location}
{attendee_lines}STATUS:CONFIRMED
SEQUENCE:0
END:VEVENT
END:VCALENDAR
"""
    return ics.strip()


# ------------- UI ------------- #
ensure_session_state()

st.title("PowerDash Interview Scheduler")

tab1, tab2, tab3 = st.tabs(
    ["New Scheduling Request", "Scheduler Inbox", "Calendar Invites"]
)

# -------- TAB 1: NEW SCHEDULING REQUEST -------- #
with tab1:
    st.subheader("New Scheduling Request")

    col_left, col_center, col_right = st.columns([1.3, 1.0, 1.3])

    # LEFT: Hiring Manager + Recruiter details
    with col_left:
        st.markdown("### Hiring Manager & Recruiter")
        st.text_input("Role Title", key="role_title")
        st.text_input("Hiring Manager Name", key="hm_name")
        st.text_input("Hiring Manager Email", key="hm_email")
        st.text_input("Recruiter Name", key="recruiter_name")
        st.text_input(
            "Recruiter / Scheduling Mailbox Email",
            key="recruiter_email",
            value=st.secrets.get("smtp_from", ""),
        )

    # CENTER: Upload + parse availability
    with col_center:
        st.markdown("### Upload Availability")
        file = st.file_uploader(
            "Free/busy screenshot (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        parse_button = st.button("Parse Availability", type="primary", use_container_width=True)

        st.markdown("#### Extracted Slots")
        if st.session_state["slots"]:
            for idx, slot in enumerate(st.session_state["slots"]):
                st.write(format_slot_label(slot, idx))
        else:
            st.info("No slots extracted yet. Upload a calendar view and click **Parse Availability**.")

    # RIGHT: Candidate details
    with col_right:
        st.markdown("### Candidate")
        st.text_input("Candidate Name", key="candidate_name")
        st.text_input("Candidate Email", key="candidate_email")

    st.markdown("---")

    if parse_button and file is not None:
        try:
            images = []
            if file.type == "application/pdf":
                with fitz.open(stream=file.read(), filetype="pdf") as doc:
                    for page_index in range(len(doc)):
                        page = doc.load_page(page_index)
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
            else:
                image = Image.open(file)
                images.append(image)

            all_slots = []
            with st.spinner("Extracting slots with GPT-4o-mini Vision..."):
                for img in images:
                    slots = parse_slots_from_image(img)
                    all_slots.extend(slots)

            st.session_state["slots"] = all_slots
            if all_slots:
                st.success(f"Extracted {len(all_slots)} slots.")
            else:
                st.warning("No slots extracted. Check the image and try again.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # ----- Generate + edit + send email -----
    st.subheader("2. Generate Candidate Scheduling Email")

    if st.session_state["slots"]:
        offer_indices = st.multiselect(
            "Slots to Offer (by position)",
            options=list(range(len(st.session_state["slots"]))),
            default=list(range(len(st.session_state["slots"]))),
            format_func=lambda i: format_slot_label(st.session_state["slots"][i], i),
        )
    else:
        offer_indices = []
        st.info("Extract availability first to offer slots.")

    # Button to build draft
    generate_draft = st.button("Generate Email Draft", type="secondary")

    if generate_draft:
        candidate_name = st.session_state.get("candidate_name", "").strip()
        candidate_email = st.session_state.get("candidate_email", "").strip()
        role = st.session_state.get("role_title", "").strip()
        hm_name = st.session_state.get("hm_name", "").strip()
        recruiter_name = st.session_state.get("recruiter_name", "").strip()

        if not all([candidate_name, candidate_email, role, hm_name, recruiter_name]):
            st.error("Please complete candidate, role, hiring manager and recruiter details before generating a draft.")
        elif not st.session_state["slots"]:
            st.error("No availability slots have been extracted.")
        elif not offer_indices:
            st.error("Please select at least one slot to offer.")
        else:
            slots_to_offer = [st.session_state["slots"][i] for i in offer_indices]
            email_body = build_scheduling_email(
                candidate_name=candidate_name,
                role=role,
                hiring_manager_name=hm_name,
                recruiter_name=recruiter_name,
                slots=slots_to_offer,
            )
            st.session_state["email_draft_subject"] = f"Interview availability – {role}"
            st.session_state["email_draft_body"] = email_body
            st.success("Draft generated below. You can edit it before sending.")

    st.markdown("### Email Draft (editable)")

    st.text_input("Email Subject", key="email_draft_subject")
    st.text_area("Email Body", key="email_draft_body", height=260)

    send_col1, send_col2 = st.columns([1, 4])
    with send_col1:
        send_email_btn = st.button("Send Email", type="primary")

    if send_email_btn:
        candidate_email = st.session_state.get("candidate_email", "").strip()
        hm_email = st.session_state.get("hm_email", "").strip()
        recruiter_email = st.session_state.get("recruiter_email", "").strip()
        subject = st.session_state.get("email_draft_subject", "").strip()
        body = st.session_state.get("email_draft_body", "").strip()

        if not all([candidate_email, hm_email, recruiter_email, subject, body]):
            st.error("Please ensure candidate, hiring manager, recruiter emails and the draft subject & body are filled in.")
        else:
            success = send_email_smtp(
                subject=subject,
                body=body,
                to_emails=[candidate_email],
                cc_emails=[recruiter_email, hm_email],
            )
            if success:
                st.success("Scheduling email sent successfully.")
                st.session_state["email_log"].append(
                    {
                        "candidate": st.session_state.get("candidate_name", ""),
                        "candidate_email": candidate_email,
                        "role": st.session_state.get("role_title", ""),
                        "subject": subject,
                        "body": body,
                    }
                )
            else:
                st.error("Failed to send scheduling email.")


# -------- TAB 2: SCHEDULER INBOX -------- #
with tab2:
    st.subheader("Monitor Scheduling Mailbox (IMAP)")
    st.write("This reads **unread** messages only and does **not** modify or delete any email.")

    if st.button("Fetch Unread Replies", type="primary"):
        emails = fetch_unread_emails_imap()
        if not emails:
            st.info("No unread emails found or failed to fetch.")
        else:
            parsed_results = []
            for em in emails:
                detection = detect_slot_choice_from_text(em["body"])
                parsed_results.append(
                    {
                        "from": em["from"],
                        "subject": em["subject"],
                        "body_preview": em["body"][:400].replace("\n", " ") + ("..." if len(em["body"]) > 400 else ""),
                        "status": detection["status"],
                        "reason": detection["reason"],
                        "choice": detection["choice"],
                    }
                )

            st.session_state["parsed_replies"] = parsed_results

    if st.session_state["parsed_replies"]:
        st.markdown("### Parsed Replies")
        for i, pr in enumerate(st.session_state["parsed_replies"]):
            with st.expander(f"{i+1}. {pr['from']} – {pr['subject']}"):
                st.write(f"**Status:** {pr['status']}")
                if pr["status"] == "ok":
                    st.write(f"**Detected choice number:** {pr['choice']}")
                else:
                    st.write(f"**Reason:** {pr['reason']}")
                st.write(f"**Body preview:** {pr['body_preview']}")
    else:
        st.info("No parsed replies yet.")


# -------- TAB 3: CALENDAR INVITES -------- #
with tab3:
    st.subheader("Generate & Send Calendar Invites (ICS)")

    if not st.session_state["slots"]:
        st.warning("No slots available. Please parse availability in the **New Scheduling Request** tab first.")
    else:
        st.markdown("### Available Slots")
        slot_index = st.radio(
            "Select slot for invite",
            options=list(range(len(st.session_state["slots"]))),
            format_func=lambda i: format_slot_label(st.session_state["slots"][i], i),
            index=0,
        )
        st.session_state["selected_slot_index"] = slot_index
        selected_slot = st.session_state["slots"][slot_index]

        st.markdown("### Invite Details")

        col_left, col_right = st.columns(2)

        with col_left:
            candidate_email_ci = st.text_input(
                "Candidate Email",
                value=st.session_state.get("candidate_email", ""),
            )
            hiring_manager_email_ci = st.text_input(
                "Hiring Manager Email",
                value=st.session_state.get("hm_email", ""),
            )
            recruiter_email_ci = st.text_input(
                "Recruiter Email (optional CC / optional attendee)",
                value=st.session_state.get("recruiter_email", st.secrets.get("smtp_from", "")),
            )

            interview_type = st.selectbox("Interview Type", ["Teams", "In-Person"])
            timezone_str = st.text_input("Timezone (IANA format)", value="Europe/London")
            use_graph = st.checkbox("Send via Microsoft 365 (Graph) (recommended)", value=True)
            include_recruiter_as_optional = st.checkbox("Include recruiter as optional attendee (if recruiter email provided)", value=True)

        with col_right:
            role_ci = st.text_input(
                "Role Title",
                value=st.session_state.get("role_title", ""),
            )
            organizer_email_ci = st.text_input(
                "Organizer / Scheduling Mailbox",
                value=st.secrets.get("smtp_from", "scheduling@powerdashhr.com"),
            )

            if interview_type == "Teams":
                teams_link = st.text_area("Teams Meeting Link / Joining Instructions")
                location = "Microsoft Teams"
                in_person_notes = ""
            else:
                teams_link = ""
                location = st.text_input("In-Person Location (address)")
                in_person_notes = st.text_area(
                    "In-Person Address & Instructions (will appear on invite)"
                )

        additional_notes = st.text_area("Additional Notes (optional)", "")

        generate_btn = st.button("Generate & Send Invite", type="primary")

        if generate_btn:
            if not all([candidate_email_ci, hiring_manager_email_ci, role_ci, organizer_email_ci]):
                st.error("Please fill in all mandatory invite fields (candidate, hiring manager, role, organizer).")
            else:
                try:
                    date_str = selected_slot["date"]
                    start_str = selected_slot["start"]
                    end_str = selected_slot["end"]

                    start_dt_local = datetime.strptime(date_str + " " + start_str, "%Y-%m-%d %H:%M")
                    end_dt_local = datetime.strptime(date_str + " " + end_str, "%Y-%m-%d %H:%M")


                    # Normalize times: store & send to Graph in UTC; display uses timezone_str
                    start_utc_iso = _to_utc_iso(start_dt_local, timezone_str)
                    end_utc_iso = _to_utc_iso(end_dt_local, timezone_str)

                    subject = f"{role_ci} Interview"
                    description_parts = [f"Interview for: {role_ci}"]

                    if interview_type == "Teams" and teams_link:
                        description_parts.append(f"Teams link: {teams_link}")
                    if interview_type == "In-Person":
                        if location:
                            description_parts.append(f"Location: {location}")
                        if in_person_notes:
                            description_parts.append(f"Instructions: {in_person_notes}")
                    if additional_notes:
                        description_parts.append(f"Notes: {additional_notes}")

                    description = "\n".join(description_parts)

                    

                    required_attendees = [candidate_email_ci, hiring_manager_email_ci]
                    optional_attendees = [recruiter_email_ci] if (recruiter_email_ci and include_recruiter_as_optional) else []

                    location_display = location if interview_type == "In-Person" else "Microsoft Teams"
                    body_text = (
                        "Interview invitation\n\n"
                        + description
                        + "\n\n"
                        + "If you need to propose a new time, please reply to the invite."
                    )

                    graph_sent = False
                    graph_error = None

                    if use_graph:
                        try:
                            graph_event = create_graph_interview_event(
                                subject=subject,
                                start_utc_iso=start_utc_iso,
                                end_utc_iso=end_utc_iso,
                                hiring_manager_email=hiring_manager_email_ci,
                                candidate_email=candidate_email_ci,
                                recruiter_email=recruiter_email_ci if include_recruiter_as_optional else None,
                                body_text=body_text,
                                location=location_display,
                                is_teams=(interview_type == "Teams"),
                            )
                            join_url = (graph_event.get("onlineMeeting", {}) or {}).get("joinUrl", "") or ""

                            st.session_state["graph_event"] = {
                                "id": graph_event.get("id"),
                                "join_url": join_url,
                                "start_utc": start_utc_iso,
                                "end_utc": end_utc_iso,
                                "subject": subject,
                                "candidate_email": candidate_email_ci,
                                "hiring_manager_email": hiring_manager_email_ci,
                                "recruiter_email": recruiter_email_ci,
                                "timezone": timezone_str,
                            }

                            audit_log(
                                "graph_create_event",
                                {
                                    "event_id": graph_event.get("id"),
                                    "candidate": candidate_email_ci,
                                    "hiring_manager": hiring_manager_email_ci,
                                    "recruiter": recruiter_email_ci,
                                    "start_utc": start_utc_iso,
                                    "end_utc": end_utc_iso,
                                    "type": interview_type,
                                },
                            )

                            st.success("Interview invite created and sent via Microsoft 365.")
                            if join_url:
                                st.markdown(f"**Teams join link:** {join_url}")
                            graph_sent = True
                        except Exception as ge:
                            graph_error = str(ge)
                            audit_log(
                                "graph_create_failed",
                                {
                                    "candidate": candidate_email_ci,
                                    "hiring_manager": hiring_manager_email_ci,
                                    "error": graph_error,
                                },
                            )
                            st.warning("Microsoft 365 invite failed — using calendar (.ics) fallback.")

                    # Candidate “Add to calendar” fallback
                    # - If Graph fails we always offer an .ics download and (optionally) email it.
                    # - For non-Teams interviews we also offer .ics even if Graph succeeded (useful for external calendars).
                    need_ics = (not graph_sent) or (interview_type != "Teams")
                    if need_ics:
                        ics_content = generate_ics(
                            start_dt_local=start_dt_local,
                            end_dt_local=end_dt_local,
                            timezone_str=timezone_str,
                            subject=subject,
                            description=description,
                            location=location_display,
                            organizer_email=organizer_email_ci,
                            required_attendees=required_attendees,
                            optional_attendees=optional_attendees,
                        )

                        st.download_button(
                            label="Download calendar invite (.ics)",
                            data=ics_content,
                            file_name="interview_invite.ics",
                            mime="text/calendar",
                        )

                    # Only send SMTP email when Graph did not send the meeting request.
                    if not graph_sent:
                        email_body = (
                            f"Please find attached the calendar invite for your interview.\n\n"
                            f"{description}\n\n"
                            f"Best regards,\nTalent Acquisition"
                        )

                        attachment = {
                            "filename": "interview_invite.ics",
                            "content": ics_content,
                            "maintype": "text",
                            "subtype": "calendar",
                            "params": {"method": "REQUEST", "name": "interview_invite.ics"},
                        }

                        success = send_email_smtp(
                            subject=subject,
                            body=email_body,
                            to_emails=[candidate_email_ci, hiring_manager_email_ci],
                            cc_emails=optional_attendees if optional_attendees else None,
                            attachment=attachment,
                        )

                        if success:
                            st.success("ICS invite emailed successfully.")
                            audit_log("smtp_sent_ics", {"candidate": candidate_email_ci, "hiring_manager": hiring_manager_email_ci})
                        else:
                            st.error("Failed to send ICS invite via email.")
                            audit_log("smtp_send_failed", {"candidate": candidate_email_ci, "hiring_manager": hiring_manager_email_ci})


                except Exception as e:
                    st.error(f"Error generating or sending ICS invite: {e}")


        # -----------------------------
        # Manage (Reschedule / Cancel)
        # -----------------------------
        if st.session_state.get("graph_event") and st.session_state["graph_event"].get("id"):
            st.markdown("### Manage Interview (Microsoft 365)")
            ev = st.session_state["graph_event"]
            ev_id = ev["id"]

            st.info(
                "Event ID stored for reschedule/cancel. Times are stored internally as UTC and shown below in your selected timezone."
            )

            current_start_local = _from_utc_iso(ev["start_utc"], timezone_str)
            current_end_local = _from_utc_iso(ev["end_utc"], timezone_str)
            st.write(f"**Current time:** {current_start_local} → {current_end_local} ({timezone_str})")

            # Reschedule
            st.markdown("#### Reschedule")
            new_slot_idx = st.selectbox(
                "Choose a new slot",
                options=list(range(len(st.session_state["slots"]))),
                format_func=lambda i: f"{st.session_state['slots'][i]['date']} {st.session_state['slots'][i]['start']}–{st.session_state['slots'][i]['end']}",
                key="resched_slot_idx",
            )

            if st.button("Reschedule (preserves meeting link)", type="secondary"):
                try:
                    new_slot = st.session_state["slots"][new_slot_idx]
                    ns = datetime.strptime(new_slot["date"] + " " + new_slot["start"], "%Y-%m-%d %H:%M")
                    ne = datetime.strptime(new_slot["date"] + " " + new_slot["end"], "%Y-%m-%d %H:%M")
                    new_start_utc = _to_utc_iso(ns, timezone_str)
                    new_end_utc = _to_utc_iso(ne, timezone_str)

                    patch_graph_event_time(ev_id, new_start_utc, new_end_utc)

                    st.session_state["graph_event"]["start_utc"] = new_start_utc
                    st.session_state["graph_event"]["end_utc"] = new_end_utc

                    audit_log(
                        "graph_reschedule_event",
                        {"event_id": ev_id, "old_start_utc": ev["start_utc"], "old_end_utc": ev["end_utc"], "new_start_utc": new_start_utc, "new_end_utc": new_end_utc},
                    )

                    st.success("Interview rescheduled successfully (updates sent).")
                except Exception as e:
                    audit_log("graph_reschedule_failed", {"event_id": ev_id, "error": str(e)})
                    st.error(f"Failed to reschedule: {e}")

            # Cancel
            st.markdown("#### Cancel interview")
            confirm_cancel = st.checkbox("I understand this will cancel the interview for all attendees.", value=False)
            if st.button("Cancel interview", type="primary", disabled=not confirm_cancel):
                try:
                    delete_graph_event(ev_id)
                    audit_log("graph_cancel_event", {"event_id": ev_id})
                    st.session_state["graph_event"] = None
                    st.success("Interview cancelled (cancellation sent automatically).")
                except Exception as e:
                    audit_log("graph_cancel_failed", {"event_id": ev_id, "error": str(e)})
                    st.error(f"Failed to cancel: {e}")

        # -----------------------------
        # Audit log viewer
        # -----------------------------
        with st.expander("Audit log"):
            rows = read_audit_log(limit=200)
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.write("No audit entries yet.")
