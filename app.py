import streamlit as st
import pandas as pd
from datetime import datetime, date
from dateutil import parser
from difflib import SequenceMatcher
import re
import io
import requests

st.set_page_config(page_title="Liva Policy Checker", page_icon="✅", layout="centered")

# ---- Pretty cards CSS ----
st.markdown("""
<style>
.result-grid {display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; margin: 6px 0 18px;}
.result-card {border:1px solid #E5E7EB; border-radius:16px; padding:14px 16px; background:#FFFFFF; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
.result-title {font-weight:700; font-size:16px; margin:0 0 6px;}
.result-sub {color:#6B7280; font-size:13px; margin:0 0 10px;}
.badge {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:12px;}
.badge-valid {background:#DCFCE7; color:#166534; border:1px solid #86EFAC;}
.badge-expired {background:#FEE2E2; color:#7F1D1D; border:1px solid #FCA5A5;}
.badge-unknown {background:#FEF3C7; color:#92400E; border:1px solid #FCD34D;}
.small {font-size:12px; color:#6B7280}
</style>
""", unsafe_allow_html=True)

# Exact required columns (including any spaces) per specification
REQUIRED_COLS = [
    "Source.Name","Policy No","Reg: No","Endorsment No","Chassis No","Pol Conc Policy No",
    "Veh Engine No","Veh Regn No Text","Insured Name","Vehicle Name","Veh Date Of Regn",
    "Issue Date","Start  Date","End Date","Policy Type","Alternate Cover Premium ","Cover Details",
    "Cover Duration ","Driver Age","Policy Term","Location Code","Location ","POL PREPARED BY","SCH DESC"
]

# Columns to display in results
DISPLAY_COLS = [
    "Policy No", "Reg: No", "Insured Name", "Vehicle Name", "Issue Date", 
    "Start  Date", "End Date", "Alternate Cover Premium ", "Cover Duration ", "Location "
]

SEARCH_OPTIONS = {
    "Customer Name": "Insured Name",
    "Car Plate Number": "Reg: No",
    "Policy Number": "Policy No",
}


def robust_parse_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if text == "":
        return None
    try:
        return parser.parse(text).date()
    except Exception:
        return None


def compute_status(end_date):
    today = date.today()
    if end_date is None:
        return "Unknown", None
    return ("Valid", (end_date - today).days) if end_date >= today else ("Expired", (today - end_date).days)

def render_status_badge(status, days):
    if status == "Valid":
        suffix = f" • {days} day(s) left" if days is not None else ""
        return f'<span class="badge badge-valid">🟢 Valid{suffix}</span>'
    if status == "Expired":
        suffix = f" • {days} day(s) ago" if days is not None else ""
        return f'<span class="badge badge-expired">🔴 Expired{suffix}</span>'
    return '<span class="badge badge-unknown">⚠️ Unknown</span>'

def render_result_cards(df):
    """Render nice summary cards ABOVE the table using Streamlit components."""
    max_cards = 12
    
    # Create cards using Streamlit columns
    cards_to_show = min(max_cards, len(df))
    cols_per_row = 3
    
    for i in range(0, cards_to_show, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < cards_to_show:
                row = df.iloc[i + j]
                name = str(row.get("Insured Name","") or "—")
                plate = str(row.get("Reg: No","") or "—")
                end_d = robust_parse_date(row.get("End Date"))
                status, days = compute_status(end_d)
                pol = str(row.get("Policy No","") or "—")
                
                with col:
                    with st.container():
                        st.markdown(f"**{name}**")
                        st.markdown(f"Reg No: **{plate}**")
                        
                        # Status badge
                        if status == "Valid":
                            if days is not None:
                                if days > 30:
                                    st.success(f"🟢 Valid ({days} days left)")
                                elif days > 7:
                                    st.warning(f"🟡 Valid ({days} days left)")
                                else:
                                    st.error(f"🔴 Valid (Expires in {days} days!)")
                            else:
                                st.success("🟢 Valid")
                        elif status == "Expired":
                            if days is not None:
                                st.error(f"❌ Expired ({days} days ago)")
                            else:
                                st.error("❌ Expired")
                        else:
                            st.warning("⚠️ Unknown")
                        
                        st.caption(f"Policy #{pol}")
    
    if len(df) > max_cards:
        st.caption(f"Showing first {max_cards} of {len(df)} matches.")


def get_policy_status(end_date):
    """Get policy status based on end date"""
    parsed_date = robust_parse_date(end_date)
    if parsed_date is None:
        return "⚠️ Unknown"
    
    today = date.today()
    if parsed_date >= today:
        days_remaining = (parsed_date - today).days
        if days_remaining > 30:
            return "🟢 Valid (30+ days remaining)"
        elif days_remaining > 7:
            return "🟡 Valid (7+ days remaining)"
        else:
            return "🔴 Valid (Expires soon!)"
    else:
        days_expired = (today - parsed_date).days
        return f"❌ Expired ({days_expired} days ago)"


def status_badge(end_date: date):
    today = date.today()
    if end_date is None:
        return ("Unknown", None, "⚠️")
    if end_date >= today:
        return ("Valid", (end_date - today).days, "🟢")
    return ("Expired", (today - end_date).days, "🔴")


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def normalize_plate(s: str) -> str:
    # Remove all non-alphanumeric and fold case
    return re.sub(r"[^A-Za-z0-9]+", "", (s or "")).casefold()


def to_text_series(series: pd.Series) -> pd.Series:
    # Convert to string, replace NaN-like with empty for display/matching
    s = series.astype(str)
    s = s.replace({"nan": "", "NaN": ""})
    s = s.fillna("")
    return s


def contains_match(series: pd.Series, query: str, normalize_fn=None):
    s = to_text_series(series)
    q = query.strip()
    if normalize_fn:
        s = s.map(normalize_fn)
        q = normalize_fn(q)
        # Already case-folded in normalize_fn
        return s.str.contains(q, case=False, na=False)
    return s.str.contains(q, case=False, na=False)


def fuzzy_match(series: pd.Series, query: str, threshold: int, normalize_fn=None):
    s = to_text_series(series)
    q = query.strip()
    if normalize_fn:
        s = s.map(normalize_fn)
        q = normalize_fn(q)
    else:
        # Case-insensitive by folding both sides
        s = s.str.casefold()
        q = q.casefold()

    def ratio(a: str, b: str) -> int:
        try:
            return int(round(SequenceMatcher(None, a, b).ratio() * 100))
        except Exception:
            return 0

    scores = s.map(lambda x: ratio(x, q))
    mask = scores >= int(threshold)
    return mask, scores


def show_policy_details(row: pd.Series):
    start_d = robust_parse_date(row.get("Start Date"))
    end_d = robust_parse_date(row.get("End Date"))
    status, days, emoji = status_badge(end_d)

    st.subheader("Policy Summary")
    cols = st.columns(3)
    cols[0].metric("Policy No", str(row.get("Policy No", "")))
    cols[1].metric("Insured Name", str(row.get("Insured Name", "")))
    cols[2].metric("Reg: No", str(row.get("Reg: No", "")))

    st.write("")
    badge_text = f"{emoji} **Status:** {status}"
    if days is not None:
        if status == "Valid":
            badge_text += f" — {days} day(s) remaining"
        elif status == "Expired":
            badge_text += f" — expired {days} day(s) ago"
    st.markdown(badge_text)

    st.markdown(
        f"**Start Date:** {start_d if start_d else 'Unknown'} \u00A0\u00A0|\u00A0\u00A0 "
        f"**End Date:** {end_d if end_d else 'Unknown'}"
    )

    st.divider()
    st.subheader("All Details")

    left, right = st.columns(2)
    items = list(row.items())
    half = (len(items) + 1) // 2
    for i, (k, v) in enumerate(items):
        target = left if i < half else right
        display_val = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        target.markdown(f"**{k}**")
        target.write(display_val if str(display_val).strip() != "" else "—")


def pick_engine(file_name: str):
    name = (file_name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        return "openpyxl"
    if name.endswith(".xls"):
        return "xlrd"
    return "openpyxl"


def read_excel_sheet(uploaded_file, sheet_name: str):
    engine = pick_engine(getattr(uploaded_file, "name", "uploaded.xlsx"))
    try:
        # Check if sheet exists
        xl_file = pd.ExcelFile(uploaded_file, engine=engine)
        if sheet_name not in xl_file.sheet_names:
            st.error(f"Sheet '{sheet_name}' not found. Available sheets: {xl_file.sheet_names}")
            st.stop()
            
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine=engine)
    except ImportError:
        st.error("""Missing Excel engine. Install:
```
pip install openpyxl xlrd
```""")
        st.stop()
    except ValueError as e:
        if "Worksheet" in str(e) or "sheet" in str(e).lower():
            st.error(f'Worksheet "{sheet_name}" not found.')
            st.stop()
        st.error(f"Failed to read file: {e}")
        st.stop()
    except Exception as e:
        st.error("""Failed to read file. If this is an Excel engine issue, install:
```
pip install openpyxl xlrd
```
""" + str(e))
        st.stop()


st.title("Liva Policy Checker")

st.info("Data source: OneDrive (auto-fetched on each search)")

# OneDrive fallback URL and helpers
DEFAULT_ONEDRIVE_URL = https://curbplus141-my.sharepoint.com/:x:/g/personal/arhab_alrahbi_curbplus_co/IQAYIhvaSYUeSY6q3dwakG4LARfFIovfejSVgJs56FgZdt0?e=CReRW9&nav=MTVfezU5MjgyQzQwLTFBRUUtNDAxNC04NjZFLUNGRjUyRUIwRDQwRn0&download=1
"

def _force_download_param(url: str) -> str:
    # For SharePoint links, append &download=1 if not present
    if "download=" in url.lower():
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}download=1"

# Helper: read Excel from OneDrive/SharePoint, prefer secret then fallback
def read_excel_from_onedrive(sheet_name: str):
    try:
        secret_url = st.secrets.get("ONEDRIVE_URL", "")  # May raise if secrets.toml is absent
    except Exception:
        secret_url = ""
    url = (secret_url or DEFAULT_ONEDRIVE_URL).strip()
    if not url:
        st.error("No OneDrive URL provided. Set ONEDRIVE_URL in secrets or update DEFAULT_ONEDRIVE_URL.")
        st.stop()

    url = _force_download_param(url)

    try:
        resp = requests.get(url, timeout=60, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Failed to download Excel from OneDrive/SharePoint. Check link/permissions.\n{e}")
        st.stop()

    bio = io.BytesIO(resp.content)

    lname = url.lower()
    engine = "openpyxl"
    if lname.endswith(".xls") and not lname.endswith((".xlsx", ".xlsm")):
        engine = "xlrd"

    try:
        return pd.read_excel(bio, sheet_name=sheet_name, engine=engine)
    except ImportError:
        st.error("Missing Excel engine. Install with: pip install openpyxl xlrd")
        st.stop()
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        st.stop()

st.markdown("## 🔍 **Search Policies**")

col1, col2 = st.columns([1, 1])
with col1:
    by = st.selectbox("**Search by:**", list(SEARCH_OPTIONS.keys()), help="Choose what field to search")
    key_col = SEARCH_OPTIONS[by]
    query = st.text_input("**Enter search value:**", placeholder="Type your search term here...")

with col2:
    match_mode = st.radio("**Match mode:**", ["Contains", "Fuzzy"], horizontal=True, index=0, help="Contains: exact substring match | Fuzzy: similarity match")
    
    norm_plate = False
    if by == "Car Plate Number":
        norm_plate = st.checkbox("🔧 Normalize plate formats", value=True, help="Remove spaces, dashes, and special characters")

    threshold = None
    if match_mode == "Fuzzy":
        threshold = st.slider("🎯 Similarity threshold", min_value=50, max_value=95, value=80, step=1, help="Higher = more strict matching")

st.markdown("---")
go = st.button("🔍 **Search Policies**", type="primary", use_container_width=True)

if go:
    if not query.strip():
        st.warning("⚠️ **Please enter a search value.**")
    else:
        with st.spinner("Loading latest data from OneDrive…"):
            df = read_excel_from_onedrive(sheet_name="Liva Monthly Reports")

            missing = validate_columns(df)
            if missing:
                st.error("The following required columns are missing:\n\n- " + "\n- ".join(missing))
                st.stop()

        with st.spinner("🔍 Searching policies..."):
            normalize_fn = normalize_plate if (by == "Car Plate Number" and norm_plate) else None

            if match_mode == "Contains":
                mask = contains_match(df[key_col], query, normalize_fn=normalize_fn)
                results = df[mask].copy()
            else:
                mask, scores = fuzzy_match(df[key_col], query, threshold=threshold, normalize_fn=normalize_fn)
                results = df[mask].copy()
                if not results.empty:
                    results["Match %"] = scores[mask].astype(int).values

            if results.empty:
                st.info("🔍 **No matching policies found.**")
                st.markdown("Try adjusting your search criteria or using fuzzy matching.")
            else:
                show_cols = [c for c in DISPLAY_COLS if c in results.columns]
                table = results[show_cols].copy()
                
                table["Status"] = table["End Date"].apply(lambda x: get_policy_status(x))
                
                if "Match %" in results.columns:
                    table["Match %"] = results["Match %"].astype(int)

                st.markdown("## 📊 **Search Results**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Found", len(results))
                with col2:
                    valid_count = len(table[table["Status"].str.contains("Valid", na=False)])
                    st.metric("Valid Policies", valid_count)
                with col3:
                    expired_count = len(table[table["Status"].str.contains("Expired", na=False)])
                    st.metric("Expired Policies", expired_count)

                st.markdown("---")
                
                st.write(f"Found {len(results)} matching policies:")
                render_result_cards(results)
                
                st.dataframe(
                    table.reset_index(drop=True), 
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### 💾 **Download Results**")
                csv_bytes = table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    data=csv_bytes,
                    file_name=f"policy_search_results_{date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="secondary"
                )



