from flask import Flask, render_template, request, url_for, redirect, session, flash
import os
import re
import json
import ast
import datetime as dt
from math import ceil
from urllib.parse import urlencode
from functools import wraps

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# --------------------------------------------------------------------------------------
# App & Config
# --------------------------------------------------------------------------------------
app = Flask(__name__)

# Prefer environment variable for secret key in production
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "lars-data-secret-check")

# CSVs used for admin auth & question/answer catalogs
# ADMIN_CSV = r"./data/admin.csv"
ADMIN_ENCODING = "UTF-8-SIG"
QUESTIONS_CSV = r"./data/questions.csv"   # columns: id, survey_id, question_text, question_order
ANSWERS_CSV   = r"./data/answers.csv"     # columns: id, survey_id, question_id, answer_text

# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------
# Prefer env vars; fall back to sensible defaults
DB_HOST    = os.getenv("DB_HOST", "localhost")           # set to RDS endpoint in AWS
DB_USER    = os.getenv("DB_USER", "postgres")
DB_PWD     = os.getenv("DB_PWD",  "larsapienv!a")
DB_NAME    = os.getenv("DB_NAME", "postgres")
DB_PORT    = int(os.getenv("DB_PORT", "5432"))
DB_SSLMODE = os.getenv("DB_SSLMODE", "")                 # e.g., "require" if needed

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
connect_args = {}
if DB_SSLMODE:
    connect_args["sslmode"] = DB_SSLMODE

try:
    db_engine = create_engine(DB_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=5,
)
except Exception as e:
    print(f"Database connection error: {e}")
    db_engine = None

# Which table holds admins?
ADMIN_TABLE = os.getenv("ADMIN_TABLE", "admin")  # default: public.admin

def qident(name: str) -> str:
    """
    Quote a SQL identifier (handles schema.table and embedded double-quotes).
    """
    name = str(name)
    def quote_one(part: str) -> str:
        return '"' + part.replace('"', '""') + '"'
    parts = name.split('.')
    return '.'.join(quote_one(p) for p in parts)


def get_data_df(table_name: str) -> pd.DataFrame:
    """Reads the full table into a pandas DataFrame using the global DB engine."""
    if db_engine is None:
        print("Error: Database engine not initialized.")
        return pd.DataFrame()
    try:
        sql = text(f"SELECT * FROM {qident(table_name)}")
        return pd.read_sql(sql, db_engine)
    except Exception as e:
        print(f"Error querying table {table_name}: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
BELONG_MAP = {1: '서울대학교병원', 2: '분당서울대학교병원', 3: '국립암센터', 6: '헤링스병원'}

def _digits_only(s):
    return "".join(ch for ch in str(s or "") if ch.isdigit())

def _fmt_phone(x):
    """숫자만 남기고 한국 전화번호 형태로 하이픈 포맷. 없으면 '-'"""
    if x is None:
        return "-"
    s = re.sub(r"\D+", "", str(x))
    if not s:
        return "-"
    if s.startswith("02") and len(s) in (9, 10):  # 서울 유선
        return f"{s[:2]}-{s[2:-4]}-{s[-4:]}"
    if len(s) == 10:
        return f"{s[:3]}-{s[3:6]}-{s[6:]}"
    if len(s) == 11:
        return f"{s[:3]}-{s[3:7]}-{s[7:]}"
    return s

def _parse_any_datetime(s):
    """Robust datetime parser. Returns pandas.Timestamp or NaT."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    dtv = pd.to_datetime(s, errors="coerce")
    if pd.notna(dtv):
        return dtv
    m = re.match(r'^\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?', s)
    if not m:
        return pd.NaT
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4) or 0); mm = int(m.group(5) or 0); ss = int(m.group(6) or 0)
    return pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)

def _parse_date(s):
    """Accept datetime-like or string and return a date object (or None)."""
    ts = _parse_any_datetime(s)
    return ts.date() if pd.notna(ts) else None

def _fmt_date(d):
    return d.strftime("%Y-%m-%d") if d else "-"

def date_or_dash(value):
    """Return 'YYYY-MM-DD' for datetime-like values, or '-' if missing."""
    try:
        if value is None or pd.isna(value):
            return "-"
    except Exception:
        pass
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        s = (str(value) or "").strip()
        return s.split(" ")[0] if s else "-"
    return ts.date().isoformat()

def _fmt_score(x):
    """for table display: NaN → '-', integers without .0, keep one decimal if needed"""
    if pd.isna(x):
        return "-"
    try:
        f = float(x)
        return str(int(f)) if f.is_integer() else f"{f:.1f}"
    except Exception:
        return "-"

def _fmt_num(x, decimals=1):
    try:
        f = float(x)
        if decimals == 0 or f.is_integer():
            return str(int(round(f)))
        return f"{f:.{decimals}f}"
    except Exception:
        return "-"

def parse_birth_age(birth_str):
    try:
        parts = [int(p) for p in re.split(r'[./-]', str(birth_str)) if p]
        year = parts[0]
        m = parts[1] if len(parts) > 1 else 1
        d = parts[2] if len(parts) > 2 else 1
        today = dt.date.today()
        age = today.year - year - ((today.month, today.day) < (m, d))
        return age
    except Exception:
        return ""

def _pick(sr, *names):
    """pandas Series sr에서 이름 후보들 중 최초로 값이 있는 컬럼을 반환"""
    for n in names:
        if n in sr and pd.notna(sr[n]) and str(sr[n]).strip() != "":
            return sr[n]
    return None

def _pick_loose(sr, names):
    """
    pandas Series sr에서 후보 열 이름을 폭넓게 대응:
    - 대소문자, 공백, 특수문자 무시
    - 정확 일치 먼저, 없으면 정규화(normalized) 일치 시도
    """
    if sr is None:
        return None

    # 1) 정확 일치 우선
    for n in names:
        if n in sr and pd.notna(sr[n]) and str(sr[n]).strip() != "":
            return sr[n]

    # 2) 정규화 일치 (공백/기호 제거 + 소문자)
    def norm(s):
        return re.sub(r"[\s_\-:/\.]", "", str(s).lower())

    norm_index = {norm(col): col for col in sr.index}
    for n in names:
        key = norm(n)
        if key in norm_index:
            col = norm_index[key]
            v = sr[col]
            if pd.notna(v) and str(v).strip() != "":
                return v
    return None

# ------------------------------ Workout helpers ------------------------------
TIER_NAME = {
    0: "워밍업/쿨다운",
    1: "하지 근력",
    2: "근력",
    3: "기능적 운동",
    4: "지구력",
    5: "제자리 걷기",
    6: "쿨다운"  # if you use 0 and/or 6 for warmup/cooldown
}

def _exercise_headers_for_user(user_id: int) -> pd.DataFrame:
    """
    Return workout_header rows for a user with id, date, recommendWorkoutId
    """
    try:
        sql = text('SELECT id, "date", "recommendWorkoutId", "userId" '
                   'FROM workout_header WHERE "userId" = :uid ORDER BY "date" ASC')
        return pd.read_sql(sql, db_engine, params={"uid": int(user_id)})
    except Exception as e:
        print(f"workout_header query error: {e}")
        return pd.DataFrame()

def _exercise_ratio_for_header(header_id: int) -> tuple[float | None, int, int]:
    """
    Compute completion ratio = sum(setNumber)/sum(setCount) * 100
    Returns (ratio, done_sets, total_sets)
    """
    try:
        # ratio
        sql = text("""
            SELECT
            COALESCE(SUM(w."setNumber"), 0) AS done_sets,
            COALESCE(SUM(rwd."setCount"), 0) AS total_sets
            FROM workout w
            JOIN recommend_workout_detail rwd
            ON w."recommendWorkoutDetailId" = rwd.id
            WHERE w."workoutHeaderId" = :hid
        """)

        df = pd.read_sql(sql, db_engine, params={"hid": int(header_id)})
        if df.empty:
            return None, 0, 0
        done = int(df.iloc[0]["done_sets"])
        total = int(df.iloc[0]["total_sets"])
        
        if total <= 0:
            return None, 0, 0
        ratio = round((done / total) * 100, 1)
        return ratio, done, total
    except Exception as e:
        print(f"ratio calc error: {e}")
        return None, 0, 0


def _exercise_series_for_user(user_id: int):
    """
    Return (rows_for_table, chart_for_trend) for Exercise.
      rows_for_table: list[dict] sorted by date DESC
        - {id, date, ratio_display, status}
      chart_for_trend: {"labels": [...], "values": [...]} sorted by date ASC
        - ratio values are percentages (0-100)
    """
    # 1) Load headers for this user (expects DataFrame with at least: id, date)
    hdr = _exercise_headers_for_user(user_id)
    if hdr is None or getattr(hdr, "empty", True):
        return [], {"labels": [], "values": []}

    # 2) Normalize date & keep only up to today (no future)
    try:
        hdr["_date_only"] = pd.to_datetime(hdr["date"], errors="coerce").dt.date
    except Exception:
        hdr["_date_only"] = pd.NaT

    today = dt.date.today()
    hdr = hdr[hdr["_date_only"].notna() & (hdr["_date_only"] <= today)]
    if hdr.empty:
        return [], {"labels": [], "values": []}

    # 3) Build chart (ASC by date) + collect per-day stats
    hdr_asc = hdr.sort_values("_date_only", ascending=True).copy()

    chart_labels: list[str] = []
    chart_values: list[float] = []
    table_rows_tmp: list[dict] = []

    for _, r in hdr_asc.iterrows():
        header_id = int(r["id"])
        row_date: dt.date = r["_date_only"]

        # You likely already have this helper; it should return done & total sets for the header/day.
        # If it also returns a ratio, we ignore it and recompute to be safe.
        _ratio_maybe, done, total = _exercise_ratio_for_header(header_id)

        # Guard against divide-by-zero
        if not total or total <= 0:
            percent = 0.0
        else:
            percent = (done / total) * 100.0

        # Status logic:
        # - 완료: all sets done (any date)
        # - 진행중: today AND not all done
        # - 미완료: past date AND not all done
        if total > 0 and done >= total:
            status = "완료"
        elif row_date == today:
            status = "진행중"
        else:
            status = "미완료"

        # Append for chart (ASC)
        chart_labels.append(row_date.isoformat())
        chart_values.append(round(percent, 1))

        # Collect for table (we'll sort DESC right after)
        table_rows_tmp.append({
            "id": header_id,
            "date": row_date.isoformat(),
            "ratio_display": f"{percent:.1f}%",
            "status": status,
        })

    # 4) Table rows should be DESC by date (newest first)
    rows_desc = sorted(table_rows_tmp, key=lambda d: d["date"], reverse=True)

    # 5) Return
    chart = {"labels": chart_labels, "values": chart_values}
    return rows_desc, chart

def _exercise_details_for_header(header_id: int) -> list[dict]:
    """
    Return per-exercise detail for a given workout_header:
      [{ id, typeName, workoutName, setNumber, setCount, countPerSet, restTime, videoLink }]
    """
    try:
        # details
        sql = text("""
            SELECT
            w.id,
            w."setNumber",
            rwd."setCount",
            rwd."countPerSet",
            wb."type",
            wb."workoutName",
            wb."restTime",
            wb."videoLink"
            FROM workout w
            JOIN recommend_workout_detail rwd
            ON w."recommendWorkoutDetailId" = rwd.id
            JOIN workout_base wb
            ON rwd."workoutBaseId" = wb.id
            WHERE w."workoutHeaderId" = :hid
            ORDER BY w.id ASC
        """)

        df = pd.read_sql(sql, db_engine, params={"hid": int(header_id)})
    except Exception as e:
        print(f"exercise detail query error: {e}")
        df = pd.DataFrame()

    details = []
    for _, r in df.iterrows():
        t = r.get("type")
        # map 'type' to friendly name; fallback to workoutName if unknown
        type_name = TIER_NAME.get(int(t), str(r.get("workoutName") or "운동"))
        details.append({
            "id": int(r["id"]),
            "typeName": type_name,
            "workoutName": r.get("workoutName"),
            "setNumber": int(r.get("setNumber") or 0),
            "setCount": int(r.get("setCount") or 0),
            "countPerSet": int(r.get("countPerSet") or 0),
            "restTime": r.get("restTime"),
            "videoLink": r.get("videoLink"),
        })
    ORDER_PRIORITY = {
        "워밍업/쿨다운": 0,
        "하지 근력": 1,
        "근력": 2,
        "기능적 운동": 3,
        "지구력": 4,
        "제자리 걷기": 5,
        "쿨다운": 6,
    }
    details.sort(key=lambda d: ORDER_PRIORITY.get(d["typeName"], 99))
    return details


# --------------------------------------------------------------------------------------
# Admin helpers
# --------------------------------------------------------------------------------------

def _admin_by_email(email: str):
    """
    Lookup admin by email using DB, with flexible column fallbacks.
    Works even if your table uses name/adminName and phone/phoneNumber/mobile, etc.
    """
    if not email or db_engine is None:
        return None

    try:
        sql = text(f"""
            SELECT to_jsonb(t) AS row
            FROM {ADMIN_TABLE} AS t
            WHERE LOWER(t.email) = LOWER(:email)
            LIMIT 1
        """)
        df = pd.read_sql(sql, db_engine, params={"email": str(email).strip()})
    except Exception as e:
        print(f"[admin lookup] DB query error: {e}")
        return None

    if df.empty:
        return None

    # psycopg2 usually returns the jsonb as a Python dict already.
    row = df.iloc[0]["row"]
    if isinstance(row, str):
        import json
        row = json.loads(row)

    # Fallbacks across possible column names
    admin_id   = row.get("id") or row.get("adminId")
    admin_mail = row.get("email") or row.get("adminEmail")
    admin_name = (
        row.get("name") or
        row.get("adminName") or
        row.get("admin_name") or
        admin_mail
    )
    phone_raw = (
        row.get("phone") or
        row.get("phoneNumber") or
        row.get("mobile") or
        row.get("tel") or
        ""
    )
    belong_val = row.get("belongId") or row.get("belong_id") or row.get("belong")

    # Normalize
    phone_digits = "".join(ch for ch in str(phone_raw) if ch.isdigit())
    try:
        belong_id = int(belong_val) if belong_val is not None and str(belong_val).strip() != "" else None
    except Exception:
        belong_id = None

    return {
        "id": admin_id,
        "email": admin_mail,
        "name": admin_name,
        "phone_digits": phone_digits,
        "belong_id": belong_id,
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "admin_email" not in session or "belong_id" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# --------------------------------------------------------------------------------------
# Decode map (optional)
# --------------------------------------------------------------------------------------
DECODE_MAP = {}

def load_decode_csv(path: str | Path = None):
    """decode.csv를 (테이블, 영문컬럼명, 코드) → 사람이 읽는 텍스트 로 맵핑"""
    global DECODE_MAP
    if path is None:
        path = Path(__file__).with_name("decode.csv")
    try:
        df = pd.read_csv(path)
        # '테이블명'이 "eq_survey, lars_score, user"처럼 다중일 수 있으므로 분해
        df["tables"] = df["테이블명"].astype(str).str.split(r"\s*,\s*")
        m = {}
        for _, r in df.iterrows():
            for t in r["tables"]:
                key = (t.strip(), str(r["영문컬럼명"]).strip(), str(r["dCode"]).strip())
                m[key] = str(r["decode_text"]).strip()
        DECODE_MAP = m
        print(f"[decode] loaded {len(DECODE_MAP)} items from {path}")
    except Exception as e:
        print(f"[decode] failed to load decode.csv: {e}")
        DECODE_MAP = {}

def decode_val(table: str, col: str, v):
    """코드값을 사람이 읽는 텍스트로. {고혈압,당뇨} 같이 세트 문자열도 파싱."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        return [tok.strip() for tok in inner.split(",") if tok.strip()] if inner else []
    txt = DECODE_MAP.get((table, col, s))
    return txt if txt is not None else s

# 앱 시작 시 1회 로드 (없으면 조용히 무시)
load_decode_csv()

# --------------------------------------------------------------------------------------
# Data Loading helpers
# --------------------------------------------------------------------------------------
_user_cache = None

def _load_users():
    df = get_data_df("user")
    df = df.rename(columns={"id": "userId"})
    m = {}
    for _, r in df.iterrows():
        try:
            uid = int(r["userId"])
        except Exception:
            continue
        name = str(r.get("name", "") or "").strip() or "-"
        reg  = str(r.get("regNum", "") or "").strip() or "-"
        m[uid] = {"name": name, "regNum": reg}
    return m

def _get_user_meta(user_id):
    global _user_cache
    if _user_cache is None:
        _user_cache = _load_users()
    if user_id is None:
        return {"name": "-", "regNum": "-"}
    try:
        return _user_cache.get(int(user_id), {"name": "-", "regNum": "-"})
    except Exception:
        return {"name": "-", "regNum": "-"}

def _parse_answer_array(s):
    """Convert the stored answer string into an ordered list of ints.
       Accepts formats like '{1.0,0.0,2.0}', '{1,0,2}', '[1,0,2]', '1,0,2'."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    t = str(s).strip()
    if not t:
        return []
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        t = t[1:-1].strip()
    if not t:
        return []
    parts = [p.strip() for p in t.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            f = float(p)
            out.append(int(round(f)))
        except Exception:
            try:
                v = ast.literal_eval(p)
                out.append(int(v))
            except Exception:
                out.append(0)
    return out
def _decode_first(table, value, col_candidates):
    if value is None or str(value).strip() == "":
        return None
    # 1) 정확히 '영문컬럼명'과 dCode 매칭
    from re import sub
    for c in col_candidates:
        t = decode_val(table, c, value)
        if t is not None:
            return t
    # 2) 컬럼명 정규화(공백/기호 제거) 후 매칭
    def norm(s): return sub(r"[\s_\-:/\.]", "", str(s).lower())
    for c in col_candidates:
        for (t, col, code), txt in DECODE_MAP.items():
            if t == table and norm(col) == norm(c) and str(code).strip() == str(value).strip():
                return txt
    return value  # 매핑 없으면 원래 코드 반환

def normalize_users(df):
    rows = []
    for _, r in df.iterrows():
        created_dt = _parse_any_datetime(r.get("createdAt"))
        rows.append({
            "id": int(r.get("id", 0)),
            "hospital": BELONG_MAP.get(int(r.get("belongId", 0)) if pd.notna(r.get("belongId", None)) else 0, "-"),
            "name": r.get("name", ""),
            "research_id": r.get("regNum", ""),
            "registered_on": date_or_dash(created_dt),
            "age": parse_birth_age(r.get("birth", "")),
            "sex": "남" if str(r.get("sex","")).lower() in ["true","1","male","남","m"] else "여",
            "tumor": r.get("cancerType", ""),
            "secret_date": None,
            "pw_reset": False,
            "status": "활성" if str(r.get("status","")).lower() in ["true","1","active","활성"] else "비활성",
        })
    return rows

def _patient_summary(user_id: int) -> dict:
    u = pd.read_sql(
        text('SELECT id, name, "belongId", "birth", "sex", "createdAt" FROM "user" WHERE id = :pid'),
        db_engine, params={"pid": int(user_id)}
    )
    if u.empty:
        return {"name":"-", "hospital":"-", "age":"-", "sex":"-", "registered_on":"-"}

    row = u.iloc[0]

    # 소속병원
    belong_id = int(row.get("belongId") or 0)
    hospital = "-"
    if belong_id:
        b = pd.read_sql(text('SELECT name FROM belong WHERE id = :bid'),
                        db_engine, params={"bid": belong_id})
        if not b.empty:
            hospital = b.iloc[0].get("name") or "-"

    # 나이
    age = "-"
    birth = row.get("birth")
    if pd.notna(birth):
        try:
            birth_dt = pd.to_datetime(birth).date()
            from datetime import date
            today = date.today()
            age = today.year - birth_dt.year - ((today.month, today.day) < (birth_dt.month, birth_dt.day))
        except Exception:
            pass

    # 성별: TRUE → 남, FALSE → 여
    sex_raw = str(row.get("sex", "")).strip().lower()
    sex = "남" if sex_raw in ("1", "true", "male", "m", "남") else "여"

    # 등록일
    registered_on = date_or_dash(row.get("createdAt"))

    return {
        "name": row.get("name") or "-",
        "hospital": hospital,
        "age": age,
        "sex": sex,
        "registered_on": registered_on,
    }

def is_super_admin():
    # works whether or not you set session["super_admin"] at login
    return (
        str(session.get("admin_email", "")).lower() == "herings@heringsglobal.com"
        or bool(session.get("super_admin"))
    )

def compute_last_survey_dates(df_lars, df_eq):
    # rows with valid scores
    lars_valid = df_lars[df_lars['score'].notna()].copy()
    lars_valid['답변일'] = lars_valid['doneDay'].apply(_parse_any_datetime)
    lars_valid.loc[lars_valid['답변일'].isna(), '답변일'] = lars_valid['updatedAt'].apply(_parse_any_datetime)
    eq_valid = df_eq[df_eq['score'].notna()].copy()
    eq_valid['답변일'] = eq_valid['doneDay'].apply(_parse_any_datetime)
    eq_valid.loc[eq_valid['답변일'].isna(), '답변일'] = eq_valid['updatedAt'].apply(_parse_any_datetime)

    lars_last = lars_valid.groupby('userId', as_index=False)['답변일'].max().rename(columns={'답변일': 'lars_last'})
    eq_last   = eq_valid.groupby('userId', as_index=False)['답변일'].max().rename(columns={'답변일': 'eq_last'})
    last = lars_last.merge(eq_last, how='outer', on='userId')
    last['last_survey'] = last[['lars_last', 'eq_last']].max(axis=1)

    return {int(r.userId): (pd.to_datetime(r.last_survey).strftime('%Y-%m-%d') if pd.notna(r.last_survey) else None)
            for r in last.itertuples(index=False)}

# --------------------------------------------------------------------------------------
# Routes: Auth
# --------------------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw    = (request.form.get("password") or "").strip()
        admin = _admin_by_email(email)
        if not admin:
            flash("No admin found for that email.", "error")
            return render_template("login.html"), 401

        last4 = admin["phone_digits"][-4:] if len(admin["phone_digits"]) >= 4 else ""
        if pw != last4 or not last4:
            flash("Incorrect password.", "error")
            return render_template("login.html"), 401

        # success
        session["admin_email"] = admin["email"]
        session["admin_name"]  = admin["name"]
        session["belong_id"]   = admin["belong_id"]
        session["admin_id"]    = admin["id"]
        # special case: super-admin view
        if admin["email"].lower() == "herings@heringsglobal.com":
            session["super_admin"] = True
        else:
            session["super_admin"] = False
        session.permanent = True
        return redirect(request.args.get("next") or url_for("patient_list"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("patient_list"))

@app.route("/patient-list")
@login_required
def patient_list():
    # Load all data sources from DB
    users   = get_data_df("user")
    df_lars = get_data_df("lars_score")
    df_eq   = get_data_df("eq_survey")

     # Super admin sees everything
    if is_super_admin():
        # no belongId filter
        pass
    else:
        # Filter by belongId from session
        try:
            belong_id = session.get("belong_id")
            if belong_id is None:
                # no belong_id means no access to any patient rows
                users = users.iloc[0:0]
            else:
                if "belongId" in users.columns:
                    # robust numeric compare even if column is str/float/NaN
                    belong_col = pd.to_numeric(users["belongId"], errors="coerce")
                    users = users[belong_col.fillna(-1).astype("Int64") == int(belong_id)]
                else:
                    # if the column doesn't exist, return empty for safety
                    users = users.iloc[0:0]
        except Exception:
            # any unexpected issue -> safe empty set (non-super admin)
            users = users.iloc[0:0]

    # Pre-compute the last survey date for all users
    last_survey_dates = compute_last_survey_dates(df_lars, df_eq)

    # simple name-only search (literal substring, case-insensitive)
    q = request.args.get("q", "").strip()
    if q and "name" in users.columns:
        users = users[users["name"].astype(str).str.contains(q, case=False, na=False, regex=False)]

    # Normalize & attach last survey date
    rows = normalize_users(users)
    for row in rows:
        row['설문 마지막 응답일'] = last_survey_dates.get(row['id'])
        
        # 연구자등록번호 순으로 정렬 (내림차순: 최근 번호 위)
    def _regnum_key(r):
        # 숫자만 추출해서 정수 변환, 실패하면 0
        try:
            return int("".join(ch for ch in str(r.get("research_id") or "") if ch.isdigit()))
        except Exception:
            return 0

    rows.sort(key=_regnum_key, reverse=True)
    # pagination
    per_page = int(request.args.get("per_page", 10))
    total = len(rows)
    page = max(int(request.args.get("page", 1)), 1)
    total_pages = max(1, ceil(total / per_page))
    page = min(page, total_pages)
    start = (page - 1) * per_page
    end = start + per_page
    paged_rows = rows[start:end]

    def page_url(n):
        params = dict(request.args)  # keeps q
        params["page"] = n
        return f"{url_for('patient_list')}?{urlencode(params)}"

    pages = [{"n": n, "url": page_url(n), "current": (n == page)}
             for n in range(1, total_pages + 1)]
    has_prev = page > 1
    has_next = page < total_pages
    prev_url = page_url(page - 1) if has_prev else None
    next_url = page_url(page + 1) if has_next else None

    # stats across all users (unfiltered)
    all_users_df = get_data_df("user")
    all_users = normalize_users(all_users_df)
    stats = {
        "전체": len(all_users),
        "활성": sum(1 for r in all_users if r["status"] == "활성"),
        "비활성": sum(1 for r in all_users if r["status"] != "활성"),
    }
    query_args = request.args.to_dict(flat=True)
    return render_template(
        "patient_list.html",
        rows=paged_rows,
        stats=stats,
        q=q,
        belong=request.args.get("belong",""),
        pages=pages,
        has_prev=has_prev,
        has_next=has_next,
        prev_url=prev_url,
        next_url=next_url,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        total=total,
        query_args=query_args,
    )

@app.route("/patient-detail/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    # Load only the target user from DB (parameterized)
    try:
        users = pd.read_sql(text('SELECT * FROM "user" WHERE id = :pid'), db_engine, params={"pid": int(patient_id)})
    except Exception:
        users = pd.DataFrame()

    pr = users.loc[users["id"] == patient_id]
    if pr.empty:
        return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404
    pr = pr.iloc[0]

    # Access control: belongId must match session
    try:
        user_belong = int(float(pr.get("belongId")))
    except Exception:
        user_belong = None
        # Access control: belongId must match session unless super admin  :contentReference[oaicite:1]{index=1}
    if not is_super_admin():  # super admin sees everything
        belong_id = session.get("belong_id")
        if belong_id is None:
            return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404
        try:
            ok = (user_belong is not None) and (int(user_belong) == int(belong_id))
        except Exception:
            ok = False
        if not ok:
            return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404


    # Basic info
    birth = _parse_date(pr.get("birth"))
    today = dt.date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day)) if birth else "-"
    sex_raw = str(pr.get("sex", "")).strip().lower()
    sex = "남" if sex_raw in ("1", "true", "male", "m", "남") else "여"
    phone_raw = _pick(pr, "phone", "phoneNumber", "mobile", "cell", "tel", "전화번호", "휴대전화", "연락처")
    phone = _fmt_phone(phone_raw)

    belong_raw = pr.get("belongId")
    try:
        belong_name = BELONG_MAP.get(int(float(belong_raw)), "-")
    except Exception:
        belong_name = "-"

    # Body info
    height_raw = _pick(pr, "height", "키", "heightCm", "height_cm", "patient_height")
    weight_raw = _pick(pr, "current_weight", "origin_weight", "weight", "체중", "weightKg", "weight_kg")

    def _to_height_m(h):
        if h is None: return None
        try:
            f = float(h)
            return f/100.0 if f > 3 else f
        except Exception:
            return None

    h_m = _to_height_m(height_raw)
    try:
        w_kg = float(weight_raw) if weight_raw is not None else None
    except Exception:
        w_kg = None
    bmi = (w_kg / (h_m**2)) if (w_kg and h_m and h_m > 0) else None
    # ---- 질병 정보 (디코딩 적용) ----
    tumor_raw   = _pick_loose(pr, ["cancerType","종양","암종"])
    tumor       = _decode_first("user", tumor_raw, ["cancerType","종양","암종"])

    histology_raw = _pick_loose(pr, ["histology","histologicType","조직형", "tumor"])
    histology     = _decode_first("user", histology_raw, ["histology","histologicType","조직형", "tumor"])

    stage_raw   = _pick_loose(pr, ["stage","병기","cancerStage"])
    stage       = _decode_first("user", stage_raw, ["stage","병기","cancerStage"])
    lesion_loc_raw = _pick_loose(pr, ["lesionLocation","병변의 위치","병변위치","place","lesion"])
    lesion_loc     = _decode_first("user", lesion_loc_raw, ["lesionLocation","병변의 위치","병변위치","lesion_site","lesion"])

    stoma_raw   = _pick_loose(pr, ["stoma","hasStoma","장루 여부","장루여부"])
    stoma_dec   = _decode_first("user", stoma_raw, ["stoma","hasStoma","장루 여부","장루여부"])
    stoma_disp  = "-" if stoma_dec in (None, "") else ("있음" if str(stoma_dec).strip().lower() in ("1","y","yes","true","있음") else stoma_dec)
    # ---- 치료 정보 (디코딩 적용) ----
    surg_date = _parse_date(_pick_loose(pr, ["surgeryDate", "수술 날짜", "수술날짜", "opDate", "operationDate"]))

    # 수술명: already includes "operation" so no change needed
    surg_name = _decode_first("user",_pick_loose(pr, ["surgeryName","수술명","operation","opName"]),["surgeryName","수술명","operation","opName"])

    # 치료 상태: already includes anticancerTreatmentYN (OK)
    treat_status_raw = _pick_loose(pr, [
        "anticancerTreatmentYN","chemoStatus","treatStatus","radiationStatus",
        "항암치료","치료 상태","치료상태"])
    treat_status = _decode_first("user", treat_status_raw, [
        "anticancerTreatmentYN","chemoStatus","treatStatus","radiationStatus",
        "항암치료","치료 상태","치료상태"])

    treat_type_raw = _pick_loose(pr, ["therapyType","chemoType","항암치료 종류","방사선 치료 종류","rtType"])
    treat_type     = _decode_first("user", treat_type_raw, ["therapyType","chemoType","항암치료 종류","방사선 치료 종류","rtType"])

    # 종양반응: include trgType explicitly (normalizer would catch it, but this is clearer)
    resp_type_raw  = _pick_loose(pr, [
        "tumorResponseType","responseType","반응유형","TRG_Type","종양 반응 유형","trgType"])
    resp_grade_raw = _pick_loose(pr, [
        "tumorResponseGrade","responseGrade","반응등급","TRG_Grade","종양 반응 등급"])
    resp_type  = _decode_first("user", resp_type_raw,  [
        "tumorResponseType","responseType","반응유형","TRG_Type","종양 반응 유형","trgType"
    ])
    resp_grade = _decode_first("user", resp_grade_raw, [
        "tumorResponseGrade","responseGrade","반응등급","TRG_Grade","종양 반응 등급"
    ])
    resp_combo = f"{resp_type or '-'} / {resp_grade or '-'}" if (resp_type or resp_grade) else "-"
    patient = {
        "id": int(pr.get("id")),
        "name": pr.get("name") or "-",
        "hospital": belong_name,
        "research_id": pr.get("regNum") or "-",
        "registered_on": date_or_dash(pr.get("createdAt")),
        "age": age,
        "sex": sex,
        "phone": phone,
        "height_cm": _fmt_num(float(h_m)*100, 0) if h_m else "-",
        "weight_kg": _fmt_num(w_kg, 1) if w_kg is not None else "-",
        "bmi": _fmt_num(bmi, 1) if bmi is not None else "-",
        "tumor": tumor or "-",
        "histology": histology or "-",
        "stage": stage or "-",
        "lesion": (lesion_loc or "-"),
        "stoma": stoma_disp,
        "surgery_date": _fmt_date(surg_date),
        "surgery_name": surg_name or "-",
        "treat_status": treat_status or "-",
        "treat_type": treat_type or "-",
        "response": resp_combo,
        "status": "활성" if str(pr.get("status")).strip() in ("1","True","true","활성") else "비활성",
        "tumor_short": pr.get("cancerType") or "-",
    }

    def build_rows_and_chart(df, survey_label_fallback):
        sub = df.loc[df["userId"] == patient_id].copy()

        # choose best available date per row
        candidates = ["doneDay", "receiveDate", "updatedAt", "createdAt"]
        def choose_date(row):
            for c in candidates:
                if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                    d = _parse_date(row[c])
                    if d:
                        return d
            return None

        sub["__date"] = sub.apply(choose_date, axis=1)

        # --- TABLE: DESC ---
        table_sub = sub.sort_values(by="__date", ascending=False, kind="mergesort")

        rows = []
        for _, r in table_sub.iterrows():
            title = r.get("title") or survey_label_fallback
            d = r.get("__date")
            score = r.get("score")
            is_nan = pd.isna(score)
            rows.append({
                "id": int(r.get("id")),
                "title": title,
                "date": _fmt_date(d),
                "score_display": _fmt_score(score),
                "status_display": "미완료" if is_nan else "완료",
            })

        # --- GRAPH: ASC ---
        chart_sub = sub[sub["__date"].notna()].sort_values(by="__date", ascending=True, kind="mergesort")

        labels = [_fmt_date(d) for d in chart_sub["__date"]]  # 'YYYY-MM-DD'
        values = [None if pd.isna(s) else float(s) if s is not None else None
                for s in chart_sub["score"]]

        return rows, {"labels": labels, "values": values}


    # Query only this patient's surveys (parameterized) and include updated/created timestamps for fallback
    try:
        lars = pd.read_sql(
            text('SELECT id, title, "receiveDate", "doneDay", "updatedAt", "createdAt", "userId", score FROM lars_score WHERE "userId" = :uid'),
            db_engine, params={"uid": int(patient_id)}
        )
    except Exception:
        lars = pd.DataFrame()

    try:
        eq = pd.read_sql(
            text('SELECT id, title, "receiveDate", "doneDay", "updatedAt", "createdAt", "userId", score FROM eq_survey WHERE "userId" = :uid'),
            db_engine, params={"uid": int(patient_id)}
        )
    except Exception:
        eq = pd.DataFrame()

    lars_rows, lars_chart = build_rows_and_chart(lars, "LARS 설문")
    eq_rows,   eq_chart   = build_rows_and_chart(eq,   "EQ-5D 설문")
        # ----- Exercise (운동 달성률) -----
    ex_rows, ex_chart = _exercise_series_for_user(patient_id)


    back_url = request.args.get("back")
    if not back_url or not back_url.startswith("/patient-list"):
        back_url = url_for("patient_list")

    return render_template(
        "patient_detail.html",
        patient=patient,
        lars_rows=lars_rows,
        lars_chart=lars_chart,
        eq_rows=eq_rows,
        eq_chart=eq_chart,
        back_url=back_url,
        ex_rows=ex_rows,
        ex_chart=ex_chart,

        title=f"{patient['name']} 상세"
    )

@app.route("/survey-detail/<int:survey_id>")
@login_required
def survey_detail(survey_id):
    # Find the survey row by id across both tables
    row = None
    try:
        eq = pd.read_sql(text("SELECT * FROM eq_survey WHERE id = :sid"), db_engine, params={"sid": int(survey_id)})
        if not eq.empty:
            row = eq.iloc[0].to_dict()
            row["__type__"] = "EQ-5D"
        else:
            lars = pd.read_sql(text("SELECT * FROM lars_score WHERE id = :sid"), db_engine, params={"sid": int(survey_id)})
            if not lars.empty:
                row = lars.iloc[0].to_dict()
                row["__type__"] = "LARS"
    except Exception as e:
        print(f"find_survey error: {e}")
        row = None

    if not row:
        return render_template("not_found.html", message="Survey not found"), 404

    survey_type = row.get("__type__") or ""
    sid_for_questions = 1 if "LARS" in survey_type.upper() else 2

    meta = {
        "ID": row.get("id"),
        "__type__": survey_type,
        "title": row.get("title"),
        "receiveDate": row.get("receiveDate") or row.get("doneDay"),
        "score": row.get("score"),
        "status": row.get("status"),
        "raw_answer": row.get("answer", "-"),
    }
    score_val = meta["score"]
    score_display = "-" if (score_val is None or pd.isna(score_val)) else score_val
    status_display = "미완료" if score_display == "-" else "완료"

    # ACCESS CONTROL: check patient's belongId matches session
    patient_id = row.get("userId") or row.get("patientId")
    try:
        patient_id = int(patient_id) if patient_id is not None else None
    except (TypeError, ValueError):
        patient_id = None

    if patient_id is not None:
        try:
            users_df = pd.read_sql(text('SELECT id, "belongId" FROM "user" WHERE id = :pid'), db_engine, params={"pid": int(patient_id)})
        except Exception:
            users_df = pd.DataFrame()
        pr = users_df.loc[users_df["id"] == patient_id]
        if pr.empty:
            return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404
        try:
            user_belong = int(float(pr.iloc[0].get("belongId")))
        except Exception:
            user_belong = None
            # ACCESS CONTROL: belong must match unless super admin  :contentReference[oaicite:4]{index=4}
    if patient_id is not None:
        try:
            users_df = pd.read_sql(text('SELECT id, "belongId" FROM "user" WHERE id = :pid'),
                                   db_engine, params={"pid": int(patient_id)})
        except Exception:
            users_df = pd.DataFrame()
        pr = users_df.loc[users_df["id"] == patient_id]
        if pr.empty:
            return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404

        try:
            user_belong = int(float(pr.iloc[0].get("belongId")))
        except Exception:
            user_belong = None

        if not is_super_admin():
            belong_id = session.get("belong_id")
            if belong_id is None:
                return render_template("not_found.html", message="설문을 찾을 수 없습니다."), 404
            try:
                ok = (user_belong is not None) and (int(user_belong) == int(belong_id))
            except Exception:
                ok = False
            if not ok:
                return render_template("not_found.html", message="설문을 찾을 수 없습니다."), 404


    user_meta = _get_user_meta(patient_id)
    back_url = url_for("patient_detail", patient_id=patient_id) if patient_id is not None else url_for("patient_list")

    selected_indices = _parse_answer_array(row.get("answer"))
    # Load questions & choices from CSV
    qdf = pd.read_csv(QUESTIONS_CSV).rename(columns={"question_text": "text", "question_order": "order"})
    qdf["order"] = qdf["order"].astype(int)
    adf = pd.read_csv(ANSWERS_CSV).sort_values(["survey_id", "question_id", "id"])

    # Build choices map
    choices_map = {}
    for (sid, qid), g in adf.groupby(["survey_id", "question_id"], sort=False):
        choices_map[(int(sid), int(qid))] = list(g["answer_text"].astype(str).tolist())

    # Questions list for this survey
    qs_for_survey = qdf[qdf["survey_id"] == sid_for_questions].copy()
    qs_for_survey = qs_for_survey.sort_values(["order", "id"])

    expanded = []
    for pos, (_, q) in enumerate(qs_for_survey.iterrows()):
        qid = int(q["id"]); qtext = str(q["text"]); qorder = int(q["order"])
        sel_idx = selected_indices[pos] if pos < len(selected_indices) else None
        option_texts = choices_map.get((sid_for_questions, qid), [])
        options = []
        selected_text = None
        for i, txt in enumerate(option_texts):
            is_sel = (sel_idx is not None and i == sel_idx)
            if is_sel:
                selected_text = txt
            options.append({"index": i, "text": txt, "selected": is_sel})
        expanded.append({
            "order": qorder,
            "question_id": qid,
            "question_text": qtext,
            "options": options,
            "selected_index": sel_idx,
            "selected_text": selected_text if selected_text is not None else "-"
        })

    # ... keep existing meta/vars above ...

    # Build base qa WITHOUT Condition yet
    qa = [
        {"q": "설문 유형",      "a": meta["__type__"]},
        {"q": "이름",          "a": user_meta["name"]},
        {"q": "연구대상자번호", "a": user_meta["regNum"]},
        {"q": "설문 제목",      "a": meta["title"]},
        {"q": "EQ-5D 점수",          "a": score_display},      # ← score row (will be pushed after Condition)
        {"q": "상태",          "a": status_display},
        {"q": "수신/완료일",    "a": meta["receiveDate"]},
    ]

    # If EQ-5D, insert Condition (EQ-VAS) BEFORE the score row
    if "EQ-5D" in (survey_type or ""):
        vas_val = row.get("condition") or row.get("eq_vas") or None
        cond = "-" if vas_val is None or pd.isna(vas_val) else str(int(round(float(vas_val))))
        qa.insert(4, {"q": "EQ-VAS", "a": cond})  # index 4 = before "점수"

    return render_template(
        "survey_detail.html",
        survey={"title": meta["title"], "id": meta["ID"], "date": meta["receiveDate"], "type": meta["__type__"]},
        qa=qa,
        expanded=expanded,
        patient_id=patient_id,
        back_url=back_url,
    )
@app.route("/exercise-detail/<int:header_id>")
@login_required
def exercise_detail(header_id):
    # find the header row and check belong access
    try:
        hdr = pd.read_sql(
            text('SELECT id, "userId", "date", "recommendWorkoutId" FROM workout_header WHERE id = :hid'),
            db_engine, params={"hid": int(header_id)}
        )
    except Exception:
        hdr = pd.DataFrame()

    if hdr.empty:
        return render_template("not_found.html", message="운동 기록을 찾을 수 없습니다."), 404

    row = hdr.iloc[0]
    user_id = int(row["userId"])
    # access control
    try:
        users_df = pd.read_sql(text('SELECT id, "belongId", name, "regNum" FROM "user" WHERE id = :pid'),
                               db_engine, params={"pid": user_id})
    except Exception:
        users_df = pd.DataFrame()

    if users_df.empty:
        return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404
    try:
        user_belong = int(float(users_df.iloc[0].get("belongId")))
    except Exception:
        user_belong = None
        # access control  :contentReference[oaicite:3]{index=3}
    if not is_super_admin():
        belong_id = session.get("belong_id")
        if belong_id is None:
            return render_template("not_found.html", message="대상자를 찾을 수 없습니다."), 404
        try:
            ok = (user_belong is not None) and (int(user_belong) == int(belong_id))
        except Exception:
            ok = False
        if not ok:
            return render_template("not_found.html", message="운동 기록을 찾을 수 없습니다."), 404

    # latest non-null LARS score for this user  :contentReference[oaicite:0]{index=0}
    try:
        df_lars = pd.read_sql(
            text('SELECT score, "doneDay", "updatedAt", "createdAt" '
                 'FROM lars_score WHERE "userId" = :uid'),
            db_engine, params={"uid": user_id}
        )
    except Exception:
        df_lars = pd.DataFrame()

    latest_lars_score, latest_lars_date = "-", "-"

    if not df_lars.empty:
        # Keep only rows with a real score
        df_valid = df_lars[df_lars["score"].notna()].copy()

        if not df_valid.empty:
            # Build tz-agnostic date using _parse_date (returns datetime.date)  :contentReference[oaicite:1]{index=1}
            def choose_date(r):
                for c in ("doneDay", "updatedAt", "createdAt"):
                    d = _parse_date(r.get(c))
                    if d:
                        return d
                return None

            df_valid["__date"] = df_valid.apply(choose_date, axis=1)
            df_valid = df_valid[df_valid["__date"].notna()]

            if not df_valid.empty:
                df_valid = df_valid.sort_values("__date", ascending=False, kind="mergesort")
                r = df_valid.iloc[0]
                latest_lars_score = _fmt_score(r.get("score"))     # helper in app.py  :contentReference[oaicite:2]{index=2}
                latest_lars_date  = _fmt_date(r.get("__date"))     # helper in app.py  :contentReference[oaicite:3]{index=3}


    # patient summary (lean)
    user_meta = {
        "name": users_df.iloc[0].get("name") or "-",
        "regNum": users_df.iloc[0].get("regNum") or "-"
    }
    ratio, done_sets, total_sets = _exercise_ratio_for_header(header_id)
    ratio_disp = "-" if (ratio is None) else f"{ratio:.1f}%"
    details = _exercise_details_for_header(header_id)
    patient_meta = _patient_summary(user_id)
    patient_meta["lars_score"] = latest_lars_score
    patient_meta["lars_date"]  = latest_lars_date
    back_url = url_for("patient_detail", patient_id=user_id)

    return render_template(
    "exercise_detail.html",
    workout_header={
        "id": int(row["id"]),
        "date": date_or_dash(row.get("date")),
        "ratio": ratio_disp,
        "done_sets": done_sets,
        "total_sets": total_sets,
    },
    patient=patient_meta,   # see #2 below
    details=details,
    back_url=back_url,
)

# in app.py (same file that defines `app = Flask(__name__)`)
@app.get("/healthz")
def healthz():
    return "ok", 200


# --------------------------------------------------------------------------------------
# Session / Cookie Settings
# --------------------------------------------------------------------------------------
SECURE = (os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("1","true","yes"))
app.config.update(
    PERMANENT_SESSION_LIFETIME=dt.timedelta(days=180),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE,
)

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
