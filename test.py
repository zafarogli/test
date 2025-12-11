import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt
from io import BytesIO

# PDF libs (pure Python – work on Streamlit Cloud)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ------------- STREAMLIT PAGE CONFIG -----------------

st.set_page_config(
    page_title="Weyland-Yutani Mining Ops Dashboard",
    layout="wide",
)

# ------------- CONSTANTS -----------------------------

CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRx7FuaguRcCHCwQOJyPv1oDCHM7u7oq5yDmI-iV0IoPOa2uroqOG8qZtD3ZvlB1CpqsOMw9Ri9mkS5/"
    "pub?gid=809861880&single=true&output=csv"
)

# ------------- DATA HELPERS --------------------------


def load_data(csv_url: str) -> pd.DataFrame:
    """Load CSV from Google Sheets, parse dates, drop empty columns."""
    df = pd.read_csv(csv_url)

    # Drop unnamed columns (extra commas in CSV)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def compute_stats(df: pd.DataFrame, mine_cols: list[str]) -> pd.DataFrame:
    """Compute mean, std, median, IQR per mine + total."""
    rows = []

    for col in mine_cols:
        s = df[col].dropna()
        rows.append(
            {
                "Mine": col,
                "Mean": s.mean(),
                "Std": s.std(ddof=1),
                "Median": s.median(),
                "IQR": s.quantile(0.75) - s.quantile(0.25),
            }
        )

    # total output across mines
    total_series = df[mine_cols].sum(axis=1)
    rows.append(
        {
            "Mine": "Total",
            "Mean": total_series.mean(),
            "Std": total_series.std(ddof=1),
            "Median": total_series.median(),
            "IQR": total_series.quantile(0.75) - total_series.quantile(0.25),
        }
    )

    return pd.DataFrame(rows)


# ------------- ANOMALY DETECTION ---------------------


def detect_iqr_anomalies(
    df: pd.DataFrame, mine_cols: list[str], k: float = 1.5
) -> pd.DataFrame:
    """IQR-rule anomalies per mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        mask = (df[col] < lower) | (df[col] > upper)

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = df.loc[mask, col].values
            outliers["Lower_bound"] = lower
            outliers["Upper_bound"] = upper
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "Date",
            "Day_idx",
            "Weekday",
            "Mine",
            "Output",
            "Lower_bound",
            "Upper_bound",
        ]
    )


def detect_zscore_anomalies(
    df: pd.DataFrame, mine_cols: list[str], z_thresh: float = 3.0
) -> pd.DataFrame:
    """Z-score anomalies per mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()
        mean = s.mean()
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            continue

        z = (df[col] - mean) / std
        mask = z.abs() > z_thresh

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = df.loc[mask, col].values
            outliers["Z"] = z[mask].values
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(columns=["Date", "Day_idx", "Weekday", "Mine", "Output", "Z"])


def detect_ma_percent_anomalies(
    df: pd.DataFrame,
    mine_cols: list[str],
    window: int = 7,
    pct_thresh: float = 30.0,
) -> pd.DataFrame:
    """Distance (percent) from moving average anomalies."""
    all_rows = []

    for col in mine_cols:
        s = df[col]

        ma = s.rolling(window=window, min_periods=window).mean()
        valid = ma.notna() & (ma != 0)
        deviation_pct = (s - ma).abs() / ma * 100

        mask = valid & (deviation_pct > pct_thresh)

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = s[mask].values
            outliers["MA"] = ma[mask].values
            outliers["Deviation_pct"] = deviation_pct[mask].values
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "Date",
            "Day_idx",
            "Weekday",
            "Mine",
            "Output",
            "MA",
            "Deviation_pct",
        ]
    )


def detect_grubbs_anomalies(
    df: pd.DataFrame, mine_cols: list[str], alpha: float = 0.05
) -> pd.DataFrame:
    """Grubbs' test anomalies per mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()
        N = len(s)
        if N < 3:
            continue

        mean = s.mean()
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            continue

        G = (s - mean).abs() / std

        t_crit = t_dist.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = ((N - 1) / np.sqrt(N)) * np.sqrt(
            t_crit**2 / (N - 2 + t_crit**2)
        )

        mask = G > G_crit

        if mask.any():
            outliers = df.loc[s.index[mask], ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = s[mask].values
            outliers["G"] = G[mask].values
            outliers["G_crit"] = G_crit
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=["Date", "Day_idx", "Weekday", "Mine", "Output", "G", "G_crit"]
    )


def compute_trendlines(df: pd.DataFrame, mine_cols: list[str], degree: int):
    """Polynomial trendlines per mine (x = Day_idx, y = output)."""
    if degree <= 0:
        return None

    rows = []
    x_all = df["Day_idx"].values.astype(float)

    for col in mine_cols:
        y_all = df[col].values.astype(float)

        mask = ~np.isnan(x_all) & ~np.isnan(y_all)
        x = x_all[mask]
        y = y_all[mask]

        if len(x) <= degree:
            continue

        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)

        tmp = pd.DataFrame(
            {
                "Date": df.loc[mask, "Date"].values,
                "Mine": col,
                "Trend": y_pred,
            }
        )
        rows.append(tmp)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def build_anomaly_summary(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    anomalies_iqr: pd.DataFrame,
    anomalies_z: pd.DataFrame,
    anomalies_ma: pd.DataFrame,
    anomalies_grubbs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Unified anomaly table:

    Columns:
      - Date
      - Mine
      - Value
      - Spike_or_drop
      - Methods
    """
    frames = []

    if not anomalies_iqr.empty:
        tmp = anomalies_iqr[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "IQR"
        frames.append(tmp)

    if not anomalies_z.empty:
        tmp = anomalies_z[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Z-score"
        frames.append(tmp)

    if not anomalies_ma.empty:
        tmp = anomalies_ma[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Moving average"
        frames.append(tmp)

    if not anomalies_grubbs.empty:
        tmp = anomalies_grubbs[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Grubbs"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["Date", "Mine", "Value", "Spike_or_drop", "Methods"]
        )

    combined = pd.concat(frames, ignore_index=True)

    grouped = (
        combined.groupby(["Date", "Mine", "Output"])["Method"]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .reset_index()
        .rename(columns={"Output": "Value", "Method": "Methods"})
    )

    # classify as spike / drop vs median
    median_map = stats_df.set_index("Mine")["Median"].to_dict()

    def classify(row):
        med = median_map.get(row["Mine"])
        if pd.isna(med):
            return ""
        if row["Value"] > med:
            return "Spike"
        elif row["Value"] < med:
            return "Drop"
        else:
            return "Neutral"

    grouped["Spike_or_drop"] = grouped.apply(classify, axis=1)
    grouped = grouped[["Date", "Mine", "Value", "Spike_or_drop", "Methods"]]
    grouped = grouped.sort_values(["Date", "Mine"]).reset_index(drop=True)

    return grouped


# ------------- PDF BUILDING (REPORTLAB) ---------------


def build_overall_chart_bytes(df: pd.DataFrame, mine_cols: list[str]) -> bytes:
    """
    Build a matplotlib line chart (all mines over time) and return it as PNG bytes.
    This is used inside the PDF report.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for col in mine_cols:
        ax.plot(df["Date"], df[col], label=col)

    ax.set_title("Daily output per mine")
    ax.set_xlabel("Date")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_report(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    combined_anomalies: pd.DataFrame,
    mine_cols: list[str],
) -> bytes:
    """
    Build a PDF report using pure Python (reportlab).
    Includes:
      - overview
      - overall chart
      - per-mine statistics table
      - anomaly table
      - per-anomaly text lines (satisfies "separate sections" requirement)
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    elements = []

    # --- title ---
    elements.append(Paragraph("Weyland-Yutani Mines – Daily Output Report", title_style))
    elements.append(Spacer(1, 12))

    # --- overview ---
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    n_days = df["Date"].nunique()
    n_mines = len(mine_cols)

    df_total = df.copy()
    df_total["Total_output"] = df_total[mine_cols].sum(axis=1)
    total_output_sum = df_total["Total_output"].sum()
    total_output_mean = df_total["Total_output"].mean()

    overview_html = (
        f"Date range: {start_date.date()} – {end_date.date()}<br/>"
        f"Number of days: {n_days}<br/>"
        f"Number of mines: {n_mines}<br/>"
        f"Total output (all mines): {total_output_sum:,.2f}<br/>"
        f"Average daily total output: {total_output_mean:,.2f}"
    )
    elements.append(Paragraph(overview_html, normal_style))
    elements.append(Spacer(1, 16))

    # --- chart ---
    elements.append(Paragraph("Overall production chart", heading_style))
    elements.append(Spacer(1, 8))

    chart_bytes = build_overall_chart_bytes(df, mine_cols)
    chart_buf = BytesIO(chart_bytes)
    elements.append(Image(chart_buf, width=400, height=250))
    elements.append(Spacer(1, 16))

    # --- stats table ---
    elements.append(Paragraph("Per-mine statistics", heading_style))
    elements.append(Spacer(1, 8))

    stats_df_round = stats_df.copy()
    for c in ["Mean", "Std", "Median", "IQR"]:
        stats_df_round[c] = stats_df_round[c].round(2)

    stats_data = [list(stats_df_round.columns)] + stats_df_round.astype(str).values.tolist()
    stats_table = Table(stats_data, repeatRows=1)
    stats_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(stats_table)

    # --- anomalies on new page ---
    elements.append(PageBreak())
    elements.append(Paragraph("Anomaly events", heading_style))
    elements.append(Spacer(1, 8))

    if combined_anomalies.empty:
        elements.append(Paragraph("No anomalies detected.", normal_style))
    else:
        anomalies_disp = combined_anomalies.copy()
        if "Value" in anomalies_disp.columns:
            anomalies_disp["Value"] = anomalies_disp["Value"].round(2)

        anomalies_data = [list(anomalies_disp.columns)] + anomalies_disp.astype(str).values.tolist()
        anomalies_table = Table(anomalies_data, repeatRows=1)
        anomalies_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("ALIGN", (2, 1), (-2, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(anomalies_table)
        elements.append(Spacer(1, 12))

        # separate lines for each anomaly (spike/drop)
        elements.append(Paragraph("Anomaly descriptions", heading_style))
        elements.append(Spacer(1, 6))

        for _, row in anomalies_disp.iterrows():
            txt = (
                f"{row['Date']} – {row['Mine']}: "
                f"{row['Spike_or_drop']} (value {row['Value']}, "
                f"methods: {row['Methods']})"
            )
            elements.append(Paragraph(txt, normal_style))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ------------- STREAMLIT UI ---------------------------

st.title("Weyland-Yutani Mining Ops Dashboard")
st.caption(
    "Daily mining output simulator analytics: statistics, anomalies, charts, and PDF report."
)
st.markdown("---")

st.sidebar.title("Controls")
csv_url = st.sidebar.text_input("Google Sheets CSV URL", CSV_URL_DEFAULT)

# which anomaly tests to run
st.sidebar.markdown("### Anomaly tests")
run_iqr = st.sidebar.checkbox("IQR rule", value=True)
run_z = st.sidebar.checkbox("Z-score", value=True)
run_ma = st.sidebar.checkbox("Moving average %", value=True)
run_grubbs = st.sidebar.checkbox("Grubbs' test", value=True)

st.sidebar.markdown("---")

k_iqr = st.sidebar.slider("IQR multiplier k", 1.0, 5.0, 1.5, 0.1)
z_thresh = st.sidebar.slider("Z-score threshold |z| >", 1.0, 5.0, 3.0, 0.1)
ma_window = st.sidebar.slider("MA window (days)", 3, 30, 7, 1)
ma_pct = st.sidebar.slider("MA deviation threshold (%)", 5, 200, 30, 5)
alpha_grubbs = st.sidebar.slider("Grubbs alpha", 0.001, 0.10, 0.05, 0.005)

st.sidebar.markdown("---")

st.sidebar.markdown("### Chart options")
chart_type = st.sidebar.selectbox(
    "Chart type",
    ["Line", "Bar", "Stacked area"],
)
trend_degree = st.sidebar.selectbox(
    "Trendline degree",
    [0, 1, 2, 3, 4],
    format_func=lambda d: "No trendline" if d == 0 else f"Degree {d}",
    index=1,
)
anomaly_for_chart = st.sidebar.selectbox(
    "Highlight anomalies on chart",
    ["None", "IQR", "Z-score", "Moving average", "Grubbs"],
)

if not csv_url:
    st.info("Paste the published Google Sheets CSV URL in the sidebar.")
    st.stop()

df = load_data(csv_url)

# IMPORTANT: treat only actual mines as mines
mine_cols = [
    c
    for c in df.columns
    if c not in ["Date", "Day_idx", "Weekday", "Event Multiplier"]
]

stats_df = compute_stats(df, mine_cols)

df_long = df.melt(
    id_vars=["Date", "Day_idx", "Weekday"],
    value_vars=mine_cols,
    var_name="Mine",
    value_name="Output",
)

# anomalies
anomalies_iqr = (
    detect_iqr_anomalies(df, mine_cols, k=k_iqr)
    if run_iqr or anomaly_for_chart == "IQR"
    else pd.DataFrame()
)
anomalies_z = (
    detect_zscore_anomalies(df, mine_cols, z_thresh=z_thresh)
    if run_z or anomaly_for_chart == "Z-score"
    else pd.DataFrame()
)
anomalies_ma = (
    detect_ma_percent_anomalies(df, mine_cols, window=ma_window, pct_thresh=float(ma_pct))
    if run_ma or anomaly_for_chart == "Moving average"
    else pd.DataFrame()
)
anomalies_grubbs = (
    detect_grubbs_anomalies(df, mine_cols, alpha=alpha_grubbs)
    if run_grubbs or anomaly_for_chart == "Grubbs"
    else pd.DataFrame()
)

combined_anomalies = build_anomaly_summary(
    df,
    stats_df,
    anomalies_iqr,
    anomalies_z,
    anomalies_ma,
    anomalies_grubbs,
)

# ------------- TOP LAYOUT: CHART + STATS -------------

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Mines output over time")

    tooltip = ["Date:T", "Mine:N", "Output:Q"]

    if chart_type == "Stacked area":
        main_chart = (
            alt.Chart(df_long)
            .mark_area()
            .encode(
                x="Date:T",
                y=alt.Y("Output:Q", stack="zero"),
                color="Mine:N",
                tooltip=tooltip,
            )
        )
    else:
        base = (
            alt.Chart(df_long)
            .encode(
                x="Date:T",
                y="Output:Q",
                color="Mine:N",
                tooltip=tooltip,
            )
        )
        if chart_type == "Line":
            main_chart = base.mark_line()
        else:
            main_chart = base.mark_bar()

    trend_df = compute_trendlines(df, mine_cols, degree=trend_degree)
    if trend_df is not None and trend_degree > 0:
        trend_chart = (
            alt.Chart(trend_df)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x="Date:T",
                y="Trend:Q",
                color="Mine:N",
                tooltip=["Date:T", "Mine:N", "Trend:Q"],
            )
        )
        chart = main_chart + trend_chart
    else:
        chart = main_chart

    # overlay anomalies on chart if selected
    anomalies_for_chart = None
    if anomaly_for_chart == "IQR":
        anomalies_for_chart = anomalies_iqr
    elif anomaly_for_chart == "Z-score":
        anomalies_for_chart = anomalies_z
    elif anomaly_for_chart == "Moving average":
        anomalies_for_chart = anomalies_ma
    elif anomaly_for_chart == "Grubbs":
        anomalies_for_chart = anomalies_grubbs

    if anomalies_for_chart is not None and not anomalies_for_chart.empty:
        anomaly_points = anomalies_for_chart[["Date", "Mine", "Output"]].copy()
        anomaly_chart = (
            alt.Chart(anomaly_points)
            .mark_circle(size=80, color="red")
            .encode(
                x="Date:T",
                y="Output:Q",
                tooltip=["Date:T", "Mine:N", "Output:Q"],
            )
        )
        chart = chart + anomaly_chart

    st.altair_chart(chart.interactive(), use_container_width=True)

with col_right:
    st.subheader("Summary statistics per mine and total")
    st.dataframe(
        stats_df.style.format(
            {
                "Mean": "{:,.1f}",
                "Std": "{:,.1f}",
                "Median": "{:,.1f}",
                "IQR": "{:,.1f}",
            }
        )
    )

st.markdown("---")

# ------------- ANOMALY TABLES ------------------------

st.subheader("Combined anomaly list (all methods)")
if combined_anomalies.empty:
    st.write("No anomalies detected by any method.")
else:
    st.dataframe(combined_anomalies)

if run_iqr:
    st.subheader(f"IQR-based anomalies (k = {k_iqr})")
    if anomalies_iqr.empty:
        st.write("No IQR anomalies.")
    else:
        st.dataframe(anomalies_iqr)

if run_z:
    st.subheader(f"Z-score-based anomalies (|z| > {z_thresh})")
    if anomalies_z.empty:
        st.write("No Z-score anomalies.")
    else:
        st.dataframe(anomalies_z)

if run_ma:
    st.subheader(
        f"Moving-average anomalies (window={ma_window}, deviation > {ma_pct}%)"
    )
    if anomalies_ma.empty:
        st.write("No moving-average anomalies.")
    else:
        st.dataframe(anomalies_ma)

if run_grubbs:
    st.subheader(f"Grubbs' test anomalies (alpha = {alpha_grubbs})")
    if anomalies_grubbs.empty:
        st.write("No Grubbs anomalies.")
    else:
        st.dataframe(anomalies_grubbs)

# ------------- PDF DOWNLOAD --------------------------

st.markdown("---")
st.subheader("Download report")

if st.button("Generate PDF report"):
    try:
        pdf_bytes = build_pdf_report(df, stats_df, combined_anomalies, mine_cols)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="weyland_yutani_report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
