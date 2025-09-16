# src/tools/tools.py
from crewai.tools import tool
from typing import Dict, Any, Union, List
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import openpyxl
from jinja2 import Template
from pathlib import Path

# -------------------------
# Utilities
# -------------------------
def df_from_records(records):
    try:
        df = pd.DataFrame(records)
        if not df.empty:
            # strip whitespace from column names
            df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def is_time_series(df: pd.DataFrame) -> bool:
    # time series if there's any column that is datetime-like and at least one numeric column
    if df is None or df.empty:
        return False
    date_cols = [c for c in df.columns if 'date' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return len(date_cols) >= 1 and len(numeric_cols) >= 1

def infer_main_numeric_and_date(df: pd.DataFrame):
    date_col = None
    num_cols = []
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc or 'day' in lc or 'period' in lc:
            date_col = c
            break
    if date_col is None:
        # try to detect datetime dtype
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    # numeric columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    return date_col, num_cols

def to_base64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    try:
        df = df.dropna(how='all').dropna(axis=1, how='all')
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df.columns = df.columns.map(lambda c: str(c).strip())
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                continue
    return df

# -------------------------
# Data Ingestion Tool
# -------------------------
class DataIngestionInput(BaseModel):
    file_path: Union[str, List[str]] = Field(..., description="File path or list of file paths.")

@tool("Data Ingestion Tool")
def data_ingestion_tool(file_path: Union[str, List[str]]) -> Dict[str, Any]:
    """Load tabular data from Excel, CSV, TSV, or JSON files into a sheet-indexed dict."""
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = [fp for fp in (file_path or []) if fp]

    if not file_paths:
        return {'error': 'No file paths provided.'}

    aggregated: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []

    for path_str in file_paths:
        resolved_path = os.path.abspath(path_str)
        if not os.path.exists(resolved_path):
            errors.append(f"File not found: {path_str}.")
            continue

        suffix = Path(resolved_path).suffix.lower()
        try:
            if suffix in {'.xlsx', '.xls', '.xlsm'}:
                sheets = pd.read_excel(resolved_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    cleaned = _clean_dataframe(df)
                    if cleaned.empty:
                        continue
                    key = (sheet_name or '').strip() or Path(resolved_path).stem
                    records = cleaned.to_dict('records')
                    aggregated.setdefault(key, []).extend(records)
            elif suffix in {'.csv'}:
                df = pd.read_csv(resolved_path)
                cleaned = _clean_dataframe(df)
                if cleaned.empty:
                    continue
                key = Path(resolved_path).stem
                aggregated.setdefault(key, []).extend(cleaned.to_dict('records'))
            elif suffix in {'.tsv', '.txt'}:
                df = pd.read_csv(resolved_path, sep=None, engine='python')
                cleaned = _clean_dataframe(df)
                if cleaned.empty:
                    continue
                key = Path(resolved_path).stem
                aggregated.setdefault(key, []).extend(cleaned.to_dict('records'))
            elif suffix in {'.json'}:
                df = pd.read_json(resolved_path)
                cleaned = _clean_dataframe(df)
                if cleaned.empty:
                    continue
                key = Path(resolved_path).stem
                aggregated.setdefault(key, []).extend(cleaned.to_dict('records'))
            else:
                errors.append(f"Unsupported file type: {path_str}.")
        except Exception as exc:
            errors.append(f"Ingestion failed for {path_str}: {exc}")

    if aggregated:
        return aggregated

    detail = '; '.join(errors) if errors else 'No readable content found.'
    return {'error': f"Ingestion failed: {detail}"}

# -------------------------
# Liquidity Analysis Tool (dynamic)
# -------------------------
class LiquidityAnalysisInput(BaseModel):
    data: Union[str, dict] = Field(..., description="JSON string or dict data.")
    objective: str = Field(..., description="Objective of analysis.")

@tool("Liquidity Analysis Tool")
def liquidity_analysis_tool(data: Union[str, dict], objective: str) -> Dict[str, Any]:
    """
    Dynamic analysis:
    - converts sheets to DataFrames
    - infers types and picks appropriate computations
    - returns tables, metrics, insights, and a list of chart_specs (not images)
    Chart spec example:
      {"name":"cash_trend","type":"line","labels":[...],"datasets":[{"label":"Net Cash","data":[...]}], "meta":{...}}
    """
    # Normalize input
    if isinstance(data, str):
        try:
            data_dict = json.loads(data)
        except Exception:
            return {'error': 'Invalid JSON input for analysis.'}
    else:
        data_dict = data

    if not isinstance(data_dict, dict):
        return {'error': 'Data must be a dict of sheets.'}

    results = {'objective': objective, 'metrics': {}, 'tables': {}, 'chart_specs': [], 'insights': []}
    dfs = {}

    # Convert all sheets to dataframes (and attempt type coercions)
    for sheet_name, records in data_dict.items():
        try:
            df = df_from_records(records)
            # coerce likely date columns
            for c in df.columns:
                if 'date' in c.lower():
                    df[c] = pd.to_datetime(df[c], errors='coerce')
            dfs[sheet_name] = df
        except Exception:
            dfs[sheet_name] = pd.DataFrame()

    # Helper to add chart spec
    def add_chart_spec(name, chart_type, labels, datasets, meta=None):
        spec = {"name": name, "type": chart_type, "labels": labels, "datasets": datasets}
        if meta:
            spec["meta"] = meta
        results['chart_specs'].append(spec)

    # Inspect sheets and build dynamic analysis
    # Heuristics based on sheet names
    keys = {k.lower(): k for k in dfs.keys()}

    # 1) Bank Cash Daily / cashflow time series
    cash_sheet = None
    for candidate in ['bank_cash_daily', 'bankcashdaily', 'cash_daily', 'daily_cash', 'cashflow_daily', 'cash_flow']:
        if candidate in keys:
            cash_sheet = keys[candidate]
            break

    if cash_sheet:
        df = dfs[cash_sheet].copy()
        # try to detect date column and net cash column
        date_col, num_cols = infer_main_numeric_and_date(df)
        if date_col is None and 'Date' in df.columns:
            date_col = 'Date'
        if date_col:
            df = df.sort_values(by=date_col)
            # try compute NetCash if Inflow/Outflow exist
            if 'Inflow' in df.columns and 'Outflow' in df.columns:
                df['NetCash'] = pd.to_numeric(df['Inflow'], errors='coerce').fillna(0) - pd.to_numeric(df['Outflow'], errors='coerce').fillna(0)
            # else try Closing Balance
            if 'Closing Balance' in df.columns:
                df['ClosingBalance'] = pd.to_numeric(df['Closing Balance'], errors='coerce')
            # fill missing numeric
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # store table (head)
            results['tables'][cash_sheet] = df.head(50).to_dict('records')
            # basic metrics
            if 'NetCash' in df.columns:
                results['metrics']['avg_daily_netcash'] = float(df['NetCash'].mean())
                # prepare line chart for net cash
                labels = df[date_col].dt.strftime('%Y-%m-%d').tolist()
                datasets = [{"label": "NetCash", "data": df['NetCash'].fillna(0).tolist()}]
                add_chart_spec("net_cash_daily", "line", labels, datasets, meta={"sheet": cash_sheet})
                results['insights'].append(f"Net cash series detected in '{cash_sheet}' ({len(df)} rows).")
            elif 'ClosingBalance' in df.columns:
                results['metrics']['latest_closing_balance'] = float(df['ClosingBalance'].dropna().iloc[-1]) if not df['ClosingBalance'].dropna().empty else 0.0
                labels = df[date_col].dt.strftime('%Y-%m-%d').tolist()
                datasets = [{"label": "Closing Balance", "data": df['ClosingBalance'].fillna(0).tolist()}]
                add_chart_spec("closing_balance", "line", labels, datasets, meta={"sheet": cash_sheet})
                results['insights'].append(f"Closing balance series detected in '{cash_sheet}'.")
        else:
            results['insights'].append(f"'{cash_sheet}' found but no date column could be inferred.")

    # 2) Cash Flow Monthly or Projections -> time series aggregation
    proj_sheet = None
    for candidate in ['projections_monthly', 'projections', 'cash_flow_monthly', 'cashflow_monthly']:
        if candidate in keys:
            proj_sheet = keys[candidate]
            break
    if proj_sheet:
        df = dfs[proj_sheet].copy()
        date_col, num_cols = infer_main_numeric_and_date(df)
        if date_col and num_cols:
            df = df.sort_values(by=date_col)
            # sum numeric per period
            series_name = num_cols[0]
            labels = df[date_col].dt.strftime('%Y-%m').tolist()
            datasets = [{"label": series_name, "data": pd.to_numeric(df[series_name], errors='coerce').fillna(0).tolist()}]
            add_chart_spec("projections", "line", labels, datasets, meta={"sheet": proj_sheet})
            results['tables'][proj_sheet] = df.head(50).to_dict('records')
            results['insights'].append(f"Projection/time series detected in '{proj_sheet}'.")
            # simple linear forecast for next 2 periods if numeric
            try:
                df2 = df.reset_index(drop=True)
                df2['idx'] = range(len(df2))
                model = LinearRegression()
                y = pd.to_numeric(df2[series_name], errors='coerce').fillna(0).values
                if len(y) >= 3:
                    model.fit(df2[['idx']], y)
                    future_idx = pd.DataFrame({'idx': [len(y), len(y)+1]})
                    preds = model.predict(future_idx)
                    results['metrics']['forecast_next_two'] = [float(preds[0]), float(preds[1])]
            except Exception:
                pass

    # 3) Balance sheet / ratios
    bs_sheet = None
    for candidate in ['balance_sheet_monthly', 'balance_sheet', 'balancesheet_monthly']:
        if candidate in keys:
            bs_sheet = keys[candidate]
            break
    if bs_sheet:
        df = dfs[bs_sheet].copy()
        # try to compute current ratio / quick ratio if columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # aggregate by date if present
        if 'Date' in df.columns:
            agg = df.groupby('Date').sum(numeric_only=True)
            if not agg.empty:
                latest = agg.iloc[-1]
                cur_assets = latest.get('Total Current Assets', None)
                cur_liab = latest.get('Total Liabilities', None)
                inv = latest.get('Inventory', 0)
                if cur_assets is not None and cur_liab is not None and cur_liab != 0:
                    cr = float(cur_assets / cur_liab)
                    qr = float((cur_assets - inv) / cur_liab) if cur_liab else None
                    results['metrics']['current_ratio'] = cr
                    results['metrics']['quick_ratio'] = qr
                    results['insights'].append(f"Balance sheet ratios computed from '{bs_sheet}'.")
                results['tables'][bs_sheet] = agg.tail(12).reset_index().to_dict('records')
        else:
            results['tables'][bs_sheet] = df.head(10).to_dict('records')

    # 4) Debt schedule -> near-term maturities
    debt_sheet = None
    for candidate in ['debt_schedule', 'debt', 'liabilities']:
        if candidate in keys:
            debt_sheet = keys[candidate]
            break
    if debt_sheet:
        df = dfs[debt_sheet].copy()
        if 'Maturity' in df.columns:
            try:
                df['Maturity'] = pd.to_datetime(df['Maturity'], errors='coerce')
                near = df[df['Maturity'] <= (pd.Timestamp.now() + pd.Timedelta(days=365))]
                results['tables'][debt_sheet] = df.head(50).to_dict('records')
                results['metrics']['near_term_debt_count'] = int(len(near))
                results['metrics']['near_term_debt_amount'] = float(pd.to_numeric(near.get('Outstanding', 0)).sum())
                results['insights'].append(f"Found {len(near)} debt items due within 12 months.")
            except Exception:
                results['tables'][debt_sheet] = df.head(10).to_dict('records')
        else:
            results['tables'][debt_sheet] = df.head(10).to_dict('records')

    # 5) AR/AP -> aging buckets
    ar_sheet = None
    ap_sheet = None
    for candidate in ['ar_aging', 'accounts_receivable', 'receivables']:
        if candidate in keys and ar_sheet is None:
            ar_sheet = keys[candidate]
    for candidate in ['ap_aging', 'accounts_payable', 'payables']:
        if candidate in keys and ap_sheet is None:
            ap_sheet = keys[candidate]
    if ar_sheet:
        df = dfs[ar_sheet].copy()
        if 'Amount' in df.columns:
            results['metrics']['ar_total'] = float(pd.to_numeric(df['Amount'], errors='coerce').fillna(0).sum())
            if 'Days Past Due' in df.columns:
                overdue = df[df['Days Past Due'] > 30]['Amount'].sum()
                results['metrics']['ar_overdue'] = float(overdue)
            results['tables'][ar_sheet] = df.head(50).to_dict('records')
            results['insights'].append(f"AR sheet '{ar_sheet}' processed.")
    if ap_sheet:
        df = dfs[ap_sheet].copy()
        if 'Amount' in df.columns:
            results['metrics']['ap_total'] = float(pd.to_numeric(df['Amount'], errors='coerce').fillna(0).sum())
            results['tables'][ap_sheet] = df.head(50).to_dict('records')
            results['insights'].append(f"AP sheet '{ap_sheet}' processed.")
    if ar_sheet and ap_sheet:
        ar_total = results['metrics'].get('ar_total', 0)
        ap_total = results['metrics'].get('ap_total', 0)
        results['metrics']['ar_ap_net'] = float(ar_total - ap_total)

    # 6) Investments
    inv_sheet = None
    for candidate in ['investments', 'investment', 'portfolio']:
        if candidate in keys:
            inv_sheet = keys[candidate]
            break
    if inv_sheet:
        df = dfs[inv_sheet].copy()
        if 'Amount' in df.columns:
            inv_val = float(pd.to_numeric(df['Amount'], errors='coerce').fillna(0).sum())
            results['metrics']['investment_value'] = inv_val
            results['tables'][inv_sheet] = df.head(50).to_dict('records')
            results['insights'].append(f"Investments total: ${inv_val:,.0f}.")

    # Final packaging: return results (no images)
    try:
        return results
    except Exception as e:
        return {'error': f'Failed to build results: {str(e)}'}

# -------------------------
# Chart Generation Tool (factory + inference)
# -------------------------
class ChartGenerationInput(BaseModel):
    chart_spec: Dict[str, Any] = Field(..., description="Chart spec produced by analysis tool.")

@tool("Chart Generation Tool")
def chart_generation_tool(chart_spec: Dict[str, Any]) -> Dict[str, str]:
    """
    Accepts a single chart_spec and returns a dict: {chart_name: base64_png}
    chart_spec = {"name": "...", "type":"line"|"bar"|"pie"|"scatter", "labels": [...], "datasets":[{"label":..,"data":[...]}]}
    """
    try:
        name = chart_spec.get('name', 'chart')
        ctype = chart_spec.get('type', 'line')
        labels = chart_spec.get('labels', [])
        datasets = chart_spec.get('datasets', [])
        title = chart_spec.get('meta', {}).get('title', name.replace('_', ' ').title())

        if isinstance(labels, (pd.Series, pd.Index)):
            labels = labels.tolist()
        elif hasattr(labels, 'tolist') and not isinstance(labels, list):
            labels = labels.tolist()
        elif isinstance(labels, tuple):
            labels = list(labels)
        elif not isinstance(labels, list):
            labels = [labels] if labels else []
        labels = [str(l) for l in labels]

        cleaned_datasets = []
        max_data_len = 0
        for ds in datasets or []:
            data_values = ds.get('data', [])
            if isinstance(data_values, (pd.Series, pd.Index)):
                data_list = data_values.tolist()
            elif hasattr(data_values, 'tolist') and not isinstance(data_values, list):
                data_list = data_values.tolist()
            elif isinstance(data_values, (tuple, set)):
                data_list = list(data_values)
            elif isinstance(data_values, list):
                data_list = data_values
            else:
                data_list = [data_values]
            max_data_len = max(max_data_len, len(data_list))
            cleaned = {**ds}
            cleaned['label'] = str(cleaned.get('label', 'Series'))
            cleaned['data'] = data_list
            cleaned_datasets.append(cleaned)

        if not labels and max_data_len:
            labels = [str(i + 1) for i in range(max_data_len)]
        label_count = len(labels)

        pad_value = float('nan')
        if label_count:
            for ds in cleaned_datasets:
                data_list = ds['data']
                if len(data_list) > label_count:
                    ds['data'] = data_list[:label_count]
                elif len(data_list) < label_count:
                    ds['data'] = data_list + [pad_value] * (label_count - len(data_list))
        elif cleaned_datasets:
            label_count = max(len(ds['data']) for ds in cleaned_datasets)
            labels = [str(i + 1) for i in range(label_count)]
            for ds in cleaned_datasets:
                data_list = ds['data']
                if len(data_list) < label_count:
                    ds['data'] = data_list + [pad_value] * (label_count - len(data_list))

        fig, ax = plt.subplots(figsize=(8,4))
        has_data = False

        if not cleaned_datasets or label_count == 0:
            ax.text(0.5, 0.5, "No chart data available", ha='center', va='center', fontsize=12)
            ax.axis('off')
        elif ctype == 'pie':
            ds = cleaned_datasets[0]
            data_pairs = [(lab, val) for lab, val in zip(labels, ds['data']) if pd.notna(val)]
            if data_pairs:
                pie_labels, pie_data = zip(*data_pairs)
                ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
                has_data = True
            else:
                ax.text(0.5, 0.5, "No chart data available", ha='center', va='center', fontsize=12)
                ax.axis('off')
        elif ctype == 'bar':
            x = list(range(len(labels)))
            width = 0.8 / max(1, len(cleaned_datasets))
            for i, ds in enumerate(cleaned_datasets):
                values = [0 if pd.isna(v) else v for v in ds['data']]
                ax.bar([xi + i * width for xi in x], values, width=width, label=ds['label'])
            ax.set_xticks([i + width * (len(cleaned_datasets) - 1) / 2 for i in x])
            ax.set_xticklabels(labels)
            has_data = True
        elif ctype == 'scatter':
            for ds in cleaned_datasets:
                ax.scatter(labels, ds['data'], label=ds['label'])
            has_data = True
        else:
            for ds in cleaned_datasets:
                ax.plot(labels, ds['data'], label=ds['label'])
            has_data = True

        ax.set_title(title)
        if has_data and ctype != 'pie':
            ax.legend()
            ax.grid(alpha=0.2)

        img_b64 = to_base64_png(fig)
        return {name: img_b64}
    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}

# -------------------------
# Report Generation Tool (Jinja2 templating)
# -------------------------
class ReportGenerationInput(BaseModel):
    analysis: Union[str, dict] = Field(..., description="Analysis dict or JSON string.")
    objective: str = Field(..., description="Objective string.")

@tool("Report Generation Tool")
def report_generation_tool(analysis: Union[str, dict], objective: str) -> str:
    """
    Generate a Markdown report using a Jinja2 template.
    The analysis dict should contain:
      - insights (list)
      - metrics (dict)
      - tables (dict of lists of records)
      - chart_specs (list)
    The report generator will call chart_generation_tool for each chart_spec and embed base64 images inline.
    """
    # normalize
    if isinstance(analysis, str):
        try:
            analysis_dict = json.loads(analysis)
        except Exception:
            return "# Error\nInvalid analysis JSON provided."
    else:
        analysis_dict = analysis

    if not isinstance(analysis_dict, dict):
        return "# Error\nInvalid analysis format."

    # Basic template with loops — easy to extend
    template_md = """
# Liquidity Risk Analysis Report

**Objective**: {{ objective }}

## Executive Summary
{% if insights %}
{% for i in insights %}
- {{ i }}
{% endfor %}
{% else %}
_No insights available._
{% endif %}

## Key Metrics
| Metric | Value |
|---:|:---|
{% for k, v in metrics.items() %}
| {{ k.replace('_',' ').title() }} | {% if v is number %}{{ "{:,.2f}".format(v) }}{% else %}{{ v }}{% endif %} |
{% endfor %}

{% if tables %}
{% for name, rows in tables.items() %}
## {{ name.replace('_',' ').title() }}
{% if rows %}
{{ tables[name] | to_markdown }}
{% else %}
_No table data._
{% endif %}
{% endfor %}
{% endif %}

{% if charts %}
## Charts
{% for c in charts %}
### {{ c.name.replace('_',' ').title() }}
![{{ c.name }}](data:image/png;base64,{{ c.img }})
{% endfor %}
{% endif %}

## Risk Assessment & Recommendations
- Review the above metrics and charts for concentration risks, covenant breaches, and near-term debt.
- Run sensitivity and scenario analyses monthly and before any major funding decision.
"""

    # utility: convert tables to markdown
    def to_markdown_filter(obj):
        try:
            df = pd.DataFrame(obj)
            return df.to_markdown(index=False)
        except Exception:
            return ""

    template = Template(template_md)
    template.environment.filters['to_markdown'] = to_markdown_filter

    # Generate charts via chart_generation_tool and collect their base64
    charts_out = []
    for spec in analysis_dict.get('chart_specs', []):
        try:
            chart_result = chart_generation_tool(spec)
            if 'error' in chart_result:
                charts_out.append({"name": spec.get('name', 'chart'), "img": "", "error": chart_result['error']})
            else:
                # chart_result is {name: base64}
                for k, v in chart_result.items():
                    charts_out.append({"name": k, "img": v})
        except Exception:
            charts_out.append({"name": spec.get('name', 'chart'), "img": ""})

    rendered = template.render(
        objective=objective,
        insights=analysis_dict.get('insights', []),
        metrics=analysis_dict.get('metrics', {}),
        tables=analysis_dict.get('tables', {}),
        charts=charts_out
    )

    return rendered
