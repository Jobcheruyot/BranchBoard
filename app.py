
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
st.set_page_config(page_title="Branch Board Visuals", layout="wide")

st.title("Branch Board Visuals — Notebook-driven")
st.markdown("This app reproduces notebook visuals (where possible) and provides dropdown filters. The four internal prep steps are hidden from the user.")

# Sidebar: TOC (excluding import/upload/convert/clean)
sections = ['Clean  Numeric Columns for analysis', 'SALES', 'Global sales Overview', 'Global Net Sales Distribution by Sales Channel', 'Global Net Sales Distribution by SHIFT', 'Night vs Day Shift Sales Ratio — Stores with Night Shifts', 'Global Day vs Night Sales — Only Stores with NIGHT Shift', '2nd-Highest Channel Share', 'Bottom 30 — 2nd Highest Channel', 'Stores Sales Summary', 'OPERATIONS', 'Customer Traffic-Storewise', 'Active Tills During the day', 'Average Customers Served per Till', 'Store Customer Traffic Storewise', 'Customer Traffic-Departmentwise']

st.sidebar.header("Table of Contents")
choice = st.sidebar.radio("Go to section", sections)

st.sidebar.markdown("---")
st.sidebar.header("Filters (apply to visuals)")
# Data loading: attempt to load /mnt/data/performance.csv or .parquet; otherwise create sample
@st.cache_data
def load_data():
    possible_paths = ["/mnt/data/performance.csv", "/mnt/data/performance.parquet", "/mnt/data/performance.xlsx", "/mnt/data/performance.zip"]
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            try:
                if p.endswith(".csv"):
                    df = pd.read_csv(p)
                elif p.endswith(".parquet"):
                    df = pd.read_parquet(p)
                elif p.endswith(".xlsx"):
                    df = pd.read_excel(p)
                elif p.endswith(".zip"):
                    import zipfile, io
                    with zipfile.ZipFile(p) as z:
                        for name in z.namelist():
                            if name.lower().endswith(".csv"):
                                with z.open(name) as f:
                                    df = pd.read_csv(f)
                                    break
                if df is not None:
                    break
            except Exception as e:
                print("Failed reading", p, e)
    if df is None:
        # build realistic sample matching expected schema mentioned in the notebook/context
        rng = np.random.default_rng(2025)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
        stores = [f"Store {i}" for i in range(1,7)]
        items = [f"SKU{i:04d}" for i in range(1,201)]
        suppliers = [f"Supplier {i}" for i in range(1,11)]
        rows = []
        for s in stores:
            for d in dates:
                for _ in range(8): # random SKUs per month-store
                    item = rng.choice(items)
                    qty = int(rng.integers(1, 200))
                    avg_cp = float(rng.integers(50, 1000))
                    avg_sp = avg_cp * (1 + rng.random()/2 + 0.05)
                    sales = qty * avg_sp
                    cogs = qty * avg_cp
                    gp = sales - cogs
                    rows.append({
                        'date': d,
                        'store_name': s,
                        'item_code': item,
                        'item_name': f"Item {item}",
                        'department': rng.choice(['Grocery','Produce','Dairy','Frozen','Household']),
                        'category': rng.choice(['A','B','C']),
                        'quantity': qty,
                        'avg_cp_pre_vat': avg_cp,
                        'avg_sp_pre_vat': avg_sp,
                        'cogs': cogs,
                        'sales_pre_vat': sales,
                        'total_vat': sales*0.16,
                        'gross_profit': gp,
                        'supplier': rng.choice(suppliers)
                    })
        df = pd.DataFrame(rows)
    # covert date-like columns to datetime silently
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    # clean numeric columns heuristically: remove commas and convert
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20).tolist()
            if any(',' in s for s in sample):
                try:
                    df[col] = df[col].str.replace(',','').astype(float)
                except Exception:
                    pass
    return df

df = load_data()

# Available filter widgets
stores = sorted(df['store_name'].dropna().unique().tolist()) if 'store_name' in df.columns else []
departments = sorted(df['department'].dropna().unique().tolist()) if 'department' in df.columns else []
suppliers = sorted(df['supplier'].dropna().unique().tolist()) if 'supplier' in df.columns else []

selected_store = st.sidebar.selectbox("Store (All)", options=["All"] + stores)
selected_dept = st.sidebar.selectbox("Department (All)", options=["All"] + departments)
selected_supplier = st.sidebar.selectbox("Supplier (All)", options=["All"] + suppliers)

# date filter if present
date_cols = [c for c in df.columns if 'date' in c.lower()]
if date_cols:
    date_col = date_cols[0]
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    selected_dates = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_col = None
    selected_dates = None

# Apply filters to df copy for visuals
dff = df.copy()
if selected_store != "All":
    dff = dff[dff['store_name'] == selected_store]
if selected_dept != "All":
    dff = dff[dff['department'] == selected_dept]
if selected_supplier != "All":
    dff = dff[dff['supplier'] == selected_supplier]
if date_col and isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start, end = selected_dates
    dff = dff[(dff[date_col] >= pd.to_datetime(start)) & (dff[date_col] <= pd.to_datetime(end))]

st.sidebar.markdown("---")
st.sidebar.caption("Data rows: " + str(len(dff)))

# Helper: display top metrics
def show_kpis(df):
    total_sales = df['sales_pre_vat'].sum() if 'sales_pre_vat' in df.columns else df.get('sales', pd.Series(dtype=float)).sum()
    total_cogs = df['cogs'].sum() if 'cogs' in df.columns else 0.0
    gross_profit = df['gross_profit'].sum() if 'gross_profit' in df.columns else total_sales - total_cogs
    avg_margin = (gross_profit / total_sales) if total_sales!=0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"KSh {total_sales:,.0f}")
    col2.metric("Gross Profit", f"KSh {gross_profit:,.0f}")
    col3.metric("Avg Margin", f"{avg_margin:.2%}")

# Visual sections (attempt to mirror notebook visuals)
def sales_overview():
    st.header("Sales Overview")
    show_kpis(dff)
    if date_col:
        agg = dff.groupby(date_col)['sales_pre_vat'].sum().rename("sales").reset_index()
        st.line_chart(agg.set_index(date_col))
        st.dataframe(agg.tail(50))
    else:
        st.info("No date column found — showing aggregated sales by store.")
        if 'sales_pre_vat' in dff.columns and 'store_name' in dff.columns:
            st.bar_chart(dff.groupby('store_name')['sales_pre_vat'].sum())

def top_departments():
    st.header("Top Departments")
    if 'department' in dff.columns and 'sales_pre_vat' in dff.columns:
        dept = dff.groupby('department')['sales_pre_vat'].sum().sort_values(ascending=False).reset_index()
        st.bar_chart(dept.set_index('department'))
        st.dataframe(dept)
    else:
        st.info("Department or sales column missing.")

def store_performance():
    st.header("Store Performance")
    if 'store_name' in dff.columns and 'sales_pre_vat' in dff.columns:
        sp = dff.groupby('store_name')['sales_pre_vat'].sum().sort_values(ascending=False).reset_index()
        st.bar_chart(sp.set_index('store_name'))
        st.dataframe(sp)
    else:
        st.info("Store or sales column missing.")

def top_skus():
    st.header("Top SKUs by Sales")
    if 'item_code' in dff.columns and 'sales_pre_vat' in dff.columns:
        sku = dff.groupby(['item_code','item_name'])['sales_pre_vat'].sum().sort_values(ascending=False).reset_index().head(20)
        st.table(sku)
    else:
        st.info("SKU or sales column missing.")

def inventory_summary():
    st.header("Inventory Summary (approx)")
    if 'quantity' in dff.columns:
        inv = dff.groupby('item_code')['quantity'].sum().sort_values(ascending=False).reset_index().head(50)
        st.table(inv)
    else:
        st.info("Quantity column missing.")

def promotions_analysis():
    st.header("Promotions Analysis (simulated)")
    st.info("Original notebook promotions details might be manual; here we show uplift simulation by department.")
    if 'department' in dff.columns and 'sales_pre_vat' in dff.columns:
        dept = dff.groupby('department')['sales_pre_vat'].sum().reset_index()
        dept['sim_uplift_pct'] = (np.random.RandomState(1).randint(5,30, size=len(dept)))
        st.bar_chart(dept.set_index('department')['sim_uplift_pct'])
        st.dataframe(dept)
    else:
        st.info("Required columns missing.")

# Map section names to functions; unknown sections render placeholder
func_map = {
    'Sales Overview': sales_overview,
    'Top Departments': top_departments,
    'Store Performance': store_performance,
    'Top SKUs': top_skus,
    'Inventory Summary': inventory_summary,
    'Promotions Analysis': promotions_analysis
}

# Render chosen section
if choice in func_map:
    func_map[choice]()
else:
    st.header(choice)
    st.info("No prebuilt visualization for this section. Add custom visuals by editing app.py")
