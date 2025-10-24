
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
st.set_page_config(page_title="Branch Board Visuals (v4)", layout="wide")

st.title("Branch Board Visuals — Upload CSV (v4)")
st.markdown("""
Upload your **CSV** file using the uploader below. The app will:
- parse date-like columns to `datetime`
- clean numeric columns (remove commas, convert to numeric)
- cache the cleaned dataset for use across visuals
The internal prep steps are hidden from the Table of Contents.
""")

# Sidebar: Table of Contents (exclude the 4 prep sections)
# These are derived from the notebook headings but the four internal steps are excluded.
sections = ['Sales Overview', 'Top Departments', 'Store Performance', 'Top SKUs', 'Inventory Summary', 'Promotions Analysis']
st.sidebar.header("Table of Contents")
choice = st.sidebar.radio("Go to section", sections)

st.sidebar.markdown("---")
st.sidebar.header("Data Controls")

# File uploader (CSV only)
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], help="Upload a CSV exported from your system (performance.csv).")

# Use session_state to store the dataframe so it persists across reruns
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.cleaned = False

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Convert date-like columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20).tolist()
            # heuristic: look for date patterns like yyyy-mm or / or -
            if any('/' in s or '-' in s or s.count('-')==2 or s.count('/')==2 for s in sample):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                except Exception:
                    pass
    # Also try converting columns with 'date' in name
    for col in df.columns:
        if 'date' in col.lower() and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    # Clean numeric columns: remove commas, % and convert to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:
            # remove commas and percent signs for conversion
            try:
                no_commas = df[col].str.replace(',', '').str.replace('%','').str.strip()
                numeric = pd.to_numeric(no_commas, errors='coerce')
                # if a good portion converts to numbers, keep it
                if numeric.notna().sum() >= len(df) * 0.3:
                    df[col] = numeric
            except Exception:
                pass
    return df

# Load CSV if uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_dataframe(df)
        st.session_state.df = df
        st.session_state.cleaned = True
        st.sidebar.success("CSV uploaded and cleaned ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
else:
    if st.session_state.df is None:
        st.sidebar.info("No CSV uploaded. Upload a CSV to enable visuals. (App can work with sample data if you choose 'Use sample data' below.)")

# Option to use sample data if desired
use_sample = st.sidebar.checkbox("Use sample data (if no CSV)", value=False)
if st.session_state.df is None and use_sample:
    # build sample similar to expected schema
    rng = np.random.default_rng(2025)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
    stores = [f"Store {i}" for i in range(1,7)]
    items = [f"SKU{i:04d}" for i in range(1,201)]
    suppliers = [f"Supplier {i}" for i in range(1,11)]
    rows = []
    for s in stores:
        for d in dates:
            for _ in range(8):
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
    st.session_state.df = pd.DataFrame(rows)
    st.session_state.cleaned = True
    st.sidebar.success("Sample data loaded")

# If data is ready, show filters and visuals
if st.session_state.df is not None and st.session_state.cleaned:
    df = st.session_state.df.copy()

    # Identify common column names with fallback mapping
    col_map = {c.lower(): c for c in df.columns}
    def col(name):
        # return column matching name (case-insensitive) or None
        return col_map.get(name.lower())

    date_cols = [c for c in df.columns if 'date' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    date_col = date_cols[0] if date_cols else None

    store_col = None
    for candidate in ['store_name','store','storename']:
        if candidate in col_map:
            store_col = col_map[candidate]
            break
    if store_col is None:
        # try fuzzy: any column containing 'store'
        for c in df.columns:
            if 'store' in c.lower():
                store_col = c; break

    dept_col = None
    for candidate in ['department','dept']:
        if candidate in col_map:
            dept_col = col_map[candidate]; break
    if dept_col is None:
        for c in df.columns:
            if 'depart' in c.lower():
                dept_col = c; break

    supplier_col = None
    for c in df.columns:
        if 'supplier' in c.lower():
            supplier_col = c; break

    sales_col = None
    for candidate in ['sales_pre_vat','sales','sales_prevat','sales_pre']:
        if candidate in col_map:
            sales_col = col_map[candidate]; break
    if sales_col is None:
        for c in df.columns:
            if 'sales' == c.lower() or 'sales' in c.lower():
                sales_col = c; break

    # Sidebar filters based on detected columns
    stores = sorted(df[store_col].dropna().unique().tolist()) if store_col else []
    departments = sorted(df[dept_col].dropna().unique().tolist()) if dept_col else []
    suppliers = sorted(df[supplier_col].dropna().unique().tolist()) if supplier_col else []

    selected_store = st.sidebar.selectbox("Store (All)", options=["All"] + stores)
    selected_dept = st.sidebar.selectbox("Department (All)", options=["All"] + departments)
    selected_supplier = st.sidebar.selectbox("Supplier (All)", options=["All"] + suppliers)

    if date_col:
        min_date = pd.to_datetime(df[date_col].min()).date()
        max_date = pd.to_datetime(df[date_col].max()).date()
        selected_dates = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        selected_dates = None

    # Apply filters
    dff = df.copy()
    if store_col and selected_store != "All":
        dff = dff[dff[store_col] == selected_store]
    if dept_col and selected_dept != "All":
        dff = dff[dff[dept_col] == selected_dept]
    if supplier_col and selected_supplier != "All":
        dff = dff[dff[supplier_col] == selected_supplier]
    if date_col and isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
        start, end = selected_dates
        dff = dff[(pd.to_datetime(dff[date_col]).dt.date >= start) & (pd.to_datetime(dff[date_col]).dt.date <= end)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Rows after filter: {len(dff):,}")

    # KPI display
    def show_kpis(df_local):
        total_sales = df_local[sales_col].sum() if sales_col and sales_col in df_local.columns else df_local.select_dtypes(include='number').sum().sum()
        total_cogs = df_local[col('cogs')].sum() if col('cogs') in df_local.columns else 0.0
        gross_profit = df_local[col('gross_profit')].sum() if col('gross_profit') in df_local.columns else (total_sales - total_cogs)
        avg_margin = (gross_profit / total_sales) if total_sales!=0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sales", f"KSh {total_sales:,.0f}")
        c2.metric("Gross Profit", f"KSh {gross_profit:,.0f}")
        c3.metric("Avg Margin", f"{avg_margin:.2%}")

    # Visuals
    def sales_overview():
        st.header("Sales Overview")
        show_kpis(dff)
        if date_col and sales_col in dff.columns:
            agg = dff.groupby(pd.Grouper(key=date_col, freq='M'))[sales_col].sum().reset_index()
            st.line_chart(agg.set_index(date_col))
            st.dataframe(agg.tail(50))
        elif sales_col in dff.columns and store_col:
            st.bar_chart(dff.groupby(store_col)[sales_col].sum())

    def top_departments():
        st.header("Top Departments")
        if dept_col and sales_col in dff.columns:
            dept = dff.groupby(dept_col)[sales_col].sum().sort_values(ascending=False).reset_index()
            st.bar_chart(dept.set_index(dept_col))
            st.dataframe(dept)
        else:
            st.info("Department or Sales column missing.")

    def store_performance():
        st.header("Store Performance")
        if store_col and sales_col in dff.columns:
            sp = dff.groupby(store_col)[sales_col].sum().sort_values(ascending=False).reset_index()
            st.bar_chart(sp.set_index(store_col))
            st.dataframe(sp)
        else:
            st.info("Store or Sales column missing.")

    def top_skus():
        st.header("Top SKUs by Sales")
        if 'item_code' in dff.columns and sales_col in dff.columns:
            sku = dff.groupby(['item_code','item_name'])[sales_col].sum().sort_values(ascending=False).reset_index().head(20)
            st.table(sku)
        else:
            st.info("item_code or sales column missing.")

    def inventory_summary():
        st.header("Inventory Summary (approx)")
        if 'quantity' in dff.columns:
            inv = dff.groupby('item_code')['quantity'].sum().sort_values(ascending=False).reset_index().head(50)
            st.table(inv)
        else:
            st.info("Quantity column missing.")

    def promotions_analysis():
        st.header("Promotions Analysis (simulated)")
        st.info("This section simulates uplift by department (replace with real promo data if available).")
        if dept_col and sales_col in dff.columns:
            dept = dff.groupby(dept_col)[sales_col].sum().reset_index()
            dept['sim_uplift_pct'] = (np.random.RandomState(1).randint(5,30, size=len(dept)))
            st.bar_chart(dept.set_index(dept_col)['sim_uplift_pct'])
            st.dataframe(dept)
        else:
            st.info("Required columns missing.")

    func_map = {
        'Sales Overview': sales_overview,
        'Top Departments': top_departments,
        'Store Performance': store_performance,
        'Top SKUs': top_skus,
        'Inventory Summary': inventory_summary,
        'Promotions Analysis': promotions_analysis
    }

    if choice in func_map:
        func_map[choice]()
    else:
        st.header(choice)
        st.info("No visualization defined for this section.")

    # Data preview and download
    with st.expander("Preview cleaned data (first 200 rows)"):
        st.dataframe(dff.head(200))

    csv_exp = dff.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", data=csv_exp, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.info("Upload a CSV (left sidebar) to enable visuals, or check 'Use sample data' to view examples.")
