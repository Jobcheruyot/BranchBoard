
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Branch Board Visuals", layout="wide")

st.title("Branch Board Visuals")
st.markdown("Auto-generated Streamlit app based on the notebook's Table of Contents (excluded items starting with '*').")

# Table of Contents (sidebar)
sections = ['Import Libraries', 'Upload Data set', 'Convert Dates to Datetime', 'Clean  Numeric Columns for analysis', 'SALES', 'Global sales Overview', 'Global Net Sales Distribution by Sales Channel', 'Global Net Sales Distribution by SHIFT', 'Night vs Day Shift Sales Ratio — Stores with Night Shifts', 'Global Day vs Night Sales — Only Stores with NIGHT Shift', '2nd-Highest Channel Share', 'Bottom 30 — 2nd Highest Channel', 'Stores Sales Summary', 'OPERATIONS', 'Customer Traffic-Storewise', 'Active Tills During the day', 'Average Customers Served per Till', 'Store Customer Traffic Storewise', 'Customer Traffic-Departmentwise']

st.sidebar.header("Table of Contents")
choice = st.sidebar.radio("Go to section", sections)

# Generate sample data (replace with your data source)
@st.cache_data
def make_sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
    stores = [f"Store {{i}}" for i in range(1,7)]
    df = pd.DataFrame({
        'date': np.tile(dates, len(stores)),
        'store': np.repeat(stores, len(dates)),
        'sales': rng.integers(10000, 100000, size=len(dates)*len(stores)),
        'cogs': rng.integers(5000, 60000, size=len(dates)*len(stores)),
        'units': rng.integers(50, 2000, size=len(dates)*len(stores))
    })
    df['gross_profit'] = df['sales'] - df['cogs']
    df['margin'] = df['gross_profit'] / df['sales']
    return df

df = make_sample_data()

# Section rendering functions
def sales_overview():
    st.header("Sales Overview")
    monthly = df.groupby('date')['sales'].sum().reset_index()
    st.line_chart(monthly.set_index('date'))
    st.dataframe(monthly)

def top_departments():
    st.header("Top Departments (simulated)")
    depts = ['Grocery','Produce','Dairy','Frozen','Household']
    vals = np.random.RandomState(1).randint(100000, 500000, len(depts))
    dept_df = pd.DataFrame({'department': depts, 'sales': vals}).sort_values('sales', ascending=False)
    st.bar_chart(dept_df.set_index('department'))
    st.table(dept_df)

def store_performance():
    st.header("Store Performance")
    pivot = df.pivot_table(values='sales', index='store', aggfunc='sum').sort_values('sales', ascending=False)
    st.bar_chart(pivot)
    st.dataframe(pivot)

def inventory_summary():
    st.header("Inventory Summary (simulated)")
    st.write("Placeholder inventory KPIs — replace with your inventory dataset.")
    inv = pd.DataFrame({
        'item': [f'Item {{i}}' for i in range(1,11)],
        'on_hand': np.random.randint(0,500,10),
        'weeks_cover': np.round(np.random.rand(10)*12,1)
    })
    st.table(inv)

def customer_metrics():
    st.header("Customer Metrics")
    cust = pd.DataFrame({
        'metric': ['New Customers','Repeat Rate','Avg Order Value'],
        'value': [np.random.randint(500,2000), f"{np.round(np.random.rand()*100,1)}%", "KSh " + str(np.random.randint(800,5000))]
    })
    st.table(cust)

def promotions_analysis():
    st.header("Promotions Analysis (simulated)")
    promos = pd.DataFrame({
        'promo': ['Promo A','Promo B','Promo C'],
        'uplift_pct': [12.4, 5.6, 20.1],
        'cost': [15000, 8000, 23000]
    })
    st.bar_chart(promos.set_index('promo')['uplift_pct'])
    st.table(promos)

# Mapping choices to functions (if a TOC heading doesn't match a function, show a placeholder)
func_map = {
    'Sales Overview': sales_overview,
    'Top Departments': top_departments,
    'Store Performance': store_performance,
    'Inventory Summary': inventory_summary,
    'Customer Metrics': customer_metrics,
    'Promotions Analysis': promotions_analysis
}

# Render the chosen section; if not in map, show a simple placeholder
if choice in func_map:
    func_map[choice]()
else:
    st.header(choice)
    st.info("No prebuilt visualization for this section. Please replace the sample data with your real dataset and add visuals as needed.")
