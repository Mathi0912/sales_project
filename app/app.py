import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="Retail Analytics", layout="wide")

st.title("🛒 Retail Demand & Pricing Analytics Dashboard")

# ----------------------------------
# LOAD DATA
# ----------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")

    # ----------------------------------
    

    # ----------------------------------
    # FEATURE SELECTION
    # ----------------------------------
    features_uc1 = [
        'Inventory_Level','Price','Discount','Promotion',
        'Competitor_Pricing','Seasonality','Epidemic'
    ]

    features_uc2 = [
        'Price','Discount','Promotion',
        'Competitor_Pricing','Seasonality','Epidemic'
    ]

    X1 = df[features_uc1]
    X2 = df[features_uc2]



    # ----------------------------------
    # SIDEBAR USE CASE SELECTION
    # ----------------------------------
    use_case = st.sidebar.radio(
        "Select Use Case",
        ("Use Case 1 – Demand Prediction", "Use Case 2 – Pricing Impact")
    )

    # ==========================================================
    # USE CASE 1
    # ==========================================================
    if use_case == "Use Case 1 – Demand Prediction":

        st.header("📦 Use Case 1: Demand Prediction")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Demand", round(df['Demand'].mean(), 2))
        col2.metric("Avg Inventory", round(df['Inventory_Level'].mean(), 2))
        col3.metric("Stock-out Risk",
                    df[df['Demand'] > df['Inventory_Level']].shape[0])
        col4.metric("Prediction Mean",
                    round(df['Predicted_Demand_UC1'].mean(), 2))

        # Actual vs Predicted
        st.subheader("Actual vs Predicted Demand")
        st.line_chart(df[['Demand', 'Predicted_Demand_UC1']].head(100))

        # Inventory vs Demand
        st.subheader("Inventory vs Demand")
        fig1 = px.scatter(df, x='Inventory_Level', y='Demand',
                          title="Inventory vs Demand")
        st.plotly_chart(fig1, use_container_width=True)

        # Promotion impact
        st.subheader("Promotion Impact")
        fig2 = px.box(df, x='Promotion', y='Demand',
                      title="Promotion vs Demand")
        st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # USE CASE 2
    # ==========================================================
    else:

        st.header("💰 Use Case 2: Pricing & Promotion Impact")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Demand (Promo ON)",
                    round(df[df['Promotion']==1]['Demand'].mean(),2))
        col2.metric("Avg Demand (Promo OFF)",
                    round(df[df['Promotion']==0]['Demand'].mean(),2))
        col3.metric("Discount Avg",
                    round(df['Discount'].mean(),2))
        col4.metric("Prediction Mean",
                    round(df['Predicted_Demand_UC2'].mean(),2))

        # Price vs Demand
        st.subheader("Price vs Demand")
        fig3 = px.scatter(df, x='Price', y='Demand',
                          title="Price Sensitivity")
        st.plotly_chart(fig3, use_container_width=True)

        # Discount vs Demand
        st.subheader("Discount vs Demand")
        fig4 = px.scatter(df, x='Discount', y='Demand',
                          title="Discount Impact")
        st.plotly_chart(fig4, use_container_width=True)

        # Actual vs Predicted
        st.subheader("Actual vs Predicted Demand")
        st.line_chart(df[['Demand', 'Predicted_Demand_UC2']].head(100))

else:
    st.info("👈 Upload dataset to begin")
