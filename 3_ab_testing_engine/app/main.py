import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.stats_utils import perform_z_test, bayesian_conversion_probability

st.set_page_config(page_title="A/B Testing Simulator", layout="wide")

st.title("🧪 A/B Testing & Experimentation Platform")
st.markdown("""
This dashboard simulates and analyzes A/B test results. 
It uses both Frequentist (Z-test) and Bayesian methods to determine statistical significance.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Group A (Control)")
    visitors_a = st.number_input("Visitors A", value=1000, step=100)
    conversions_a = st.number_input("Conversions A", value=100, step=10)
    rate_a = conversions_a / visitors_a if visitors_a > 0 else 0
    st.metric("Conversion Rate", f"{rate_a:.2%}")

with col2:
    st.header("Group B (Variant)")
    visitors_b = st.number_input("Visitors B", value=1000, step=100)
    conversions_b = st.number_input("Conversions B", value=120, step=10)
    rate_b = conversions_b / visitors_b if visitors_b > 0 else 0
    st.metric("Conversion Rate", f"{rate_b:.2%}")

st.divider()

# Analysis
if st.button("Run Statistical Analysis"):
    z_stat, p_val = perform_z_test(rate_a, visitors_a, rate_b, visitors_b)
    prob_b_better = bayesian_conversion_probability(conversions_a, visitors_a, conversions_b, visitors_b)
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.subheader("Frequentist Result")
        st.write(f"**P-Value:** {p_val:.4f}")
        if p_val < 0.05:
            st.success("Statistically Significant")
        else:
            st.warning("Not Significant")
            
    with res_col2:
        st.subheader("Bayesian Result")
        st.write(f"**Prob(B > A):** {prob_b_better:.2%}")
        if prob_b_better > 0.95:
            st.success("Strong Evidence for B")
        elif prob_b_better < 0.05:
            st.error("Strong Evidence for A")
        else:
            st.info("Inconclusive")
            
    with res_col3:
        lift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0
        st.subheader("Observed Lift")
        st.metric("Relative Lift", f"{lift:.2%}")

    # Visualization
    st.subheader("Conversion Distributions (Bayesian)")
    x = np.linspace(0, max(rate_a, rate_b) * 2, 500)
    y_a = stats.beta.pdf(x, conversions_a + 1, visitors_a - conversions_a + 1)
    y_b = stats.beta.pdf(x, conversions_b + 1, visitors_b - conversions_b + 1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_a, fill='tozeroy', name='Group A'))
    fig.add_trace(go.Scatter(x=x, y=y_b, fill='tozeroy', name='Group B'))
    fig.update_layout(title="Posterior Distributions of Conversion Rates", xaxis_title="Conversion Rate", yaxis_title="Probability Density")
    st.plotly_chart(fig, use_container_width=True)
