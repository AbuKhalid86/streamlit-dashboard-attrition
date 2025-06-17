import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Set page config
st.set_page_config(
    page_title="Analisis Attrition Jaya Jaya Maju",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("processed_data.csv")
    return data

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
dept_filter = st.sidebar.multiselect(
    "Pilih Departemen",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

age_filter = st.sidebar.slider(
    "Pilih Rentang Usia",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

gender_filter = st.sidebar.multiselect(
    "Pilih Jenis Kelamin",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

# Apply filters
filtered_df = df[
    (df['Department'].isin(dept_filter)) &
    (df['Age'].between(age_filter[0], age_filter[1])) &
    (df['Gender'].isin(gender_filter))
]

# Calculate metrics
total_employees = len(filtered_df)
attrition_rate = filtered_df['Attrition'].mean() * 100
avg_age = filtered_df['Age'].mean()

# Main page
st.title("📊 Dashboard Analisis Attrition Jaya Jaya Maju")
st.markdown("""
    Dashboard ini memberikan analisis komprehensif tentang tingkat attrition karyawan di perusahaan Jaya Jaya Maju.
    Gunakan filter di sidebar untuk menyesuaikan tampilan data.
""")

# Overview Section
st.header("1. Overview Attrition")
col1, col2, col3 = st.columns(3)
col1.metric("Total Karyawan", total_employees)
col2.metric("Tingkat Attrition", f"{attrition_rate:.1f}%")
col3.metric("Rata-rata Usia", f"{avg_age:.1f} Tahun")

# Attrition Distribution
fig1 = px.pie(
    filtered_df, 
    names='Attrition', 
    title='Distribusi Attrition',
    color='Attrition',
    color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
)
st.plotly_chart(fig1, use_container_width=True)

# Demographic Analysis
st.header("2. Analisis Demografis")

tab1, tab2, tab3, tab4 = st.tabs(["Usia", "Jenis Kelamin", "Status Perkawinan", "Pendidikan"])

with tab1:
    fig2 = px.histogram(
        filtered_df, 
        x='Age', 
        color='Attrition',
        nbins=20,
        title='Distribusi Usia berdasarkan Attrition',
        barmode='overlay',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    gender_attrition = filtered_df.groupby(['Gender', 'Attrition']).size().unstack()
    fig3 = px.bar(
        gender_attrition,
        barmode='group',
        title='Attrition berdasarkan Jenis Kelamin',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    fig4 = px.pie(
        filtered_df[filtered_df['Attrition'] == 1],
        names='MaritalStatus',
        title='Status Perkawinan Karyawan yang Attrit'
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    fig5 = px.bar(
        filtered_df.groupby(['Education', 'Attrition']).size().unstack(),
        barmode='group',
        title='Tingkat Pendidikan berdasarkan Attrition',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig5, use_container_width=True)

# Job Analysis
st.header("3. Analisis Pekerjaan")

tab5, tab6, tab7, tab8 = st.tabs(["Departemen", "Peran Pekerjaan", "Tingkat Pekerjaan", "Lama Bekerja"])

with tab5:
    dept_attrition = filtered_df.groupby(['Department', 'Attrition']).size().unstack()
    fig6 = px.bar(
        dept_attrition,
        barmode='group',
        title='Attrition berdasarkan Departemen',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig6, use_container_width=True)

with tab6:
    fig7 = px.treemap(
        filtered_df[filtered_df['Attrition'] == 1],
        path=['Department', 'JobRole'],
        title='Peran Pekerjaan yang Paling Sering Attrit'
    )
    st.plotly_chart(fig7, use_container_width=True)

with tab7:
    fig8 = px.bar(
        filtered_df.groupby(['JobLevel', 'Attrition']).size().unstack(),
        barmode='group',
        title='Attrition berdasarkan Tingkat Pekerjaan',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig8, use_container_width=True)

with tab8:
    fig9 = px.histogram(
        filtered_df,
        x='YearsAtCompany',
        color='Attrition',
        nbins=15,
        title='Lama Bekerja di Perusahaan',
        barmode='overlay',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig9, use_container_width=True)

# Satisfaction Factors
st.header("4. Faktor Kepuasan")

# Prepare correlation matrix
corr_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'Attrition']
corr_matrix = filtered_df[corr_cols].corr()

tab9, tab10, tab11 = st.tabs(["Heatmap Korelasi", "Kepuasan Kerja vs Lingkungan", "Distribusi Kepuasan"])

with tab9:
    fig10 = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Korelasi antara Faktor Kepuasan dan Attrition'
    )
    st.plotly_chart(fig10, use_container_width=True)

with tab10:
    fig11 = px.scatter(
        filtered_df,
        x='JobSatisfaction',
        y='EnvironmentSatisfaction',
        color='Attrition',
        title='Kepuasan Kerja vs Kepuasan Lingkungan',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig11, use_container_width=True)

with tab11:
    satisfaction_col = st.selectbox(
        "Pilih Indikator Kepuasan",
        ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    )
    fig12 = px.box(
        filtered_df,
        x='Attrition',
        y=satisfaction_col,
        color='Attrition',
        title=f'Distribusi {satisfaction_col} berdasarkan Attrition',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig12, use_container_width=True)

# Compensation Analysis
st.header("5. Analisis Kompensasi")

tab12, tab13, tab14 = st.tabs(["Pendapatan Bulanan", "Kenaikan Gaji", "Opsi Saham"])

with tab12:
    fig13 = px.box(
        filtered_df,
        x='Attrition',
        y='MonthlyIncome',
        color='Attrition',
        title='Distribusi Pendapatan Bulanan',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig13, use_container_width=True)

with tab13:
    fig14 = px.box(
        filtered_df,
        x='Attrition',
        y='PercentSalaryHike',
        color='Attrition',
        title='Distribusi Kenaikan Gaji',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig14, use_container_width=True)

with tab14:
    fig15 = px.bar(
        filtered_df.groupby(['StockOptionLevel', 'Attrition']).size().unstack(),
        barmode='group',
        title='Distribusi Opsi Saham',
        color_discrete_map={1: '#FF6B6B', 0: '#4ECDC4'}
    )
    st.plotly_chart(fig15, use_container_width=True)

# SECTION: Feature Importance
st.header("6. 🔍 Feature Importance (Pengaruh Fitur terhadap Attrition)")

# Load feature importance
@st.cache_data
def load_feature_importance():
    return pd.read_csv("feature_importance.csv")

feature_importance_df = load_feature_importance()

# Visualisasi Top 15 Fitur Terpenting
fig16 = px.bar(
    feature_importance_df.sort_values(by="Importance", ascending=False).head(15),
    x="Importance",
    y="Feature",
    orientation='h',
    title='Top 15 Fitur yang Paling Mempengaruhi Attrition',
    color="Importance",
    color_continuous_scale='Tealrose'
)
fig16.update_layout(yaxis=dict(categoryorder='total ascending'))

st.plotly_chart(fig16, use_container_width=True)
