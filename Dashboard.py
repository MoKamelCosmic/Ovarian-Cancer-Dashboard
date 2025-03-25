import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout
st.set_page_config(page_title='Ovarian Cancer Classification Dashboard', layout='wide')

# Load dataset clearly
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

df = load_data()

# Sidebar Navigation
st.sidebar.title('📌 Dashboard Navigation')
page = st.sidebar.radio("Choose page:", ["📊 EDA Overview", "🛠️ Handcrafted Model", "🔬 ABMIL/Otsu Model"])

# --- 📊 EDA Overview Page ---
if page == "📊 EDA Overview":
    st.title('📊 Dataset Overview & Exploratory Data Analysis')

    # Dataset Summary
    st.subheader('🔍 Dataset Summary')
    col1, col2 = st.columns(2)
    col1.metric("Total Samples", len(df))
    col2.metric("Unique Subtypes", df['label'].nunique())
    col1.metric("TMA Images", df['is_tma'].sum())
    col2.metric("WSI Images", (~df['is_tma']).sum())

    # Class Distribution
    st.subheader('📈 Class Distribution')
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x='label', data=df, palette='Blues_r', order=df['label'].value_counts().index, ax=ax)
    st.pyplot(fig)

    # Image Size Distribution
    st.subheader('📐 Image Size Distribution')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    sns.histplot(df['image_width'], bins=30, kde=True, ax=ax1, color='navy')
    ax1.set_title('Width Distribution')
    sns.histplot(df['image_height'], bins=30, kde=True, ax=ax2, color='darkorange')
    ax2.set_title('Height Distribution')
    st.pyplot(fig)

    # TMA vs WSI Analysis
    st.subheader('🧬 TMA vs WSI Analysis')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x='label', hue='is_tma', data=df, palette='Set2', ax=ax)
    ax.legend(title='Is TMA', labels=['WSI','TMA'])
    st.pyplot(fig)

    # Aspect Ratios
    st.subheader('🔲 Aspect Ratios')
    df['aspect_ratio'] = df['image_width'] / df['image_height']
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x='label', y='aspect_ratio', data=df, palette='Set3', ax=ax)
    st.pyplot(fig)

    # Outlier Detection
    st.subheader('🚨 Outlier Detection')
    df['area'] = df['image_width'] * df['image_height']
    Q1, Q3 = df['area'].quantile(0.25), df['area'].quantile(0.75)
    IQR = Q3 - Q1
    area_outliers = df[(df['area'] < (Q1 - 1.5 * IQR)) | (df['area'] > (Q3 + 1.5 * IQR))]
    st.write(f"Area Outliers (IQR): **{len(area_outliers)}**")
    st.table(area_outliers[['image_id', 'label', 'area']])

    # Class Imbalance
    st.subheader('⚖️ Class Imbalance')
    class_counts = df['label'].value_counts()
    fig, ax = plt.subplots(figsize=(7,5))
    ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    st.pyplot(fig)

# --- 🛠️ Handcrafted Model Analysis ---
elif page == "🛠️ Handcrafted Model":
    st.title('🛠️ Handcrafted Pipeline Analysis')

    # Overall Metrics
    st.subheader('📌 Overall Metrics')
    st.write('Accuracy: **59%**, Balanced Accuracy: **58.74%**')

    # Filters
    filter_option = st.sidebar.radio("Filtering method (Handcrafted):", ["By Model", "By Class"])
    classwise_df = pd.DataFrame({
        'Class': ['CC', 'EC', 'HGSC', 'LGSC', 'MC'],
        'Precision': [0.61, 0.43, 0.61, 0.67, 0.83],
        'Recall': [0.65, 0.42, 0.65, 0.67, 0.56]
    })

    if filter_option == "By Model":
        metric = st.radio("Metric:", ["Precision", "Recall"])
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='Class', y=metric, data=classwise_df, palette='viridis', ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

    else:
        selected_class = st.selectbox("Class:", classwise_df['Class'])
        metrics = classwise_df[classwise_df['Class'] == selected_class].melt(id_vars='Class')
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x='variable', y='value', data=metrics, palette='coolwarm', ax=ax)
        ax.set_ylim(0,1)
        ax.set_title(f'{selected_class} Metrics')
        st.pyplot(fig)

# --- 🔬 ABMIL/Otsu Model Analysis ---
elif page == "🔬 ABMIL/Otsu Model":
    st.title('🔬 ABMIL/Otsu Pipeline Analysis')

    # Overall Metrics
    st.subheader('📌 Overall Metrics')
    st.write('Accuracy: **82%**, Balanced Accuracy: **75.62%**')

    # Filters
    filter_option = st.sidebar.radio("Filtering method (ABMIL/Otsu):", ["By Model", "By Class"])
    classwise_df = pd.DataFrame({
        'Class': ['CC', 'EC', 'HGSC', 'LGSC', 'MC'],
        'Precision': [0.82, 0.70, 0.83, 1.0, 1.0],
        'Recall': [0.88, 0.74, 0.92, 0.50, 0.75]
    })

    if filter_option == "By Model":
        metric = st.radio("Metric:", ["Precision", "Recall"])
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='Class', y=metric, data=classwise_df, palette='magma', ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

    else:
        selected_class = st.selectbox("Class:", classwise_df['Class'])
        metrics = classwise_df[classwise_df['Class'] == selected_class].melt(id_vars='Class')
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x='variable', y='value', data=metrics, palette='coolwarm', ax=ax)
        ax.set_ylim(0,1)
        ax.set_title(f'{selected_class} Metrics')
        st.pyplot(fig)

# --- Questionnaire Link clearly ---
st.sidebar.markdown("---")
st.sidebar.markdown("[📝 **Usability Questionnaire (Microsoft Forms)**](<https://forms.office.com/Pages/ResponsePage.aspx?id=8l9CbGVo30Kk245q9jSBPQ8IjV1C0JZPvUuSGlQO22hUQlVPT0s0WUxZQThUNkcyVkI1TDlLSzVCMC4u>)")
