import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Marketing Customer Segmentation Dashboard")
st.markdown("Segment customers using Machine Learning (KMeans Clustering)")


@st.cache_data
def load_data():
    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
    return df

df = load_data()


df.columns = df.columns.str.strip()
df = df.dropna(subset=['Income'])


features = ['Income', 'Recency']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


st.sidebar.header("⚙️ Controls")

k = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)
run = st.sidebar.button("Run Clustering")


if run:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # SAVE MODEL FOR PREDICTION (IMPORTANT FIX)
    st.session_state["model"] = kmeans
    st.session_state["scaler"] = scaler


    st.subheader("📌 Dashboard Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Clusters", k)
    col3.metric("Features Used", len(features))

    
    st.subheader("📄 Clustered Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    
    st.subheader("📈 Customer Segments")

    fig, ax = plt.subplots()

    ax.scatter(
        df['Income'],
        df['Recency'],
        c=df['Cluster'],
        cmap='viridis',
        s=50,
        alpha=0.7
    )

    ax.set_xlabel("Income")
    ax.set_ylabel("Recency")
    ax.set_title("Customer Segmentation (KMeans)")

    # Centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        marker='X',
        s=200,
        label='Centroids'
    )

    ax.legend()

    st.pyplot(fig)


st.subheader("🔮 Predict Customer Segment")

income = st.number_input("Enter Income", min_value=0.0)
recency = st.number_input("Enter Recency", min_value=0.0)

if st.button("Predict"):

    if "model" not in st.session_state:
        st.error("⚠️ Please run clustering first!")
    else:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]

        new_data = scaler.transform([[income, recency]])
        cluster = model.predict(new_data)

        st.success(f"✅ Customer belongs to Cluster {cluster[0]}")