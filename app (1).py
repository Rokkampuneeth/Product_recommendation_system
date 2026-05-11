"""
P669 — Product Recommendation System
Streamlit Deployment App
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from scipy.cluster.hierarchy import dendrogram, linkage

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P669 — Product Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .metric-card {
        background: linear-gradient(135deg, #1E2A3A, #16213E);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2D3F5A;
        margin: 6px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4FC3F7; }
    .metric-label { font-size: 0.85rem; color: #90A4AE; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #1565C0, #0277BD);
        padding: 10px 18px;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        margin: 16px 0 12px 0;
    }
    .winner-badge {
        background: linear-gradient(135deg, #F57F17, #FF8F00);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1565C0, #0277BD);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }
    div[data-testid="stSidebar"] { background-color: #0D1B2A; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE       = {'KMeans': '#4C72B0', 'Hierarchical': '#DD8452', 'DBSCAN': '#55A868'}
CLUSTER_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_data(uploaded_file, min_user_ratings=50, min_prod_ratings=5):
    df = pd.read_csv(uploaded_file, header=None,
                     names=['user_id', 'prod_id', 'rating', 'timestamp'])
    df.drop(columns=['timestamp'], inplace=True)
    df.dropna(inplace=True)

    user_counts   = df['user_id'].value_counts()
    active_users  = user_counts[user_counts >= min_user_ratings].index
    df = df[df['user_id'].isin(active_users)]

    prod_counts   = df['prod_id'].value_counts()
    popular_prods = prod_counts[prod_counts >= min_prod_ratings].index
    df = df[df['prod_id'].isin(popular_prods)]

    return df


@st.cache_data(show_spinner="Building user features…")
def build_features(_df):
    user_features = _df.groupby('user_id').agg(
        avg_rating  = ('rating', 'mean'),
        num_ratings = ('rating', 'count')
    ).reset_index()
    scaler = StandardScaler()
    X = scaler.fit_transform(user_features[['avg_rating', 'num_ratings']])
    return user_features, X, scaler


@st.cache_data(show_spinner="Fitting clustering models…")
def fit_clusters(_X, best_k=3):
    km        = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_km = km.fit_predict(_X)
    sil_km    = silhouette_score(_X, labels_km)

    hc        = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels_hc = hc.fit_predict(_X)
    sil_hc    = silhouette_score(_X, labels_hc)

    db        = DBSCAN(eps=0.5, min_samples=5)
    labels_db = db.fit_predict(_X)
    n_db      = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    sil_db    = silhouette_score(_X, labels_db) if n_db > 1 else None

    return km, labels_km, sil_km, hc, labels_hc, sil_hc, db, labels_db, n_db, sil_db


@st.cache_data(show_spinner="Building recommendation engine…")
def build_recommender(_df, top_n_users=500):
    top_users     = _df['user_id'].value_counts().head(top_n_users).index
    df_sub        = _df[_df['user_id'].isin(top_users)]
    user_item_mat = df_sub.pivot_table(index='user_id', columns='prod_id',
                                       values='rating', fill_value=0)
    sim_matrix    = cosine_similarity(user_item_mat.values)
    sim_df        = pd.DataFrame(sim_matrix,
                                 index=user_item_mat.index,
                                 columns=user_item_mat.index)
    return user_item_mat, sim_df


@st.cache_data(show_spinner="Training classifiers…")
def train_classifiers(_df):
    le_user = LabelEncoder()
    le_prod = LabelEncoder()
    df2 = _df.copy()
    df2['user_enc'] = le_user.fit_transform(df2['user_id'])
    df2['prod_enc'] = le_prod.fit_transform(df2['prod_id'])

    user_feat = df2.groupby('user_id').agg(
        avg_rating  = ('rating', 'mean'),
        num_ratings = ('rating', 'count')
    ).reset_index()
    df2 = df2.merge(user_feat, on='user_id', how='left')

    prod_avg = df2.groupby('prod_id')['rating'].mean().reset_index()
    prod_avg.columns = ['prod_id', 'avg_prod_rating']
    df2 = df2.merge(prod_avg, on='prod_id', how='left')

    df2['high_rating'] = (df2['rating'] >= 4).astype(int)

    features = ['user_enc', 'prod_enc', 'avg_rating', 'num_ratings', 'avg_prod_rating']
    X_clf = df2[features]
    y_clf = df2['high_rating']
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf,
                                                         test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt  = dt.predict(X_test)
    dt_acc     = accuracy_score(y_test, y_pred_dt)
    dt_report  = classification_report(y_test, y_pred_dt,
                                        target_names=['Low (0)', 'High (1)'],
                                        output_dict=True)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf  = rf.predict(X_test)
    rf_acc     = accuracy_score(y_test, y_pred_rf)
    rf_report  = classification_report(y_test, y_pred_rf,
                                        target_names=['Low (0)', 'High (1)'],
                                        output_dict=True)

    feat_imp = pd.DataFrame({'Feature': features,
                              'Importance': rf.feature_importances_}).sort_values(
                                  'Importance', ascending=False)
    return dt_acc, dt_report, rf_acc, rf_report, feat_imp


def recommend_products(user_id, user_item_matrix, sim_df, n=5):
    if user_id not in sim_df.index:
        return []
    similar_users  = sim_df[user_id].sort_values(ascending=False)[1:]
    top_similar    = similar_users.head(10).index
    already_rated  = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    scores = {}
    for sim_user in top_similar:
        sim_score = similar_users[sim_user]
        rated     = user_item_matrix.loc[sim_user]
        for prod, rating in rated.items():
            if rating > 0 and prod not in already_rated:
                scores[prod] = scores.get(prod, 0) + sim_score * rating
    return sorted(scores, key=scores.get, reverse=True)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
    st.title("P669 · Settings")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload ratings_.csv", type=["csv"])
    st.markdown("---")

    st.subheader("⚙️ Data Filters")
    min_user_ratings = st.slider("Min ratings per user",  10, 200, 50, 10)
    min_prod_ratings = st.slider("Min ratings per product", 1, 20,  5, 1)

    st.subheader("🔬 Clustering")
    best_k = st.slider("KMeans: number of clusters (k)", 2, 8, 3, 1)

    st.subheader("🛒 Recommendations")
    top_n = st.slider("Number of recommendations", 3, 20, 5, 1)

    st.markdown("---")
    st.caption("P669 · Product Recommendation System")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛒 P669 — Product Recommendation System")
st.caption("Amazon Ratings Dataset · Clustering + Collaborative Filtering + Classification")

if uploaded_file is None:
    st.info("👈  Upload your **ratings_.csv** file in the sidebar to get started.")
    st.markdown("""
    **Expected CSV format (no header):**
    ```
    userId, productId, rating, timestamp
    A3SGXH7..., B00002..., 1, 1342080000
    ```
    """)
    st.stop()

# ── Load & process ────────────────────────────────────────────────────────────
df           = load_data(uploaded_file, min_user_ratings, min_prod_ratings)
user_features, X, scaler = build_features(df)
km, labels_km, sil_km, hc, labels_hc, sil_hc, db, labels_db, n_db, sil_db = fit_clusters(X, best_k)
user_item_matrix, sim_df = build_recommender(df)
dt_acc, dt_report, rf_acc, rf_report, feat_imp = train_classifiers(df)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔬 Cluster Analysis",
    "🌳 Dendrogram",
    "🤖 Classifiers",
    "🛒 Recommend",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Total Ratings",  f"{len(df):,}"),
        (c2, "Unique Users",   f"{df['user_id'].nunique():,}"),
        (c3, "Unique Products",f"{df['prod_id'].nunique():,}"),
        (c4, "Avg Rating",     f"{df['rating'].mean():.2f} ★"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1E2A3A')
        ax.set_facecolor('#1E2A3A')
        rating_counts = df['rating'].value_counts().sort_index()
        bars = ax.bar([str(int(r)) for r in rating_counts.index], rating_counts.values,
                      color=['#4C72B0','#DD8452','#55A868','#C44E52','#FFD700'],
                      edgecolor='white', linewidth=1)
        for bar, cnt in zip(bars, rating_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(df)*0.003,
                    f'{cnt:,}', ha='center', va='bottom', color='white', fontsize=8)
        ax.set_title('Star Rating Frequency', color='white', fontsize=11)
        ax.set_xlabel('Star Rating', color='#90A4AE')
        ax.set_ylabel('Count', color='#90A4AE')
        ax.tick_params(colors='#90A4AE')
        for spine in ax.spines.values(): spine.set_color('#2D3F5A')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Top 10 Most-Rated Products")
        top_prods = df['prod_id'].value_counts().head(10).reset_index()
        top_prods.columns = ['Product ID', 'Rating Count']
        top_prods['Product ID'] = top_prods['Product ID'].str[:14] + '…'
        st.dataframe(top_prods, use_container_width=True, hide_index=True)

    st.subheader("User Activity Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#1E2A3A')
    for ax in axes: ax.set_facecolor('#1E2A3A')

    user_counts_col = df.groupby('user_id')['rating'].count()
    axes[0].hist(user_counts_col, bins=40, color='#4C72B0', edgecolor='white', linewidth=0.5)
    axes[0].set_title('Ratings per User', color='white', fontsize=10)
    axes[0].set_xlabel('Number of Ratings', color='#90A4AE')
    axes[0].set_ylabel('Users', color='#90A4AE')

    prod_counts_col = df.groupby('prod_id')['rating'].count()
    axes[1].hist(prod_counts_col, bins=40, color='#DD8452', edgecolor='white', linewidth=0.5)
    axes[1].set_title('Ratings per Product', color='white', fontsize=10)
    axes[1].set_xlabel('Number of Ratings', color='#90A4AE')
    axes[1].set_ylabel('Products', color='#90A4AE')

    for ax in axes:
        ax.tick_params(colors='#90A4AE')
        for spine in ax.spines.values(): spine.set_color('#2D3F5A')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Comparative Cluster Analysis Dashboard</div>',
                unsafe_allow_html=True)

    # ── Silhouette score cards ────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    for col, algo, sil_val, color in [
        (s1, "KMeans",       sil_km, "#4C72B0"),
        (s2, "Hierarchical", sil_hc, "#DD8452"),
        (s3, "DBSCAN",       sil_db, "#55A868"),
    ]:
        val_str = f"{sil_val:.4f}" if sil_val else "N/A"
        badge   = " 🏆" if (sil_val and sil_val == max(
            v for v in [sil_km, sil_hc, sil_db] if v)) else ""
        col.markdown(f"""<div class="metric-card" style="border-color:{color}">
            <div class="metric-value" style="color:{color}">{val_str}{badge}</div>
            <div class="metric-label">{algo} · Silhouette Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Side-by-side scatters ─────────────────────────────────────────────────
    st.subheader("Cluster Assignments in Feature Space")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1E2A3A', sharey=True)
    algo_sets = [('KMeans', labels_km, sil_km), ('Hierarchical', labels_hc, sil_hc),
                 ('DBSCAN', labels_db, sil_db)]

    for ax, (algo, labels, sil_val) in zip(axes, algo_sets):
        ax.set_facecolor('#1E2A3A')
        for i, lbl in enumerate(sorted(set(labels))):
            mask  = labels == lbl
            c     = '#AAAAAA' if lbl == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            name  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
            ax.scatter(X[mask, 0], X[mask, 1], c=c, label=name,
                       alpha=0.6, s=20, edgecolors='none')
        if algo == 'KMeans':
            ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                       s=180, c='red', marker='X', zorder=5, label='Centroids')
        sil_str = f'{sil_val:.4f}' if sil_val else 'N/A'
        ax.set_title(f'{algo}  ·  Silhouette: {sil_str}',
                     color=PALETTE[algo], fontsize=11, fontweight='bold')
        ax.set_xlabel('avg_rating (std)', color='#90A4AE', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.4)
        ax.tick_params(colors='#90A4AE')
        ax.grid(True, linestyle='--', alpha=0.2)
        for spine in ax.spines.values(): spine.set_color('#2D3F5A')

    axes[0].set_ylabel('num_ratings (std)', color='#90A4AE', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Distribution bar charts ───────────────────────────────────────────────
    st.subheader("Cluster Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor='#1E2A3A')
    for ax, (algo, labels, _) in zip(axes, algo_sets):
        ax.set_facecolor('#1E2A3A')
        uniq, cnts = np.unique(labels, return_counts=True)
        bar_lbls   = ['Noise' if l == -1 else f'Cluster {l}' for l in uniq]
        bar_clrs   = ['#AAAAAA' if l == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                      for i, l in enumerate(uniq)]
        bars = ax.bar(bar_lbls, cnts, color=bar_clrs, edgecolor='white', linewidth=1.2)
        total = len(labels)
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{cnt:,}\n({cnt/total*100:.1f}%)', ha='center', va='bottom',
                    color='white', fontsize=8)
        ax.set_title(algo, color=PALETTE[algo], fontsize=11, fontweight='bold')
        ax.set_ylabel('Users', color='#90A4AE', fontsize=9)
        ax.set_ylim(0, total * 1.2)
        ax.tick_params(colors='#90A4AE')
        ax.grid(True, axis='y', linestyle='--', alpha=0.2)
        for spine in ax.spines.values(): spine.set_color('#2D3F5A')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Cluster profiles ──────────────────────────────────────────────────────
    st.subheader("Cluster Profiles (Mean Feature Values)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#1E2A3A')
    for ax, (algo, labels) in zip(axes, [('KMeans', labels_km), ('Hierarchical', labels_hc)]):
        ax.set_facecolor('#1E2A3A')
        profile_df = user_features.copy()
        profile_df['cluster'] = labels
        cluster_means = profile_df.groupby('cluster')[['avg_rating', 'num_ratings']].mean()
        x = np.arange(2)
        width = 0.35
        for i, (cid, row) in enumerate(cluster_means.iterrows()):
            norm = (row - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
            bars = ax.bar(x + i * width, norm.values, width,
                          label=f'Cluster {cid}',
                          color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                          edgecolor='white', alpha=0.85)
            for b, actual in zip(bars, row.values):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                        f'{actual:.2f}', ha='center', va='bottom', color='white', fontsize=8)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(['Avg Rating', 'No. Ratings'], color='#90A4AE')
        ax.set_ylabel('Normalised (0–1)', color='#90A4AE')
        ax.set_title(algo, color=PALETTE[algo], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.tick_params(colors='#90A4AE')
        ax.grid(True, axis='y', linestyle='--', alpha=0.2)
        for spine in ax.spines.values(): spine.set_color('#2D3F5A')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Algorithm Comparison Table")
    metrics_df = pd.DataFrame({
        'Algorithm':        ['KMeans', 'Hierarchical', 'DBSCAN'],
        'Clusters Found':   [best_k,   2,              n_db],
        'Silhouette Score': [round(sil_km, 4), round(sil_hc, 4),
                             round(sil_db, 4) if sil_db else 'N/A'],
        'Needs k upfront':  ['Yes', 'Optional', 'No'],
        'Handles Noise':    ['No', 'No', 'Yes'],
        'Winner':           ['', '🏆' if sil_hc >= sil_km else '', ''],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DENDROGRAM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🌳 Dendrogram — Hierarchical Clustering Tree</div>',
                unsafe_allow_html=True)
    st.info("The dendrogram shows the order in which user groups were merged. "
            "The red dashed line represents the optimal cut for 2 clusters.")

    sample_size = st.slider("Sample size for dendrogram", 50, 500, 200, 50)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample   = X[sample_idx]
    Z          = linkage(X_sample, method='ward')

    fig, ax = plt.subplots(figsize=(18, 6), facecolor='#1E2A3A')
    ax.set_facecolor('#1E2A3A')

    dendrogram(
        Z, ax=ax,
        leaf_rotation=90,
        leaf_font_size=0,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color='#666666',
    )

    cut_height = sorted(Z[:, 2])[-1] * 0.55
    ax.axhline(y=cut_height, color='#FF4444', linestyle='--', linewidth=2,
               label=f'Cut → 2 clusters (h ≈ {cut_height:.2f})')

    ax.set_title(f'Dendrogram — Hierarchical Clustering (Ward Linkage)  ·  '
                 f'Sample n={sample_size}',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Users', color='#90A4AE')
    ax.set_ylabel('Merge Distance (Ward)', color='#90A4AE')
    ax.tick_params(colors='#90A4AE')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.2)
    for spine in ax.spines.values(): spine.set_color('#2D3F5A')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📖 How to read a Dendrogram"):
        st.markdown("""
        - **Each leaf** at the bottom represents one user (or a group already merged)
        - **Height of a horizontal bar** = the distance at which two groups merged; the higher the bar, the more different the groups were
        - **Cutting the tree** with a horizontal line determines the number of clusters — each vertical line that crosses the cut = one cluster
        - The **red dashed line** is the optimal cut for 2 clusters (based on the largest gap in merge distances)
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🤖 Classification Models</div>',
                unsafe_allow_html=True)
    st.caption("Predicting whether a user will give a **high rating (4–5 ★)**")

    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class="metric-card" style="border-color:#4C72B0">
        <div class="metric-value" style="color:#4C72B0">{dt_acc*100:.1f}%</div>
        <div class="metric-label">Decision Tree Accuracy</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card" style="border-color:#DD8452">
        <div class="metric-value" style="color:#DD8452">{rf_acc*100:.1f}%</div>
        <div class="metric-label">Random Forest Accuracy 🏆</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🌳 Decision Tree")
        dt_summary = {k: v for k, v in dt_report.items() if k in ['Low (0)', 'High (1)']}
        dt_df = pd.DataFrame(dt_summary).T[['precision', 'recall', 'f1-score', 'support']]
        dt_df = dt_df.round(4)
        st.dataframe(dt_df, use_container_width=True)

    with col_right:
        st.subheader("🌲 Random Forest")
        rf_summary = {k: v for k, v in rf_report.items() if k in ['Low (0)', 'High (1)']}
        rf_df = pd.DataFrame(rf_summary).T[['precision', 'recall', 'f1-score', 'support']]
        rf_df = rf_df.round(4)
        st.dataframe(rf_df, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1E2A3A')
    ax.set_facecolor('#1E2A3A')
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(feat_imp))]
    bars   = ax.barh(feat_imp['Feature'], feat_imp['Importance'],
                     color=colors, edgecolor='white', linewidth=1)
    for bar, val in zip(bars, feat_imp['Importance']):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', color='white', fontsize=9)
    ax.set_title('Feature Importance — Random Forest', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score', color='#90A4AE')
    ax.tick_params(colors='#90A4AE')
    ax.invert_yaxis()
    for spine in ax.spines.values(): spine.set_color('#2D3F5A')
    ax.grid(True, axis='x', linestyle='--', alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🛒 Product Recommendation Engine</div>',
                unsafe_allow_html=True)
    st.caption("User-Based Collaborative Filtering · Cosine Similarity")

    available_users = list(user_item_matrix.index)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox("Select a User", options=available_users,
                                     format_func=lambda x: f"👤 {x[:20]}…" if len(x) > 20 else f"👤 {x}")
    with col2:
        n_recs = st.slider("Recommendations", 3, 20, top_n, 1)

    if st.button("🔍 Get Recommendations"):
        recs = recommend_products(selected_user, user_item_matrix, sim_df, n=n_recs)

        if not recs:
            st.warning("No recommendations found for this user.")
        else:
            st.success(f"Top {len(recs)} recommendations for **{selected_user[:30]}**")
            rec_df = pd.DataFrame({
                'Rank':        range(1, len(recs) + 1),
                'Product ID':  recs,
                'Category':    ['Recommended'] * len(recs),
            })
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

            # Similarity bar chart
            user_sim = sim_df[selected_user].sort_values(ascending=False)[1:11]
            fig, ax  = plt.subplots(figsize=(10, 3.5), facecolor='#1E2A3A')
            ax.set_facecolor('#1E2A3A')
            ax.barh([u[:20] for u in user_sim.index], user_sim.values,
                    color='#4C72B0', edgecolor='white', linewidth=0.8)
            ax.set_title(f'Top 10 Most Similar Users to {selected_user[:24]}',
                         color='white', fontsize=11)
            ax.set_xlabel('Cosine Similarity', color='#90A4AE')
            ax.tick_params(colors='#90A4AE')
            ax.invert_yaxis()
            for spine in ax.spines.values(): spine.set_color('#2D3F5A')
            ax.grid(True, axis='x', linestyle='--', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with st.expander("📖 How Collaborative Filtering Works"):
        st.markdown("""
        1. **Build the User-Item Matrix** — rows = users, columns = products, cells = ratings
        2. **Compute Cosine Similarity** — measure the angle between each pair of user rating vectors
        3. **Find Similar Users** — for the selected user, find the top-10 most similar users
        4. **Aggregate Ratings** — weight each similar user's ratings by their similarity score
        5. **Filter** — exclude products the selected user has already rated
        6. **Rank & Return** — return the top-N highest-scored unseen products
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("P669 · Product Recommendation System · Amazon Ratings Dataset")
