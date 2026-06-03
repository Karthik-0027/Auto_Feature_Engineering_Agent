import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import io
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ==============================
# Premium Explanation Generator
# ==============================
def generate_detailed_explanation(feat_name, target_col, improvement, base_score):
    imp_pct = improvement * 100
    if imp_pct >= 1.0:
        impact_level = "High"
        impact_color = "#7f5af0"  # Violet
        impact_bg = "#3a305e"     # Lighter violet
    elif imp_pct >= 0.5:
        impact_level = "Moderate"
        impact_color = "#5c7cfa"  # Indigo
        impact_bg = "#2a3a6e"     # Lighter indigo
    elif imp_pct > 0:
        impact_level = "Positive"
        impact_color = "#0f766e"  # Teal
        impact_bg = "#0a4a45"     # Lighter teal
    else:
        impact_level = "None"
        impact_color = "#94a3b8"  # Light gray
        impact_bg = "#1e293b"     # Dark slate

    if '_x_' in feat_name:
        feat_type = "Interaction"
        parts = feat_name.split('_x_')
        formula = f"{parts[0]} × {parts[1]}"
        insight = f"This interaction reveals how {parts[0]} and {parts[1]} jointly influence {target_col}."
    elif '_div_' in feat_name:
        feat_type = "Ratio"
        parts = feat_name.split('_div_')
        formula = f"{parts[0]} / ({parts[1]} + ε)"
        insight = f"Normalizing {parts[0]} by {parts[1]} exposes relative importance over absolute scale."
    elif '_sq' in feat_name:
        feat_type = "Squared"
        base = feat_name.replace('_sq', '')
        formula = f"{base}²"
        insight = f"Squaring {base} emphasizes extreme values that strongly correlate with {target_col}."
    elif '_cube' in feat_name:
        feat_type = "Cubed"
        base = feat_name.replace('_cube', '')
        formula = f"{base}³"
        insight = f"Cubing {base} amplifies outliers to capture non-linear effects on {target_col}."
    elif '_log' in feat_name:
        feat_type = "Log-Transform"
        base = feat_name.replace('_log', '')
        formula = f"sign({base}) × log(1 + |{base}|)"
        insight = f"The log transform stabilizes variance and handles skewed distributions in {base}."
    elif '_freq' in feat_name:
        feat_type = "Frequency Encoding"
        base = feat_name.replace('_freq', '')
        formula = f"Frequency of category in '{base}'"
        insight = f"Replacing categories with their frequency helps generalize rare versus common patterns."
    elif 'word_count' in feat_name:
        feat_type = "Text Length"
        formula = "Word count"
        insight = f"Text length provides signal about detail level or complexity related to {target_col}."
    else:
        feat_type = "Custom"
        formula = feat_name
        insight = f"This engineered feature improves prediction of {target_col}."

    return {
        "feature": feat_name,
        "type": feat_type,
        "impact_level": impact_level,
        "impact_pct": f"+{imp_pct:.2f}%",
        "impact_color": impact_color,
        "impact_bg": impact_bg,
        "formula": formula,
        "insight": insight
    }

# ==============================
# Core Agents (Fully Fixed)
# ==============================
class DataAnalyzerAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.text_cols = []

    def detect_types(self):
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if col_data.empty:
                continue
            if self._is_datetime(col_data):
                self.datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            elif col_data.nunique() / len(col_data) < 0.5 and col_data.nunique() < 50:
                self.categorical_cols.append(col)
            elif col_data.astype(str).str.len().mean() < 100:
                self.text_cols.append(col)
        return {
            'numeric': self.numeric_cols,
            'categorical': self.categorical_cols,
            'datetime': self.datetime_cols,
            'text': self.text_cols
        }

    def _is_datetime(self, series):
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except (ValueError, TypeError):
            return False

class FeatureGeneratorAgent:
    def __init__(self, df, col_info):
        self.df = df.copy()
        self.col_info = col_info
        self.candidate_features = {}

    def generate_features(self):
        for col in self.col_info['numeric']:
            self._add_numeric_features(col)
        for col in self.col_info['categorical']:
            self._add_categorical_features(col)
        for col in self.col_info['datetime']:
            self._add_datetime_features(col)
        for col in self.col_info['text']:
            self._add_text_features(col)
        self._add_pairwise_interactions()
        return self.candidate_features

    def _add_numeric_features(self, col):
        base = self.df[col]
        unique_vals = base.dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False})
        self.candidate_features[f"{col}_sq"] = base ** 2
        self.candidate_features[f"{col}_cube"] = base ** 3
        if not is_binary:
            base_float = base.astype(float)
            self.candidate_features[f"{col}_log"] = np.log1p(np.abs(base_float)) * np.sign(base_float)

    def _add_categorical_features(self, col):
        freq_map = self.df[col].value_counts().to_dict()
        self.candidate_features[f"{col}_freq"] = self.df[col].map(freq_map)

    def _add_datetime_features(self, col):
        dt_series = pd.to_datetime(self.df[col], errors='coerce')
        # ✅ FIXED: Convert to numeric nullable integers
        self.candidate_features[f"{col}_year"] = pd.to_numeric(dt_series.dt.year, errors='coerce').astype('Int64')
        self.candidate_features[f"{col}_month"] = pd.to_numeric(dt_series.dt.month, errors='coerce').astype('Int64')
        self.candidate_features[f"{col}_weekday"] = pd.to_numeric(dt_series.dt.weekday, errors='coerce').astype('Int64')

    def _add_text_features(self, col):
        text_series = self.df[col].astype(str)
        self.candidate_features[f"{col}_word_count"] = text_series.str.split().str.len()
        for kw in ["good", "bad", "high", "low"]:
            self.candidate_features[f"{col}_has_{kw}"] = text_series.str.contains(kw, case=False).astype(int)

    def _add_pairwise_interactions(self):
        nums = self.col_info['numeric']
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                a, b = nums[i], nums[j]
                self.candidate_features[f"{a}_x_{b}"] = self.df[a] * self.df[b]
                self.candidate_features[f"{a}_div_{b}"] = self.df[a] / (self.df[b] + 1e-8)
                self.candidate_features[f"{a}_minus_{b}"] = self.df[a] - self.df[b]

class EvaluatorAgent:
    def __init__(self, df, target_col, candidate_features):
        self.df = df.copy()
        self.target_col = target_col
        self.candidate_features = candidate_features
        self.results = []

    def evaluate_all(self, cv_desired=3):
        y = self.df[self.target_col]
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_base = self.df.drop(columns=[self.target_col])
        X_base = pd.get_dummies(X_base, drop_first=True)

        # 🔥 CRITICAL FIX: Ensure all columns are numeric and handle NaN
        for col in X_base.columns:
            if X_base[col].dtype == 'object':
                X_base[col] = pd.to_numeric(X_base[col], errors='coerce')
            if X_base[col].isna().any():
                fill_val = X_base[col].median() if X_base[col].notna().any() else 0
                X_base[col].fillna(fill_val, inplace=True)

        class_counts = Counter(y)
        min_class_count = min(class_counts.values())

        if min_class_count >= cv_desired:
            cv = cv_desired
            use_stratified = True
        elif min_class_count >= 2:
            cv = min_class_count
            use_stratified = True
        else:
            cv = min(cv_desired, len(y) // 2)
            cv = max(2, cv)
            use_stratified = False

        model = RandomForestClassifier(n_estimators=50, random_state=42)

        if use_stratified:
            base_score = np.mean(cross_val_score(model, X_base, y, cv=cv, scoring='accuracy'))
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
            base_score = np.mean(cross_val_score(model, X_base, y, cv=kfold, scoring='accuracy'))

        for feat_name, feat_series in self.candidate_features.items():
            X_temp = X_base.copy()
            feat_series = feat_series.reindex(X_temp.index)
            X_temp[feat_name] = feat_series

            # Apply same numeric cleanup to new feature
            if X_temp[feat_name].dtype == 'object':
                X_temp[feat_name] = pd.to_numeric(X_temp[feat_name], errors='coerce')
            if X_temp[feat_name].isna().any():
                fill_val = X_temp[feat_name].median() if X_temp[feat_name].notna().any() else 0
                X_temp[feat_name].fillna(fill_val, inplace=True)

            X_temp = X_temp.replace([np.inf, -np.inf], np.nan)
            X_temp = X_temp.dropna()
            y_temp = y[X_temp.index]

            if len(y_temp) < 2:
                score = 0.0
            else:
                subset_counts = Counter(y_temp)
                min_subset = min(subset_counts.values()) if subset_counts else 0

                if min_subset >= 2:
                    local_cv = min(cv, min_subset)
                    local_use_strat = True
                else:
                    local_cv = min(cv, len(y_temp) // 2)
                    local_cv = max(2, local_cv)
                    local_use_strat = False

                try:
                    if local_use_strat:
                        score = np.mean(cross_val_score(model, X_temp, y_temp, cv=local_cv, scoring='accuracy'))
                    else:
                        kfold_local = KFold(n_splits=local_cv, shuffle=True, random_state=42)
                        score = np.mean(cross_val_score(model, X_temp, y_temp, cv=kfold_local, scoring='accuracy'))
                except Exception:
                    score = 0.0

            improvement = score - base_score
            self.results.append({
                'feature': feat_name,
                'cv_score': float(score),
                'improvement': float(improvement)
            })

        self.results.sort(key=lambda x: x['improvement'], reverse=True)
        return base_score, self.results

# ==============================
# Streamlit App — Premium Dark Theme (Fully Fixed)
# ==============================
st.set_page_config(page_title="Feature Engineering Agent", layout="wide", page_icon="🚀")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #e0e6f0;
    background: #0b0f1a;
}
.main-header {
    background: linear-gradient(135deg, #1e1f3a 0%, #0b0f1a 100%);
    padding: 28px 0;
    border-radius: 16px;
    margin-bottom: 28px;
    color: #f0f5ff;
    text-align: center;
}
.main-header h1 {
    font-weight: 700;
    font-size: 2.2em;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #7f5af0, #5c7cfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.main-header p {
    font-weight: 400;
    opacity: 0.8;
    max-width: 700px;
    margin: 0 auto;
    font-size: 1.1em;
}
.explanation-card {
    background: rgba(20, 25, 48, 0.8);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.3s ease;
    color: #cbd5e1;
}
.explanation-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.7);
}
.feature-title {
    font-size: 1.35em;
    font-weight: 700;
    color: #f0f5ff;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
}
.feature-formula {
    font-family: 'SFMono-Regular', Consolas, monospace;
    background: rgba(255, 255, 255, 0.05);
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.95em;
    color: #a5b4fc;
    display: inline-block;
    margin: 6px 0;
    border: 1px solid rgba(165, 180, 252, 0.2);
}
.impact-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.88em;
    color: white;
    margin-left: 12px;
    font-family: 'Inter', sans-serif;
}
.section-header {
    color: #e0e6f0;
    font-weight: 700;
    font-size: 1.6em;
    margin: 36px 0 20px 0;
    padding-bottom: 8px;
    position: relative;
}
.section-header:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #7f5af0, #5c7cfa);
    border-radius: 3px;
}
.stButton>button {
    background: linear-gradient(135deg, #7f5af0, #5c7cfa);
    color: #f0f5ff;
    font-weight: 600;
    border: none;
    padding: 10px 24px;
    border-radius: 12px;
    font-size: 1em;
    transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(127, 90, 240, 0.6);
}
.stSuccess {
    background: rgba(15, 118, 110, 0.15);
    color: #f0f5ff;
    border: 1px solid rgba(15, 118, 110, 0.4);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stSidebar"] {
    background: #1a1e2c;
    color: #f0f5ff;
}
strong {
    color: #f0f5ff;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Feature Engineering Agent</h1>
    <p>Enterprise-grade automated feature engineering with explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📥 Data Input")
input_option = st.sidebar.radio("Source", ("Upload File", "Enter URL"))

df = None
if input_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    url = st.sidebar.text_input("Dataset URL")
    if url:
        try:
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            else:
                df = pd.read_excel(url)
        except Exception as e:
            st.error(f"Error: {e}")

if df is not None:
    st.success(f"✅ Loaded **{df.shape[0]:,} records** with **{df.shape[1]} features**")
    
    target_col = st.selectbox("🎯 Target Variable", df.columns)
    max_features = st.slider("Top features to generate", 1, 20, 10)
    
    if st.button("⚡ Generate Premium Features", type="primary"):
        with st.spinner("Engineering features with enterprise-grade AI..."):
            try:
                analyzer = DataAnalyzerAgent(df)
                col_info = analyzer.detect_types()
                generator = FeatureGeneratorAgent(df, col_info)
                candidates = generator.generate_features()
                
                evaluator = EvaluatorAgent(df, target_col, candidates)
                base_score, results = evaluator.evaluate_all()
                top_results = [r for r in results if r['improvement'] > 0][:max_features]
                
                if not top_results:
                    st.warning("⚠️ No features improved model performance.")
                else:
                    # Bar Chart
                    results_df = pd.DataFrame(top_results)
                    fig1 = px.bar(
                        results_df,
                        x='improvement',
                        y='feature',
                        orientation='h',
                        title=f"Top {len(top_results)} Features by Accuracy Gain",
                        labels={'improvement': 'Accuracy Improvement', 'feature': 'Feature'},
                        color='improvement',
                        color_continuous_scale=['#5c7cfa', '#7f5af0']
                    )
                    fig1.update_layout(
                        height=420,
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e0e6f0'}
                    )
                    
                    # Gauge Chart
                    top_score = top_results[0]['cv_score']
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=top_score,
                        delta={'reference': base_score, 'increasing': {'color': "#0f766e"}},
                        gauge={
                            'axis': {'range': [None, 1], 'tickcolor': "#94a3b8"},
                            'bar': {'color': "#7f5af0"},
                            'steps': [
                                {'range': [0, base_score], 'color': "rgba(92, 124, 250, 0.1)"},
                                {'range': [base_score, 1], 'color': "rgba(15, 118, 110, 0.1)"}
                            ],
                        },
                        title={'text': "Model Accuracy", 'font': {'size': 20, 'color': '#f0f5ff'}}
                    ))
                    fig2.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)')
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Explanations
                    st.markdown('<div class="section-header">🧠 AI-Generated Feature Insights</div>', unsafe_allow_html=True)
                    
                    for r in top_results:
                        exp = generate_detailed_explanation(r['feature'], target_col, r['improvement'], base_score)
                        st.markdown(f"""
                        <div class="explanation-card">
                            <div class="feature-title">
                                {exp['feature']}
                                <span class="impact-badge" style="background-color: {exp['impact_bg']};">
                                    {exp['impact_pct']}
                                </span>
                            </div>
                            <p><strong>Feature Type:</strong> {exp['type']}</p>
                            <p><strong>Mathematical Form:</strong> <span class="feature-formula">{exp['formula']}</span></p>
                            <p><strong>AI Insight:</strong> {exp['insight']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Heatmap
                    if len(top_results) >= 2:
                        st.markdown('<div class="section-header">📊 Feature Correlation Matrix</div>', unsafe_allow_html=True)
                        final_df = df.copy()
                        top_feats = [r['feature'] for r in top_results[:5]]
                        for feat in top_feats:
                            final_df[feat] = candidates[feat].reindex(final_df.index)
                        # Apply numeric cleanup to final_df too
                        for col in final_df.columns:
                            if final_df[col].dtype == 'object':
                                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                            if final_df[col].isna().any():
                                fill_val = final_df[col].median() if final_df[col].notna().any() else 0
                                final_df[col].fillna(fill_val, inplace=True)
                        corr_cols = top_feats + [target_col]
                        corr_df = final_df[corr_cols].select_dtypes(include=[np.number]).dropna()
                        if not corr_df.empty and corr_df.shape[1] > 1:
                            corr = corr_df.corr()
                            fig3 = px.imshow(
                                corr, 
                                text_auto='.2f', 
                                color_continuous_scale='RdBu_r',
                                aspect="auto",
                                title="Correlation: Top Features & Target"
                            )
                            fig3.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#e0e6f0'}
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    # Downloads
                    st.markdown('<div class="section-header">📥 Export Enterprise Assets</div>', unsafe_allow_html=True)
                    final_df = df.copy()
                    for r in top_results:
                        final_df[r['feature']] = candidates[r['feature']].reindex(final_df.index)
                    # Final cleanup for download
                    for col in final_df.columns:
                        if final_df[col].dtype == 'object':
                            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                        if final_df[col].isna().any():
                            fill_val = final_df[col].median() if final_df[col].notna().any() else 0
                            final_df[col].fillna(fill_val, inplace=True)
                    
                    report_md = f"""# AutoFE-Agent Pro Report

## Model Performance
- **Baseline Accuracy**: {base_score:.4f}
- **Enhanced Accuracy**: {top_score:.4f} (+{top_score - base_score:.4f})

## Top Engineered Features
"""
                    for r in top_results:
                        exp = generate_detailed_explanation(r['feature'], target_col, r['improvement'], base_score)
                        report_md += f"\n### {exp['feature']} ({exp['impact_pct']})\n"
                        report_md += f"- **Type**: {exp['type']}\n"
                        report_md += f"- **Formula**: `{exp['formula']}`\n"
                        report_md += f"- **Insight**: {exp['insight']}\n"

                    csv_buffer = io.BytesIO()
                    final_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    md_buffer = io.BytesIO()
                    md_buffer.write(report_md.encode('utf-8'))
                    md_buffer.seek(0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📥 Engineered Dataset (CSV)", data=csv_buffer, file_name="autofe_premium_features.csv")
                    with col2:
                        st.download_button("📄 Executive Report (Markdown)", data=md_buffer, file_name="autofe_executive_report.md")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
else:
    st.info("👈 Upload a dataset or enter a URL to unlock premium feature engineering.")

st.markdown("---")
st.caption("AutoFE-Agent Pro • Enterprise AI for Automated Feature Engineering")