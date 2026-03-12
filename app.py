import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# 
# PAGE CONFIGURATION
# 
st.set_page_config(
    page_title="Learning Pathway System",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 
# CUSTOM CSS
# 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# 
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# 
def download_from_huggingface(url, output_path):
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Download error: {e}")
        return False

@st.cache_resource
def load_models():
    model_path = "best_hybrid_model.pkl"
    scaler_path = "scaler.pkl"
    le_path = "label_encoder.pkl"

    # Download model from Hugging Face if not exists
    if not os.path.exists(model_path):
        st.info(" Downloading model... Please wait (86MB)")
        url = "https://huggingface.co/Reserach/learning-pathway-model/resolve/main/best_hybrid_model.pkl"
        success = download_from_huggingface(url, model_path)
        if not success:
            return None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        return model, scaler, le
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

model, scaler, le = load_models()

# 
# ALL FUNCTIONS
# 
def get_cluster_label(avg_score, engagement_ratio, days_active):
    if avg_score >= 80 and engagement_ratio >= 0.7:
        return 3
    elif avg_score >= 60 and engagement_ratio >= 0.5:
        return 1
    elif avg_score >= 40 and engagement_ratio >= 0.3:
        return 2
    else:
        return 0

def detect_weak_areas(student_data):
    weak_areas = []
    if student_data['avg_score'] < 50:
        weak_areas.append('low_avg_score')
    if student_data['avg_tma_score'] < 60:
        weak_areas.append('low_tma_score')
    if student_data['avg_cma_score'] < 40:
        weak_areas.append('low_cma_score')
    if student_data['avg_exam_score'] > 0 and student_data['avg_exam_score'] < 50:
        weak_areas.append('low_exam_score')
    if student_data['total_clicks'] < 500:
        weak_areas.append('low_engagement')
    if student_data['days_active'] < 50:
        weak_areas.append('low_active_days')
    if student_data['num_assessments_submitted'] < 5:
        weak_areas.append('low_assessments')
    if student_data['engagement_ratio'] < 0.5:
        weak_areas.append('low_engagement_ratio')
    if student_data['num_of_prev_attempts'] > 1:
        weak_areas.append('multiple_attempts')
    return weak_areas

def generate_recommendations(performance_category, weak_areas, student_data):
    recommendations = []
    pathway = []

    if performance_category == 'Excellent':
        recommendations.append(" Congratulations! You are performing at an Excellent level.")
        recommendations.append("Your academic performance demonstrates strong dedication and consistent effort.")
        pathway.append(" Continue maintaining your current study habits.")
        pathway.append(" Challenge yourself with advanced course materials.")
        pathway.append(" Consider helping peers who may be struggling.")
        pathway.append(" Focus on achieving distinction in all assessments.")
        if student_data['avg_exam_score'] < 70:
            pathway.append(" Consider dedicating more time to exam preparation.")

    elif performance_category == 'Good':
        recommendations.append(" You are performing at a Good level.")
        recommendations.append("You are on the right track. With focused improvements, you can achieve excellent performance.")
        if 'low_exam_score' in weak_areas:
            pathway.append(" Your exam scores need improvement. Dedicate at least 2 extra hours per week to exam preparation.")
        if 'low_tma_score' in weak_areas:
            pathway.append(" Focus on improving your TMA scores. Review feedback from previous TMAs carefully.")
        if 'low_cma_score' in weak_areas:
            pathway.append(" Your CMA scores need attention. Practice more online quizzes.")
        if 'low_engagement' in weak_areas:
            pathway.append(" Increase your interaction with course materials. Try to engage daily.")
        if not weak_areas:
            pathway.append(" Keep up your consistent performance across all areas.")
            pathway.append(" Push yourself slightly harder to move towards Excellent.")

    elif performance_category == 'At-Risk':
        recommendations.append(" You are currently identified as At-Risk.")
        recommendations.append("Immediate action is recommended to improve your academic performance.")
        if 'low_avg_score' in weak_areas:
            pathway.append(" URGENT: Your overall scores are critically low. Contact your academic advisor immediately.")
        if 'low_engagement' in weak_areas:
            pathway.append(" Your engagement is very low. Set a daily target of at least 1 hour of active learning.")
        if 'low_active_days' in weak_areas:
            pathway.append(" You have not been consistently active. Commit to logging in every day.")
        if 'low_assessments' in weak_areas:
            pathway.append(" You have submitted very few assessments. Prioritize completing all pending assessments.")
        if 'low_exam_score' in weak_areas:
            pathway.append(" Your exam scores are critically low. Seek immediate support from your tutor.")
        if 'multiple_attempts' in weak_areas:
            pathway.append(" You have attempted this course multiple times. Consider speaking with a student counselor.")
        pathway.append(" Remember: Early intervention is key to academic recovery.")

    return recommendations, pathway

def predict_student(student_data):
    feature_cols = [
        'avg_score', 'avg_tma_score', 'avg_cma_score', 'avg_exam_score',
        'total_clicks', 'avg_clicks_per_day', 'days_active',
        'unique_materials_accessed', 'engagement_ratio',
        'num_assessments_submitted', 'studied_credits',
        'num_of_prev_attempts', 'cluster_label',
        'score_consistency', 'assessment_completion_rate',
        'weighted_performance', 'engagement_quality', 'early_performance'
    ]

    student_data['score_consistency'] = 1 / (student_data.get('std_score', 1) + 1)
    student_data['assessment_completion_rate'] = (
        student_data['num_assessments_submitted'] /
        (student_data['num_assessments_submitted'] + 1))
    student_data['weighted_performance'] = (
        student_data['avg_tma_score'] * 0.3 +
        student_data['avg_cma_score'] * 0.3 +
        student_data['avg_exam_score'] * 0.4)
    student_data['engagement_quality'] = (
        student_data['avg_clicks_per_day'] * student_data['engagement_ratio'])
    student_data['early_performance'] = (
        student_data.get('early_engagement', 0) * student_data['avg_score'])
    student_data['cluster_label'] = get_cluster_label(
        student_data['avg_score'],
        student_data['engagement_ratio'],
        student_data['days_active'])

    input_df = pd.DataFrame([student_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    performance_category = le.inverse_transform([pred_encoded])[0]
    confidence = max(pred_proba) * 100
    weak_areas = detect_weak_areas(student_data)
    recommendations, pathway = generate_recommendations(
        performance_category, weak_areas, student_data)

    return {
        'performance_category': performance_category,
        'confidence': confidence,
        'weak_areas': weak_areas,
        'recommendations': recommendations,
        'pathway': pathway,
        'probabilities': dict(zip(le.classes_, pred_proba))
    }

# 
# SIDEBAR NAVIGATION
# 
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title(" Learning Pathway")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dashboard", "Get Recommendation", "Research Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Research Info:**")
st.sidebar.info("ML-Based Personalized Learning Pathway System for Undergraduate Academic Performance")

# 
# PAGE 1  HOME
# 
if page == "Home":
    st.markdown('<div class="main-header"> Personalized Learning Pathway System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-Based Academic Performance Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", "24,998", "Dataset Size")
    with col2:
        st.metric("Model Accuracy", "85.00%", "Stacking Hybrid")
    with col3:
        st.metric("Recommendation Acc", "92.20%", "System Performance")
    with col4:
        st.metric("Features Used", "18", "ML Features")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" About This Research")
        st.write("""
        This system uses Machine Learning to analyze undergraduate 
        student performance data and generate personalized learning 
        pathways to improve academic outcomes.
        
        **Key Features:**
        -  Hybrid ML Model (XGBoost + RF + KNN)
        -  85% Prediction Accuracy
        -  92.2% Recommendation Accuracy
        -  Personalized Learning Pathways
        -  GPA Tracking System
        """)

    with col2:
        st.subheader(" Research Methodology")
        st.write("""
        **Phase 1:** Exploratory Data Analysis + Clustering
        
        **Phase 2:** ML Model Development
        - 5 models tested and compared
        - Top 3 selected for hybrid
        - Stacking Hybrid achieved 85% accuracy
        
        **Phase 3:** Recommendation Engine
        
        **Phase 4:** Learning Pathway Generation
        
        **Phase 5:** Evaluation & Validation
        """)

    st.markdown("---")
    st.subheader(" How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Step 1**\n\nEnter your academic performance data")
    with col2:
        st.info("**Step 2**\n\nML model predicts your performance category")
    with col3:
        st.info("**Step 3**\n\nSystem identifies your weak areas")
    with col4:
        st.info("**Step 4**\n\nPersonalized learning pathway generated")

# 
# PAGE 2  DASHBOARD
# 
elif page == "Dashboard":
    st.title(" Research Dashboard")
    st.markdown("---")

    st.subheader(" Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "85.00%", "+15.86% after tuning")
    with col2:
        st.metric("Precision", "85.10%", "Weighted Average")
    with col3:
        st.metric("Recall", "85.00%", "Weighted Average")
    with col4:
        st.metric("F1-Score", "84.79%", "Weighted Average")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Model Comparison")
        models_data = {
            'Model': ['XGBoost\n(Tuned)', 'Stacking\nHybrid', 'Voting\nHybrid', 'Random\nForest', 'KNN\n(Tuned)'],
            'Accuracy': [0.8528, 0.8500, 0.8468, 0.8432, 0.8082],
            'Type': ['Individual', 'Hybrid', 'Hybrid', 'Individual', 'Individual']
        }
        df_models = pd.DataFrame(models_data)
        colors = ['#9b59b6' if t == 'Hybrid' else '#3498db' for t in df_models['Type']]
        fig = go.Figure(data=[go.Bar(
            x=df_models['Model'],
            y=df_models['Accuracy'],
            marker_color=colors,
            text=[f'{v:.4f}' for v in df_models['Accuracy']],
            textposition='outside'
        )])
        fig.update_layout(title='Model Accuracy Comparison', yaxis_range=[0.75, 0.90], height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(" Class Performance")
        class_data = {
            'Class': ['At-Risk', 'Good', 'Excellent'],
            'Precision': [0.93, 0.79, 0.73],
            'Recall': [0.89, 0.88, 0.54],
            'F1-Score': [0.91, 0.83, 0.62]
        }
        df_class = pd.DataFrame(class_data)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Precision', x=df_class['Class'], y=df_class['Precision'], marker_color='#2ecc71'))
        fig2.add_trace(go.Bar(name='Recall', x=df_class['Class'], y=df_class['Recall'], marker_color='#3498db'))
        fig2.add_trace(go.Bar(name='F1-Score', x=df_class['Class'], y=df_class['F1-Score'], marker_color='#e74c3c'))
        fig2.update_layout(barmode='group', title='Per Class Performance', height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" 10-Fold Cross Validation")
        cv_data = {
            'Fold': [f'Fold {i}' for i in range(1, 11)],
            'Accuracy': [0.8560, 0.8415, 0.8535, 0.8365, 0.8400, 0.8450, 0.8380, 0.8415, 0.8524, 0.8234]
        }
        df_cv = pd.DataFrame(cv_data)
        fig3 = px.line(df_cv, x='Fold', y='Accuracy', markers=True,
                       title='Cross Validation Scores', color_discrete_sequence=['#9b59b6'])
        fig3.add_hline(y=0.8428, line_dash="dash", line_color="red", annotation_text="Mean: 0.8428")
        fig3.update_layout(yaxis_range=[0.80, 0.87], height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader(" Performance Distribution")
        dist_data = {'Category': ['At-Risk', 'Good', 'Excellent'], 'Count': [2405, 2085, 510]}
        df_dist = pd.DataFrame(dist_data)
        fig4 = px.pie(df_dist, values='Count', names='Category',
                      title='Test Set Distribution',
                      color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71'])
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader(" Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", "24,998")
        st.metric("Training Set", "19,998 (80%)")
    with col2:
        st.metric("Test Set", "5,000 (20%)")
        st.metric("Features", "18")
    with col3:
        st.metric("CV Folds", "10")
        st.metric("Std Deviation", "0.0091")

# 
# PAGE 3  GET RECOMMENDATION
# 
elif page == "Get Recommendation":
    st.title(" Get Your Personalized Learning Pathway")
    st.markdown("---")

    if model is None:
        st.error(" Model files not found! Please check your folder.")
    else:
        with st.form("student_form"):
            st.subheader(" Academic Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                current_year = st.selectbox("Current Year", [1, 2, 3, 4])
            with col2:
                current_semester = st.selectbox("Current Semester", [1, 2])
            with col3:
                cumulative_gpa = st.slider("Cumulative GPA", 0.0, 4.0, 2.5, 0.01)

            st.markdown("---")
            st.subheader(" Current Semester Scores")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = st.number_input("Average Score", 0.0, 100.0, 70.0)
            with col2:
                avg_tma_score = st.number_input("TMA (Tutor Marked Assessment) Score", 0.0, 100.0, 70.0)
            with col3:
                avg_cma_score = st.number_input("CMA (Computer Marked Assessment) Score", 0.0, 100.0, 70.0)
            with col4:
                avg_exam_score = st.number_input("Exam Score (0 if not taken)", 0.0, 100.0, 0.0)

            st.markdown("---")
            st.subheader(" Engagement Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_clicks = st.number_input("Total LMS Clicks", 0, 10000, 1000)
                avg_clicks_per_day = st.number_input("Avg Clicks Per Day", 0.0, 100.0, 10.0)
            with col2:
                days_active = st.number_input("Days Active", 0, 365, 100)
                unique_materials = st.number_input("Unique Materials Accessed", 0, 200, 30)
            with col3:
                engagement_ratio = st.slider("Engagement Ratio", 0.0, 1.0, 0.5, 0.01)

            st.markdown("---")
            st.subheader(" Assessment Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                num_assessments = st.number_input("Assessments Submitted", 0, 50, 8)
            with col2:
                studied_credits = st.number_input("Studied Credits", 0, 300, 120)
            with col3:
                prev_attempts = st.number_input("Previous Attempts", 0, 10, 0)

            submitted = st.form_submit_button(" Get My Learning Pathway", use_container_width=True)

        if submitted:
            student_data = {
                'avg_score': avg_score,
                'avg_tma_score': avg_tma_score,
                'avg_cma_score': avg_cma_score,
                'avg_exam_score': avg_exam_score,
                'total_clicks': float(total_clicks),
                'avg_clicks_per_day': avg_clicks_per_day,
                'days_active': float(days_active),
                'unique_materials_accessed': float(unique_materials),
                'engagement_ratio': engagement_ratio,
                'num_assessments_submitted': float(num_assessments),
                'studied_credits': float(studied_credits),
                'num_of_prev_attempts': float(prev_attempts),
                'std_score': 5.0,
                'early_engagement': engagement_ratio * 0.8
            }

            with st.spinner(" Analyzing your performance..."):
                result = predict_student(student_data)

            st.markdown("---")
            st.subheader(" Your Results")

            category = result['performance_category']
            confidence = result['confidence']

            col1, col2, col3 = st.columns(3)
            with col1:
                if category == 'Excellent':
                    st.success(f" Performance: {category}")
                elif category == 'Good':
                    st.info(f" Performance: {category}")
                else:
                    st.error(f" Performance: {category}")
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with col3:
                st.metric("GPA Status",
                          " Excellent" if cumulative_gpa >= 3.5
                          else " Good" if cumulative_gpa >= 3.0
                          else " Average" if cumulative_gpa >= 2.5
                          else " Low")

            st.markdown("---")
            st.subheader(f" Year {current_year} GPA Advice")

            if current_year == 1:
                if cumulative_gpa >= 3.0:
                    st.success(" Great start! Maintain this momentum.")
                else:
                    st.warning(" Low GPA in Year 1 is still recoverable! Act now.")
            elif current_year == 2:
                if cumulative_gpa >= 3.0:
                    st.success(" Strong performance  aim for 3.5+!")
                else:
                    st.warning(" Seek academic support before Year 3.")
            elif current_year == 3:
                if cumulative_gpa >= 3.0:
                    st.success(" Good position  strong finish needed!")
                else:
                    st.error(" URGENT: Contact your advisor immediately!")
            elif current_year == 4:
                if cumulative_gpa >= 3.5:
                    st.success(" Outstanding! On track for first class!")
                elif cumulative_gpa >= 3.0:
                    st.success(" Good standing  push for distinction!")
                elif cumulative_gpa >= 2.5:
                    st.warning(" You can still achieve good degree  focus!")
                else:
                    st.error(" CRITICAL: Maximum effort required NOW!")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(" Performance Probabilities")
                probs = result['probabilities']
                fig = go.Figure(go.Bar(
                    x=list(probs.keys()),
                    y=[v * 100 for v in probs.values()],
                    marker_color=['#e74c3c', '#2ecc71', '#3498db'],
                    text=[f'{v*100:.1f}%' for v in probs.values()],
                    textposition='outside'
                ))
                fig.update_layout(title='Prediction Confidence',
                                  yaxis_title='Probability (%)',
                                  yaxis_range=[0, 110], height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader(" Weak Areas Detected")
                area_names = {
                    'low_avg_score': ' Overall Score',
                    'low_tma_score': ' TMA (Tutor Marked Assessment) Score',
                    'low_cma_score': ' CMA (Computer Marked Assessment) Score',
                    'low_exam_score': ' Exam Score',
                    'low_engagement': ' LMS Engagement',
                    'low_active_days': ' Active Days',
                    'low_assessments': ' Assessments',
                    'low_engagement_ratio': ' Engagement Ratio',
                    'multiple_attempts': ' Multiple Attempts'
                }
                if result['weak_areas']:
                    for area in result['weak_areas']:
                        st.warning(area_names.get(area, area))
                else:
                    st.success(" No significant weak areas!")

            st.markdown("---")
            st.subheader(" Personalized Recommendations")
            for rec in result['recommendations']:
                st.write(rec)

            st.markdown("---")
            st.subheader(" Your Personalized Learning Pathway")
            for i, step in enumerate(result['pathway'], 1):
                if '' in step:
                    st.error(f"**Step {i}:** {step}")
                elif '' in step:
                    st.warning(f"**Step {i}:** {step}")
                elif '' in step:
                    st.success(f"**Step {i}:** {step}")
                else:
                    st.info(f"**Step {i}:** {step}")

            st.markdown("---")
            st.success(" Good luck with your studies! ")

# 
# PAGE 4  RESEARCH RESULTS
# 
elif page == "Research Results":
    st.title(" Research Results & Validation")
    st.markdown("---")

    st.subheader(" Final Research Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ** Dataset:**
        - Total Students: 24,998
        - Features Used: 18
        - Performance Classes: 3
        - Training Set: 19,998 (80%)
        - Test Set: 5,000 (20%)
        """)
        st.markdown("""
        ** Model Performance:**
        - Best Model: Stacking Hybrid
        - Models Combined: XGBoost + RF + KNN
        - Test Accuracy: 85.00%
        - CV Mean Accuracy: 84.28%
        - CV Std Deviation: 0.0091
        """)

    with col2:
        st.markdown("""
        ** Recommendation System:**
        - Students Evaluated: 500
        - Recommendation Accuracy: 92.20%
        - Performance Classes: 3
        - Max Pathway Steps: 8
        """)
        st.markdown("""
        ** Research Objectives Met:**
        -  Student data analyzed
        -  ML model developed (85%)
        -  Recommendations generated
        -  Learning pathways created
        -  System evaluated & validated
        """)

    st.markdown("---")
    st.subheader(" Detailed Class Performance")
    report_data = {
        'Class': ['At-Risk', 'Good', 'Excellent', 'Weighted Avg'],
        'Precision': [0.93, 0.79, 0.73, 0.85],
        'Recall': [0.89, 0.88, 0.54, 0.85],
        'F1-Score': [0.91, 0.83, 0.62, 0.85],
        'Support': [2405, 2085, 510, 5000]
    }
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True)

    st.markdown("---")
    st.subheader(" Cross Validation Results")
    cv_data = {
        'Fold': [f'Fold {i}' for i in range(1, 11)],
        'Accuracy': [0.8560, 0.8415, 0.8535, 0.8365, 0.8400,
                     0.8450, 0.8380, 0.8415, 0.8524, 0.8234]
    }
    df_cv = pd.DataFrame(cv_data)
    fig = px.bar(df_cv, x='Fold', y='Accuracy',
                 title='10-Fold Cross Validation Scores',
                 color='Accuracy', color_continuous_scale='viridis')
    fig.add_hline(y=0.8428, line_dash="dash", line_color="red",
                  annotation_text="Mean Accuracy: 0.8428")
    fig.update_layout(yaxis_range=[0.80, 0.87])
    st.plotly_chart(fig, use_container_width=True)