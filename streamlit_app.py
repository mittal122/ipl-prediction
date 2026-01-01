"""
IPL First Innings Score Prediction - Streamlit Web App
Interactive UI for predicting IPL first innings scores
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# Page configuration
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the data loading and model training
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the IPL dataset"""
    df = pd.read_csv('dataset/ipl.csv')
    
    # Data cleaning
    columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
    df.drop(labels=columns_to_remove, axis=1, inplace=True)
    
    # Keep only consistent teams
    consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                        'Delhi Daredevils', 'Sunrisers Hyderabad']
    
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    
    # Remove first 5 overs data
    df = df[df['overs'] >= 5.0]
    
    # Convert date column
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    
    return df, consistent_teams

@st.cache_resource
def train_models(df):
    """Train all machine learning models"""
    # One-hot encoding
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    
    # Rearrange columns
    encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 
                             'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders', 
                             'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                             'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                             'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 
                             'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders', 
                             'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                             'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
                             'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    
    # Train-test split
    X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
    X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
    
    y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
    y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values
    
    X_train = X_train.drop(labels='date', axis=1)
    X_test = X_test.drop(labels='date', axis=1)
    
    # Train models
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models['Linear Regression'] = {
        'model': lr,
        'mae': mae(y_test, y_pred_lr),
        'rmse': np.sqrt(mse(y_test, y_pred_lr))
    }
    
    # Decision Tree
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    models['Decision Tree'] = {
        'model': dt,
        'mae': mae(y_test, y_pred_dt),
        'rmse': np.sqrt(mse(y_test, y_pred_dt))
    }
    
    # Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    models['Random Forest'] = {
        'model': rf,
        'mae': mae(y_test, y_pred_rf),
        'rmse': np.sqrt(mse(y_test, y_pred_rf))
    }
    
    # AdaBoost
    adb = AdaBoostRegressor(estimator=LinearRegression(), n_estimators=100)
    adb.fit(X_train, y_train)
    y_pred_adb = adb.predict(X_test)
    models['AdaBoost'] = {
        'model': adb,
        'mae': mae(y_test, y_pred_adb),
        'rmse': np.sqrt(mse(y_test, y_pred_adb))
    }
    
    return models, X_train.shape[0], X_test.shape[0]

def predict_score(model, batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    """Predict the final score using the trained model"""
    temp_array = []
    
    # Batting team encoding
    teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 
             'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
             'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    
    temp_array.extend([1 if batting_team == team else 0 for team in teams])
    temp_array.extend([1 if bowling_team == team else 0 for team in teams])
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])
    
    temp_array = np.array([temp_array])
    return int(model.predict(temp_array)[0])

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🏏 IPL First Innings Score Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict IPL match scores using Machine Learning</div>', unsafe_allow_html=True)
    
    # Load data and train models
    with st.spinner('Loading data and training models...'):
        df, teams = load_and_prepare_data()
        models, train_size, test_size = train_models(df)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=0,
        help="Choose the machine learning model for prediction"
    )
    
    selected_model = models[model_name]['model']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("MAE", f"{models[model_name]['mae']:.2f}")
    st.sidebar.metric("RMSE", f"{models[model_name]['rmse']:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Dataset Info")
    st.sidebar.info(f"**Training Data:** {train_size:,} records\n\n**Test Data:** {test_size:,} records")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Model Comparison", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Enter Match Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏏 Teams")
            batting_team = st.selectbox("Batting Team", teams, key="batting")
            bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowling")
        
        with col2:
            st.markdown("#### 📊 Current Match Status")
            overs = st.number_input("Overs Completed", min_value=5.0, max_value=19.5, value=10.0, step=0.1)
            runs = st.number_input("Current Runs", min_value=0, max_value=300, value=80, step=1)
            wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1)
        
        with col3:
            st.markdown("#### 📈 Last 5 Overs Stats")
            runs_in_prev_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=45, step=1)
            wickets_in_prev_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, value=1, step=1)
        
        st.markdown("---")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            if st.button("🔮 Predict Final Score", use_container_width=True, type="primary"):
                with st.spinner('Making prediction...'):
                    predicted_score = predict_score(
                        selected_model, batting_team, bowling_team, 
                        overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5
                    )
                    
                    # Display prediction
                    st.markdown(f"""
                        <div class="prediction-box">
                            <div style="font-size: 1rem; margin-bottom: 0.5rem;">Predicted Final Score</div>
                            <div style="font-size: 3rem; font-weight: bold;">{predicted_score}</div>
                            <div style="font-size: 1rem; margin-top: 0.5rem;">Range: {predicted_score - 10} - {predicted_score + 5}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    st.markdown("### 📊 Match Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Run Rate", f"{(runs/overs):.2f}")
                    with col2:
                        st.metric("Required Run Rate", f"{((predicted_score - runs)/(20 - overs)):.2f}")
                    with col3:
                        st.metric("Projected Score", predicted_score)
                    with col4:
                        wickets_left = 10 - wickets
                        st.metric("Wickets in Hand", wickets_left)
                    
                    # Progress visualization
                    overs_completed_pct = (overs / 20) * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = overs_completed_pct,
                        title = {'text': "Match Progress"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"},
                                {'range': [66, 100], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 100
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🏆 Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = {
            'Model': list(models.keys()),
            'MAE': [models[m]['mae'] for m in models.keys()],
            'RMSE': [models[m]['rmse'] for m in models.keys()]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MAE comparison
            fig_mae = px.bar(
                comparison_df, 
                x='Model', 
                y='MAE',
                title='Mean Absolute Error (Lower is Better)',
                color='MAE',
                color_continuous_scale='RdYlGn_r'
            )
            fig_mae.update_layout(showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                comparison_df, 
                x='Model', 
                y='RMSE',
                title='Root Mean Squared Error (Lower is Better)',
                color='RMSE',
                color_continuous_scale='RdYlGn_r'
            )
            fig_rmse.update_layout(showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Display metrics table
        st.markdown("### 📋 Detailed Metrics")
        st.dataframe(
            comparison_df.style.highlight_min(axis=0, subset=['MAE', 'RMSE'], color='lightgreen'),
            use_container_width=True
        )
    
    with tab3:
        st.markdown("### 📖 About This Project")
        
        st.markdown("""
        #### 🎯 Project Overview
        This application uses **Machine Learning** to predict the final score of IPL first innings based on:
        - Current match situation (overs, runs, wickets)
        - Team performance
        - Recent momentum (last 5 overs statistics)
        
        #### 🔬 Models Used
        - **Linear Regression**: Simple and effective baseline model
        - **Decision Tree**: Captures non-linear patterns
        - **Random Forest**: Ensemble method for robust predictions
        - **AdaBoost**: Boosting algorithm for improved accuracy
        
        #### 📊 Dataset
        - **Training Period**: IPL Seasons 2008-2016
        - **Testing Period**: IPL Season 2017+
        - **Teams Included**: 8 consistent IPL teams
        
        #### 🚀 Features
        - Real-time score predictions
        - Multiple ML models to choose from
        - Interactive visualizations
        - Match analysis metrics
        
        #### ⚠️ Note
        IPL matches are highly unpredictable! The game can change dramatically in just a few overs.
        Use these predictions as estimates, not certainties.
        
        ---
        
        **Made with ❤️ using Streamlit and Scikit-learn**
        """)
        
        # Display sample predictions
        st.markdown("### 🎲 Sample Historical Predictions")
        
        sample_data = {
            'Match': [
                'KKR vs DD (2018)',
                'SRH vs RCB (2018)',
                'MI vs KXIP (2018)',
                'MI vs KXIP (2019)'
            ],
            'Actual Score': [200, 146, 186, 176],
            'Predicted Range': ['159-174', '138-153', '180-195', '179-194']
        }
        
        st.table(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
