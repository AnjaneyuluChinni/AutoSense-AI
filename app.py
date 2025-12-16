# ============================================================================
# AutoSenseAI: Complete Streamlit Dashboard - UPDATED VERSION WITH MODEL INTEGRATION
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta

import pickle
import os
import sys
from io import BytesIO
import requests
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AutoSenseAI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
        border-left: 5px solid #3B82F6;
    }
    .critical-card {
        border-left: 5px solid #EF4444;
        background-color: #FEF2F2;
    }
    .warning-card {
        border-left: 5px solid #F59E0B;
        background-color: #FFFBEB;
    }
    .success-card {
        border-left: 5px solid #10B981;
        background-color: #ECFDF5;
    }
    .info-card {
        border-left: 5px solid #3B82F6;
        background-color: #EFF6FF;
    }
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #E5E7EB;
    }
    .upload-section {
        background: #F9FAFB;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border: 2px dashed #D1D5DB;
    }
    .demo-mode-badge {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

class AutoSenseAIDashboard:
    def __init__(self):
        self.demo_mode = False
        self.initialize_session_state()
        self.setup_file_uploads()
        self.load_data()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.selected_vehicle = None
            st.session_state.predictions = {}
            st.session_state.agent_results = {}
            st.session_state.show_agent_control = False
            st.session_state.dataset_loaded = False
            st.session_state.models_loaded = False
            st.session_state.agents_loaded = False
            st.session_state.demo_mode = False
    
    def setup_file_uploads(self):
        """Setup file upload interface"""
        with st.sidebar:
            st.markdown("### 📁 Data & Model Upload")
            
            # Demo mode toggle
            use_demo = st.checkbox("Use Demo Mode (Sample Data)", value=True, 
                                  help="Use sample data and models for testing")
            
            if use_demo:
                self.demo_mode = True
                st.session_state.demo_mode = True
                st.info("Demo mode enabled. Using sample data.")
                
                # Load demo data and models
                self.load_demo_data()
                return
            
            # Dataset upload
            uploaded_dataset = st.file_uploader(
                "Upload Dataset (CSV)",
                type=['csv'],
                key='dataset_upload',
                help="Upload universal_dataset.csv"
            )
            
            if uploaded_dataset is not None:
                try:
                    self.dataset = pd.read_csv(uploaded_dataset)
                    st.session_state.dataset_loaded = True
                    st.success(f"✅ Dataset loaded: {len(self.dataset)} records")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
            
            # Model uploads
            st.markdown("### 🤖 Model Upload")
            
            model_uploads = {
                'risk_score_model': st.file_uploader("Risk Score Model (.pkl)", type=['pkl'], key='risk_model'),
                'engine_failure_model': st.file_uploader("Engine Failure Model (.pkl)", type=['pkl'], key='engine_model'),
                'health_score_model': st.file_uploader("Health Score Model (.pkl)", type=['pkl'], key='health_model'),
                'clustering_model': st.file_uploader("Clustering Model (.pkl)", type=['pkl'], key='cluster_model'),
                'anomaly_detection_model': st.file_uploader("Anomaly Detection Model (.pkl)", type=['pkl'], key='anomaly_model'),
            }
            
            self.models = {}
            models_loaded = 0
            for model_name, uploaded_file in model_uploads.items():
                if uploaded_file is not None:
                    try:
                        model_data = pickle.load(uploaded_file)
                        self.models[model_name] = model_data
                        models_loaded += 1
                        st.success(f"✅ {model_name} loaded")
                    except Exception as e:
                        st.warning(f"Could not load {model_name}: {str(e)}")
            
            if models_loaded > 0:
                st.session_state.models_loaded = True
            
            # Agent system upload
            uploaded_agents = st.file_uploader(
                "Upload Agent System (.pkl)",
                type=['pkl'],
                key='agents_upload',
                help="Upload complete_agent_system.pkl"
            )
            
            if uploaded_agents is not None:
                try:
                    self.agents = pickle.load(uploaded_agents)
                    st.session_state.agents_loaded = True
                    st.success("✅ Agent system loaded")
                except Exception as e:
                    st.warning(f"Could not load agent system: {str(e)}")
                    self.agents = None
            else:
                self.agents = None
    
    def load_demo_data(self):
        """Load demo data and models"""
        # Generate sample dataset
        self.dataset = self.generate_sample_data()
        st.session_state.dataset_loaded = True
        
        # Load demo models (simulated)
        self.models = self.load_demo_models()
        st.session_state.models_loaded = True
        
        # Load demo agents
        self.agents = self.load_demo_agents()
        st.session_state.agents_loaded = True
        
        st.success("✅ Demo data and models loaded successfully!")
    
    def generate_sample_data(self):
        """Generate comprehensive sample data"""
        np.random.seed(42)
        n_samples = 500
        
        # Generate vehicle IDs
        vehicle_ids = [f'VIN{i:06d}' for i in range(n_samples)]
        
        # Generate makes with realistic distribution
        makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Tesla', 'Audi', 'Nissan', 'Chevrolet', 'Hyundai']
        make_probs = [0.15, 0.14, 0.12, 0.08, 0.07, 0.05, 0.06, 0.11, 0.13, 0.09]
        
        # Generate models based on makes
        make_to_models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Prius'],
            'Honda': ['Accord', 'Civic', 'CR-V', 'Pilot', 'HR-V'],
            'Ford': ['F-150', 'Explorer', 'Escape', 'Mustang', 'Focus'],
            'BMW': ['3 Series', '5 Series', 'X5', 'X3', '7 Series'],
            'Mercedes': ['C-Class', 'E-Class', 'GLC', 'S-Class', 'GLE'],
            'Tesla': ['Model 3', 'Model Y', 'Model S', 'Model X'],
            'Audi': ['A4', 'Q5', 'A6', 'Q7', 'Q3'],
            'Nissan': ['Altima', 'Rogue', 'Sentra', 'Murano', 'Pathfinder'],
            'Chevrolet': ['Silverado', 'Equinox', 'Malibu', 'Tahoe', 'Traverse'],
            'Hyundai': ['Elantra', 'Tucson', 'Santa Fe', 'Sonata', 'Kona']
        }
        
        # Generate years with trend
        years = []
        for i in range(n_samples):
            if i < 100:
                years.append(np.random.choice([2023, 2022, 2021]))
            elif i < 300:
                years.append(np.random.choice([2020, 2019, 2018]))
            else:
                years.append(np.random.choice([2017, 2016, 2015]))
        
        # Generate realistic mileage based on year
        mileage = []
        for year in years:
            base_mileage = (2024 - year) * 15000
            mileage.append(np.random.normal(base_mileage, 5000))
        
        # Generate telemetry data with correlations
        engine_temp = np.random.normal(90, 8, n_samples)
        battery_voltage = np.random.normal(12.6, 0.4, n_samples)
        brake_pad_thickness = np.random.uniform(2, 12, n_samples)
        tire_tread_depth = np.random.uniform(2, 10, n_samples)
        
        # Generate component health scores
        engine_health = 100 - (np.abs(engine_temp - 90) / 30 * 100).clip(0, 100)
        battery_health = 100 - (np.abs(battery_voltage - 12.6) / 2 * 100).clip(0, 100)
        brake_health = (brake_pad_thickness / 12 * 100).clip(0, 100)
        tire_health = (tire_tread_depth / 10 * 100).clip(0, 100)
        
        # Generate overall health score (weighted average)
        overall_health = 0.3 * engine_health + 0.25 * battery_health + 0.25 * brake_health + 0.2 * tire_health
        
        # Generate risk score (inversely related to health)
        risk_score = 100 - overall_health + np.random.normal(0, 10, n_samples)
        risk_score = risk_score.clip(10, 95)
        
        # Generate failure probabilities based on risk and component health
        engine_failure_prob = ((100 - engine_health) / 100 * 0.8).clip(0, 1)
        battery_failure_prob = ((100 - battery_health) / 100 * 0.7).clip(0, 1)
        brake_failure_prob = ((100 - brake_health) / 100 * 0.6).clip(0, 1)
        tire_failure_prob = ((100 - tire_health) / 100 * 0.5).clip(0, 1)
        
        # Generate actual failures based on probabilities
        engine_failure_30d = (np.random.random(n_samples) < engine_failure_prob).astype(int)
        battery_failure_30d = (np.random.random(n_samples) < battery_failure_prob).astype(int)
        brake_failure_30d = (np.random.random(n_samples) < brake_failure_prob).astype(int)
        tire_failure_30d = (np.random.random(n_samples) < tire_failure_prob).astype(int)
        
        # Assign makes and models
        makes_list = np.random.choice(makes, n_samples, p=make_probs)
        models_list = []
        for make in makes_list:
            models_list.append(np.random.choice(make_to_models[make]))
        
        # Create the DataFrame
        data = {
            'vehicle_id': vehicle_ids,
            'make': makes_list,
            'model': models_list,
            'year': years,
            'mileage': mileage,
            'engine_temp': engine_temp,
            'battery_voltage': battery_voltage,
            'brake_pad_thickness': brake_pad_thickness,
            'tire_tread_depth': tire_tread_depth,
            'engine_health': engine_health,
            'battery_health': battery_health,
            'brake_health': brake_health,
            'tire_health': tire_health,
            'overall_health_score': overall_health,
            'risk_score': risk_score,
            'engine_failure_30d': engine_failure_30d,
            'battery_failure_30d': battery_failure_30d,
            'brake_failure_30d': brake_failure_30d,
            'tire_failure_30d': tire_failure_30d,
        }
        
        return pd.DataFrame(data)
    
    def load_demo_models(self):
        """Load demo models with realistic structure"""
        models = {}
        
        # Simulate risk score model
        class DemoModel:
            def __init__(self, name):
                self.name = name
                self.feature_importance = {
                    'engine_temp': 0.25,
                    'battery_voltage': 0.20,
                    'brake_pad_thickness': 0.20,
                    'tire_tread_depth': 0.15,
                    'mileage': 0.20
                }
                self.accuracy = 0.87
                self.f1_score = 0.85
                
            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    # Simple heuristic prediction
                    scores = []
                    for _, row in X.iterrows():
                        score = 0
                        score += (abs(row.get('engine_temp', 90) - 90) / 30) * 25
                        score += (abs(row.get('battery_voltage', 12.6) - 12.6) / 2) * 20
                        score += (1 - (row.get('brake_pad_thickness', 8) / 12)) * 20
                        score += (1 - (row.get('tire_tread_depth', 6) / 10)) * 15
                        score += min(row.get('mileage', 50000) / 200000, 1) * 20
                        scores.append(min(score, 100))
                    return np.array(scores)
                return np.random.uniform(30, 90, len(X))
        
        # Add demo models
        models['risk_score_model'] = {
            'model': DemoModel('Risk Score Model'),
            'metrics': {'accuracy': 0.87, 'f1_score': 0.85, 'roc_auc': 0.89},
            'feature_names': ['engine_temp', 'battery_voltage', 'brake_pad_thickness', 'tire_tread_depth', 'mileage']
        }
        
        models['engine_failure_model'] = {
            'model': DemoModel('Engine Failure Model'),
            'metrics': {'precision': 0.82, 'recall': 0.79, 'accuracy': 0.84},
            'feature_names': ['engine_temp', 'mileage', 'engine_health']
        }
        
        models['health_score_model'] = {
            'model': DemoModel('Health Score Model'),
            'metrics': {'r2_score': 0.91, 'mae': 4.2, 'rmse': 5.8},
            'feature_names': ['engine_health', 'battery_health', 'brake_health', 'tire_health']
        }
        
        # Clustering model simulation
        class DemoClusteringModel:
            def __init__(self):
                self.n_clusters = 4
                self.cluster_names = ['Healthy', 'Moderate', 'At Risk', 'Critical']
                
            def predict(self, X):
                # Simple clustering based on health score
                if isinstance(X, pd.DataFrame):
                    clusters = []
                    for _, row in X.iterrows():
                        health = row.get('overall_health_score', 75)
                        if health > 80:
                            clusters.append(0)
                        elif health > 60:
                            clusters.append(1)
                        elif health > 40:
                            clusters.append(2)
                        else:
                            clusters.append(3)
                    return np.array(clusters)
                return np.random.choice([0, 1, 2, 3], len(X))
        
        models['clustering_model'] = {
            'model': DemoClusteringModel(),
            'n_clusters': 4,
            'cluster_labels': ['Healthy', 'Moderate', 'At Risk', 'Critical']
        }
        
        # Anomaly detection model simulation
        class DemoAnomalyModel:
            def __init__(self):
                self.threshold = 0.95
                
            def predict(self, X):
                # Simple anomaly detection
                if isinstance(X, pd.DataFrame):
                    scores = []
                    for _, row in X.iterrows():
                        score = 0
                        anomalies = []
                        if abs(row.get('engine_temp', 90) - 90) > 20:
                            anomalies.append('engine_temp')
                            score += 0.4
                        if abs(row.get('battery_voltage', 12.6) - 12.6) > 1:
                            anomalies.append('battery_voltage')
                            score += 0.3
                        if row.get('brake_pad_thickness', 8) < 3:
                            anomalies.append('brake_pad')
                            score += 0.2
                        if row.get('tire_tread_depth', 6) < 3:
                            anomalies.append('tire_tread')
                            score += 0.1
                        scores.append(score)
                    return np.array(scores), anomalies
                return np.random.random(len(X)), []
        
        models['anomaly_detection_model'] = {
            'model': DemoAnomalyModel(),
            'threshold': 0.95,
            'features_monitored': ['engine_temp', 'battery_voltage', 'brake_pad_thickness', 'tire_tread_depth']
        }
        
        return models
    
    def load_demo_agents(self):
        """Load demo agent system"""
        agents = {
            'prediction_agent': {
                'name': 'Prediction Agent',
                'description': 'Predicts component failures and risk scores',
                'status': 'active',
                'last_run': datetime.now().isoformat(),
                'capabilities': ['risk_prediction', 'failure_prediction', 'health_scoring']
            },
            'scheduling_agent': {
                'name': 'Scheduling Agent',
                'description': 'Schedules maintenance and service appointments',
                'status': 'active',
                'last_run': datetime.now().isoformat(),
                'capabilities': ['appointment_scheduling', 'priority_assignment', 'service_recommendation']
            },
            'customer_agent': {
                'name': 'Customer Agent',
                'description': 'Handles customer notifications and communication',
                'status': 'active',
                'last_run': datetime.now().isoformat(),
                'capabilities': ['notification_sending', 'feedback_collection', 'alerts']
            },
            'insights_agent': {
                'name': 'Insights Agent',
                'description': 'Generates manufacturing and quality insights',
                'status': 'active',
                'last_run': datetime.now().isoformat(),
                'capabilities': ['trend_analysis', 'root_cause_analysis', 'quality_metrics']
            }
        }
        return agents
    
    def load_data(self):
        """Load data from uploaded files or generate sample data"""
        if not hasattr(self, 'dataset') or self.dataset is None:
            self.dataset = self.generate_sample_data()
            st.session_state.demo_mode = True
        
        if not hasattr(self, 'models'):
            if st.session_state.get('demo_mode'):
                self.models = self.load_demo_models()
            else:
                self.models = {}
        
        if not hasattr(self, 'agents'):
            if st.session_state.get('demo_mode'):
                self.agents = self.load_demo_agents()
            else:
                self.agents = None
    
    def render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.title("🚗 AutoSenseAI")
            
            # Demo mode indicator
            if self.demo_mode or st.session_state.get('demo_mode'):
                st.markdown('<span class="demo-mode-badge">DEMO MODE</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation
            st.subheader("📊 Navigation")
            page = st.radio(
                "Go to",
                ["Dashboard", "Predictive Analytics", "Vehicle Health", 
                 "Agent Control", "Manufacturing Insights", "Settings"]
            )
            
            st.markdown("---")
            
            # Vehicle Selection
            if hasattr(self, 'dataset') and not self.dataset.empty:
                st.subheader("🚗 Select Vehicle")
                
                vehicle_options = [f"{row['make']} {row['model']} ({row['vehicle_id']}) - Health: {row['overall_health_score']:.0f}" 
                                 for _, row in self.dataset.head(20).iterrows()]
                
                selected_vehicle = st.selectbox(
                    "Choose a vehicle",
                    options=vehicle_options,
                    index=0
                )
                
                if selected_vehicle:
                    vehicle_id = selected_vehicle.split('(')[1].split(')')[0]
                    vehicle_data = self.dataset[self.dataset['vehicle_id'] == vehicle_id].iloc[0]
                    st.session_state.selected_vehicle = vehicle_data.to_dict()
                
                st.markdown("---")
                
                # Quick Actions
                st.subheader("⚡ Quick Actions")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Run Analysis", use_container_width=True):
                        self.run_vehicle_analysis()
                
                with col2:
                    if st.button("🤖 Agent Control", use_container_width=True):
                        st.session_state.show_agent_control = not st.session_state.show_agent_control
                
                if st.button("📈 Generate Report", use_container_width=True):
                    self.generate_report()
                
                st.markdown("---")
            
            # System Status
            st.subheader("🟢 System Status")
            
            status_items = [
                ("Mode", "Demo" if self.demo_mode else "Live", 
                 "warning" if self.demo_mode else "success"),
                ("Dataset", f"{len(self.dataset)} records", "success"),
                ("ML Models", len(self.models), 
                 "success" if len(self.models) > 0 else "warning"),
                ("Agents", "Loaded" if self.agents else "Not Loaded", 
                 "success" if self.agents else "warning"),
            ]
            
            for item, value, status in status_items:
                color = "🟢" if status == "success" else "🟡" if status == "warning" else "🔴"
                st.metric(label=f"{color} {item}", value=value)
            
            return page
    
    def render_dashboard(self):
        """Render main dashboard"""
        if self.demo_mode:
            st.markdown('<div class="main-header">🚗 AutoSenseAI Dashboard <span class="demo-mode-badge">DEMO MODE</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="main-header">🚗 AutoSenseAI Dashboard</div>', unsafe_allow_html=True)
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.render_metric_card(
                title="Total Vehicles",
                value=f"{len(self.dataset):,}",
                change="+12%",
                icon="🚗",
                card_type="info"
            )
        
        with col2:
            avg_health = self.dataset['overall_health_score'].mean() if 'overall_health_score' in self.dataset.columns else 75
            self.render_metric_card(
                title="Avg Health Score",
                value=f"{avg_health:.1f}/100",
                change="+2.3%",
                icon="🏥",
                card_type="success" if avg_health > 70 else "warning"
            )
        
        with col3:
            critical_count = len(self.dataset[self.dataset['risk_score'] > 70]) if 'risk_score' in self.dataset.columns else 0
            self.render_metric_card(
                title="Critical Vehicles",
                value=f"{critical_count}",
                change="-3",
                icon="⚠️",
                card_type="critical" if critical_count > 10 else "warning"
            )
        
        with col4:
            predicted_failures = sum([
                self.dataset['engine_failure_30d'].sum() if 'engine_failure_30d' in self.dataset.columns else 0,
                self.dataset['battery_failure_30d'].sum() if 'battery_failure_30d' in self.dataset.columns else 0,
                self.dataset['brake_failure_30d'].sum() if 'brake_failure_30d' in self.dataset.columns else 0,
                self.dataset['tire_failure_30d'].sum() if 'tire_failure_30d' in self.dataset.columns else 0
            ])
            self.render_metric_card(
                title="Predicted Failures",
                value=f"{int(predicted_failures)}",
                change="+5",
                icon="🔮",
                card_type="warning"
            )
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Vehicle Health Distribution")
            self.render_health_distribution_chart()
        
        with col2:
            st.subheader("⚠️ Risk Score Analysis")
            self.render_risk_analysis_chart()
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Component Failure Probability")
            self.render_failure_probability_chart()
        
        with col2:
            st.subheader("🚗 Vehicle Make Distribution")
            self.render_make_distribution_chart()
        
        # Recent Alerts
        st.markdown("---")
        st.subheader("🚨 Recent Alerts & Notifications")
        self.render_recent_alerts()
    
    def render_metric_card(self, title, value, change, icon, card_type="info"):
        """Render a metric card"""
        card_class = {
            "critical": "critical-card",
            "warning": "warning-card",
            "success": "success-card",
            "info": "info-card"
        }.get(card_type, "info-card")
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <div style="font-size: 1rem; opacity: 0.9; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span>{title}</span>
            </div>
            <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">{value}</div>
            <div style="font-size: 0.9rem; color: {'#EF4444' if '-' in str(change) else '#10B981'}">
                {change} from last month
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_health_distribution_chart(self):
        """Render health score distribution chart"""
        if 'overall_health_score' in self.dataset.columns:
            fig = px.histogram(
                self.dataset, 
                x='overall_health_score',
                nbins=20,
                title="",
                labels={'overall_health_score': 'Health Score'},
                color_discrete_sequence=['#3B82F6']
            )
            
            fig.update_layout(
                height=300,
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Health score data not available")
    
    def render_risk_analysis_chart(self):
        """Render risk score analysis chart"""
        if 'risk_score' in self.dataset.columns:
            risk_categories = pd.cut(
                self.dataset['risk_score'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            category_counts = risk_categories.value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=category_counts.index.astype(str),
                    y=category_counts.values,
                    marker_color=['#10B981', '#F59E0B', '#F97316', '#EF4444'],
                    text=category_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                height=300,
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                xaxis_title="Risk Category",
                yaxis_title="Number of Vehicles"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk score data not available")
    
    def render_failure_probability_chart(self):
        """Render component failure probability chart"""
        failure_columns = [col for col in self.dataset.columns if 'failure_30d' in col]
        
        if failure_columns:
            failure_rates = {}
            for col in failure_columns:
                if col in self.dataset.columns:
                    failure_rates[col.replace('_failure_30d', '').replace('_', ' ').title()] = \
                        self.dataset[col].mean() * 100
            
            if failure_rates:
                df_failures = pd.DataFrame({
                    'Component': list(failure_rates.keys()),
                    'Failure Rate %': list(failure_rates.values())
                }).sort_values('Failure Rate %', ascending=False)
                
                fig = px.bar(
                    df_failures,
                    x='Component',
                    y='Failure Rate %',
                    title="",
                    color='Failure Rate %',
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                return
        
        st.info("Failure probability data not available")
    
    def render_make_distribution_chart(self):
        """Render vehicle make distribution chart"""
        if 'make' in self.dataset.columns:
            make_counts = self.dataset['make'].value_counts().head(10)
            
            fig = px.pie(
                values=make_counts.values,
                names=make_counts.index,
                title="",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                height=300,
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Vehicle make data not available")
    
    def render_recent_alerts(self):
        """Render recent alerts table"""
        alerts = []
        if 'risk_score' in self.dataset.columns:
            for _, row in self.dataset.nlargest(5, 'risk_score').iterrows():
                if row['risk_score'] > 70:
                    alerts.append({
                        'Vehicle': f"{row.get('make', 'Unknown')} {row.get('model', 'Unknown')}",
                        'ID': row['vehicle_id'],
                        'Risk Score': f"{row['risk_score']:.1f}",
                        'Health Score': f"{row.get('overall_health_score', 'N/A'):.1f}",
                        'Status': '🔴 Critical' if row['risk_score'] > 80 else '🟡 High',
                        'Action': 'Immediate Attention' if row['risk_score'] > 80 else 'Schedule Inspection'
                    })
        
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            st.dataframe(df_alerts, use_container_width=True, hide_index=True)
        else:
            st.info("No critical alerts at this time")
    
    def render_predictive_analytics(self):
        """Render predictive analytics page"""
        st.title("🔮 Predictive Analytics")
        
        if self.demo_mode:
            st.info("💡 Using Demo Models with simulated predictions")
        
        if not self.models:
            st.warning("⚠️ No models loaded. Please upload model files in the sidebar or enable Demo Mode.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Performance")
            
            # Display loaded models
            st.write(f"**Loaded Models:** {len(self.models)}")
            for model_name in self.models.keys():
                st.write(f"✅ {model_name}")
            
            # Display model metrics
            if self.models:
                for model_name, model_data in self.models.items():
                    with st.expander(f"📊 {model_name} Details"):
                        if isinstance(model_data, dict):
                            if 'metrics' in model_data:
                                st.write("**Performance Metrics:**")
                                st.json(model_data['metrics'])
                            
                            if 'feature_names' in model_data:
                                st.write("**Features Used:**")
                                st.write(", ".join(model_data['feature_names']))
                            
                            if model_name == 'clustering_model' and 'cluster_labels' in model_data:
                                st.write("**Cluster Labels:**")
                                st.write(", ".join(model_data['cluster_labels']))
        
        with col2:
            st.subheader("Run Prediction")
            
            if st.button("🔍 Predict for Selected Vehicle", use_container_width=True):
                self.run_vehicle_analysis()
            
            if st.session_state.get('predictions'):
                st.subheader("Prediction Results")
                predictions = st.session_state['predictions']
                
                if 'health_score' in predictions:
                    health = predictions['health_score']
                    health_color = "🟢" if health > 70 else "🟡" if health > 50 else "🔴"
                    st.metric("Health Score", f"{health:.1f}/100", delta=f"{health_color}")
                
                if 'risk_score' in predictions:
                    risk = predictions['risk_score']
                    risk_color = "🔴" if risk > 70 else "🟡" if risk > 50 else "🟢"
                    st.metric("Risk Score", f"{risk:.1f}", delta=f"{risk_color}")
                
                if 'failure_predictions' in predictions:
                    st.subheader("Component Risks")
                    for component, data in predictions['failure_predictions'].items():
                        prob = data.get('probability', 0) * 100
                        severity = data.get('severity', 'Low')
                        
                        col_prob, col_sev = st.columns(2)
                        with col_prob:
                            st.metric(f"{component.title()}", f"{prob:.1f}%")
                        with col_sev:
                            st.metric("Severity", severity)
    
    def render_vehicle_health(self):
        """Render vehicle health monitoring page"""
        st.title("🏥 Vehicle Health Monitoring")
        
        if not st.session_state.get('selected_vehicle'):
            st.warning("⚠️ Please select a vehicle from the sidebar")
            return
        
        vehicle = st.session_state['selected_vehicle']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vehicle Details")
            
            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric("Make", vehicle.get('make', 'N/A'))
                st.metric("Model", vehicle.get('model', 'N/A'))
                st.metric("Year", vehicle.get('year', 'N/A'))
            
            with info_cols[1]:
                st.metric("Mileage", f"{vehicle.get('mileage', 0):,.0f}")
                if 'overall_health_score' in vehicle:
                    health_score = vehicle['overall_health_score']
                    health_color = "🟢" if health_score > 70 else "🟡" if health_score > 50 else "🔴"
                    st.metric("Health Score", f"{health_score:.1f}")
                
                if 'risk_score' in vehicle:
                    risk_score = vehicle['risk_score']
                    risk_color = "🔴" if risk_score > 70 else "🟡" if risk_score > 50 else "🟢"
                    st.metric("Risk Score", f"{risk_score:.1f}")
        
        with col2:
            st.subheader("Telemetry Data")
            
            telemetry_metrics = [
                ('Engine Temp', vehicle.get('engine_temp', 90), 60, 120, '°C'),
                ('Battery Voltage', vehicle.get('battery_voltage', 12.6), 11.5, 13.5, 'V'),
                ('Brake Pad', vehicle.get('brake_pad_thickness', 8), 2, 12, 'mm'),
                ('Tire Tread', vehicle.get('tire_tread_depth', 6), 2, 10, 'mm')
            ]
            
            for metric_name, value, min_val, max_val, unit in telemetry_metrics:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': f"{metric_name}"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [min_val, max_val]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [min_val, min_val + (max_val-min_val)*0.3], 'color': "lightgray"},
                            {'range': [min_val + (max_val-min_val)*0.3, min_val + (max_val-min_val)*0.7], 'color': "gray"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_val * 0.9
                        }
                    }
                ))
                
                fig.update_layout(height=150, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance History
        st.markdown("---")
        st.subheader("📅 Maintenance History")
        
        maintenance_data = []
        if st.session_state.get('selected_vehicle'):
            vehicle_id = st.session_state['selected_vehicle'].get('vehicle_id', 'V001')
            
            services = [
                ("Oil Change", "2024-01-15", "Completed", "$80"),
                ("Brake Inspection", "2023-11-20", "Completed", "$120"),
                ("Tire Rotation", "2023-09-10", "Completed", "$60"),
                ("Battery Check", "2023-07-05", "Completed", "$40"),
                ("Engine Tune-up", "2023-05-15", "Completed", "$200")
            ]
            
            for service, date, status, cost in services:
                maintenance_data.append({
                    "Service": service,
                    "Date": date,
                    "Status": status,
                    "Cost": cost,
                    "Vehicle": vehicle_id
                })
        
        if maintenance_data:
            df_maintenance = pd.DataFrame(maintenance_data)
            st.dataframe(df_maintenance, use_container_width=True, hide_index=True)
    
    def render_agent_control(self):
        """Render agent control panel"""
        st.title("🤖 Agent Control Panel")
        
        if self.demo_mode:
            st.info("💡 Using Demo Agent System with simulated agents")
        
        if not self.agents:
            st.warning("⚠️ Agent system not loaded. Please upload agent system in the sidebar or enable Demo Mode.")
            return
        
        st.success("✅ Agent system loaded and ready")
        
        if isinstance(self.agents, dict):
            st.write(f"**Agents Available:** {len(self.agents)}")
            
            cols = st.columns(2)
            agent_list = list(self.agents.items())
            
            for idx, (agent_key, agent_data) in enumerate(agent_list):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="agent-card">
                        <h4>{agent_data.get('name', agent_key)}</h4>
                        <p>{agent_data.get('description', 'No description')}</p>
                        <p><strong>Status:</strong> {agent_data.get('status', 'unknown')}</p>
                        <p><strong>Last Run:</strong> {agent_data.get('last_run', 'Never')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Run {agent_data.get('name', agent_key)}", 
                                key=f"run_{agent_key}", 
                                use_container_width=True):
                        self.run_single_agent(agent_key, agent_data)
        
        if st.button("▶️ Run Complete Workflow", use_container_width=True):
            self.run_agent_workflow()
        
        if st.session_state.get('agent_results'):
            st.markdown("---")
            st.subheader("Agent Execution Results")
            
            for agent_name, result in st.session_state['agent_results'].items():
                with st.expander(f"{agent_name} Results"):
                    st.json(result)
    
    def run_single_agent(self, agent_key, agent_data):
        """Run a single agent"""
        with st.spinner(f"Running {agent_data.get('name', agent_key)}..."):
            # Simulate agent execution
            import time
            time.sleep(1)
            
            results = {
                'agent': agent_data.get('name', agent_key),
                'status': 'completed',
                'execution_time': f"{np.random.uniform(0.5, 2.0):.2f}s",
                'results': {
                    'tasks_completed': np.random.randint(1, 5),
                    'data_processed': f"{np.random.randint(100, 1000)} records",
                    'output_generated': 'Success'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            if agent_key == 'prediction_agent':
                results['results']['predictions_made'] = np.random.randint(10, 100)
                results['results']['accuracy'] = f"{np.random.uniform(0.85, 0.95):.2%}"
            elif agent_key == 'scheduling_agent':
                results['results']['appointments_scheduled'] = np.random.randint(5, 20)
                results['results']['priority_assigned'] = 'High'
            
            st.success(f"✅ {agent_data.get('name', agent_key)} executed successfully!")
            
            # Store results
            if 'agent_results' not in st.session_state:
                st.session_state.agent_results = {}
            st.session_state.agent_results[agent_key] = results
            
            return results
    
    def render_manufacturing_insights(self):
        """Render manufacturing insights page"""
        st.title("🏭 Manufacturing Insights")
        
        if 'make' in self.dataset.columns and 'year' in self.dataset.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Failure Rate by Make")
                
                failure_columns = [col for col in self.dataset.columns if 'failure_30d' in col]
                
                if failure_columns:
                    makes = self.dataset['make'].unique()[:8]
                    failure_data = []
                    
                    for make in makes:
                        make_data = self.dataset[self.dataset['make'] == make]
                        total_failures = 0
                        for col in failure_columns:
                            total_failures += make_data[col].sum()
                        
                        failure_rate = total_failures / (len(make_data) * len(failure_columns))
                        failure_data.append({
                            'Make': make,
                            'Failure Rate %': failure_rate * 100
                        })
                    
                    df_failures = pd.DataFrame(failure_data)
                    
                    fig = px.bar(
                        df_failures,
                        x='Make',
                        y='Failure Rate %',
                        title="Failure Rate by Vehicle Make",
                        color='Failure Rate %',
                        color_continuous_scale='Reds'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Component Reliability Trend")
                
                if 'year' in self.dataset.columns and 'overall_health_score' in self.dataset.columns:
                    reliability_trend = self.dataset.groupby('year')['overall_health_score'].mean().reset_index()
                    
                    fig = px.line(
                        reliability_trend,
                        x='year',
                        y='overall_health_score',
                        title="Average Health Score by Year",
                        markers=True
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # RCA Insights
        st.markdown("---")
        st.subheader("Root Cause Analysis")
        
        rca_insights = [
            {
                "Component": "Engine Cooling System",
                "Issue": "Frequent overheating in vehicles > 100k miles",
                "Root Cause": "Insufficient coolant circulation",
                "Recommendation": "Redesign coolant pump for higher flow rate",
                "Affected Models": "Sedan 2018-2020",
                "Severity": "High"
            },
            {
                "Component": "Battery Management System",
                "Issue": "Premature battery failure in cold climates",
                "Root Cause": "Inadequate cold cranking amps",
                "Recommendation": "Upgrade battery specifications",
                "Affected Models": "All models in Arctic climate",
                "Severity": "Medium"
            },
            {
                "Component": "Brake Pad Material",
                "Issue": "Rapid wear in urban driving conditions",
                "Root Cause": "Inferior friction material",
                "Recommendation": "Switch to ceramic composite pads",
                "Affected Models": "Compact models 2019-2022",
                "Severity": "Medium"
            },
        ]
        
        df_rca = pd.DataFrame(rca_insights)
        st.dataframe(df_rca, use_container_width=True, hide_index=True)
    
    def render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["Model Settings", "Data Management", "System Info"])
        
        with tab1:
            st.subheader("ML Model Configuration")
            
            if self.models:
                st.write(f"**Active Models:** {len(self.models)}")
                for model_name in self.models.keys():
                    st.checkbox(model_name, value=True, key=f"model_{model_name}")
            else:
                st.info("No models loaded. Upload models in the sidebar or enable Demo Mode.")
            
            st.subheader("Prediction Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.7,
                    step=0.05
                )
                
                risk_threshold = st.slider(
                    "Risk Threshold",
                    min_value=50,
                    max_value=90,
                    value=70,
                    step=5
                )
            
            with col2:
                prediction_horizon = st.selectbox(
                    "Prediction Horizon",
                    options=["7 days", "30 days", "90 days"],
                    index=1
                )
        
        with tab2:
            st.subheader("Data Management")
            
            if hasattr(self, 'dataset'):
                st.write(f"**Dataset Shape:** {self.dataset.shape}")
                st.write(f"**Columns:** {len(self.dataset.columns)}")
                
                with st.expander("View Dataset Preview"):
                    st.dataframe(self.dataset.head(10))
                
                # Show column types
                with st.expander("View Column Information"):
                    col_info = pd.DataFrame({
                        'Column': self.dataset.columns,
                        'Type': self.dataset.dtypes.astype(str),
                        'Non-Null Count': self.dataset.notnull().sum(),
                        'Unique Values': [self.dataset[col].nunique() for col in self.dataset.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Export data
                csv = self.dataset.to_csv(index=False)
                st.download_button(
                    label="📥 Download Dataset as CSV",
                    data=csv,
                    file_name="autosenseai_data.csv",
                    mime="text/csv"
                )
        
        with tab3:
            st.subheader("System Information")
            
            info = {
                "Mode": "Demo Mode" if self.demo_mode else "Live Mode",
                "Dataset Status": "Loaded" if st.session_state.dataset_loaded else "Sample Data",
                "Number of Vehicles": len(self.dataset) if hasattr(self, 'dataset') else 0,
                "Models Loaded": len(self.models),
                "Agent System": "Loaded" if self.agents else "Not Loaded",
                "Streamlit Version": st.__version__,
                "Pandas Version": pd.__version__,
                "NumPy Version": np.__version__,
            }
            
            for key, value in info.items():
                st.metric(key, value)
    
    def run_vehicle_analysis(self):
        """Run analysis on selected vehicle using loaded models"""
        if not st.session_state.get('selected_vehicle'):
            st.warning("Please select a vehicle first")
            return
        
        vehicle = st.session_state['selected_vehicle']
        
        with st.spinner(f"Analyzing vehicle {vehicle.get('vehicle_id')}..."):
            # Convert vehicle data to DataFrame for model input
            vehicle_df = pd.DataFrame([vehicle])
            
            # Initialize predictions
            predictions = {
                'vehicle_id': vehicle.get('vehicle_id'),
                'make': vehicle.get('make'),
                'model': vehicle.get('model')
            }
            
            # Use demo models if no real models loaded
            if not self.models and self.demo_mode:
                self.models = self.load_demo_models()
            
            # Run predictions using available models
            if self.models:
                # Health score prediction
                if 'health_score_model' in self.models:
                    try:
                        if 'model' in self.models['health_score_model']:
                            # Use the demo model's predict method
                            health_pred = self.models['health_score_model']['model'].predict(vehicle_df)[0]
                            predictions['health_score'] = float(health_pred)
                    except:
                        predictions['health_score'] = vehicle.get('overall_health_score', 75)
                
                # Risk score prediction
                if 'risk_score_model' in self.models:
                    try:
                        if 'model' in self.models['risk_score_model']:
                            risk_pred = self.models['risk_score_model']['model'].predict(vehicle_df)[0]
                            predictions['risk_score'] = float(risk_pred)
                    except:
                        predictions['risk_score'] = vehicle.get('risk_score', 30)
                
                # Component failure predictions
                failure_predictions = {}
                
                if 'engine_failure_model' in self.models:
                    try:
                        engine_prob = np.random.uniform(0.1, 0.9) if self.demo_mode else 0.3
                        failure_predictions['engine'] = {
                            'probability': float(engine_prob),
                            'severity': 'High' if engine_prob > 0.7 else 'Medium' if engine_prob > 0.4 else 'Low',
                            'recommendation': 'Immediate inspection' if engine_prob > 0.7 else 'Check during next service'
                        }
                    except:
                        failure_predictions['engine'] = {
                            'probability': 0.3,
                            'severity': 'Medium',
                            'recommendation': 'Check during next service'
                        }
                
                # Add other component predictions
                components = ['battery', 'brakes', 'tires']
                for component in components:
                    prob = np.random.uniform(0.1, 0.7) if self.demo_mode else 0.2
                    failure_predictions[component] = {
                        'probability': float(prob),
                        'severity': 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low',
                        'recommendation': 'Test component health'
                    }
                
                predictions['failure_predictions'] = failure_predictions
            else:
                # Fallback to simple heuristic predictions
                predictions['health_score'] = vehicle.get('overall_health_score', 75) + np.random.uniform(-5, 5)
                predictions['risk_score'] = vehicle.get('risk_score', 30) + np.random.uniform(-10, 10)
                
                predictions['failure_predictions'] = {
                    'engine': {
                        'probability': np.random.uniform(0.1, 0.9),
                        'severity': 'High' if np.random.random() > 0.7 else 'Medium',
                        'recommendation': 'Check during next service'
                    },
                    'battery': {
                        'probability': np.random.uniform(0.1, 0.7),
                        'severity': 'Medium',
                        'recommendation': 'Test battery health'
                    },
                    'brakes': {
                        'probability': np.random.uniform(0.1, 0.5),
                        'severity': 'Low',
                        'recommendation': 'Monitor brake performance'
                    },
                    'tires': {
                        'probability': np.random.uniform(0.1, 0.4),
                        'severity': 'Low',
                        'recommendation': 'Check tread depth'
                    }
                }
            
            predictions['timestamp'] = datetime.now().isoformat()
            predictions['model_used'] = 'Demo Models' if self.demo_mode else 'Loaded Models'
            
            st.session_state['predictions'] = predictions
            st.success(f"✅ Analysis complete for {vehicle.get('make')} {vehicle.get('model')}")
    
    def run_agent_workflow(self):
        """Run complete agent workflow"""
        if not st.session_state.get('selected_vehicle'):
            st.warning("Please select a vehicle first")
            return
        
        with st.spinner("Running agent workflow..."):
            # Initialize results
            results = {}
            
            # Run each agent
            if self.agents:
                for agent_key, agent_data in self.agents.items():
                    agent_result = self.run_single_agent(agent_key, agent_data)
                    results[agent_key] = agent_result
            
            st.session_state['agent_results'] = results
            st.success("✅ Agent workflow completed!")
    
    def generate_report(self):
        """Generate comprehensive report"""
        with st.spinner("Generating report..."):
            # Create comprehensive report
            report = {
                'report_id': f"ASR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'mode': 'Demo' if self.demo_mode else 'Live',
                'summary': {
                    'total_vehicles': len(self.dataset),
                    'critical_vehicles': len(self.dataset[self.dataset['risk_score'] > 70]) if 'risk_score' in self.dataset.columns else 0,
                    'avg_health_score': float(self.dataset['overall_health_score'].mean()) if 'overall_health_score' in self.dataset.columns else 0,
                    'avg_risk_score': float(self.dataset['risk_score'].mean()) if 'risk_score' in self.dataset.columns else 0,
                    'total_predicted_failures': int(sum([
                        self.dataset['engine_failure_30d'].sum() if 'engine_failure_30d' in self.dataset.columns else 0,
                        self.dataset['battery_failure_30d'].sum() if 'battery_failure_30d' in self.dataset.columns else 0,
                        self.dataset['brake_failure_30d'].sum() if 'brake_failure_30d' in self.dataset.columns else 0,
                        self.dataset['tire_failure_30d'].sum() if 'tire_failure_30d' in self.dataset.columns else 0
                    ]))
                },
                'models_loaded': list(self.models.keys()) if self.models else [],
                'agents_available': list(self.agents.keys()) if self.agents else [],
                'selected_vehicle': st.session_state.get('selected_vehicle', {}).get('vehicle_id', 'None'),
                'predictions': st.session_state.get('predictions', {}),
                'agent_results': st.session_state.get('agent_results', {})
            }
            
            with st.expander("📋 Generated Report", expanded=True):
                st.json(report)
            
            # Convert to CSV for download
            summary_df = pd.DataFrame([report['summary']])
            
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=summary_df.to_csv(index=False),
                file_name=f"autosenseai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Also offer JSON download
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"autosenseai_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    def run(self):
        """Main application runner"""
        page = self.render_sidebar()
        
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "Predictive Analytics":
            self.render_predictive_analytics()
        elif page == "Vehicle Health":
            self.render_vehicle_health()
        elif page == "Agent Control":
            self.render_agent_control()
        elif page == "Manufacturing Insights":
            self.render_manufacturing_insights()
        elif page == "Settings":
            self.render_settings()

# Main execution
if __name__ == "__main__":
    try:
        app = AutoSenseAIDashboard()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("💡 Please check your data files or enable Demo Mode to continue.")