from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Optional, Any
import json
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State definition for agent workflow"""
    vehicle_id: str
    vehicle_data: Dict[str, Any]
    telemetry_data: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    risk_score: float
    recommended_actions: List[str]
    scheduled_service: Optional[Dict[str, Any]]
    customer_notification: Optional[Dict[str, Any]]
    customer_feedback: Optional[Dict[str, Any]]
    manufacturing_insights: Optional[Dict[str, Any]]
    current_step: str
    workflow_completed: bool
    timestamp: str

class MasterAgent:
    def __init__(self):
        self.workflow = self._create_workflow()
        self.agent_registry = self._initialize_agents()
        logger.info("Master Agent initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all worker agents"""
        return {
            "telemetry_agent": TelemetryAgent(),
            "prediction_agent": PredictionAgent(),
            "diagnosis_agent": DiagnosisAgent(),
            "scheduling_agent": SchedulingAgent(),
            "customer_agent": CustomerAgent(),
            "insights_agent": ManufacturingInsightsAgent()
        }
    
    def _create_workflow(self):
        """Create LangGraph workflow for agent orchestration"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("collect_telemetry", self.collect_telemetry_data)
        workflow.add_node("analyze_health", self.analyze_vehicle_health)
        workflow.add_node("predict_failures", self.predict_potential_failures)
        workflow.add_node("diagnose_issues", self.diagnose_issues)
        workflow.add_node("calculate_risk", self.calculate_risk_score)
        workflow.add_node("plan_actions", self.plan_recommended_actions)
        workflow.add_node("decide_scheduling", self.decide_scheduling_needs)
        workflow.add_node("schedule_service", self.schedule_service)
        workflow.add_node("notify_customer", self.notify_customer)
        workflow.add_node("collect_feedback", self.collect_customer_feedback)
        workflow.add_node("generate_insights", self.generate_manufacturing_insights)
        
        # Define workflow edges
        workflow.set_entry_point("collect_telemetry")
        
        # Main flow
        workflow.add_edge("collect_telemetry", "analyze_health")
        workflow.add_edge("analyze_health", "predict_failures")
        workflow.add_edge("predict_failures", "diagnose_issues")
        workflow.add_edge("diagnose_issues", "calculate_risk")
        workflow.add_edge("calculate_risk", "plan_actions")
        
        # Conditional branching based on risk
        workflow.add_conditional_edges(
            "plan_actions",
            self._should_schedule_service,
            {
                "immediate": "schedule_service",
                "soon": "decide_scheduling",
                "monitor": "generate_insights"
            }
        )
        
        workflow.add_edge("decide_scheduling", "schedule_service")
        workflow.add_edge("schedule_service", "notify_customer")
        workflow.add_edge("notify_customer", "collect_feedback")
        workflow.add_edge("collect_feedback", "generate_insights")
        workflow.add_edge("generate_insights", END)
        
        return workflow.compile()
    
    def collect_telemetry_data(self, state: AgentState) -> AgentState:
        """Collect and process telemetry data"""
        logger.info(f"Collecting telemetry for vehicle {state.get('vehicle_id')}")
        
        telemetry_agent = self.agent_registry["telemetry_agent"]
        telemetry_data = telemetry_agent.collect_data(state["vehicle_id"])
        
        state["telemetry_data"] = telemetry_data
        state["current_step"] = "collect_telemetry"
        state["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Collected {len(telemetry_data)} telemetry parameters")
        return state
    
    def analyze_vehicle_health(self, state: AgentState) -> AgentState:
        """Analyze overall vehicle health"""
        logger.info("Analyzing vehicle health...")
        
        # Simple health scoring based on telemetry
        telemetry = state["telemetry_data"]
        health_score = 100
        
        # Adjust based on critical parameters
        if telemetry.get("engine_temp", 90) > 110:
            health_score -= 20
        if telemetry.get("oil_pressure", 45) < 30:
            health_score -= 25
        if telemetry.get("battery_voltage", 12.6) < 11.8:
            health_score -= 15
        
        state["vehicle_data"]["health_score"] = max(health_score, 0)
        state["current_step"] = "analyze_health"
        
        logger.info(f"Vehicle health score: {health_score}/100")
        return state
    
    def predict_potential_failures(self, state: AgentState) -> AgentState:
        """Predict potential component failures"""
        logger.info("Predicting potential failures...")
        
        prediction_agent = self.agent_registry["prediction_agent"]
        predictions = prediction_agent.predict(
            state["vehicle_id"],
            state["telemetry_data"],
            state["vehicle_data"]
        )
        
        state["predictions"] = predictions
        state["current_step"] = "predict_failures"
        
        logger.info(f"Generated {len(predictions)} predictions")
        return state
    
    def diagnose_issues(self, state: AgentState) -> AgentState:
        """Diagnose root causes of predicted issues"""
        logger.info("Diagnosing issues...")
        
        diagnosis_agent = self.agent_registry["diagnosis_agent"]
        diagnosed_issues = diagnosis_agent.diagnose(state["predictions"])
        
        # Add diagnosis to predictions
        for i, prediction in enumerate(state["predictions"]):
            if i < len(diagnosed_issues):
                prediction["diagnosis"] = diagnosed_issues[i]
                prediction["root_cause"] = diagnosis_agent.identify_root_cause(prediction)
        
        state["current_step"] = "diagnose_issues"
        return state
    
    def calculate_risk_score(self, state: AgentState) -> AgentState:
        """Calculate overall risk score for the vehicle"""
        logger.info("Calculating risk score...")
        
        risk_score = 0
        for prediction in state["predictions"]:
            severity = prediction.get("severity", "low")
            probability = prediction.get("probability", 0)
            
            # Weighted risk calculation
            severity_weights = {"low": 1, "medium": 2, "high": 3, "critical": 5}
            risk_score += probability * severity_weights.get(severity, 1) * 20
        
        state["risk_score"] = min(risk_score, 100)  # Cap at 100
        state["current_step"] = "calculate_risk"
        
        logger.info(f"Overall risk score: {state['risk_score']}/100")
        return state
    
    def plan_recommended_actions(self, state: AgentState) -> AgentState:
        """Plan recommended actions based on predictions"""
        logger.info("Planning recommended actions...")
        
        actions = []
        
        for prediction in state["predictions"]:
            severity = prediction.get("severity", "low")
            component = prediction.get("component", "unknown")
            
            if severity == "critical":
                actions.append(f"Immediate replacement of {component}")
            elif severity == "high":
                actions.append(f"Schedule {component} inspection within 7 days")
            elif severity == "medium":
                actions.append(f"Monitor {component} closely during next service")
            else:
                actions.append(f"Routine check for {component}")
        
        state["recommended_actions"] = actions
        state["current_step"] = "plan_actions"
        
        logger.info(f"Planned {len(actions)} recommended actions")
        return state
    
    def _should_schedule_service(self, state: AgentState) -> str:
        """Decision node: should we schedule service?"""
        risk_score = state.get("risk_score", 0)
        
        if risk_score >= 70:
            return "immediate"
        elif risk_score >= 40:
            return "soon"
        else:
            return "monitor"
    
    def decide_scheduling_needs(self, state: AgentState) -> AgentState:
        """Decide on scheduling needs"""
        logger.info("Deciding scheduling needs...")
        
        # Based on predictions and vehicle usage
        state["scheduling_needs"] = {
            "priority": "medium",
            "timeline": "within_30_days",
            "estimated_duration": "2_hours"
        }
        state["current_step"] = "decide_scheduling"
        
        return state
    
    def schedule_service(self, state: AgentState) -> AgentState:
        """Schedule maintenance service"""
        logger.info("Scheduling service...")
        
        scheduling_agent = self.agent_registry["scheduling_agent"]
        
        scheduled_service = scheduling_agent.schedule(
            vehicle_id=state["vehicle_id"],
            predictions=state["predictions"],
            priority="high" if state["risk_score"] >= 70 else "medium"
        )
        
        state["scheduled_service"] = scheduled_service
        state["current_step"] = "schedule_service"
        
        logger.info(f"Scheduled service: {scheduled_service.get('service_id')}")
        return state
    
    def notify_customer(self, state: AgentState) -> AgentState:
        """Notify customer about scheduled service"""
        logger.info("Notifying customer...")
        
        customer_agent = self.agent_registry["customer_agent"]
        
        notification = customer_agent.notify(
            vehicle_id=state["vehicle_id"],
            predictions=state["predictions"],
            scheduled_service=state.get("scheduled_service"),
            preferred_channel="voice"  # Could be SMS, email, app notification
        )
        
        state["customer_notification"] = notification
        state["current_step"] = "notify_customer"
        
        logger.info(f"Customer notified via {notification.get('channel')}")
        return state
    
    def collect_customer_feedback(self, state: AgentState) -> AgentState:
        """Collect feedback from customer"""
        logger.info("Collecting customer feedback...")
        
        customer_agent = self.agent_registry["customer_agent"]
        
        # Simulate feedback collection
        feedback = customer_agent.collect_feedback(
            service_id=state.get("scheduled_service", {}).get("service_id"),
            customer_id=state["vehicle_data"].get("owner_id")
        )
        
        state["customer_feedback"] = feedback
        state["current_step"] = "collect_feedback"
        
        logger.info(f"Collected feedback with rating: {feedback.get('rating')}")
        return state
    
    def generate_manufacturing_insights(self, state: AgentState) -> AgentState:
        """Generate insights for manufacturing team"""
        logger.info("Generating manufacturing insights...")
        
        insights_agent = self.agent_registry["insights_agent"]
        
        insights = insights_agent.generate_insights(
            vehicle_data=state["vehicle_data"],
            predictions=state["predictions"],
            feedback=state.get("customer_feedback", {})
        )
        
        state["manufacturing_insights"] = insights
        state["current_step"] = "generate_insights"
        state["workflow_completed"] = True
        
        logger.info(f"Generated {len(insights.get('rca_insights', []))} RCA insights")
        return state
    
    def run(self, vehicle_id: str, vehicle_data: Dict) -> Dict:
        """Execute the complete workflow for a vehicle"""
        logger.info(f"Starting workflow for vehicle {vehicle_id}")
        
        initial_state: AgentState = {
            "vehicle_id": vehicle_id,
            "vehicle_data": vehicle_data,
            "telemetry_data": {},
            "predictions": [],
            "risk_score": 0,
            "recommended_actions": [],
            "scheduled_service": None,
            "customer_notification": None,
            "customer_feedback": None,
            "manufacturing_insights": None,
            "current_step": "",
            "workflow_completed": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            logger.info(f"Workflow completed successfully for {vehicle_id}")
            return result
        except Exception as e:
            logger.error(f"Workflow failed for {vehicle_id}: {str(e)}")
            return initial_state

# Worker Agent Classes

class TelemetryAgent:
    def collect_data(self, vehicle_id: str) -> Dict[str, Any]:
        """Collect telemetry data from vehicle"""
        # Simulated telemetry data - in production, this would come from IoT sensors
        import random
        from datetime import datetime
        
        return {
            "timestamp": datetime.now().isoformat(),
            "engine_temp": random.uniform(85, 115),  # Celsius
            "oil_pressure": random.uniform(25, 55),  # PSI
            "battery_voltage": random.uniform(11.5, 13.5),  # Volts
            "tire_pressure": {
                "front_left": random.uniform(28, 36),
                "front_right": random.uniform(28, 36),
                "rear_left": random.uniform(28, 36),
                "rear_right": random.uniform(28, 36)
            },
            "fuel_level": random.uniform(10, 100),  # Percentage
            "engine_rpm": random.uniform(800, 3500),
            "speed": random.uniform(0, 120),  # km/h
            "mileage": random.uniform(5000, 150000),
            "brake_fluid_level": random.uniform(0.7, 1.0),
            "coolant_level": random.uniform(0.7, 1.0),
            "transmission_temp": random.uniform(70, 110)
        }

class PredictionAgent:
    def __init__(self):
        # Load trained models
        self.models = self._load_models()
    
    def _load_models(self):
        """Load pre-trained ML models"""
        # In production, load actual trained models
        return {
            "engine_failure": {"accuracy": 0.87, "threshold": 0.7},
            "battery_failure": {"accuracy": 0.92, "threshold": 0.65},
            "brake_failure": {"accuracy": 0.85, "threshold": 0.75},
            "tire_failure": {"accuracy": 0.88, "threshold": 0.6}
        }
    
    def predict(self, vehicle_id: str, telemetry: Dict, vehicle_data: Dict) -> List[Dict]:
        """Predict potential failures"""
        import random
        
        # Simulate predictions - in production, use actual ML models
        components = ["Engine", "Battery", "Brakes", "Tires", "Transmission", "Alternator"]
        predictions = []
        
        for component in components:
            # Simulate probability based on telemetry
            base_prob = random.uniform(0.1, 0.9)
            
            # Adjust based on telemetry values
            if component == "Engine" and telemetry.get("engine_temp", 90) > 105:
                base_prob = min(base_prob + 0.3, 0.95)
            elif component == "Battery" and telemetry.get("battery_voltage", 12.6) < 12.0:
                base_prob = min(base_prob + 0.4, 0.98)
            elif component == "Brakes" and telemetry.get("brake_fluid_level", 1.0) < 0.8:
                base_prob = min(base_prob + 0.25, 0.9)
            
            # Determine severity
            if base_prob >= 0.8:
                severity = "critical"
            elif base_prob >= 0.6:
                severity = "high"
            elif base_prob >= 0.4:
                severity = "medium"
            else:
                severity = "low"
            
            predictions.append({
                "component": component,
                "probability": round(base_prob, 2),
                "severity": severity,
                "predicted_failure_window": self._calculate_failure_window(base_prob),
                "confidence": random.uniform(0.75, 0.95)
            })
        
        return predictions
    
    def _calculate_failure_window(self, probability: float) -> str:
        """Calculate estimated failure window"""
        if probability >= 0.8:
            return "within_7_days"
        elif probability >= 0.6:
            return "within_30_days"
        elif probability >= 0.4:
            return "within_90_days"
        else:
            return "beyond_90_days"

class DiagnosisAgent:
    def diagnose(self, predictions: List[Dict]) -> List[str]:
        """Diagnose issues based on predictions"""
        diagnoses = []
        
        for prediction in predictions:
            component = prediction["component"]
            probability = prediction["probability"]
            
            if probability > 0.7:
                diagnoses.append(f"High likelihood of {component} malfunction requiring immediate attention")
            elif probability > 0.5:
                diagnoses.append(f"Potential {component} degradation detected - preventive action recommended")
            else:
                diagnoses.append(f"{component} operating within normal parameters")
        
        return diagnoses
    
    def identify_root_cause(self, prediction: Dict) -> str:
        """Identify root cause of potential failure"""
        component = prediction["component"]
        causes = {
            "Engine": "High operating temperatures causing wear",
            "Battery": "Frequent deep discharge cycles",
            "Brakes": "Worn brake pads and fluid degradation",
            "Tires": "Uneven wear and pressure fluctuations",
            "Transmission": "Fluid contamination and overheating",
            "Alternator": "Voltage regulator failure"
        }
        return causes.get(component, "Unknown cause - requires inspection")

class SchedulingAgent:
    def schedule(self, vehicle_id: str, predictions: List[Dict], priority: str = "medium") -> Dict:
        """Schedule maintenance service"""
        from datetime import datetime, timedelta
        import random
        
        # Determine service type based on predictions
        critical_components = [p["component"] for p in predictions if p["severity"] in ["critical", "high"]]
        
        if critical_components:
            service_type = f"Emergency repair: {', '.join(critical_components[:2])}"
            schedule_date = datetime.now() + timedelta(days=random.randint(1, 3))
        else:
            service_type = "Predictive maintenance check"
            schedule_date = datetime.now() + timedelta(days=random.randint(7, 14))
        
        # Select service center
        service_centers = ["AutoCare Central", "QuickFix Garage", "Premium Motors", "City Auto Service"]
        
        return {
            "service_id": f"SVC-{vehicle_id[-6:]}-{datetime.now().strftime('%Y%m%d')}",
            "vehicle_id": vehicle_id,
            "service_type": service_type,
            "scheduled_date": schedule_date.strftime("%Y-%m-%d %H:%M"),
            "service_center": random.choice(service_centers),
            "estimated_duration": "2 hours" if "Emergency" in service_type else "1 hour",
            "priority": priority,
            "status": "pending_confirmation",
            "estimated_cost": f"${random.randint(150, 800)}"
        }

class CustomerAgent:
    def notify(self, vehicle_id: str, predictions: List[Dict], scheduled_service: Dict, preferred_channel: str = "voice") -> Dict:
        """Notify customer about service needs"""
        # Compose notification message
        critical_issues = [p for p in predictions if p["severity"] in ["critical", "high"]]
        
        if critical_issues:
            message = f"URGENT: Your vehicle {vehicle_id[-6:]} needs immediate attention. "
            message += f"Critical issue detected: {critical_issues[0]['component']}. "
        else:
            message = f"Your vehicle {vehicle_id[-6:]} is due for preventive maintenance. "
        
        if scheduled_service:
            message += f"Scheduled for {scheduled_service['scheduled_date']} at {scheduled_service['service_center']}."
        
        return {
            "customer_id": f"CUST-{vehicle_id[-6:]}",
            "vehicle_id": vehicle_id,
            "channel": preferred_channel,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "sent",
            "response_required": len(critical_issues) > 0
        }
    
    def collect_feedback(self, service_id: str, customer_id: str) -> Dict:
        """Collect feedback after service"""
        import random
        
        return {
            "service_id": service_id,
            "customer_id": customer_id,
            "rating": random.randint(3, 5),  # 1-5 scale
            "comments": random.choice([
                "Great service, very professional",
                "Fixed the issue quickly",
                "Could improve communication",
                "Excellent predictive maintenance",
                "Slightly expensive but worth it"
            ]),
            "timestamp": datetime.now().isoformat(),
            "would_recommend": random.choice([True, True, True, False])  # 75% would recommend
        }

class ManufacturingInsightsAgent:
    def generate_insights(self, vehicle_data: Dict, predictions: List[Dict], feedback: Dict) -> Dict:
        """Generate insights for manufacturing team"""
        # Analyze failure patterns
        frequent_failures = {}
        for prediction in predictions:
            component = prediction["component"]
            if prediction["probability"] > 0.6:
                frequent_failures[component] = frequent_failures.get(component, 0) + 1
        
        # RCA Analysis
        rca_insights = []
        for component, count in frequent_failures.items():
            if count > 0:
                rca_insights.append({
                    "component": component,
                    "failure_rate": "high" if count > 2 else "medium",
                    "likely_causes": self._identify_manufacturing_causes(component),
                    "recommendation": self._generate_capa_recommendation(component)
                })
        
        # Quality metrics
        quality_metrics = {
            "mean_time_between_failures": f"{random.uniform(6, 18):.1f} months",
            "warranty_claims_rate": f"{random.uniform(2, 8):.1f}%",
            "customer_satisfaction_score": feedback.get("rating", 4),
            "most_failing_component": max(frequent_failures.items(), key=lambda x: x[1])[0] if frequent_failures else "None"
        }
        
        return {
            "vehicle_model": vehicle_data.get("model", "Unknown"),
            "manufacturing_batch": vehicle_data.get("batch", "2024-Q1"),
            "rca_insights": rca_insights,
            "quality_metrics": quality_metrics,
            "design_recommendations": self._generate_design_recommendations(rca_insights),
            "timestamp": datetime.now().isoformat()
        }
    
    def _identify_manufacturing_causes(self, component: str) -> List[str]:
        """Identify potential manufacturing causes"""
        causes = {
            "Engine": ["Insufficient cooling system", "Suboptimal material quality", "Assembly tolerances"],
            "Battery": ["Poor quality cells", "Inadequate cooling", "Faulty BMS"],
            "Brakes": ["Material wear issues", "Fluid quality", "Calibration problems"],
            "Tires": ["Rubber compound quality", "Tread design", "Manufacturing defects"]
        }
        return causes.get(component, ["Unknown - requires investigation"])
    
    def _generate_capa_recommendation(self, component: str) -> str:
        """Generate Corrective and Preventive Action recommendations"""
        recommendations = {
            "Engine": "Improve cooling system design and use higher grade materials",
            "Battery": "Implement better thermal management and upgrade BMS software",
            "Brakes": "Use higher quality brake pads and improve fluid specifications",
            "Tires": "Enhance rubber compound and improve quality control processes"
        }
        return recommendations.get(component, "Conduct detailed failure analysis")
    
    def _generate_design_recommendations(self, rca_insights: List[Dict]) -> List[str]:
        """Generate design improvement recommendations"""
        recommendations = []
        
        for insight in rca_insights:
            component = insight["component"]
            if insight["failure_rate"] == "high":
                recommendations.append(f"Redesign {component} for better durability")
            else:
                recommendations.append(f"Improve {component} quality control processes")
        
        return recommendations

# Utility function to run the agent system
def run_agent_system():
    """Run the complete agent system with sample data"""
    # Sample vehicle data
    sample_vehicle = {
        "vin": "1HGCM82633A123456",
        "make": "Honda",
        "model": "Accord",
        "year": 2023,
        "mileage": 15234,
        "owner_id": "CUST001",
        "last_service": "2024-01-15",
        "batch": "2023-Q4"
    }
    
    # Initialize and run master agent
    master_agent = MasterAgent()
    result = master_agent.run(sample_vehicle["vin"], sample_vehicle)
    
    # Print results
    print("\n" + "="*60)
    print("AUTOSENSEAI AGENT EXECUTION RESULTS")
    print("="*60)
    
    print(f"\nVehicle: {result['vehicle_data']['make']} {result['vehicle_data']['model']}")
    print(f"VIN: {result['vehicle_id']}")
    print(f"Health Score: {result['vehicle_data'].get('health_score', 'N/A')}/100")
    print(f"Risk Score: {result['risk_score']}/100")
    
    print(f"\n📊 PREDICTIONS:")
    for pred in result['predictions'][:3]:  # Show top 3
        print(f"  • {pred['component']}: {pred['probability']*100:.0f}% ({pred['severity']})")
    
    if result.get('scheduled_service'):
        print(f"\n📅 SCHEDULED SERVICE:")
        svc = result['scheduled_service']
        print(f"  Type: {svc['service_type']}")
        print(f"  Date: {svc['scheduled_date']}")
        print(f"  Center: {svc['service_center']}")
    
    if result.get('customer_notification'):
        print(f"\n📱 CUSTOMER NOTIFICATION:")
        print(f"  Status: {result['customer_notification']['status']}")
        print(f"  Channel: {result['customer_notification']['channel']}")
    
    if result.get('manufacturing_insights'):
        print(f"\n🏭 MANUFACTURING INSIGHTS:")
        insights = result['manufacturing_insights']
        print(f"  Model: {insights['vehicle_model']}")
        print(f"  Batch: {insights['manufacturing_batch']}")
        if insights['rca_insights']:
            print(f"  RCA Findings: {len(insights['rca_insights'])} issues identified")
    
    print(f"\n✅ Workflow completed: {result['current_step']}")
    print("="*60)
    
    return result

if __name__ == "__main__":
    # Run a demo
    run_agent_system()