import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List, Any
from predictive_maintenance import PredictiveMaintenanceModel
from master_agent import MasterAgent, run_agent_system

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentTrainer:
    def __init__(self):
        self.pm_model = None
        self.master_agent = None
        self.training_data = []
        self.evaluation_results = {}
        
    def setup_environment(self):
        """Set up the training environment"""
        logger.info("Setting up training environment...")
        
        # Initialize predictive maintenance model
        self.pm_model = PredictiveMaintenanceModel(model_type='ensemble')
        
        # Train or load the PM model
        try:
            self.pm_model.load_models('models/')
            logger.info("Loaded pre-trained predictive maintenance models")
        except:
            logger.info("Training new predictive maintenance models...")
            self.pm_model.train_models()
        
        # Initialize master agent
        self.master_agent = MasterAgent()
        logger.info("Master agent initialized")
        
    def generate_training_scenarios(self, n_scenarios=100):
        """Generate diverse training scenarios for agents"""
        logger.info(f"Generating {n_scenarios} training scenarios...")
        
        scenarios = []
        
        for i in range(n_scenarios):
            # Create different types of scenarios
            scenario_type = np.random.choice([
                'normal_operation',
                'engine_problem',
                'battery_issue',
                'brake_wear',
                'tire_problem',
                'multiple_issues',
                'false_positive',
                'critical_failure'
            ])
            
            # Generate vehicle data based on scenario type
            vehicle_data = self._create_scenario_vehicle(scenario_type, i)
            
            scenarios.append({
                'scenario_id': f"SCEN_{i:04d}",
                'type': scenario_type,
                'vehicle_data': vehicle_data,
                'expected_actions': self._get_expected_actions(scenario_type),
                'difficulty': np.random.choice(['easy', 'medium', 'hard'])
            })
        
        self.training_data = scenarios
        logger.info(f"Generated {len(scenarios)} training scenarios")
        return scenarios
    
    def _create_scenario_vehicle(self, scenario_type: str, idx: int) -> Dict:
        """Create vehicle data for a specific scenario"""
        base_data = {
            'vin': f'TESTVIN{idx:06d}',
            'make': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']),
            'model': np.random.choice(['Sedan', 'SUV', 'Truck']),
            'year': np.random.randint(2018, 2024),
            'mileage': np.random.randint(5000, 150000),
            'owner_id': f'OWNER{idx:04d}',
            'last_service': (datetime.now() - pd.Timedelta(days=np.random.randint(30, 180))).strftime('%Y-%m-%d'),
            'batch': f"202{np.random.randint(3,5)}-Q{np.random.randint(1,5)}"
        }
        
        # Add scenario-specific telemetry
        telemetry = self._generate_scenario_telemetry(scenario_type)
        
        return {**base_data, **telemetry}
    
    def _generate_scenario_telemetry(self, scenario_type: str) -> Dict:
        """Generate telemetry data specific to scenario type"""
        if scenario_type == 'normal_operation':
            return {
                'engine_temp': np.random.normal(90, 5),
                'oil_pressure': np.random.normal(45, 3),
                'battery_voltage': np.random.normal(12.6, 0.2),
                'brake_pad_thickness': np.random.uniform(8, 12),
                'tire_tread_depth': np.random.uniform(6, 10),
                'vibration_level': np.random.uniform(0.5, 1.5)
            }
        elif scenario_type == 'engine_problem':
            return {
                'engine_temp': np.random.uniform(110, 125),
                'oil_pressure': np.random.uniform(25, 35),
                'battery_voltage': np.random.normal(12.6, 0.2),
                'brake_pad_thickness': np.random.uniform(8, 12),
                'tire_tread_depth': np.random.uniform(6, 10),
                'vibration_level': np.random.uniform(1.5, 2.5)
            }
        elif scenario_type == 'battery_issue':
            return {
                'engine_temp': np.random.normal(90, 5),
                'oil_pressure': np.random.normal(45, 3),
                'battery_voltage': np.random.uniform(11.5, 12.0),
                'brake_pad_thickness': np.random.uniform(8, 12),
                'tire_tread_depth': np.random.uniform(6, 10),
                'vibration_level': np.random.uniform(0.5, 1.5)
            }
        elif scenario_type == 'critical_failure':
            return {
                'engine_temp': np.random.uniform(120, 140),
                'oil_pressure': np.random.uniform(20, 30),
                'battery_voltage': np.random.uniform(11.0, 11.5),
                'brake_pad_thickness': np.random.uniform(2, 4),
                'tire_tread_depth': np.random.uniform(1, 3),
                'vibration_level': np.random.uniform(2.5, 4.0)
            }
        else:
            return {
                'engine_temp': np.random.normal(90, 8),
                'oil_pressure': np.random.normal(45, 5),
                'battery_voltage': np.random.normal(12.6, 0.4),
                'brake_pad_thickness': np.random.uniform(4, 12),
                'tire_tread_depth': np.random.uniform(3, 10),
                'vibration_level': np.random.uniform(0.5, 2.5)
            }
    
    def _get_expected_actions(self, scenario_type: str) -> List[str]:
        """Get expected agent actions for each scenario type"""
        actions = {
            'normal_operation': ['monitor', 'no_action'],
            'engine_problem': ['predict_engine_failure', 'schedule_service', 'notify_customer'],
            'battery_issue': ['predict_battery_failure', 'schedule_service', 'notify_customer'],
            'brake_wear': ['predict_brake_failure', 'recommend_inspection'],
            'tire_problem': ['predict_tire_failure', 'schedule_replacement'],
            'multiple_issues': ['predict_multiple_failures', 'schedule_comprehensive_service', 'high_priority'],
            'false_positive': ['verify_readings', 'no_action_if_normal'],
            'critical_failure': ['predict_critical_failure', 'immediate_action', 'emergency_notification']
        }
        return actions.get(scenario_type, ['monitor'])
    
    def train_agents(self, n_iterations=50):
        """Train agents through iterative learning"""
        logger.info(f"Starting agent training with {n_iterations} iterations...")
        
        training_history = []
        
        for iteration in range(n_iterations):
            logger.info(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Select random scenarios for this iteration
            batch_size = min(10, len(self.training_data))
            batch = np.random.choice(self.training_data, batch_size, replace=False)
            
            iteration_results = []
            
            for scenario in batch:
                # Run agent system for this scenario
                result = self.master_agent.run(
                    vehicle_id=scenario['vehicle_data']['vin'],
                    vehicle_data=scenario['vehicle_data']
                )
                
                # Evaluate agent performance
                evaluation = self._evaluate_agent_performance(scenario, result)
                iteration_results.append(evaluation)
                
                # Log learning
                learning_point = self._extract_learning_point(scenario, result, evaluation)
                if learning_point:
                    self._update_agent_knowledge(learning_point)
            
            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(iteration_results)
            training_history.append({
                'iteration': iteration + 1,
                'metrics': iteration_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Iteration {iteration + 1} metrics: {iteration_metrics}")
            
            # Save checkpoint every 10 iterations
            if (iteration + 1) % 10 == 0:
                self._save_checkpoint(iteration + 1, training_history)
        
        self.evaluation_results = training_history
        logger.info("Agent training completed!")
        return training_history
    
    def _evaluate_agent_performance(self, scenario: Dict, result: Dict) -> Dict:
        """Evaluate agent performance for a scenario"""
        evaluation = {
            'scenario_id': scenario['scenario_id'],
            'scenario_type': scenario['type'],
            'difficulty': scenario['difficulty']
        }
        
        # Check if agent identified critical issues
        risk_score = result.get('risk_score', 0)
        predictions = result.get('predictions', [])
        
        # Determine if agent took appropriate action
        expected_actions = scenario['expected_actions']
        actual_actions = []
        
        if result.get('scheduled_service'):
            actual_actions.append('schedule_service')
        if result.get('customer_notification'):
            actual_actions.append('notify_customer')
        if risk_score > 70:
            actual_actions.append('high_priority')
        if any(p.get('severity') == 'critical' for p in predictions):
            actual_actions.append('critical_action')
        
        # Calculate action accuracy
        correct_actions = set(actual_actions).intersection(set(expected_actions))
        action_accuracy = len(correct_actions) / len(expected_actions) if expected_actions else 1.0
        
        evaluation.update({
            'risk_score': risk_score,
            'action_accuracy': action_accuracy,
            'expected_actions': expected_actions,
            'actual_actions': actual_actions,
            'health_score': result.get('vehicle_data', {}).get('health_score', 0),
            'has_manufacturing_insights': bool(result.get('manufacturing_insights'))
        })
        
        return evaluation
    
    def _extract_learning_point(self, scenario: Dict, result: Dict, evaluation: Dict) -> Dict:
        """Extract learning points from agent execution"""
        if evaluation['action_accuracy'] < 0.7:
            # Agent made mistakes - learn from them
            return {
                'type': 'correction',
                'scenario_type': scenario['type'],
                'expected_actions': scenario['expected_actions'],
                'actual_actions': evaluation['actual_actions'],
                'risk_score': evaluation['risk_score'],
                'timestamp': datetime.now().isoformat()
            }
        elif evaluation['action_accuracy'] > 0.9:
            # Agent performed well - reinforce learning
            return {
                'type': 'reinforcement',
                'scenario_type': scenario['type'],
                'actions': evaluation['actual_actions'],
                'risk_score': evaluation['risk_score'],
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def _update_agent_knowledge(self, learning_point: Dict):
        """Update agent knowledge based on learning points"""
        # In a real implementation, this would update agent decision rules
        # or model parameters. For now, we'll just log it.
        logger.debug(f"Learning point: {learning_point['type']} for {learning_point['scenario_type']}")
        
        # You could implement:
        # 1. Adjusting decision thresholds
        # 2. Updating action selection probabilities
        # 3. Adding new rules to the knowledge base
        # 4. Retraining ML models with new data
    
    def _calculate_iteration_metrics(self, iteration_results: List[Dict]) -> Dict:
        """Calculate metrics for an iteration"""
        if not iteration_results:
            return {}
        
        metrics = {
            'total_scenarios': len(iteration_results),
            'avg_action_accuracy': np.mean([r['action_accuracy'] for r in iteration_results]),
            'avg_risk_score': np.mean([r['risk_score'] for r in iteration_results]),
            'scenarios_with_insights': sum(1 for r in iteration_results if r['has_manufacturing_insights']),
            'success_rate': sum(1 for r in iteration_results if r['action_accuracy'] > 0.8) / len(iteration_results)
        }
        
        # Add difficulty-specific metrics
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in iteration_results if r['difficulty'] == difficulty]
            if diff_results:
                metrics[f'{difficulty}_accuracy'] = np.mean([r['action_accuracy'] for r in diff_results])
        
        return metrics
    
    def _save_checkpoint(self, iteration: int, training_history: List[Dict]):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'training_history': training_history,
            'agent_state': self._get_agent_state(),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'checkpoints/training_checkpoint_{iteration:04d}.json'
        import os
        os.makedirs('checkpoints', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filename}")
    
    def _get_agent_state(self) -> Dict:
        """Get current state of agents for checkpointing"""
        # This would capture the current state of all agents
        # For now, return a simplified version
        return {
            'master_agent': 'active',
            'worker_agents': ['telemetry', 'prediction', 'scheduling', 'customer', 'insights'],
            'last_training': datetime.now().isoformat(),
            'model_versions': {k: '1.0' for k in self.pm_model.models.keys()} if self.pm_model else {}
        }
    
    def analyze_training_results(self):
        """Analyze and visualize training results"""
        if not self.evaluation_results:
            logger.warning("No training results to analyze")
            return
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING RESULTS ANALYSIS")
        logger.info("="*60)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'iteration': r['iteration'],
            'avg_accuracy': r['metrics']['avg_action_accuracy'],
            'success_rate': r['metrics']['success_rate'],
            'avg_risk': r['metrics']['avg_risk_score']
        } for r in self.evaluation_results])
        
        # Calculate improvements
        if len(df) > 1:
            start_accuracy = df['avg_accuracy'].iloc[0]
            end_accuracy = df['avg_accuracy'].iloc[-1]
            improvement = ((end_accuracy - start_accuracy) / start_accuracy) * 100
            
            logger.info(f"Training iterations: {len(df)}")
            logger.info(f"Starting accuracy: {start_accuracy:.2%}")
            logger.info(f"Final accuracy: {end_accuracy:.2%}")
            logger.info(f"Improvement: {improvement:+.1f}%")
            logger.info(f"Best iteration: {df['avg_accuracy'].idxmax() + 1} ({df['avg_accuracy'].max():.2%})")
        
        # Generate recommendations for improvement
        self._generate_improvement_recommendations()
        
        return df
    
    def _generate_improvement_recommendations(self):
        """Generate recommendations for improving agent performance"""
        logger.info("\nImprovement Recommendations:")
        
        recommendations = [
            "1. Add more diverse failure scenarios to training data",
            "2. Implement reinforcement learning for action selection",
            "3. Improve telemetry data quality and frequency",
            "4. Add more context awareness to agents",
            "5. Implement human-in-the-loop validation for critical decisions",
            "6. Optimize decision thresholds based on real-world outcomes",
            "7. Add more sophisticated anomaly detection",
            "8. Implement transfer learning from similar domains"
        ]
        
        for rec in recommendations:
            logger.info(rec)
    
    def deploy_trained_agents(self):
        """Deploy trained agents to production"""
        logger.info("\nDeploying trained agents...")
        
        # Save final agent state
        deployment_package = {
            'agents': {
                'master_agent': 'trained_v1.0',
                'prediction_agent': 'trained_v1.0',
                'scheduling_agent': 'trained_v1.0',
                'customer_agent': 'trained_v1.0',
                'insights_agent': 'trained_v1.0'
            },
            'models': list(self.pm_model.models.keys()) if self.pm_model else [],
            'training_stats': {
                'total_iterations': len(self.evaluation_results),
                'final_accuracy': self.evaluation_results[-1]['metrics']['avg_action_accuracy'] if self.evaluation_results else 0,
                'deployment_date': datetime.now().isoformat()
            },
            'configuration': {
                'risk_thresholds': {
                    'low': 30,
                    'medium': 50,
                    'high': 70,
                    'critical': 85
                },
                'notification_channels': ['voice', 'sms', 'email', 'app'],
                'scheduling_window': '7-14 days'
            }
        }
        
        # Save deployment package
        with open('deployment/agent_deployment_v1.0.json', 'w') as f:
            json.dump(deployment_package, f, indent=2)
        
        logger.info("Agents deployed successfully!")
        logger.info("Deployment package saved to: deployment/agent_deployment_v1.0.json")
        
        return deployment_package

def main():
    """Main training function"""
    logger.info("Starting AutoSenseAI Agent Training Pipeline")
    
    # Initialize trainer
    trainer = AgentTrainer()
    
    # Setup environment
    trainer.setup_environment()
    
    # Generate training scenarios
    scenarios = trainer.generate_training_scenarios(n_scenarios=200)
    logger.info(f"Generated {len(scenarios)} training scenarios")
    
    # Train agents
    training_history = trainer.train_agents(n_iterations=100)
    
    # Analyze results
    results_df = trainer.analyze_training_results()
    
    # Deploy trained agents
    deployment_package = trainer.deploy_trained_agents()
    
    # Run a test with the trained agents
    logger.info("\n" + "="*60)
    logger.info("FINAL AGENT TEST")
    logger.info("="*60)
    
    # Test with a critical scenario
    test_scenario = next((s for s in scenarios if s['type'] == 'critical_failure'), scenarios[0])
    
    logger.info(f"Testing with scenario: {test_scenario['scenario_id']} ({test_scenario['type']})")
    
    result = trainer.master_agent.run(
        vehicle_id=test_scenario['vehicle_data']['vin'],
        vehicle_data=test_scenario['vehicle_data']
    )
    
    logger.info(f"Test completed. Risk score: {result.get('risk_score', 0):.1f}")
    logger.info(f"Actions taken: {[k for k, v in result.items() if v and k.endswith('_service') or k.endswith('_notification')]}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ AGENT TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    
    return trainer, results_df, deployment_package

if __name__ == "__main__":
    # Run the training pipeline
    trainer, results, deployment = main()
    
    # Save detailed results
    import pandas as pd
    
    results_df = pd.DataFrame(trainer.evaluation_results)
    results_df.to_csv('training_results.csv', index=False)
    print("\nDetailed results saved to 'training_results.csv'")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total iterations: {len(results_df)}")
    print(f"Final accuracy: {results_df['metrics'].iloc[-1]['avg_action_accuracy']:.2%}")
    print(f"Success rate: {results_df['metrics'].iloc[-1]['success_rate']:.2%}")