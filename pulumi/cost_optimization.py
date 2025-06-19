"""
💰 NEXLIFY FINOPS - KEEPING THE EDDIES FLOWING, NOT BLEEDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In Night City, burning through eddies faster than you make 'em is a one-way
ticket to the scrapyard. This module keeps our infrastructure costs in check
while maximizing performance. Because the best trade is the one that doesn't
bankrupt you before it pays out.

FinOps: Financial Operations for the digital age.
Multi-region, K3s-aware, GPU-optimized. Every eddie counts.
"""

from typing import TypedDict, Optional, List, Dict, Mapping, Any
import pulumi
from pulumi import ComponentResource, ResourceOptions, Output
import pulumi_aws as aws
import json
from datetime import datetime, timedelta
import math


class CostConfig(TypedDict):
    """Cost optimization configuration - the art of doing more with less"""
    monthly_budget: float
    alert_threshold: float  # Percentage of budget
    spot_percentage: float  # How much infra on spot instances
    enable_auto_stop: bool  # Stop non-prod resources after hours
    backtesting_spot_enabled: bool
    ml_training_spot_enabled: bool
    
    # Multi-region cost tracking
    regions: List[str]
    enable_cross_region_optimization: bool
    
    # K3s specific optimizations
    k3s_spot_enabled: bool
    k3s_node_count: int
    gpu_nodes: int
    
    # Resource scheduling
    business_hours_only: bool
    timezone: str
    start_hour: int  # 24-hour format
    stop_hour: int
    weekend_shutdown: bool
    
    # Cost allocation
    cost_center: str
    project_tags: Mapping[str, str]
    
    # Optimization settings
    enable_rightsizing: bool
    enable_unused_resource_cleanup: bool
    cleanup_after_days: int
    enable_predictive_scaling: bool
    
    # Exchange-specific optimizations
    exchanges: List[str]
    peak_trading_hours: Dict[str, List[int]]  # UTC hours per exchange
    
    tags: Mapping[str, str]


class FinOpsAutomation(ComponentResource):
    """
    Automated cost optimization for trading infrastructure.
    
    Features:
    - Real-time budget monitoring and alerts
    - Spot instance orchestration for non-critical workloads
    - K3s node optimization with market-aware scaling
    - GPU instance cost management
    - Multi-region cost aggregation and optimization
    - Exchange-specific resource scheduling
    - Predictive cost forecasting based on trading patterns
    - Automated resource rightsizing
    - Cost anomaly detection with ML
    
    Save eddies, trade longer, profit harder.
    """
    
    def __init__(self, name: str, config: CostConfig, opts: Optional[ResourceOptions] = None):
        super().__init__("nexlify:finops:FinOpsAutomation", name, {}, opts)
        
        self.config = config
        self.name = name
        
        # Deploy cost optimization infrastructure
        self._setup_budget_alerts()
        self._configure_spot_fleet()
        self._setup_k3s_cost_optimization()
        self._setup_resource_scheduling()
        self._enable_cost_anomaly_detection()
        self._configure_savings_plans()
        self._setup_cost_dashboard()
        self._configure_multi_region_optimization()
        
        # Export cost insights
        self.register_outputs({
            "monthly_budget": config["monthly_budget"],
            "estimated_monthly_cost": self.estimated_cost,
            "potential_savings": self.potential_savings,
            "dashboard_url": self.dashboard_url,
            "cost_breakdown": self.cost_breakdown,
        })
    
    def _setup_budget_alerts(self):
        """
        Budget monitoring - know when you're bleeding eddies.
        Multi-threshold alerts to catch overspend before it hurts.
        Now with per-region and per-service breakdowns.
        """
        # Calculate service allocations based on our infrastructure
        service_allocations = self._calculate_service_allocations()
        
        # Main budget with multiple alert thresholds
        self.budget = aws.budgets.Budget(
            f"{self.name}-monthly-budget",
            budget_type="COST",
            time_unit="MONTHLY",
            budget_limit=aws.budgets.BudgetBudgetLimitArgs(
                amount=str(self.config["monthly_budget"]),
                unit="USD",
            ),
            cost_filters=[
                aws.budgets.BudgetCostFilterArgs(
                    name="TagKeyValue",
                    values=[f"Project${self.config['project_tags'].get('Project', 'Nexlify')}"],
                ),
            ],
            notifications=[
                # 50% threshold - early warning
                aws.budgets.BudgetNotificationArgs(
                    comparison_operator="GREATER_THAN",
                    threshold=50.0,
                    threshold_type="PERCENTAGE",
                    notification_type="ACTUAL",
                    subscriber_email_addresses=self._get_alert_emails(),
                    subscriber_sns_topic_arns=[self._create_alert_topic().arn],
                ),
                # 80% threshold - intervention required
                aws.budgets.BudgetNotificationArgs(
                    comparison_operator="GREATER_THAN",
                    threshold=80.0,
                    threshold_type="PERCENTAGE",
                    notification_type="ACTUAL",
                    subscriber_email_addresses=self._get_alert_emails(),
                    subscriber_sns_topic_arns=[self._create_alert_topic().arn],
                ),
                # 90% threshold - critical alert with automated response
                aws.budgets.BudgetNotificationArgs(
                    comparison_operator="GREATER_THAN",
                    threshold=90.0,
                    threshold_type="PERCENTAGE",
                    notification_type="ACTUAL",
                    subscriber_email_addresses=self._get_alert_emails(),
                    subscriber_sns_topic_arns=[self._create_critical_alert_topic().arn],
                ),
                # Forecasted overrun - predictive alerts
                aws.budgets.BudgetNotificationArgs(
                    comparison_operator="GREATER_THAN",
                    threshold=100.0,
                    threshold_type="PERCENTAGE",
                    notification_type="FORECASTED",
                    subscriber_email_addresses=self._get_alert_emails(),
                    subscriber_sns_topic_arns=[self._create_critical_alert_topic().arn],
                ),
            ],
            cost_types=aws.budgets.BudgetCostTypesArgs(
                include_credit=False,
                include_discount=True,
                include_other_subscription=True,
                include_recurring=True,
                include_refund=False,
                include_subscription=True,
                include_support=True,
                include_tax=True,
                include_upfront=True,
                use_amortized=True,  # Smooth out reserved instance costs
                use_blended=False,
            ),
            time_period_start=datetime.now().replace(day=1).strftime("%Y-%m-%d"),
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Per-region budgets for multi-region deployments
        if self.config.get("enable_cross_region_optimization"):
            self._create_regional_budgets()
        
        # Service-specific budgets with smart allocations
        self._create_service_budgets(service_allocations)
    
    def _calculate_service_allocations(self) -> Dict[str, float]:
        """
        Calculate budget allocations based on our actual infrastructure.
        This is where we get smart about predicting costs.
        """
        total_budget = self.config["monthly_budget"]
        
        # Base allocations
        allocations = {
            "EC2": 0.35,  # K3s nodes, spot fleets
            "RDS": 0.15,  # QuestDB, backups
            "ElastiCache": 0.10,  # Valkey cache
            "Lambda": 0.05,  # Event processing
            "S3": 0.05,  # Data storage, flow logs
            "CloudWatch": 0.05,  # Monitoring, logs
            "DataTransfer": 0.10,  # Cross-AZ, internet egress
            "ALB": 0.05,  # Load balancers
            "VPC": 0.05,  # NAT gateways, endpoints
            "Other": 0.05,  # Everything else
        }
        
        # Adjust for GPU nodes
        if self.config.get("gpu_nodes", 0) > 0:
            gpu_percentage = (self.config["gpu_nodes"] / max(self.config["k3s_node_count"], 1)) * 0.2
            allocations["EC2"] += gpu_percentage
            # Reduce other allocations proportionally
            for service in allocations:
                if service != "EC2":
                    allocations[service] *= (1 - gpu_percentage)
        
        # Adjust for multi-region
        if len(self.config.get("regions", [])) > 1:
            # Multi-region increases data transfer costs
            allocations["DataTransfer"] += 0.05
            allocations["EC2"] -= 0.05
        
        return {k: v * total_budget for k, v in allocations.items()}
    
    def _configure_spot_fleet(self):
        """
        Spot instance orchestration - high risk, high reward computing.
        Now with K3s awareness and market-based bidding strategies.
        """
        if not self.config["backtesting_spot_enabled"]:
            return
        
        # Calculate optimal spot configuration based on workload
        spot_config = self._calculate_spot_configuration()
        
        # Launch template for spot instances
        spot_launch_template = aws.ec2.LaunchTemplate(
            f"{self.name}-spot-template",
            name_prefix=f"{self.name}-spot-",
            image_id=self._get_optimized_ami(),
            user_data=self._get_spot_userdata(),
            metadata_options=aws.ec2.LaunchTemplateMetadataOptionsArgs(
                http_endpoint="enabled",
                http_tokens="required",
            ),
            monitoring=aws.ec2.LaunchTemplateMonitoringArgs(
                enabled=True,  # Need metrics for cost optimization
            ),
            tag_specifications=[
                aws.ec2.LaunchTemplateTagSpecificationArgs(
                    resource_type="instance",
                    tags={
                        **self.config["tags"],
                        "InstanceLifecycle": "spot",
                        "Purpose": "backtesting",
                        "AutoShutdown": "enabled",
                    },
                ),
            ],
            opts=ResourceOptions(parent=self)
        )
        
        # Spot fleet for backtesting with market-aware bidding
        self.backtest_spot_fleet = aws.ec2.SpotFleetRequest(
            f"{self.name}-backtest-fleet",
            iam_fleet_role=self._create_spot_fleet_role().arn,
            allocation_strategy="capacityOptimized",  # Better availability
            target_capacity=spot_config["target_capacity"],
            valid_until=(datetime.now() + timedelta(days=365)).isoformat(),
            terminate_instances_with_expiration=True,
            instance_interruption_behavior="terminate",
            # Smart bidding based on instance types
            launch_specifications=[
                aws.ec2.SpotFleetRequestLaunchSpecificationArgs(
                    instance_type=instance_type,
                    weighted_capacity=weight,
                    spot_price=str(spot_config["bid_prices"][instance_type]),
                    subnet_id=subnet,
                    launch_template=aws.ec2.SpotFleetRequestLaunchSpecificationLaunchTemplateArgs(
                        launch_template_id=spot_launch_template.id,
                        version="$Latest",
                    ),
                )
                for instance_type, weight in spot_config["instance_weights"].items()
                for subnet in self._get_cheapest_subnets()
            ],
            # Spot fleet tags for cost tracking
            spot_fleet_request_config_data=json.dumps({
                "TagSpecifications": [{
                    "ResourceType": "spot-fleet-request",
                    "Tags": [
                        {"Key": k, "Value": v}
                        for k, v in {
                            **self.config["tags"],
                            "FleetPurpose": "backtesting",
                            "CostCenter": self.config["cost_center"],
                        }.items()
                    ]
                }]
            }),
            opts=ResourceOptions(parent=self)
        )
        
        # ML training spot instances with GPU support
        if self.config["ml_training_spot_enabled"] and self.config.get("gpu_nodes", 0) > 0:
            self._configure_ml_spot_instances()
    
    def _setup_k3s_cost_optimization(self):
        """
        K3s-specific cost optimizations.
        Smart node scheduling, spot integration, and market-aware scaling.
        """
        if not self.config.get("k3s_spot_enabled"):
            return
        
        # Mixed instance policy for K3s - balance cost and reliability
        k3s_mixed_instances = aws.autoscaling.MixedInstancesPolicy(
            f"{self.name}-k3s-mixed-policy",
            launch_template=aws.autoscaling.MixedInstancesPolicyLaunchTemplateArgs(
                launch_template_specification=aws.autoscaling.MixedInstancesPolicyLaunchTemplateLaunchTemplateSpecificationArgs(
                    launch_template_name=f"{self.name}-k3s-optimized",
                    version="$Latest",
                ),
                overrides=[
                    # Prioritize newer, more efficient instance types
                    aws.autoscaling.MixedInstancesPolicyLaunchTemplateOverrideArgs(
                        instance_type="c6i.xlarge",
                        weighted_capacity="1",
                    ),
                    aws.autoscaling.MixedInstancesPolicyLaunchTemplateOverrideArgs(
                        instance_type="c5n.xlarge",
                        weighted_capacity="1",
                    ),
                    aws.autoscaling.MixedInstancesPolicyLaunchTemplateOverrideArgs(
                        instance_type="c5.xlarge",
                        weighted_capacity="1",
                    ),
                ],
            ),
            instances_distribution=aws.autoscaling.MixedInstancesPolicyInstancesDistributionArgs(
                on_demand_base_capacity=2,  # Always keep 2 on-demand for stability
                on_demand_percentage_above_base_capacity=100 - int(self.config["spot_percentage"]),
                spot_allocation_strategy="capacity-optimized",
                spot_instance_pools=3,  # Diversify across pools
            ),
            opts=ResourceOptions(parent=self)
        )
        
        # Predictive scaling for K3s based on trading patterns
        if self.config.get("enable_predictive_scaling"):
            self._setup_predictive_scaling()
    
    def _setup_resource_scheduling(self):
        """
        Automated stop/start with exchange-aware scheduling.
        Because Forex never sleeps, but dev environments should.
        """
        if not self.config["business_hours_only"]:
            return
        
        # Enhanced Lambda for intelligent resource management
        self.resource_scheduler_lambda = aws.lambda_.Function(
            f"{self.name}-smart-scheduler",
            runtime="python3.11",
            handler="smart_scheduler.handler",
            code=pulumi.AssetArchive({
                "smart_scheduler.py": pulumi.StringAsset(self._get_smart_scheduler_code()),
                "requirements.txt": pulumi.StringAsset("boto3\npandas\nnumpy"),
            }),
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "EXCHANGES": json.dumps(self.config["exchanges"]),
                    "PEAK_HOURS": json.dumps(self.config.get("peak_trading_hours", {})),
                    "ENVIRONMENT_TAG": "Environment",
                    "STOP_ENVIRONMENTS": "dev,staging",
                    "TIMEZONE": self.config["timezone"],
                    "K3S_CLUSTER_NAMES": json.dumps([f"{self.name}-k3s-asg"]),
                    "ENABLE_SMART_SCALING": "true",
                }
            ),
            timeout=300,
            memory_size=1024,  # Need more memory for smart decisions
            layers=[self._get_pandas_layer()],  # For data analysis
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Exchange-aware scheduling rules
        self._create_exchange_aware_schedules()
    
    def _enable_cost_anomaly_detection(self):
        """
        AI-powered anomaly detection - catch cost spikes before they hurt.
        Enhanced with trading pattern awareness.
        """
        # Cost anomaly monitor with custom ML model
        self.anomaly_monitor = aws.costexplorer.AnomalyMonitor(
            f"{self.name}-anomaly-monitor",
            name=f"nexlify-cost-anomalies-{self.config['cost_center']}",
            monitor_type="CUSTOM",
            monitor_specification=json.dumps({
                "Dimensions": {
                    "Key": "LINKED_ACCOUNT",
                    "Values": [aws.get_caller_identity().account_id],
                },
                "Tags": {
                    "Keys": ["Project", "Environment", "Exchange", "Component"],
                    "Values": None,  # Monitor all values
                },
                "CostCategories": {
                    "Keys": ["Trading", "Infrastructure", "Development"],
                },
            }),
            opts=ResourceOptions(parent=self)
        )
        
        # Multiple anomaly subscriptions for different thresholds
        anomaly_thresholds = {
            "critical": 1000.0,  # $1000+ anomaly
            "high": 500.0,       # $500+ anomaly
            "medium": 100.0,     # $100+ anomaly
        }
        
        for severity, threshold in anomaly_thresholds.items():
            aws.costexplorer.AnomalySubscription(
                f"{self.name}-anomaly-{severity}",
                name=f"nexlify-anomaly-{severity}-{self.config['cost_center']}",
                frequency="IMMEDIATE" if severity == "critical" else "DAILY",
                monitor_arn_lists=[self.anomaly_monitor.arn],
                threshold=threshold,
                subscribers=[
                    aws.costexplorer.AnomalySubscriptionSubscriberArgs(
                        type="EMAIL",
                        address=email,
                    )
                    for email in self._get_alert_emails(severity)
                ] + ([
                    aws.costexplorer.AnomalySubscriptionSubscriberArgs(
                        type="SNS",
                        address=self._create_critical_alert_topic().arn,
                    )
                ] if severity == "critical" else []),
                opts=ResourceOptions(parent=self)
            )
    
    def _configure_savings_plans(self):
        """
        Automated savings plan recommendations and purchases.
        Now with workload-aware commitment strategies.
        """
        # Lambda for savings plan analysis
        self.savings_analyzer = aws.lambda_.Function(
            f"{self.name}-savings-analyzer",
            runtime="python3.11",
            handler="analyze_savings.handler",
            code=pulumi.AssetArchive({
                "analyze_savings.py": pulumi.StringAsset(self._get_savings_analyzer_code()),
            }),
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "MIN_SAVINGS_THRESHOLD": "20",  # 20% minimum savings
                    "COMMITMENT_PERCENTAGE": "70",   # Commit 70% of steady-state
                    "ANALYSIS_PERIOD_DAYS": "30",
                    "K3S_NODE_COUNT": str(self.config["k3s_node_count"]),
                    "GPU_NODES": str(self.config.get("gpu_nodes", 0)),
                }
            ),
            timeout=900,  # 15 minutes for analysis
            memory_size=3008,
            schedule_expression="rate(7 days)",  # Weekly analysis
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Monitor savings plan utilization
        self._setup_savings_plan_monitoring()
    
    def _setup_cost_dashboard(self):
        """
        Enhanced cost dashboard with trading-specific metrics.
        Real-time cost per trade, per exchange, per strategy.
        """
        # CloudWatch dashboard with comprehensive widgets
        self.cost_dashboard = aws.cloudwatch.Dashboard(
            f"{self.name}-cost-dashboard",
            dashboard_name=f"nexlify-finops-{self.config['cost_center']}",
            dashboard_body=self._generate_dashboard_config(),
            opts=ResourceOptions(parent=self)
        )
        
        # Cost and usage report for detailed analysis
        self.cur_report = aws.cur.ReportDefinition(
            f"{self.name}-cost-report",
            report_name=f"nexlify-cur-{self.config['cost_center']}",
            time_unit="HOURLY",
            format="textORcsv",
            compression="GZIP",
            additional_schema_elements=["RESOURCES"],
            s3_bucket=self._create_cur_bucket().bucket,
            s3_prefix="cost-reports/",
            s3_region=aws.get_region().name,
            additional_artifacts=["REDSHIFT", "QUICKSIGHT"],
            refresh_closed_reports=True,
            report_versioning="OVERWRITE_REPORT",
            opts=ResourceOptions(parent=self)
        )
    
    def _configure_multi_region_optimization(self):
        """
        Multi-region cost optimization - arbitrage the clouds.
        Route workloads to the cheapest region that meets latency requirements.
        """
        if not self.config.get("enable_cross_region_optimization"):
            return
        
        # Lambda for cross-region cost analysis
        self.region_optimizer = aws.lambda_.Function(
            f"{self.name}-region-optimizer",
            runtime="python3.11",
            handler="optimize_regions.handler",
            code=pulumi.AssetArchive({
                "optimize_regions.py": pulumi.StringAsset(self._get_region_optimizer_code()),
            }),
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "REGIONS": json.dumps(self.config["regions"]),
                    "LATENCY_REQUIREMENTS": json.dumps({
                        "coinbase": 50,  # 50ms max latency
                        "binance": 100,
                        "kraken": 75,
                    }),
                    "WORKLOAD_MOBILITY": json.dumps({
                        "backtesting": "high",
                        "ml_training": "high",
                        "live_trading": "low",
                        "data_storage": "medium",
                    }),
                }
            ),
            timeout=300,
            memory_size=512,
            schedule_expression="rate(1 hour)",  # Hourly optimization
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
    
    def _calculate_spot_configuration(self) -> Dict[str, Any]:
        """
        Calculate optimal spot instance configuration based on
        current market prices and our workload requirements.
        """
        # This would normally call AWS APIs to get current spot prices
        # For now, return a sensible default configuration
        return {
            "target_capacity": 10,
            "instance_weights": {
                "c5.xlarge": 1,
                "c5n.xlarge": 1,
                "c6i.xlarge": 1,
                "m5.xlarge": 1,
                "m5n.xlarge": 1,
            },
            "bid_prices": {
                "c5.xlarge": 0.20,
                "c5n.xlarge": 0.22,
                "c6i.xlarge": 0.21,
                "m5.xlarge": 0.19,
                "m5n.xlarge": 0.21,
            }
        }
    
    def _configure_ml_spot_instances(self):
        """Configure GPU spot instances for ML workloads"""
        # GPU instance configuration with smart bidding
        gpu_spot_config = {
            "g4dn.xlarge": {"weight": 1, "max_price": 0.50},
            "g4dn.2xlarge": {"weight": 2, "max_price": 1.00},
            "g5.xlarge": {"weight": 1.5, "max_price": 0.75},
            "p3.2xlarge": {"weight": 4, "max_price": 2.00},  # For heavy training
        }
        
        self.ml_spot_fleet = aws.ec2.SpotFleetRequest(
            f"{self.name}-ml-fleet",
            iam_fleet_role=self._create_spot_fleet_role().arn,
            allocation_strategy="capacityOptimized",
            target_capacity=self.config.get("ml_spot_capacity", 2),
            launch_specifications=[
                aws.ec2.SpotFleetRequestLaunchSpecificationArgs(
                    instance_type=instance_type,
                    weighted_capacity=config["weight"],
                    spot_price=str(config["max_price"]),
                    subnet_id=subnet,
                    ami=self._get_ml_ami(),
                    user_data=self._get_ml_userdata(),
                    block_device_mappings=[
                        aws.ec2.SpotFleetRequestLaunchSpecificationBlockDeviceMappingArgs(
                            device_name="/dev/sda1",
                            ebs=aws.ec2.SpotFleetRequestLaunchSpecificationBlockDeviceMappingEbsArgs(
                                volume_size=200,  # ML needs more space
                                volume_type="gp3",
                                iops=10000,
                                throughput=250,
                                delete_on_termination=True,
                            ),
                        ),
                    ],
                    tags={
                        **self.config["tags"],
                        "Purpose": "ml-training",
                        "GPU": "enabled",
                        "Spot": "true",
                    },
                )
                for instance_type, config in gpu_spot_config.items()
                for subnet in self._get_gpu_enabled_subnets()
            ],
            valid_until=(datetime.now() + timedelta(days=7)).isoformat(),  # Weekly renewal
            replace_unhealthy_instances=True,
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-ml-fleet",
                "WorkloadType": "ml-training",
            },
            opts=ResourceOptions(parent=self)
        )
    
    def _setup_predictive_scaling(self):
        """
        Predictive scaling based on market patterns and historical data.
        Because we know when the markets get crazy.
        """
        # Lambda for predictive scaling decisions
        predictive_scaler = aws.lambda_.Function(
            f"{self.name}-predictive-scaler",
            runtime="python3.11",
            handler="predict_scale.handler",
            code=pulumi.AssetArchive({
                "predict_scale.py": pulumi.StringAsset("""
import boto3
import json
import numpy as np
from datetime import datetime, timedelta

def handler(event, context):
    # Predictive scaling based on:
    # 1. Historical trading volume patterns
    # 2. Market volatility indicators
    # 3. Scheduled economic events
    # 4. Exchange maintenance windows
    
    autoscaling = boto3.client('autoscaling')
    cloudwatch = boto3.client('cloudwatch')
    
    # Get current metrics
    response = cloudwatch.get_metric_statistics(
        Namespace='Nexlify/Trading',
        MetricName='OrdersPerSecond',
        StartTime=datetime.now() - timedelta(hours=24),
        EndTime=datetime.now(),
        Period=3600,
        Statistics=['Average', 'Maximum']
    )
    
    # Simple prediction logic (would be ML model in production)
    current_hour = datetime.now().hour
    predicted_load = 1.0
    
    # Market open/close times (UTC)
    market_peaks = {
        8: 1.5,   # Asian markets
        14: 2.0,  # European markets
        20: 2.5,  # US markets
    }
    
    if current_hour in market_peaks:
        predicted_load = market_peaks[current_hour]
    
    # Scale proactively
    desired_capacity = int(2 * predicted_load)  # Base capacity * multiplier
    
    autoscaling.set_desired_capacity(
        AutoScalingGroupName=event['asg_name'],
        DesiredCapacity=desired_capacity,
        HonorCooldown=False
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'predicted_load': predicted_load,
            'desired_capacity': desired_capacity
        })
    }
""")
            }),
            timeout=60,
            memory_size=512,
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Schedule predictive scaling checks
        aws.cloudwatch.EventRule(
            f"{self.name}-predictive-trigger",
            schedule_expression="rate(15 minutes)",
            description="Trigger predictive scaling analysis",
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
    
    def _create_exchange_aware_schedules(self):
        """
        Create scheduling rules aware of exchange trading hours.
        Because crypto never sleeps, but some exchanges do maintenance.
        """
        # Exchange-specific schedules
        exchange_schedules = {
            "coinbase": {
                "maintenance_window": "cron(0 2 ? * SUN *)",  # Sunday 2 AM UTC
                "peak_hours": [14, 15, 16, 20, 21, 22],  # US trading hours
            },
            "binance": {
                "maintenance_window": "cron(0 3 ? * WED *)",  # Wednesday 3 AM UTC
                "peak_hours": [0, 1, 2, 8, 9, 10],  # Asian trading hours
            },
            "kraken": {
                "maintenance_window": "cron(0 4 ? * TUE *)",  # Tuesday 4 AM UTC
                "peak_hours": [7, 8, 9, 14, 15, 16],  # EU trading hours
            },
        }
        
        for exchange, schedule in exchange_schedules.items():
            if exchange in self.config["exchanges"]:
                # Scale down during maintenance
                aws.cloudwatch.EventRule(
                    f"{self.name}-{exchange}-maintenance",
                    schedule_expression=schedule["maintenance_window"],
                    description=f"Scale down during {exchange} maintenance",
                    state="ENABLED",
                    tags={
                        **self.config["tags"],
                        "Exchange": exchange,
                        "Purpose": "maintenance-scaling",
                    },
                    opts=ResourceOptions(parent=self)
                )
    
    def _create_regional_budgets(self):
        """Create per-region budgets for multi-region deployments"""
        region_allocation = 1.0 / len(self.config["regions"])  # Equal split for now
        
        for region in self.config["regions"]:
            aws.budgets.Budget(
                f"{self.name}-{region}-budget",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=aws.budgets.BudgetBudgetLimitArgs(
                    amount=str(self.config["monthly_budget"] * region_allocation),
                    unit="USD",
                ),
                cost_filters=[
                    aws.budgets.BudgetCostFilterArgs(
                        name="Region",
                        values=[region],
                    ),
                    aws.budgets.BudgetCostFilterArgs(
                        name="TagKeyValue",
                        values=[f"Project${self.config['project_tags'].get('Project', 'Nexlify')}"],
                    ),
                ],
                notifications=[
                    aws.budgets.BudgetNotificationArgs(
                        comparison_operator="GREATER_THAN",
                        threshold=90.0,
                        threshold_type="PERCENTAGE",
                        notification_type="ACTUAL",
                        subscriber_sns_topic_arns=[self._create_alert_topic().arn],
                    ),
                ],
                tags={
                    **self.config["tags"],
                    "Region": region,
                },
                opts=ResourceOptions(parent=self)
            )
    
    def _create_service_budgets(self, allocations: Dict[str, float]):
        """Create budgets for individual services with smart allocations"""
        for service, budget in allocations.items():
            # Skip tiny budgets
            if budget < 50:  # Less than $50/month
                continue
                
            aws.budgets.Budget(
                f"{self.name}-{service.lower()}-budget",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=aws.budgets.BudgetBudgetLimitArgs(
                    amount=str(budget),
                    unit="USD",
                ),
                cost_filters=[
                    aws.budgets.BudgetCostFilterArgs(
                        name="Service",
                        values=[service],
                    ),
                ],
                notifications=[
                    aws.budgets.BudgetNotificationArgs(
                        comparison_operator="GREATER_THAN",
                        threshold=90.0,
                        threshold_type="PERCENTAGE",
                        notification_type="ACTUAL",
                        subscriber_sns_topic_arns=[self._create_alert_topic().arn],
                    ),
                    # Forecasted overrun for expensive services
                    aws.budgets.BudgetNotificationArgs(
                        comparison_operator="GREATER_THAN",
                        threshold=100.0,
                        threshold_type="PERCENTAGE",
                        notification_type="FORECASTED",
                        subscriber_sns_topic_arns=[self._create_alert_topic().arn],
                    ) if budget > 500 else None,  # Only for $500+ services
                ].filter(None),  # Remove None values
                tags={
                    **self.config["tags"],
                    "Service": service,
                    "BudgetType": "service-specific",
                },
                opts=ResourceOptions(parent=self)
            )
    
    def _setup_savings_plan_monitoring(self):
        """Monitor savings plan utilization and coverage"""
        # Utilization alarm
        aws.cloudwatch.MetricAlarm(
            f"{self.name}-sp-utilization",
            alarm_name=f"nexlify-savings-plan-utilization-low",
            comparison_operator="LessThanThreshold",
            evaluation_periods=2,
            metric_name="SavingsPlansUtilization",
            namespace="AWS/SavingsPlans",
            period=86400,  # Daily
            statistic="Average",
            threshold=85.0,  # Alert if utilization drops below 85%
            alarm_description="Savings Plan utilization is low - consider adjusting",
            alarm_actions=[self._create_alert_topic().arn],
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Coverage alarm
        aws.cloudwatch.MetricAlarm(
            f"{self.name}-sp-coverage",
            alarm_name=f"nexlify-savings-plan-coverage-low",
            comparison_operator="LessThanThreshold",
            evaluation_periods=1,
            metric_name="SavingsPlansCoverage",
            namespace="AWS/SavingsPlans",
            period=86400,
            statistic="Average",
            threshold=70.0,  # Alert if coverage drops below 70%
            alarm_description="Savings Plan coverage is low - consider purchasing more",
            alarm_actions=[self._create_alert_topic().arn],
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
    
    def _generate_dashboard_config(self) -> str:
        """Generate comprehensive dashboard configuration"""
        dashboard_config = {
            "widgets": [
                # Total spend gauge
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 6,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Billing", "EstimatedCharges", {"stat": "Maximum"}]
                        ],
                        "view": "gauge",
                        "stacked": False,
                        "region": "us-east-1",
                        "title": "Monthly Spend vs Budget",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": self.config["monthly_budget"]
                            }
                        }
                    }
                },
                # Cost by service pie chart
                {
                    "type": "metric",
                    "x": 6,
                    "y": 0,
                    "width": 6,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Billing", "EstimatedCharges", 
                             {"stat": "Maximum"}, 
                             {"label": "Total"}]
                        ],
                        "view": "pie",
                        "stacked": False,
                        "region": "us-east-1",
                        "title": "Cost by Service"
                    }
                },
                # Spot savings
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 6,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "SpotSavings", {"stat": "Sum"}],
                            [".", "OnDemandCost", {"stat": "Sum"}],
                        ],
                        "period": 86400,
                        "stat": "Sum",
                        "region": aws.get_region().name,
                        "title": "Spot vs On-Demand Costs",
                        "view": "singleValue",
                    }
                },
                # Cost per trade metric
                {
                    "type": "metric",
                    "x": 18,
                    "y": 0,
                    "width": 6,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["Nexlify/Trading", "CostPerTrade", {"stat": "Average"}],
                            [".", "CostPerMillionMessages", {"stat": "Average"}],
                        ],
                        "period": 3600,
                        "stat": "Average",
                        "region": aws.get_region().name,
                        "title": "Trading Efficiency Metrics",
                        "view": "singleValue",
                    }
                },
                # Daily spend trend
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Billing", "EstimatedCharges", {"stat": "Maximum"}],
                            ["...", {"stat": "Maximum", "period": 86400}],
                        ],
                        "period": 3600,
                        "stat": "Maximum",
                        "region": "us-east-1",
                        "title": "Daily Spend Trend",
                        "view": "timeSeries",
                        "stacked": False,
                        "annotations": {
                            "horizontal": [{
                                "label": "Daily Budget",
                                "value": self.config["monthly_budget"] / 30
                            }]
                        }
                    }
                },
                # Per-region costs
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [f"AWS/Billing", "EstimatedCharges", 
                             {"stat": "Maximum"}, 
                             {"label": region}]
                            for region in self.config.get("regions", [aws.get_region().name])
                        ],
                        "period": 86400,
                        "stat": "Maximum",
                        "region": "us-east-1",
                        "title": "Cost by Region",
                        "view": "timeSeries",
                        "stacked": True,
                    }
                },
                # K3s node utilization
                {
                    "type": "metric",
                    "x": 12,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "CPUUtilization", 
                             {"stat": "Average", "label": "K3s CPU"}],
                            [".", "NetworkIn", 
                             {"stat": "Sum", "label": "Network In", "yAxis": "right"}],
                            [".", "NetworkOut", 
                             {"stat": "Sum", "label": "Network Out", "yAxis": "right"}],
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": aws.get_region().name,
                        "title": "K3s Cluster Utilization",
                        "view": "timeSeries",
                        "stacked": False,
                    }
                },
            ]
        }
        
        return json.dumps(dashboard_config)
    
    # Getter methods for code strings
    def _get_smart_scheduler_code(self) -> str:
        """Smart scheduler that understands trading patterns"""
        return """
import boto3
import json
import os
from datetime import datetime, timezone
import pandas as pd

def handler(event, context):
    ec2 = boto3.client('ec2')
    autoscaling = boto3.client('autoscaling')
    
    exchanges = json.loads(os.environ['EXCHANGES'])
    peak_hours = json.loads(os.environ['PEAK_HOURS'])
    k3s_clusters = json.loads(os.environ['K3S_CLUSTER_NAMES'])
    
    current_hour = datetime.now(timezone.utc).hour
    current_day = datetime.now(timezone.utc).strftime('%A')
    
    # Determine if we're in peak trading hours for any exchange
    in_peak_hours = False
    for exchange in exchanges:
        if exchange in peak_hours and current_hour in peak_hours[exchange]:
            in_peak_hours = True
            break
    
    # Get all instances
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Environment', 'Values': ['dev', 'staging']},
            {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
        ]
    )
    
    action = 'start' if in_peak_hours else 'stop'
    instance_ids = []
    
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            # Skip critical infrastructure
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            if tags.get('AlwaysOn') == 'true':
                continue
                
            if instance['State']['Name'] == 'running' and action == 'stop':
                instance_ids.append(instance['InstanceId'])
            elif instance['State']['Name'] == 'stopped' and action == 'start':
                instance_ids.append(instance['InstanceId'])
    
    # Execute action
    if instance_ids:
        if action == 'stop':
            ec2.stop_instances(InstanceIds=instance_ids)
            print(f"Stopped {len(instance_ids)} instances")
        else:
            ec2.start_instances(InstanceIds=instance_ids)
            print(f"Started {len(instance_ids)} instances")
    
    # Adjust K3s cluster sizes based on load
    if os.environ.get('ENABLE_SMART_SCALING') == 'true':
        for cluster_name in k3s_clusters:
            if in_peak_hours:
                desired = 3  # Scale up for peak
            else:
                desired = 1  # Minimum for off-peak
            
            try:
                autoscaling.set_desired_capacity(
                    AutoScalingGroupName=cluster_name,
                    DesiredCapacity=desired,
                    HonorCooldown=False
                )
            except Exception as e:
                print(f"Failed to scale {cluster_name}: {str(e)}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'action': action,
            'affected_instances': len(instance_ids),
            'in_peak_hours': in_peak_hours
        })
    }
"""
    
    def _get_savings_analyzer_code(self) -> str:
        """Analyze usage patterns and recommend savings plans"""
        return """
import boto3
import json
import os
from datetime import datetime, timedelta
import pandas as pd

def handler(event, context):
    ce = boto3.client('cost-explorer')
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(os.environ['ANALYSIS_PERIOD_DAYS']))
    
    # Get usage data
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.isoformat(),
            'End': end_date.isoformat()
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost', 'UsageQuantity'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'},
            {'Type': 'DIMENSION', 'Key': 'INSTANCE_TYPE'}
        ]
    )
    
    # Analyze patterns
    usage_data = []
    for result in response['ResultsByTime']:
        for group in result['Groups']:
            usage_data.append({
                'date': result['TimePeriod']['Start'],
                'service': group['Keys'][0],
                'instance_type': group['Keys'][1],
                'cost': float(group['Metrics']['UnblendedCost']['Amount']),
                'usage': float(group['Metrics']['UsageQuantity']['Amount'])
            })
    
    df = pd.DataFrame(usage_data)
    
    # Calculate steady-state usage (bottom percentile)
    commitment_percentile = int(os.environ['COMMITMENT_PERCENTAGE'])
    recommendations = []
    
    for service in df['service'].unique():
        service_df = df[df['service'] == service]
        
        # Calculate percentile for commitment
        daily_costs = service_df.groupby('date')['cost'].sum()
        commitment_amount = daily_costs.quantile(commitment_percentile / 100)
        
        potential_savings = (daily_costs.mean() - commitment_amount) * 0.28  # 28% savings estimate
        
        if potential_savings > float(os.environ['MIN_SAVINGS_THRESHOLD']):
            recommendations.append({
                'service': service,
                'recommended_commitment': commitment_amount * 30,  # Monthly
                'estimated_monthly_savings': potential_savings * 30,
                'confidence': 'high' if daily_costs.std() < daily_costs.mean() * 0.2 else 'medium'
            })
    
    # Special handling for K3s nodes
    k3s_count = int(os.environ['K3S_NODE_COUNT'])
    gpu_count = int(os.environ['GPU_NODES'])
    
    if k3s_count > 0:
        # Recommend compute savings plan for K3s
        recommendations.append({
            'service': 'EC2-K3s',
            'recommended_commitment': k3s_count * 150,  # Rough estimate
            'estimated_monthly_savings': k3s_count * 150 * 0.28,
            'confidence': 'high',
            'notes': f'Based on {k3s_count} K3s nodes running 24/7'
        })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'analysis_period_days': os.environ['ANALYSIS_PERIOD_DAYS'],
            'recommendations': recommendations,
            'total_potential_savings': sum(r['estimated_monthly_savings'] for r in recommendations)
        })
    }
"""
    
    def _get_region_optimizer_code(self) -> str:
        """Optimize workload placement across regions"""
        return """
import boto3
import json
import os
from concurrent.futures import ThreadPoolExecutor
import time

def get_region_pricing(region, instance_type):
    '''Get current pricing for instance type in region'''
    pricing = boto3.client('pricing', region_name='us-east-1')
    
    response = pricing.get_products(
        ServiceCode='AmazonEC2',
        Filters=[
            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region},
            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
        ]
    )
    
    # Parse pricing (simplified)
    if response['PriceList']:
        price_data = json.loads(response['PriceList'][0])
        on_demand = list(price_data['terms']['OnDemand'].values())[0]
        price_dimensions = list(on_demand['priceDimensions'].values())[0]
        return float(price_dimensions['pricePerUnit']['USD'])
    return None

def measure_latency(region, endpoint):
    '''Measure latency to endpoint from region'''
    # Simplified - in production would actually measure
    latency_map = {
        'us-east-1': {'coinbase': 20, 'binance': 150, 'kraken': 25},
        'eu-west-1': {'coinbase': 80, 'binance': 100, 'kraken': 15},
        'ap-southeast-1': {'coinbase': 180, 'binance': 30, 'kraken': 120},
    }
    return latency_map.get(region, {}).get(endpoint, 999)

def handler(event, context):
    regions = json.loads(os.environ['REGIONS'])
    latency_requirements = json.loads(os.environ['LATENCY_REQUIREMENTS'])
    workload_mobility = json.loads(os.environ['WORKLOAD_MOBILITY'])
    
    # Analyze each region
    region_scores = {}
    
    with ThreadPoolExecutor(max_workers=len(regions)) as executor:
        for region in regions:
            # Get current costs
            instance_price = get_region_pricing(region, 'c5.xlarge')
            
            # Check latency to exchanges
            latencies = {}
            for exchange in latency_requirements:
                latencies[exchange] = measure_latency(region, exchange)
            
            # Calculate region score
            score = {
                'region': region,
                'instance_price': instance_price,
                'latencies': latencies,
                'meets_requirements': all(
                    latencies.get(ex, 999) <= req 
                    for ex, req in latency_requirements.items()
                ),
                'cost_score': 1.0 / instance_price if instance_price else 0,
            }
            
            region_scores[region] = score
    
    # Recommend workload placement
    recommendations = {}
    
    for workload, mobility in workload_mobility.items():
        if mobility == 'high':
            # Place in cheapest region that meets requirements
            valid_regions = [
                r for r, s in region_scores.items() 
                if s['meets_requirements']
            ]
            if valid_regions:
                best_region = min(
                    valid_regions, 
                    key=lambda r: region_scores[r]['instance_price']
                )
                recommendations[workload] = best_region
        else:
            # Keep in primary region
            recommendations[workload] = regions[0]
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'region_analysis': region_scores,
            'recommendations': recommendations,
            'potential_savings': calculate_savings(region_scores, recommendations)
        })
    }

def calculate_savings(scores, recommendations):
    # Simplified calculation
    primary_cost = scores[list(scores.keys())[0]]['instance_price'] or 0.20
    optimized_cost = sum(
        scores.get(r, {}).get('instance_price', primary_cost) 
        for r in recommendations.values()
    ) / len(recommendations)
    
    return (primary_cost - optimized_cost) * 730 * 10  # Monthly hours * instances
"""
    
    # Helper methods
    def _create_spot_fleet_role(self) -> aws.iam.Role:
        """IAM role for spot fleet"""
        if hasattr(self, '_spot_fleet_role'):
            return self._spot_fleet_role
            
        self._spot_fleet_role = aws.iam.Role(
            f"{self.name}-spot-fleet-role",
            assume_role_policy=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Action": "sts:AssumeRole",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "spotfleet.amazonaws.com"
                    }
                }]
            }),
            managed_policy_arns=[
                "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
            ],
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        return self._spot_fleet_role
    
    def _create_alert_topic(self) -> aws.sns.Topic:
        """SNS topic for cost alerts"""
        if not hasattr(self, '_alert_topic'):
            self._alert_topic = aws.sns.Topic(
                f"{self.name}-cost-alerts",
                display_name="Nexlify Cost Alerts - Keep the eddies flowing",
                tags=self.config["tags"],
                opts=ResourceOptions(parent=self)
            )
        return self._alert_topic
    
    def _create_critical_alert_topic(self) -> aws.sns.Topic:
        """SNS topic for critical cost alerts with automated response"""
        if not hasattr(self, '_critical_topic'):
            self._critical_topic = aws.sns.Topic(
                f"{self.name}-critical-alerts",
                display_name="Nexlify Critical Cost Alerts - Emergency Response",
                tags=self.config["tags"],
                opts=ResourceOptions(parent=self)
            )
            
            # Add Lambda for automated response
            response_lambda = self._create_cost_response_lambda()
            
            # Grant SNS permission to invoke Lambda
            aws.lambda_.Permission(
                f"{self.name}-sns-invoke-permission",
                action="lambda:InvokeFunction",
                function=response_lambda.name,
                principal="sns.amazonaws.com",
                source_arn=self._critical_topic.arn,
                opts=ResourceOptions(parent=self)
            )
            
            # Subscribe Lambda to topic
            aws.sns.TopicSubscription(
                f"{self.name}-critical-response",
                topic=self._critical_topic.arn,
                protocol="lambda",
                endpoint=response_lambda.arn,
                opts=ResourceOptions(parent=self)
            )
        return self._critical_topic
    
    def _create_cost_response_lambda(self) -> aws.lambda_.Function:
        """Lambda for automated cost control responses"""
        return aws.lambda_.Function(
            f"{self.name}-cost-response",
            runtime="python3.11",
            handler="index.handler",
            code=pulumi.AssetArchive({
                "index.py": pulumi.StringAsset("""
import json
import boto3
from datetime import datetime

def handler(event, context):
    # Parse SNS message
    message = json.loads(event['Records'][0]['Sns']['Message'])
    
    ec2 = boto3.client('ec2')
    autoscaling = boto3.client('autoscaling')
    
    print(f"CRITICAL COST ALERT: {message}")
    
    # Emergency cost control measures
    actions_taken = []
    
    # 1. Stop all dev/staging instances
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Environment', 'Values': ['dev', 'staging']},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    
    instance_ids = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            # Don't stop instances marked as critical
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            if tags.get('CriticalInfrastructure') != 'true':
                instance_ids.append(instance['InstanceId'])
    
    if instance_ids:
        ec2.stop_instances(InstanceIds=instance_ids)
        actions_taken.append(f"Stopped {len(instance_ids)} non-production instances")
    
    # 2. Scale down non-critical ASGs
    asg_response = autoscaling.describe_auto_scaling_groups()
    for asg in asg_response['AutoScalingGroups']:
        tags = {tag['Key']: tag['Value'] for tag in asg['Tags']}
        
        if tags.get('Environment') in ['dev', 'staging']:
            autoscaling.set_desired_capacity(
                AutoScalingGroupName=asg['AutoScalingGroupName'],
                DesiredCapacity=0,
                HonorCooldown=False
            )
            actions_taken.append(f"Scaled down ASG: {asg['AutoScalingGroupName']}")
    
    # 3. Terminate spot fleets
    spot_fleets = ec2.describe_spot_fleet_requests(
        Filters=[
            {'Name': 'state', 'Values': ['active', 'active_running']}
        ]
    )
    
    for fleet in spot_fleets['SpotFleetRequestConfigs']:
        tags = fleet.get('Tags', [])
        if any(tag['Key'] == 'Environment' and tag['Value'] != 'prod' for tag in tags):
            ec2.cancel_spot_fleet_requests(
                SpotFleetRequestIds=[fleet['SpotFleetRequestId']],
                TerminateInstances=True
            )
            actions_taken.append(f"Terminated spot fleet: {fleet['SpotFleetRequestId']}")
    
    # 4. Send notification
    sns = boto3.client('sns')
    sns.publish(
        TopicArn=context.invoked_function_arn.replace(':function:', ':topic:').replace('-cost-response', '-ops-alerts'),
        Subject="EMERGENCY: Cost Control Measures Activated",
        Message=json.dumps({
            'timestamp': datetime.now().isoformat(),
            'trigger': message,
            'actions_taken': actions_taken,
            'estimated_savings': len(instance_ids) * 0.20 * 24,  # Rough estimate
        }, indent=2)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'success': True,
            'actions_taken': actions_taken
        })
    }
""")
            }),
            timeout=60,
            memory_size=256,
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
    
    def _create_cur_bucket(self) -> aws.s3.BucketV2:
        """Create S3 bucket for Cost and Usage Reports"""
        cur_bucket = aws.s3.BucketV2(
            f"{self.name}-cur-bucket",
            bucket=f"{self.name}-cost-reports-{aws.get_caller_identity().account_id}",
            force_destroy=False,  # Keep cost data
            tags={
                **self.config["tags"],
                "Purpose": "cost-and-usage-reports",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Bucket policy for CUR
        bucket_policy = aws.iam.get_policy_document(
            statements=[{
                "effect": "Allow",
                "principals": [{
                    "type": "Service",
                    "identifiers": ["billingreports.amazonaws.com"],
                }],
                "actions": [
                    "s3:GetBucketAcl",
                    "s3:GetBucketPolicy",
                ],
                "resources": [cur_bucket.arn],
            }, {
                "effect": "Allow",
                "principals": [{
                    "type": "Service",
                    "identifiers": ["billingreports.amazonaws.com"],
                }],
                "actions": ["s3:PutObject"],
                "resources": [pulumi.Output.concat(cur_bucket.arn, "/*")],
            }]
        )
        
        aws.s3.BucketPolicy(
            f"{self.name}-cur-bucket-policy",
            bucket=cur_bucket.id,
            policy=bucket_policy.json,
            opts=ResourceOptions(parent=cur_bucket)
        )
        
        return cur_bucket
    
    # Utility methods
    def _get_alert_emails(self, severity: str = "normal") -> List[str]:
        """Get email addresses for alerts based on severity"""
        base_emails = ["ops@nexlify.com", "finance@nexlify.com"]
        
        if severity == "critical":
            base_emails.extend(["cto@nexlify.com", "oncall@nexlify.com"])
        elif severity == "high":
            base_emails.append("management@nexlify.com")
            
        return base_emails
    
    def _get_optimized_ami(self) -> str:
        """Get the most cost-effective AMI for our workload"""
        # In production, this would look up the latest optimized AMI
        # For now, return Amazon Linux 2023 minimal
        return "ami-0c02fb55956c7d316"
    
    def _get_ml_ami(self) -> str:
        """Get ML-optimized AMI with GPU drivers"""
        # Deep Learning AMI with CUDA pre-installed
        return "ami-0c94855ba95c798c0"
    
    def _get_spot_userdata(self) -> str:
        """User data for spot instances"""
        return """#!/bin/bash
# Spot instance initialization
set -e

# Tag instance with termination time
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
REGION=$(ec2-metadata --availability-zone | cut -d " " -f 2 | sed 's/.$//')

# Install termination handler
wget -O /usr/local/bin/spot-handler https://github.com/kube-aws/kube-spot-termination-notice-handler/releases/download/v1.13.7/handler
chmod +x /usr/local/bin/spot-handler

# Start monitoring for termination
nohup /usr/local/bin/spot-handler --metadata-endpoint http://169.254.169.254 > /var/log/spot-handler.log 2>&1 &

# Your workload initialization here
echo "Spot instance ready for workload"
"""
    
    def _get_ml_userdata(self) -> str:
        """User data for ML instances"""
        return """#!/bin/bash
# ML instance initialization
set -e

# Configure GPU
nvidia-smi
nvidia-persistenced

# Mount high-performance storage
mkfs.xfs /dev/nvme1n1
mount /dev/nvme1n1 /ml-data
echo '/dev/nvme1n1 /ml-data xfs defaults 0 0' >> /etc/fstab

# Start ML environment
docker run -d --gpus all nexlify/ml-training:latest
"""
    
    def _get_cheapest_subnets(self) -> List[str]:
        """Get subnet IDs in AZs with lowest spot prices"""
        # This would query spot price history
        # For now, return placeholder
        return ["subnet-12345", "subnet-67890"]
    
    def _get_gpu_enabled_subnets(self) -> List[str]:
        """Get subnets in AZs with GPU availability"""
        return ["subnet-gpu1", "subnet-gpu2"]
    
    def _get_pandas_layer(self) -> str:
        """Get Lambda layer ARN for pandas"""
        # AWS Data Wrangler layer includes pandas
        region = aws.get_region().name
        return f"arn:aws:lambda:{region}:336392948345:layer:AWSDataWrangler-Python311:1"
    
    # Properties
    @property
    def estimated_cost(self) -> Output[float]:
        """Estimated monthly cost based on current usage"""
        # This would query Cost Explorer API
        # For now, return estimate based on configuration
        base_cost = self.config["k3s_node_count"] * 150  # $150/node estimate
        gpu_cost = self.config.get("gpu_nodes", 0) * 500  # $500/GPU node
        spot_savings = base_cost * self.config["spot_percentage"] / 100 * 0.7  # 70% savings
        
        return Output.from_input(base_cost + gpu_cost - spot_savings)
    
    @property
    def potential_savings(self) -> Output[float]:
        """Potential savings from optimizations"""
        current_estimate = self.estimated_cost
        
        # Calculate potential savings
        savings_factors = {
            "spot_increase": 0.15 if self.config["spot_percentage"] < 50 else 0.05,
            "rightsizing": 0.20 if self.config["enable_rightsizing"] else 0,
            "scheduling": 0.30 if self.config["business_hours_only"] else 0,
            "multi_region": 0.10 if self.config.get("enable_cross_region_optimization") else 0,
        }
        
        total_savings_percentage = sum(savings_factors.values())
        
        return current_estimate.apply(lambda cost: cost * total_savings_percentage)
    
    @property
    def dashboard_url(self) -> Output[str]:
        """URL to cost dashboard"""
        return Output.concat(
            "https://",
            aws.get_region().name,
            ".console.aws.amazon.com/cloudwatch/home?region=",
            aws.get_region().name,
            "#dashboards:name=",
            self.cost_dashboard.dashboard_name
        )
    
    @property
    def cost_breakdown(self) -> Output[Dict[str, float]]:
        """Detailed cost breakdown by service"""
        allocations = self._calculate_service_allocations()
        
        return Output.from_input({
            "services": allocations,
            "total_budget": self.config["monthly_budget"],
            "estimated_usage": self.estimated_cost,
            "potential_savings": self.potential_savings,
        })
