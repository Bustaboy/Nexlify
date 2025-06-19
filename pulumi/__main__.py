"""
🌃 NEXLIFY TRADING MATRIX - INFRASTRUCTURE ORCHESTRATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the market bleeds, we scale. When opportunity knocks, we're already there.
This is where silicon meets soul, where infrastructure becomes intelligence.

Author: Your favorite street samurai turned cloud architect
License: MIT (because even rebels believe in freedom)
"""

from typing import TypedDict, Optional, List, Mapping
import pulumi
from pulumi import Config, Output, ResourceOptions
import pulumi_aws as aws
import pulumi_azure_native as azure
import pulumi_kubernetes as k8s

# Local neural modules - each one a piece of our digital soul
from .trading_platform import TradingPlatform, TradingPlatformArgs
from .networking import MultiCloudNetwork, NetworkConfig
from .security import ZeroTrustSecurity, SecurityConfig
from .database import TradingDataLayer, DatabaseConfig
from .websocket_api import RealtimeMarketAPI, WebSocketConfig
from .cost_optimization import FinOpsAutomation, CostConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TYPE-SAFE CONFIGURATION - Because runtime errors are for amateurs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NexlifyConfig(TypedDict):
    """
    The neural blueprint of our trading matrix. Every field here represents
    a decision that could make or break our operation. Choose wisely, choom.
    """
    # Core identity - who we are in this digital warzone
    project_name: str
    environment: str  # dev/staging/prod - each one a different battlefield
    region_primary: str
    region_dr: str  # Disaster recovery - because shit happens
    
    # Trading configuration - the heart of our operation
    exchanges: List[str]  # ["coinbase", "binance", "kraken"]
    initial_capital_usd: float
    max_position_size: float
    risk_percentage: float
    
    # Infrastructure sizing - balance power with cost
    k3s_node_count: int
    gpu_nodes: int
    spot_instance_percentage: float  # How much risk for cost savings?
    
    # Security & compliance - the chrome that keeps us safe
    enable_cloudhsm: bool
    compliance_mode: str  # "SOC2", "PCI-DSS", "MINIMAL"
    audit_retention_days: int
    
    # Feature flags - evolution in action
    enable_ml_trading: bool
    enable_backtesting: bool
    enable_paper_trading: bool
    enable_multi_region: bool
    
    # Cost controls - because eddies don't grow on trees
    monthly_budget_usd: float
    cost_alert_threshold: float  # Percentage of budget
    
    # Monitoring - our digital eyes and ears
    enable_enhanced_monitoring: bool
    log_retention_days: int
    metric_retention_days: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION LOADING - Where dreams meet reality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_nexlify_config() -> NexlifyConfig:
    """
    Load configuration from Pulumi config with defaults that won't flatline
    your wallet or your operation. Every default here is battle-tested.
    """
    config = Config()
    
    # The streets taught me - always have a fallback plan
    return NexlifyConfig(
        project_name=config.get("project_name") or "nexlify-trading-matrix",
        environment=pulumi.get_stack(),  # Stack name IS the environment
        region_primary=config.get("region_primary") or "us-east-1",
        region_dr=config.get("region_dr") or "eu-west-1",
        
        # Trading params - conservative defaults for survival
        exchanges=config.get_object("exchanges") or ["coinbase"],
        initial_capital_usd=config.get_float("initial_capital") or 10000.0,
        max_position_size=config.get_float("max_position_size") or 0.1,  # 10% max
        risk_percentage=config.get_float("risk_percentage") or 0.02,  # 2% risk
        
        # Infrastructure - start small, scale with success
        k3s_node_count=config.get_int("k3s_nodes") or 3,
        gpu_nodes=config.get_int("gpu_nodes") or 0,  # GPUs are expensive, choom
        spot_instance_percentage=config.get_float("spot_percentage") or 0.3,
        
        # Security - paranoid by default
        enable_cloudhsm=config.get_bool("enable_cloudhsm") or False,
        compliance_mode=config.get("compliance_mode") or "MINIMAL",
        audit_retention_days=config.get_int("audit_retention") or 90,
        
        # Features - crawl before you run
        enable_ml_trading=config.get_bool("enable_ml") or False,
        enable_backtesting=config.get_bool("enable_backtesting") or True,
        enable_paper_trading=config.get_bool("enable_paper") or True,
        enable_multi_region=config.get_bool("multi_region") or False,
        
        # Cost management - don't let the cloud drain your eddies
        monthly_budget_usd=config.get_float("monthly_budget") or 1000.0,
        cost_alert_threshold=config.get_float("cost_alert") or 0.8,
        
        # Monitoring - see everything, miss nothing
        enable_enhanced_monitoring=config.get_bool("enhanced_monitoring") or True,
        log_retention_days=config.get_int("log_retention") or 30,
        metric_retention_days=config.get_int("metric_retention") or 90,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN DEPLOYMENT - Where the magic happens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def deploy_nexlify_infrastructure():
    """
    The main deployment function. This is where we bring our digital
    trading empire to life. Every component here is a piece of our
    collective neural network.
    
    Remember: In Night City, you're either predator or prey.
    This infrastructure makes sure we're the former.
    """
    
    # Load our battle plan
    config = load_nexlify_config()
    
    # Tag everything - accountability in the digital age
    base_tags = {
        "Project": config["project_name"],
        "Environment": config["environment"],
        "ManagedBy": "Pulumi",
        "CostCenter": "trading-operations",
        "Creator": pulumi.get_project(),  # Know who built what
        "Cyberpunk": "2077",  # Never forget our roots
    }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: NETWORKING - The digital highways of our empire
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    network = MultiCloudNetwork(
        "nexlify-network",
        NetworkConfig(
            primary_region=config["region_primary"],
            dr_region=config["region_dr"] if config["enable_multi_region"] else None,
            enable_private_endpoints=True,
            enable_flow_logs=config["enable_enhanced_monitoring"],
            tags=base_tags,
        ),
        opts=ResourceOptions(
            protect=config["environment"] == "prod",  # Can't delete prod by accident
        )
    )
    
    pulumi.export("vpc_id", network.primary_vpc.id)
    pulumi.export("private_subnet_ids", network.private_subnet_ids)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: SECURITY - The chrome that keeps the wolves at bay
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    security = ZeroTrustSecurity(
        "nexlify-security",
        SecurityConfig(
            vpc_id=network.primary_vpc.id,
            enable_cloudhsm=config["enable_cloudhsm"],
            compliance_mode=config["compliance_mode"],
            audit_retention_days=config["audit_retention_days"],
            allowed_exchanges=config["exchanges"],
            tags=base_tags,
        ),
        opts=ResourceOptions(
            parent=network,
            protect=True,  # NEVER accidentally delete security
        )
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: DATA LAYER - Where market memories become profitable futures
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    data_layer = TradingDataLayer(
        "nexlify-data",
        DatabaseConfig(
            subnet_ids=network.private_subnet_ids,
            security_group_id=security.database_sg.id,
            instance_class="db.r6g.xlarge" if config["environment"] == "prod" else "db.t4g.medium",
            retention_days=config["metric_retention_days"],
            enable_multi_az=config["environment"] == "prod",
            tags=base_tags,
        ),
        opts=ResourceOptions(parent=network)
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: TRADING PLATFORM - The beating heart of our operation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    platform = TradingPlatform(
        "nexlify-platform",
        TradingPlatformArgs(
            vpc_id=network.primary_vpc.id,
            subnet_ids=network.private_subnet_ids,
            k3s_node_count=config["k3s_node_count"],
            gpu_nodes=config["gpu_nodes"],
            exchanges=config["exchanges"],
            enable_ml=config["enable_ml_trading"],
            enable_backtesting=config["enable_backtesting"],
            database_endpoint=data_layer.questdb_endpoint,
            cache_endpoint=data_layer.valkey_endpoint,
            tags=base_tags,
        ),
        opts=ResourceOptions(
            depends_on=[network, security, data_layer],
            ignore_changes=["ami_id"] if config["environment"] == "prod" else None,
        )
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 5: REAL-TIME APIS - Where milliseconds become millions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if config["exchanges"]:  # Only if we're actually trading
        websocket_api = RealtimeMarketAPI(
            "nexlify-websocket",
            WebSocketConfig(
                vpc_id=network.primary_vpc.id,
                platform_endpoint=platform.api_endpoint,
                enable_throttling=True,
                max_connections=100000,  # Dream big, scale bigger
                tags=base_tags,
            ),
            opts=ResourceOptions(parent=platform)
        )
        
        pulumi.export("websocket_url", websocket_api.websocket_url)
        pulumi.export("api_gateway_url", websocket_api.api_url)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 6: COST OPTIMIZATION - Keep the eddies flowing, not bleeding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    finops = FinOpsAutomation(
        "nexlify-finops",
        CostConfig(
            monthly_budget=config["monthly_budget_usd"],
            alert_threshold=config["cost_alert_threshold"],
            spot_percentage=config["spot_instance_percentage"],
            enable_auto_stop=config["environment"] != "prod",
            backtesting_spot_enabled=config["enable_backtesting"],
            tags=base_tags,
        ),
        opts=ResourceOptions(
            depends_on=[platform],
            retain_on_delete=True,  # Keep cost data even if we tear down
        )
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OUTPUTS - The neural pathways to our infrastructure
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Critical endpoints - guard these with your life
    pulumi.export("k3s_endpoint", platform.k3s_endpoint)
    pulumi.export("trading_api_endpoint", platform.api_endpoint)
    pulumi.export("questdb_endpoint", data_layer.questdb_endpoint)
    pulumi.export("valkey_endpoint", data_layer.valkey_endpoint)
    
    # Monitoring & observability
    pulumi.export("grafana_url", platform.grafana_url)
    pulumi.export("prometheus_url", platform.prometheus_url)
    
    # Cost tracking - know where every eddie goes
    pulumi.export("estimated_monthly_cost", finops.estimated_monthly_cost)
    pulumi.export("cost_dashboard_url", finops.dashboard_url)
    
    # Status report - our infrastructure's vital signs
    pulumi.export("deployment_status", {
        "environment": config["environment"],
        "primary_region": config["region_primary"],
        "dr_enabled": config["enable_multi_region"],
        "ml_enabled": config["enable_ml_trading"],
        "exchanges_configured": config["exchanges"],
        "security_mode": config["compliance_mode"],
        "k3s_nodes": config["k3s_node_count"],
        "gpu_nodes": config["gpu_nodes"],
    })
    
    # The final message - a reminder of what we've built
    pulumi.export("message", 
        f"🌃 Nexlify Trading Matrix ONLINE in {config['environment']} mode. "
        f"May your trades be swift and your profits legendary."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT - Where the journey begins
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# This is it, choom. The moment where code becomes infrastructure,
# where dreams become digital reality. Jack in and let's ride.
deploy_nexlify_infrastructure()
