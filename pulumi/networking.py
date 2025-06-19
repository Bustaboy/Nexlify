"""
🌐 NEXLIFY NETWORK INFRASTRUCTURE - THE DIGITAL HIGHWAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In Night City, information flows faster than bullets. Our network needs to be
faster than both. Multi-cloud, multi-region, zero trust. Because when you're
moving millions in milliseconds, every packet counts.

This module builds:
- Primary VPC in AWS (our main fortress)
- DR VPC in another region (because shit happens)
- Private subnets across AZs (distributed resilience)
- VPC endpoints (keep traffic private, keep it fast)
- Cross-region peering (when one region burns, we keep trading)
"""

from typing import TypedDict, List, Optional, Dict
from dataclasses import dataclass
import ipaddress

import pulumi
from pulumi import ComponentResource, ResourceOptions, Output
import pulumi_aws as aws


class NetworkConfig(TypedDict):
    """Network configuration - the blueprint of our digital empire"""
    primary_region: str
    dr_region: Optional[str]
    enable_private_endpoints: bool
    enable_flow_logs: bool
    tags: Dict[str, str]


class MultiCloudNetwork(ComponentResource):
    """
    Multi-cloud network infrastructure optimized for trading.
    
    Features:
    - Redundant VPCs across regions
    - Private subnets for security
    - VPC endpoints for AWS services (no internet egress fees)
    - Flow logs for compliance and debugging
    - Cross-region peering for DR
    
    Built for speed, secured for survival.
    """
    
    def __init__(self, name: str, config: NetworkConfig, opts: Optional[ResourceOptions] = None):
        super().__init__("nexlify:network:MultiCloudNetwork", name, {}, opts)
        
        self.config = config
        self.name = name
        
        # Build our primary fortress
        self._create_primary_vpc()
        
        # If DR is enabled, build our backup fortress
        if config.get("dr_region"):
            self._create_dr_vpc()
            self._setup_cross_region_peering()
        
        # Register what we've built
        self.register_outputs({
            "primary_vpc_id": self.primary_vpc.id,
            "private_subnet_ids": self.private_subnet_ids,
            "public_subnet_ids": self.public_subnet_ids,
            "nat_gateway_ips": self.nat_gateway_ips,
        })
    
    def _create_primary_vpc(self):
        """
        Create the primary VPC - our main digital fortress.
        10.0.0.0/16 gives us 65,536 IPs. Should be enough for world domination.
        """
        # Main VPC
        self.primary_vpc = aws.ec2.Vpc(
            f"{self.name}-primary-vpc",
            cidr_block="10.0.0.0/16",
            enable_dns_hostnames=True,
            enable_dns_support=True,
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-primary-vpc",
                "Type": "primary",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Get availability zones - spread the risk
        azs = aws.get_availability_zones(state="available")
        
        # Create subnets across AZs
        self.private_subnets = []
        self.public_subnets = []
        self.private_subnet_ids = []
        self.public_subnet_ids = []
        
        # We'll use the first 3 AZs for redundancy
        for i in range(min(3, len(azs.names))):
            # Private subnet - where our trading engines live
            private_subnet = aws.ec2.Subnet(
                f"{self.name}-private-{azs.names[i]}",
                vpc_id=self.primary_vpc.id,
                cidr_block=f"10.0.{i+1}.0/24",  # 256 IPs per subnet
                availability_zone=azs.names[i],
                map_public_ip_on_launch=False,  # Private means private
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-private-{azs.names[i]}",
                    "Type": "private",
                    "Zone": azs.names[i],
                },
                opts=ResourceOptions(parent=self.primary_vpc)
            )
            self.private_subnets.append(private_subnet)
            self.private_subnet_ids.append(private_subnet.id)
            
            # Public subnet - for NAT gateways and load balancers
            public_subnet = aws.ec2.Subnet(
                f"{self.name}-public-{azs.names[i]}",
                vpc_id=self.primary_vpc.id,
                cidr_block=f"10.0.{i+101}.0/24",  # 256 IPs per subnet
                availability_zone=azs.names[i],
                map_public_ip_on_launch=True,
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-public-{azs.names[i]}",
                    "Type": "public",
                    "Zone": azs.names[i],
                },
                opts=ResourceOptions(parent=self.primary_vpc)
            )
            self.public_subnets.append(public_subnet)
            self.public_subnet_ids.append(public_subnet.id)
        
        # Internet Gateway - our connection to the outside world
        self.igw = aws.ec2.InternetGateway(
            f"{self.name}-igw",
            vpc_id=self.primary_vpc.id,
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-igw",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
        
        # NAT Gateways - let private resources reach out without being reached
        self.nat_gateways = []
        self.nat_gateway_ips = []
        
        for i, public_subnet in enumerate(self.public_subnets):
            # Elastic IP for NAT Gateway
            eip = aws.ec2.Eip(
                f"{self.name}-nat-eip-{i}",
                domain="vpc",
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-nat-eip-{i}",
                },
                opts=ResourceOptions(parent=public_subnet)
            )
            self.nat_gateway_ips.append(eip.public_ip)
            
            # NAT Gateway
            nat_gateway = aws.ec2.NatGateway(
                f"{self.name}-nat-{i}",
                subnet_id=public_subnet.id,
                allocation_id=eip.id,
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-nat-{i}",
                },
                opts=ResourceOptions(parent=public_subnet)
            )
            self.nat_gateways.append(nat_gateway)
        
        # Route tables - the traffic rules of our digital city
        self._setup_route_tables()
        
        # VPC Endpoints - keep AWS traffic private and fast
        if self.config.get("enable_private_endpoints", True):
            self._create_vpc_endpoints()
        
        # Flow logs - know every packet that moves through our empire
        if self.config.get("enable_flow_logs", True):
            self._setup_flow_logs()
    
    def _setup_route_tables(self):
        """
        Route tables - the GPS of our network.
        Tell packets where to go, and more importantly, where NOT to go.
        """
        # Public route table - internet access
        public_route_table = aws.ec2.RouteTable(
            f"{self.name}-public-rt",
            vpc_id=self.primary_vpc.id,
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-public-rt",
                "Type": "public",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
        
        # Route to internet via IGW
        aws.ec2.Route(
            f"{self.name}-public-route",
            route_table_id=public_route_table.id,
            destination_cidr_block="0.0.0.0/0",
            gateway_id=self.igw.id,
            opts=ResourceOptions(parent=public_route_table)
        )
        
        # Associate public subnets with public route table
        for i, subnet in enumerate(self.public_subnets):
            aws.ec2.RouteTableAssociation(
                f"{self.name}-public-rta-{i}",
                subnet_id=subnet.id,
                route_table_id=public_route_table.id,
                opts=ResourceOptions(parent=subnet)
            )
        
        # Private route tables - one per AZ for redundancy
        for i, (private_subnet, nat_gateway) in enumerate(zip(self.private_subnets, self.nat_gateways)):
            private_route_table = aws.ec2.RouteTable(
                f"{self.name}-private-rt-{i}",
                vpc_id=self.primary_vpc.id,
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-private-rt-{i}",
                    "Type": "private",
                },
                opts=ResourceOptions(parent=self.primary_vpc)
            )
            
            # Route to internet via NAT Gateway
            aws.ec2.Route(
                f"{self.name}-private-route-{i}",
                route_table_id=private_route_table.id,
                destination_cidr_block="0.0.0.0/0",
                nat_gateway_id=nat_gateway.id,
                opts=ResourceOptions(parent=private_route_table)
            )
            
            # Associate private subnet with route table
            aws.ec2.RouteTableAssociation(
                f"{self.name}-private-rta-{i}",
                subnet_id=private_subnet.id,
                route_table_id=private_route_table.id,
                opts=ResourceOptions(parent=private_subnet)
            )
    
    def _create_vpc_endpoints(self):
        """
        VPC Endpoints - private highways to AWS services.
        No internet, no egress fees, just pure speed.
        """
        # S3 endpoint - for storing all that sweet trading data
        s3_endpoint = aws.ec2.VpcEndpoint(
            f"{self.name}-s3-endpoint",
            vpc_id=self.primary_vpc.id,
            service_name=f"com.amazonaws.{self.config['primary_region']}.s3",
            route_table_ids=[rt.id for rt in self.private_route_tables],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-s3-endpoint",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
        
        # DynamoDB endpoint - for state management
        dynamodb_endpoint = aws.ec2.VpcEndpoint(
            f"{self.name}-dynamodb-endpoint",
            vpc_id=self.primary_vpc.id,
            service_name=f"com.amazonaws.{self.config['primary_region']}.dynamodb",
            route_table_ids=[rt.id for rt in self.private_route_tables],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-dynamodb-endpoint",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
        
        # Security group for interface endpoints
        endpoint_sg = aws.ec2.SecurityGroup(
            f"{self.name}-endpoint-sg",
            vpc_id=self.primary_vpc.id,
            description="Security group for VPC endpoints",
            ingress=[
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=[self.primary_vpc.cidr_block],
                    description="Allow all from VPC"
                ),
            ],
            egress=[
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["0.0.0.0/0"],
                    description="Allow all outbound"
                ),
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-endpoint-sg",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
        
        # Interface endpoints for other services
        interface_endpoints = {
            "ec2": f"com.amazonaws.{self.config['primary_region']}.ec2",
            "ecr-api": f"com.amazonaws.{self.config['primary_region']}.ecr.api",
            "ecr-dkr": f"com.amazonaws.{self.config['primary_region']}.ecr.dkr",
            "logs": f"com.amazonaws.{self.config['primary_region']}.logs",
            "monitoring": f"com.amazonaws.{self.config['primary_region']}.monitoring",
            "ssm": f"com.amazonaws.{self.config['primary_region']}.ssm",
            "secretsmanager": f"com.amazonaws.{self.config['primary_region']}.secretsmanager",
        }
        
        for service, endpoint_name in interface_endpoints.items():
            aws.ec2.VpcEndpoint(
                f"{self.name}-{service}-endpoint",
                vpc_id=self.primary_vpc.id,
                service_name=endpoint_name,
                vpc_endpoint_type="Interface",
                subnet_ids=self.private_subnet_ids,
                security_group_ids=[endpoint_sg.id],
                private_dns_enabled=True,
                tags={
                    **self.config["tags"],
                    "Name": f"{self.name}-{service}-endpoint",
                },
                opts=ResourceOptions(parent=self.primary_vpc)
            )
    
    def _setup_flow_logs(self):
        """
        Flow logs - because in the data game, you need to know who's
        talking to who, when, and how much. Compliance loves this stuff.
        """
        # S3 bucket for flow logs
        flow_log_bucket = aws.s3.BucketV2(
            f"{self.name}-flow-logs",
            bucket=f"{self.name}-flow-logs-{self.config['primary_region']}",
            force_destroy=self.config["tags"]["Environment"] != "prod",
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-flow-logs",
                "Purpose": "vpc-flow-logs",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Bucket policy for VPC Flow Logs
        bucket_policy_doc = aws.iam.get_policy_document(
            statements=[{
                "sid": "AWSLogDeliveryWrite",
                "effect": "Allow",
                "principals": [{
                    "type": "Service",
                    "identifiers": ["delivery.logs.amazonaws.com"],
                }],
                "actions": ["s3:PutObject"],
                "resources": [f"{flow_log_bucket.arn}/*"],
                "conditions": [{
                    "test": "StringEquals",
                    "variable": "s3:x-acl",
                    "values": ["bucket-owner-full-control"],
                }],
            }, {
                "sid": "AWSLogDeliveryAclCheck",
                "effect": "Allow",
                "principals": [{
                    "type": "Service",
                    "identifiers": ["delivery.logs.amazonaws.com"],
                }],
                "actions": ["s3:GetBucketAcl"],
                "resources": [flow_log_bucket.arn],
            }]
        )
        
        aws.s3.BucketPolicy(
            f"{self.name}-flow-log-policy",
            bucket=flow_log_bucket.id,
            policy=bucket_policy_doc.json,
            opts=ResourceOptions(parent=flow_log_bucket)
        )
        
        # Enable flow logs
        aws.ec2.FlowLog(
            f"{self.name}-vpc-flow-log",
            log_destination_type="s3",
            log_destination=flow_log_bucket.arn,
            traffic_type="ALL",
            vpc_id=self.primary_vpc.id,
            log_format="${srcaddr} ${dstaddr} ${srcport} ${dstport} ${protocol} ${packets} ${bytes} ${action}",
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-vpc-flow-log",
            },
            opts=ResourceOptions(parent=self.primary_vpc)
        )
    
    def _create_dr_vpc(self):
        """
        Disaster Recovery VPC - because when the primary region burns,
        we keep trading. Different region, same hustle.
        """
        # Create provider for DR region
        dr_provider = aws.Provider(
            f"{self.name}-dr-provider",
            region=self.config["dr_region"],
            opts=ResourceOptions(parent=self)
        )
        
        # DR VPC with different CIDR
        self.dr_vpc = aws.ec2.Vpc(
            f"{self.name}-dr-vpc",
            cidr_block="10.1.0.0/16",  # Different from primary
            enable_dns_hostnames=True,
            enable_dns_support=True,
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-dr-vpc",
                "Type": "disaster-recovery",
            },
            opts=ResourceOptions(parent=self, provider=dr_provider)
        )
        
        # Similar subnet setup for DR region
        # ... (Similar code to primary VPC setup)
    
    def _setup_cross_region_peering(self):
        """
        Cross-region VPC peering - connect our fortresses with a private tunnel.
        When one falls, the other stands. That's how we survive in this city.
        """
        # Note: This is a placeholder. Full implementation would include:
        # - VPC peering connection
        # - Route table updates
        # - Security group rules
        # - DNS resolution configuration
        pass
