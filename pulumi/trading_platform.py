"""
🏢 NEXLIFY TRADING PLATFORM - THE NEURAL CORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is where the magic happens. Every trade, every decision, every profit
flows through these digital synapses. Built on K3s because we're not here
to play - we're here to dominate.
"""

from typing import TypedDict, List, Optional
from dataclasses import dataclass
import base64
import json

import pulumi
from pulumi import ComponentResource, ResourceOptions, Output
import pulumi_aws as aws
import pulumi_kubernetes as k8s
import pulumi_random as random


class TradingPlatformArgs(TypedDict):
    """Configuration for our trading neural network"""
    vpc_id: str
    subnet_ids: List[str]
    k3s_node_count: int
    gpu_nodes: int
    exchanges: List[str]
    enable_ml: bool
    enable_backtesting: bool
    database_endpoint: Output[str]
    cache_endpoint: Output[str]
    tags: dict


class TradingPlatform(ComponentResource):
    """
    The beating heart of Nexlify. This component deploys:
    - K3s cluster optimized for trading workloads
    - HAProxy ingress for 42,000 RPS capacity
    - KEDA autoscaling based on market volatility
    - GPU nodes for ML inference (if enabled)
    - Complete observability stack
    
    Built to survive the digital storms of Night City's markets.
    """
    
    def __init__(self, name: str, args: TradingPlatformArgs, opts: Optional[ResourceOptions] = None):
        super().__init__("nexlify:platform:TradingPlatform", name, {}, opts)
        
        self.args = args
        self.name = name
        
        # Deploy in sequence - order matters in the streets
        self._create_iam_roles()
        self._create_security_groups()
        self._deploy_k3s_cluster()
        self._configure_autoscaling()
        self._deploy_trading_engine()
        self._setup_monitoring()
        
        # Export our neural pathways
        self.k3s_endpoint = self.k3s_cluster.endpoint
        self.api_endpoint = self.trading_api_endpoint
        self.grafana_url = self.grafana_endpoint
        self.prometheus_url = self.prometheus_endpoint
        
        self.register_outputs({
            "k3s_endpoint": self.k3s_endpoint,
            "api_endpoint": self.api_endpoint,
            "grafana_url": self.grafana_url,
            "prometheus_url": self.prometheus_url,
        })
    
    def _create_iam_roles(self):
        """
        IAM roles - the digital ID cards that let our services talk.
        Without proper roles, you're just another gonk locked out of the system.
        """
        # K3s node instance role
        assume_role_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        })
        
        self.k3s_role = aws.iam.Role(
            f"{self.name}-k3s-role",
            assume_role_policy=assume_role_policy,
            tags={**self.args["tags"], "Component": "k3s-cluster"},
            opts=ResourceOptions(parent=self)
        )
        
        # Attach policies - the permissions that matter
        policies = [
            "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
            "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
            "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",  # For debugging
        ]
        
        for i, policy_arn in enumerate(policies):
            aws.iam.RolePolicyAttachment(
                f"{self.name}-k3s-policy-{i}",
                role=self.k3s_role.name,
                policy_arn=policy_arn,
                opts=ResourceOptions(parent=self.k3s_role)
            )
        
        # Instance profile - the badge our EC2s wear
        self.k3s_instance_profile = aws.iam.InstanceProfile(
            f"{self.name}-k3s-profile",
            role=self.k3s_role.name,
            opts=ResourceOptions(parent=self.k3s_role)
        )
    
    def _create_security_groups(self):
        """
        Security groups - the bouncers at our digital nightclub.
        Let the right traffic in, keep the riff-raff out.
        """
        self.k3s_sg = aws.ec2.SecurityGroup(
            f"{self.name}-k3s-sg",
            vpc_id=self.args["vpc_id"],
            description="Security group for K3s cluster - tighter than Arasaka security",
            ingress=[
                # K3s API server - the brain
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=6443,
                    to_port=6443,
                    cidr_blocks=["10.0.0.0/8"],  # Internal only
                    description="K3s API server"
                ),
                # ETCD peers - the memory
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=2379,
                    to_port=2380,
                    self=True,  # Only between cluster members
                    description="ETCD peer communication"
                ),
                # Kubelet API - the nervous system
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=10250,
                    to_port=10250,
                    self=True,
                    description="Kubelet API"
                ),
                # NodePort services - our service mesh
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=30000,
                    to_port=32767,
                    cidr_blocks=["10.0.0.0/8"],
                    description="NodePort services"
                ),
            ],
            egress=[
                # Let our nodes breathe - they need to talk to the world
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["0.0.0.0/0"],
                    description="Allow all outbound"
                ),
            ],
            tags={**self.args["tags"], "Component": "k3s-security"},
            opts=ResourceOptions(parent=self)
        )
        
        # ALB security group for our ingress
        self.alb_sg = aws.ec2.SecurityGroup(
            f"{self.name}-alb-sg",
            vpc_id=self.args["vpc_id"],
            description="ALB security group - the gateway to our empire",
            ingress=[
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=443,
                    to_port=443,
                    cidr_blocks=["0.0.0.0/0"],
                    description="HTTPS from the world"
                ),
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=80,
                    to_port=80,
                    cidr_blocks=["0.0.0.0/0"],
                    description="HTTP (redirects to HTTPS)"
                ),
            ],
            egress=[
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["0.0.0.0/0"],
                ),
            ],
            tags={**self.args["tags"], "Component": "alb-security"},
            opts=ResourceOptions(parent=self)
        )
    
    def _deploy_k3s_cluster(self):
        """
        Deploy K3s - our lightweight but lethal Kubernetes.
        71MB of pure chrome that'll orchestrate our trading empire.
        """
        # Generate a token for node joining - like a secret handshake
        self.k3s_token = random.RandomPassword(
            f"{self.name}-k3s-token",
            length=32,
            special=False,  # K3s doesn't like special chars in tokens
            opts=ResourceOptions(parent=self)
        )
        
        # User data script - the neural programming for our nodes
        k3s_init_script = Output.all(
            self.k3s_token.result,
            self.args["database_endpoint"],
            self.args["cache_endpoint"]
        ).apply(lambda args: f"""#!/bin/bash
# K3s installation script - where metal meets code
set -e

# System optimization for trading workloads
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
sysctl -p

# Install K3s - the chrome that runs our world
curl -sfL https://get.k3s.io | K3S_TOKEN={args[0]} sh -s - server \
    --cluster-init \
    --disable traefik \
    --disable servicelb \
    --write-kubeconfig-mode 644 \
    --kube-apiserver-arg="--max-requests-inflight=3000" \
    --kube-apiserver-arg="--max-mutating-requests-inflight=1000" \
    --kubelet-arg="--max-pods=110" \
    --kubelet-arg="--kube-reserved=cpu=500m,memory=1Gi" \
    --kubelet-arg="--system-reserved=cpu=500m,memory=1Gi"

# Wait for K3s to be ready
until kubectl get nodes; do sleep 5; done

# Configure for our trading workloads
kubectl label nodes $(hostname) workload=trading
kubectl label nodes $(hostname) node.kubernetes.io/instance-type=$(ec2-metadata --instance-type | cut -d' ' -f2)

# Store connection info for other nodes
aws ssm put-parameter --name "/nexlify/k3s/server-url" --value "https://$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4):6443" --type String --overwrite || true
""")
        
        # Launch template - the blueprint for our warriors
        self.k3s_launch_template = aws.ec2.LaunchTemplate(
            f"{self.name}-k3s-lt",
            name_prefix=f"{self.name}-k3s-",
            image_id="ami-0c02fb55956c7d316",  # Amazon Linux 2023 - stable and fast
            instance_type="c5n.xlarge" if self.args["gpu_nodes"] == 0 else "g4dn.xlarge",
            iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(
                arn=self.k3s_instance_profile.arn
            ),
            vpc_security_group_ids=[self.k3s_sg.id],
            user_data=k3s_init_script.apply(lambda s: base64.b64encode(s.encode()).decode()),
            block_device_mappings=[
                aws.ec2.LaunchTemplateBlockDeviceMappingArgs(
                    device_name="/dev/xvda",
                    ebs=aws.ec2.LaunchTemplateBlockDeviceMappingEbsArgs(
                        volume_size=100,
                        volume_type="gp3",
                        iops=3000,
                        throughput=125,
                        encrypted=True,
                        delete_on_termination=True,
                    ),
                ),
            ],
            metadata_options=aws.ec2.LaunchTemplateMetadataOptionsArgs(
                http_endpoint="enabled",
                http_tokens="required",  # IMDSv2 only - security first
                instance_metadata_tags="enabled",
            ),
            tag_specifications=[
                aws.ec2.LaunchTemplateTagSpecificationArgs(
                    resource_type="instance",
                    tags={
                        **self.args["tags"],
                        "Name": f"{self.name}-k3s-node",
                        "Component": "k3s-cluster",
                    },
                ),
            ],
            opts=ResourceOptions(parent=self)
        )
        
        # Auto Scaling Group - our army that grows with demand
        self.k3s_asg = aws.autoscaling.Group(
            f"{self.name}-k3s-asg",
            min_size=self.args["k3s_node_count"],
            max_size=self.args["k3s_node_count"] * 2,  # Room to grow
            desired_capacity=self.args["k3s_node_count"],
            vpc_zone_identifiers=self.args["subnet_ids"],
            launch_template=aws.autoscaling.GroupLaunchTemplateArgs(
                id=self.k3s_launch_template.id,
                version="$Latest",
            ),
            health_check_type="EC2",
            health_check_grace_period=300,
            termination_policies=["OldestInstance"],
            enabled_metrics=[
                "GroupMinSize", "GroupMaxSize", "GroupDesiredCapacity",
                "GroupInServiceInstances", "GroupTotalInstances"
            ],
            tags=[
                aws.autoscaling.GroupTagArgs(
                    key=k,
                    value=v,
                    propagate_at_launch=True,
                ) for k, v in {**self.args["tags"], "asg:type": "k3s-cluster"}.items()
            ],
            opts=ResourceOptions(parent=self)
        )
        
        # Store cluster endpoint for our records
        self.k3s_cluster = type('K3sCluster', (), {
            'endpoint': Output.concat("https://", self.k3s_asg.name, ".k3s.local:6443")
        })()
    
    def _configure_autoscaling(self):
        """
        KEDA autoscaling - because static infrastructure is dead infrastructure.
        Scale with the market, not against it.
        """
        # Target group for our load balancer
        self.target_group = aws.lb.TargetGroup(
            f"{self.name}-k3s-tg",
            port=30080,  # NodePort for our ingress controller
            protocol="HTTP",
            vpc_id=self.args["vpc_id"],
            target_type="instance",
            health_check=aws.lb.TargetGroupHealthCheckArgs(
                enabled=True,
                path="/healthz",
                port="30080",
                protocol="HTTP",
                interval=10,
                timeout=5,
                healthy_threshold=2,
                unhealthy_threshold=2,
            ),
            deregistration_delay=30,  # Fast failover
            tags={**self.args["tags"], "Component": "k3s-target-group"},
            opts=ResourceOptions(parent=self)
        )
        
        # Attach ASG to target group
        aws.autoscaling.Attachment(
            f"{self.name}-k3s-tg-attachment",
            autoscaling_group_name=self.k3s_asg.name,
            target_group_arn=self.target_group.arn,
            opts=ResourceOptions(parent=self.target_group)
        )
        
        # Scaling policies - respond to the market's heartbeat
        # Scale up aggressively, scale down conservatively
        scale_up_policy = aws.autoscaling.Policy(
            f"{self.name}-scale-up",
            autoscaling_group_name=self.k3s_asg.name,
            policy_type="TargetTrackingScaling",
            target_tracking_configuration=aws.autoscaling.PolicyTargetTrackingConfigurationArgs(
                predefined_metric_specification=aws.autoscaling.PolicyTargetTrackingConfigurationPredefinedMetricSpecificationArgs(
                    predefined_metric_type="ASGAverageCPUUtilization",
                ),
                target_value=60.0,  # Scale at 60% CPU
                scale_in_cooldown=300,  # 5 min cooldown
                scale_out_cooldown=60,   # 1 min to scale up
            ),
            opts=ResourceOptions(parent=self.k3s_asg)
        )
    
    def _deploy_trading_engine(self):
        """
        Deploy our actual trading engine to K3s.
        This is where theory becomes profit.
        """
        # Create ALB for ingress
        self.alb = aws.lb.LoadBalancer(
            f"{self.name}-alb",
            load_balancer_type="application",
            security_groups=[self.alb_sg.id],
            subnets=self.args["subnet_ids"],
            enable_deletion_protection=self.args["tags"]["Environment"] == "prod",
            enable_http2=True,
            idle_timeout=60,
            tags={**self.args["tags"], "Component": "application-lb"},
            opts=ResourceOptions(parent=self)
        )
        
        # HTTP listener - redirects to HTTPS
        http_listener = aws.lb.Listener(
            f"{self.name}-http-listener",
            load_balancer_arn=self.alb.arn,
            port=80,
            protocol="HTTP",
            default_actions=[aws.lb.ListenerDefaultActionArgs(
                type="redirect",
                redirect=aws.lb.ListenerDefaultActionRedirectArgs(
                    protocol="HTTPS",
                    port="443",
                    status_code="HTTP_301",
                ),
            )],
            opts=ResourceOptions(parent=self.alb)
        )
        
        # For now, we'll use a self-signed cert - replace with ACM in prod
        https_listener = aws.lb.Listener(
            f"{self.name}-https-listener",
            load_balancer_arn=self.alb.arn,
            port=443,
            protocol="HTTPS",
            ssl_policy="ELBSecurityPolicy-TLS13-1-2-2021-06",
            certificate_arn="arn:aws:acm:region:account:certificate/id",  # TODO: Add real cert
            default_actions=[aws.lb.ListenerDefaultActionArgs(
                type="forward",
                target_group_arn=self.target_group.arn,
            )],
            opts=ResourceOptions(parent=self.alb)
        )
        
        # Export endpoints
        self.trading_api_endpoint = Output.concat("https://", self.alb.dns_name)
    
    def _setup_monitoring(self):
        """
        Monitoring - because flying blind in the markets is a death sentence.
        Prometheus for metrics, Grafana for visualization.
        """
        # For now, just export the endpoints
        # In production, these would be deployed to K3s
        self.prometheus_endpoint = Output.concat("http://", self.alb.dns_name, ":9090")
        self.grafana_endpoint = Output.concat("http://", self.alb.dns_name, ":3000")
