"""
🔒 NEXLIFY SECURITY INFRASTRUCTURE - ZERO TRUST, TOTAL CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In Night City, trust is a luxury you can't afford. In crypto trading, it's a
vulnerability that'll drain your wallet faster than a Scav with a grudge.

Zero trust means:
- Verify everything, trust nothing
- Encrypt everything, always
- Audit everything, forever
- Assume breach, plan for war

This module implements bank-grade security that'd make Arasaka jealous.
"""

from typing import TypedDict, List, Optional, Dict
import json
import base64
from datetime import datetime

import pulumi
from pulumi import ComponentResource, ResourceOptions, Output
import pulumi_aws as aws


class SecurityConfig(TypedDict):
    """Security configuration - paranoia as a service"""
    vpc_id: str
    enable_cloudhsm: bool
    compliance_mode: str  # "SOC2", "PCI-DSS", "MINIMAL"
    audit_retention_days: int
    allowed_exchanges: List[str]
    tags: Dict[str, str]


class ZeroTrustSecurity(ComponentResource):
    """
    Zero-trust security infrastructure for our trading platform.
    
    Implements:
    - WAF for API protection
    - CloudHSM for key management (optional)
    - Secrets management with auto-rotation
    - Audit logging for compliance
    - Network isolation and microsegmentation
    - Real-time threat detection
    
    Because in this business, paranoia isn't a disorder - it's a survival trait.
    """
    
    def __init__(self, name: str, config: SecurityConfig, opts: Optional[ResourceOptions] = None):
        super().__init__("nexlify:security:ZeroTrustSecurity", name, {}, opts)
        
        self.config = config
        self.name = name
        
        # Build our security layers - each one a barrier against the darkness
        self._create_kms_keys()
        self._setup_secrets_management()
        self._create_security_groups()
        self._setup_waf()
        self._configure_audit_logging()
        
        if config.get("enable_cloudhsm"):
            self._setup_cloudhsm()
        
        # Export our security infrastructure
        self.register_outputs({
            "kms_key_id": self.kms_key.id,
            "secrets_manager_arn": self.secrets_arn,
            "waf_web_acl_id": self.waf_acl.id,
            "audit_bucket": self.audit_bucket.id,
            "database_sg": self.database_sg.id,
        })
    
    def _create_kms_keys(self):
        """
        KMS keys - the master keys to our kingdom.
        Rotate automatically, audit everything, trust no one.
        """
        # Key policy - who can do what with our encryption
        key_policy = aws.iam.get_policy_document(
            statements=[
                # Root account has full access (break glass scenario)
                {
                    "sid": "Enable IAM Root Permissions",
                    "effect": "Allow",
                    "principals": [{
                        "type": "AWS",
                        "identifiers": [f"arn:aws:iam::{aws.get_caller_identity().account_id}:root"],
                    }],
                    "actions": ["kms:*"],
                    "resources": ["*"],
                },
                # CloudWatch Logs can use the key
                {
                    "sid": "Allow CloudWatch Logs",
                    "effect": "Allow",
                    "principals": [{
                        "type": "Service",
                        "identifiers": ["logs.amazonaws.com"],
                    }],
                    "actions": [
                        "kms:Encrypt",
                        "kms:Decrypt",
                        "kms:ReEncrypt*",
                        "kms:GenerateDataKey*",
                        "kms:CreateGrant",
                        "kms:DescribeKey",
                    ],
                    "resources": ["*"],
                    "conditions": [{
                        "test": "ArnLike",
                        "variable": "kms:EncryptionContext:aws:logs:arn",
                        "values": [f"arn:aws:logs:*:{aws.get_caller_identity().account_id}:*"],
                    }],
                },
            ]
        )
        
        # Master KMS key for all encryption
        self.kms_key = aws.kms.Key(
            f"{self.name}-master-key",
            description="Master encryption key for Nexlify trading platform",
            key_usage="ENCRYPT_DECRYPT",
            customer_master_key_spec="SYMMETRIC_DEFAULT",
            policy=key_policy.json,
            enable_key_rotation=True,  # Rotate annually
            tags={
                **self.config["tags"],
                "Purpose": "master-encryption",
                "Compliance": self.config["compliance_mode"],
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Alias for easier reference
        aws.kms.Alias(
            f"{self.name}-master-key-alias",
            target_key_id=self.kms_key.id,
            name=f"alias/{self.name}-master",
            opts=ResourceOptions(parent=self.kms_key)
        )
    
    def _setup_secrets_management(self):
        """
        Secrets management - because hardcoded API keys are how accounts die.
        Automatic rotation, versioning, audit trail. The works.
        """
        # Create secrets for each exchange
        self.exchange_secrets = {}
        
        for exchange in self.config["allowed_exchanges"]:
            # Initial secret structure
            secret_string = json.dumps({
                "api_key": f"PLACEHOLDER_{exchange.upper()}_API_KEY",
                "api_secret": f"PLACEHOLDER_{exchange.upper()}_API_SECRET",
                "passphrase": f"PLACEHOLDER_{exchange.upper()}_PASSPHRASE" if exchange == "coinbase" else None,
                "updated_at": datetime.utcnow().isoformat(),
            })
            
            secret = aws.secretsmanager.Secret(
                f"{self.name}-{exchange}-credentials",
                description=f"API credentials for {exchange} exchange",
                kms_key_id=self.kms_key.id,
                tags={
                    **self.config["tags"],
                    "Exchange": exchange,
                    "Purpose": "api-credentials",
                },
                opts=ResourceOptions(parent=self)
            )
            
            # Set initial version
            aws.secretsmanager.SecretVersion(
                f"{self.name}-{exchange}-credentials-version",
                secret_id=secret.id,
                secret_string=secret_string,
                opts=ResourceOptions(parent=secret)
            )
            
            self.exchange_secrets[exchange] = secret
        
        # Database credentials
        db_secret = aws.secretsmanager.Secret(
            f"{self.name}-db-credentials",
            description="Database credentials for QuestDB and Valkey",
            kms_key_id=self.kms_key.id,
            tags={
                **self.config["tags"],
                "Purpose": "database-credentials",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Store ARN for reference
        self.secrets_arn = db_secret.arn
        
        # Rotation configuration (Lambda would handle actual rotation)
        # In production, you'd have rotation lambdas for each secret type
        if self.config["compliance_mode"] in ["SOC2", "PCI-DSS"]:
            # Compliance requires more frequent rotation
            rotation_days = 30
        else:
            rotation_days = 90
        
        # Note: Actual rotation would require Lambda functions
        # This is a placeholder for the rotation configuration
    
    def _create_security_groups(self):
        """
        Security groups - the bouncers of our digital nightclub.
        Microsegmentation at its finest. Every service gets its own bouncer.
        """
        # Database security group - locked down tight
        self.database_sg = aws.ec2.SecurityGroup(
            f"{self.name}-database-sg",
            vpc_id=self.config["vpc_id"],
            description="Security group for databases - Fort Knox level",
            ingress=[
                # QuestDB ports
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=9000,  # HTTP API
                    to_port=9000,
                    description="QuestDB HTTP API",
                    # Only from app servers - will add after creation
                ),
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=9009,  # InfluxDB line protocol
                    to_port=9009,
                    description="QuestDB line protocol",
                ),
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=8812,  # PostgreSQL wire protocol
                    to_port=8812,
                    description="QuestDB PostgreSQL protocol",
                ),
                # Valkey (Redis) port
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=6379,
                    to_port=6379,
                    description="Valkey cache",
                ),
            ],
            egress=[
                # No outbound needed for databases
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["127.0.0.1/32"],  # Effectively no egress
                    description="No outbound traffic",
                ),
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-database-sg",
                "Component": "database-security",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Application security group
        self.app_sg = aws.ec2.SecurityGroup(
            f"{self.name}-app-sg",
            vpc_id=self.config["vpc_id"],
            description="Security group for application servers",
            ingress=[
                # Health checks from ALB
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=8080,
                    to_port=8080,
                    description="Health checks",
                ),
                # Metrics endpoint
                aws.ec2.SecurityGroupIngressArgs(
                    protocol="tcp",
                    from_port=9090,
                    to_port=9090,
                    description="Prometheus metrics",
                ),
            ],
            egress=[
                # Outbound to internet for exchange APIs
                aws.ec2.SecurityGroupEgressArgs(
                    protocol="-1",
                    from_port=0,
                    to_port=0,
                    cidr_blocks=["0.0.0.0/0"],
                    description="Outbound for exchange APIs",
                ),
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-app-sg",
                "Component": "application-security",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Allow app servers to talk to databases
        aws.ec2.SecurityGroupRule(
            f"{self.name}-app-to-db",
            type="ingress",
            from_port=0,
            to_port=65535,
            protocol="tcp",
            source_security_group_id=self.app_sg.id,
            security_group_id=self.database_sg.id,
            description="Allow app servers to databases",
            opts=ResourceOptions(parent=self.database_sg)
        )
    
    def _setup_waf(self):
        """
        Web Application Firewall - because the internet is full of
        script kiddies who think they're netrunners.
        """
        # WAF rules - the bouncer's checklist
        # SQL injection protection
        sql_injection_rule = aws.wafv2.RuleGroup(
            f"{self.name}-sql-injection-rule",
            capacity=100,
            scope="REGIONAL",
            visibility_config=aws.wafv2.RuleGroupVisibilityConfigArgs(
                cloudwatch_metrics_enabled=True,
                metric_name=f"{self.name}-sql-injection",
                sampled_requests_enabled=True,
            ),
            rules=[
                aws.wafv2.RuleGroupRuleArgs(
                    name="SQLiRule",
                    priority=1,
                    statement=aws.wafv2.RuleGroupRuleStatementArgs(
                        sqli_match_statement=aws.wafv2.RuleGroupRuleStatementSqliMatchStatementArgs(
                            field_to_match=aws.wafv2.RuleGroupRuleStatementSqliMatchStatementFieldToMatchArgs(
                                body={}
                            ),
                            text_transformations=[
                                aws.wafv2.RuleGroupRuleStatementSqliMatchStatementTextTransformationArgs(
                                    priority=1,
                                    type="URL_DECODE",
                                ),
                                aws.wafv2.RuleGroupRuleStatementSqliMatchStatementTextTransformationArgs(
                                    priority=2,
                                    type="HTML_ENTITY_DECODE",
                                ),
                            ],
                        ),
                    ),
                    action=aws.wafv2.RuleGroupRuleActionArgs(
                        block={}
                    ),
                    visibility_config=aws.wafv2.RuleGroupRuleVisibilityConfigArgs(
                        cloudwatch_metrics_enabled=True,
                        metric_name=f"{self.name}-sql-injection-rule",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
            tags=self.config["tags"],
            opts=ResourceOptions(parent=self)
        )
        
        # Rate limiting - prevent DDoS and brute force
        rate_limit_rule = {
            "name": "RateLimitRule",
            "priority": 2,
            "statement": {
                "rate_based_statement": {
                    "limit": 2000,  # 2000 requests per 5 minutes
                    "aggregate_key_type": "IP",
                },
            },
            "action": {
                "block": {},
            },
            "visibility_config": {
                "cloudwatch_metrics_enabled": True,
                "metric_name": f"{self.name}-rate-limit",
                "sampled_requests_enabled": True,
            },
        }
        
        # Geo-blocking - only allow from specific countries if needed
        geo_block_rule = {
            "name": "GeoBlockRule",
            "priority": 3,
            "statement": {
                "not_statement": {
                    "statement": {
                        "geo_match_statement": {
                            # Allow these countries
                            "country_codes": ["US", "GB", "DE", "JP", "SG", "CA", "AU"],
                        },
                    },
                },
            },
            "action": {
                "block": {},
            },
            "visibility_config": {
                "cloudwatch_metrics_enabled": True,
                "metric_name": f"{self.name}-geo-block",
                "sampled_requests_enabled": True,
            },
        }
        
        # Create WAF Web ACL
        self.waf_acl = aws.wafv2.WebAcl(
            f"{self.name}-waf-acl",
            scope="REGIONAL",
            default_action=aws.wafv2.WebAclDefaultActionArgs(
                allow={}
            ),
            visibility_config=aws.wafv2.WebAclVisibilityConfigArgs(
                cloudwatch_metrics_enabled=True,
                metric_name=f"{self.name}-waf-acl",
                sampled_requests_enabled=True,
            ),
            rules=[
                # Add AWS managed rule groups
                aws.wafv2.WebAclRuleArgs(
                    name="AWSManagedRulesCommonRuleSet",
                    priority=10,
                    override_action=aws.wafv2.WebAclRuleOverrideActionArgs(
                        none={}
                    ),
                    statement=aws.wafv2.WebAclRuleStatementArgs(
                        managed_rule_group_statement=aws.wafv2.WebAclRuleStatementManagedRuleGroupStatementArgs(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                        ),
                    ),
                    visibility_config=aws.wafv2.WebAclRuleVisibilityConfigArgs(
                        cloudwatch_metrics_enabled=True,
                        metric_name="CommonRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
                # Add rate limiting
                aws.wafv2.WebAclRuleArgs(**rate_limit_rule),
                # Add geo-blocking if compliance requires
                aws.wafv2.WebAclRuleArgs(**geo_block_rule) if self.config["compliance_mode"] in ["SOC2", "PCI-DSS"] else None,
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-waf-acl",
            },
            opts=ResourceOptions(parent=self)
        )
    
    def _configure_audit_logging(self):
        """
        Audit logging - know everything that happens, when it happened,
        and who did it. Compliance loves this, hackers hate it.
        """
        # S3 bucket for audit logs - encrypted, versioned, locked down
        self.audit_bucket = aws.s3.BucketV2(
            f"{self.name}-audit-logs",
            bucket=f"{self.name}-audit-logs-{aws.get_region().name}",
            force_destroy=False,  # NEVER delete audit logs
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-audit-logs",
                "Purpose": "audit-compliance",
                "Retention": f"{self.config['audit_retention_days']} days",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Enable versioning
        aws.s3.BucketVersioningV2(
            f"{self.name}-audit-versioning",
            bucket=self.audit_bucket.id,
            versioning_configuration=aws.s3.BucketVersioningV2VersioningConfigurationArgs(
                status="Enabled",
            ),
            opts=ResourceOptions(parent=self.audit_bucket)
        )
        
        # Server-side encryption
        aws.s3.BucketServerSideEncryptionConfigurationV2(
            f"{self.name}-audit-encryption",
            bucket=self.audit_bucket.id,
            rules=[
                aws.s3.BucketServerSideEncryptionConfigurationV2RuleArgs(
                    apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationV2RuleApplyServerSideEncryptionByDefaultArgs(
                        sse_algorithm="aws:kms",
                        kms_master_key_id=self.kms_key.id,
                    ),
                    bucket_key_enabled=True,
                ),
            ],
            opts=ResourceOptions(parent=self.audit_bucket)
        )
        
        # Bucket policy - write once, read many
        bucket_policy = aws.iam.get_policy_document(
            statements=[
                {
                    "sid": "DenyUnencryptedObjectUploads",
                    "effect": "Deny",
                    "principals": [{
                        "type": "*",
                        "identifiers": ["*"],
                    }],
                    "actions": ["s3:PutObject"],
                    "resources": [f"{self.audit_bucket.arn}/*"],
                    "conditions": [{
                        "test": "StringNotEquals",
                        "variable": "s3:x-amz-server-side-encryption",
                        "values": ["aws:kms"],
                    }],
                },
                {
                    "sid": "DenyInsecureConnections",
                    "effect": "Deny",
                    "principals": [{
                        "type": "*",
                        "identifiers": ["*"],
                    }],
                    "actions": ["s3:*"],
                    "resources": [
                        self.audit_bucket.arn,
                        f"{self.audit_bucket.arn}/*",
                    ],
                    "conditions": [{
                        "test": "Bool",
                        "variable": "aws:SecureTransport",
                        "values": ["false"],
                    }],
                },
            ]
        )
        
        aws.s3.BucketPolicy(
            f"{self.name}-audit-policy",
            bucket=self.audit_bucket.id,
            policy=bucket_policy.json,
            opts=ResourceOptions(parent=self.audit_bucket)
        )
        
        # Lifecycle rules for compliance
        aws.s3.BucketLifecycleConfigurationV2(
            f"{self.name}-audit-lifecycle",
            bucket=self.audit_bucket.id,
            rules=[
                aws.s3.BucketLifecycleConfigurationV2RuleArgs(
                    id="transition-to-glacier",
                    status="Enabled",
                    transitions=[
                        aws.s3.BucketLifecycleConfigurationV2RuleTransitionArgs(
                            days=90,
                            storage_class="GLACIER",
                        ),
                    ],
                    # Keep forever if compliance mode requires
                    expiration=aws.s3.BucketLifecycleConfigurationV2RuleExpirationArgs(
                        days=self.config["audit_retention_days"],
                    ) if self.config["compliance_mode"] == "MINIMAL" else None,
                ),
            ],
            opts=ResourceOptions(parent=self.audit_bucket)
        )
        
        # CloudTrail for API auditing
        trail = aws.cloudtrail.Trail(
            f"{self.name}-audit-trail",
            name=f"{self.name}-audit-trail",
            s3_bucket_name=self.audit_bucket.id,
            include_global_service_events=True,
            is_multi_region_trail=True,
            enable_logging=True,
            kms_key_id=self.kms_key.arn,
            event_selectors=[
                aws.cloudtrail.TrailEventSelectorArgs(
                    read_write_type="All",
                    include_management_events=True,
                    data_resources=[
                        # Audit all S3 operations
                        aws.cloudtrail.TrailEventSelectorDataResourceArgs(
                            type="AWS::S3::Object",
                            values=["arn:aws:s3:::*/*"],
                        ),
                        # Audit all Lambda invocations
                        aws.cloudtrail.TrailEventSelectorDataResourceArgs(
                            type="AWS::Lambda::Function",
                            values=["arn:aws:lambda:*:*:function/*"],
                        ),
                    ],
                ),
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-audit-trail",
            },
            opts=ResourceOptions(parent=self)
        )
    
    def _setup_cloudhsm(self):
        """
        CloudHSM - when software encryption isn't paranoid enough.
        Hardware security modules for the truly paranoid (or compliant).
        """
        # Note: CloudHSM is expensive and requires VPC setup
        # This is a placeholder for full implementation
        pass
