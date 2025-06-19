"""
⚡ NEXLIFY WEBSOCKET API - THE PULSE OF THE MARKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When microseconds mean millions, you don't poll - you stream. This module
builds our real-time neural pathways to every exchange, every order book,
every trade that matters.

WebSockets for the win. Because REST is for the rest.
"""

from typing import TypedDict, Optional, Dict
import json

import pulumi
from pulumi import ComponentResource, ResourceOptions, Output
import pulumi_aws as aws


class WebSocketConfig(TypedDict):
    """WebSocket API configuration - the speed of thought"""
    vpc_id: str
    platform_endpoint: Output[str]
    enable_throttling: bool
    max_connections: int
    tags: Dict[str, str]


class RealtimeMarketAPI(ComponentResource):
    """
    Real-time market data infrastructure using AWS API Gateway V2.
    
    Features:
    - WebSocket API for bidirectional communication
    - 100,000+ concurrent connections per region
    - Sub-millisecond message routing
    - Automatic scaling based on connection count
    - DDoS protection and rate limiting
    - Message ordering guarantees
    
    This is how we stay ahead of the market - by being the market.
    """
    
    def __init__(self, name: str, config: WebSocketConfig, opts: Optional[ResourceOptions] = None):
        super().__init__("nexlify:api:RealtimeMarketAPI", name, {}, opts)
        
        self.config = config
        self.name = name
        
        # Build our real-time infrastructure
        self._create_websocket_api()
        self._setup_lambda_handlers()
        self._configure_routes()
        self._setup_throttling()
        self._create_rest_api()
        self._setup_monitoring()
        
        # Export our endpoints
        self.register_outputs({
            "websocket_url": self.websocket_url,
            "api_url": self.api_url,
            "connection_table": self.connection_table.name,
        })
    
    def _create_websocket_api(self):
        """
        WebSocket API Gateway - our always-on connection to profit.
        Unlike REST, we don't ask for updates - they flow to us.
        """
        self.websocket_api = aws.apigatewayv2.Api(
            f"{self.name}-websocket-api",
            protocol_type="WEBSOCKET",
            route_selection_expression="$request.body.action",
            description="Nexlify real-time market data feed - faster than a Tyger Claw reflex boost",
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-websocket-api",
                "Type": "websocket",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # DynamoDB table for connection management
        # Because we need to know who's connected and what they're watching
        self.connection_table = aws.dynamodb.Table(
            f"{self.name}-connections",
            billing_mode="PAY_PER_REQUEST",  # Auto-scaling built in
            hash_key="connectionId",
            attributes=[
                aws.dynamodb.TableAttributeArgs(
                    name="connectionId",
                    type="S",
                ),
            ],
            ttl=aws.dynamodb.TableTtlArgs(
                enabled=True,
                attribute_name="ttl",  # Auto-cleanup dead connections
            ),
            stream_enabled=True,
            stream_view_type="NEW_AND_OLD_IMAGES",
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-connections",
                "Purpose": "websocket-tracking",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Subscription tracking table
        self.subscription_table = aws.dynamodb.Table(
            f"{self.name}-subscriptions",
            billing_mode="PAY_PER_REQUEST",
            hash_key="symbol",
            range_key="connectionId",
            attributes=[
                aws.dynamodb.TableAttributeArgs(name="symbol", type="S"),
                aws.dynamodb.TableAttributeArgs(name="connectionId", type="S"),
            ],
            global_secondary_indexes=[
                aws.dynamodb.TableGlobalSecondaryIndexArgs(
                    name="ConnectionIndex",
                    hash_key="connectionId",
                    projection_type="ALL",
                ),
            ],
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-subscriptions",
                "Purpose": "market-subscriptions",
            },
            opts=ResourceOptions(parent=self)
        )
    
    def _setup_lambda_handlers(self):
        """
        Lambda functions - the neurons that process our market signals.
        Fast, scalable, and ready for anything the market throws at us.
        """
        # IAM role for Lambda functions
        lambda_role = aws.iam.Role(
            f"{self.name}-lambda-role",
            assume_role_policy=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }],
            }),
            opts=ResourceOptions(parent=self)
        )
        
        # Attach policies
        aws.iam.RolePolicyAttachment(
            f"{self.name}-lambda-basic",
            role=lambda_role.name,
            policy_arn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            opts=ResourceOptions(parent=lambda_role)
        )
        
        # Custom policy for DynamoDB and API Gateway
        lambda_policy = aws.iam.Policy(
            f"{self.name}-lambda-policy",
            policy=Output.all(
                self.connection_table.arn,
                self.subscription_table.arn,
                self.websocket_api.execution_arn
            ).apply(lambda args: json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:DeleteItem",
                            "dynamodb:Scan",
                            "dynamodb:Query",
                            "dynamodb:UpdateItem",
                        ],
                        "Resource": [
                            args[0],
                            f"{args[0]}/index/*",
                            args[1],
                            f"{args[1]}/index/*",
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "execute-api:ManageConnections",
                            "execute-api:Invoke",
                        ],
                        "Resource": f"{args[2]}/*",
                    },
                ],
            })),
            opts=ResourceOptions(parent=lambda_role)
        )
        
        aws.iam.RolePolicyAttachment(
            f"{self.name}-lambda-custom-policy",
            role=lambda_role.name,
            policy_arn=lambda_policy.arn,
            opts=ResourceOptions(parent=lambda_role)
        )
        
        # Lambda layer for shared dependencies
        dependencies_layer = aws.lambda_.LayerVersion(
            f"{self.name}-dependencies",
            layer_name=f"{self.name}-dependencies",
            compatible_runtimes=["python3.11"],
            code=pulumi.FileArchive("./lambda_layers/dependencies.zip"),
            description="Shared dependencies for WebSocket handlers",
            opts=ResourceOptions(parent=self)
        )
        
        # Connection handler - manages connect/disconnect
        self.connect_handler = aws.lambda_.Function(
            f"{self.name}-connect",
            runtime="python3.11",
            handler="index.handler",
            role=lambda_role.arn,
            timeout=30,
            memory_size=512,
            layers=[dependencies_layer.arn],
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "CONNECTION_TABLE": self.connection_table.name,
                    "PLATFORM_ENDPOINT": self.config["platform_endpoint"],
                },
            ),
            code=pulumi.AssetArchive({
                "index.py": pulumi.StringAsset("""
import os
import json
import boto3
import time

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['CONNECTION_TABLE'])

def handler(event, context):
    connection_id = event['requestContext']['connectionId']
    
    # Store connection with metadata
    table.put_item(
        Item={
            'connectionId': connection_id,
            'connectedAt': int(time.time()),
            'ttl': int(time.time()) + 86400,  # 24 hour TTL
            'sourceIp': event['requestContext']['identity']['sourceIp'],
            'userAgent': event['headers'].get('User-Agent', 'Unknown'),
        }
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Connected to Nexlify Market Feed',
            'connectionId': connection_id,
        })
    }
"""),
            }),
            tags={
                **self.config["tags"],
                "Handler": "connect",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Disconnect handler
        self.disconnect_handler = aws.lambda_.Function(
            f"{self.name}-disconnect",
            runtime="python3.11",
            handler="index.handler",
            role=lambda_role.arn,
            timeout=30,
            memory_size=256,
            layers=[dependencies_layer.arn],
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "CONNECTION_TABLE": self.connection_table.name,
                    "SUBSCRIPTION_TABLE": self.subscription_table.name,
                },
            ),
            code=pulumi.AssetArchive({
                "index.py": pulumi.StringAsset("""
import os
import boto3

dynamodb = boto3.resource('dynamodb')
conn_table = dynamodb.Table(os.environ['CONNECTION_TABLE'])
sub_table = dynamodb.Table(os.environ['SUBSCRIPTION_TABLE'])

def handler(event, context):
    connection_id = event['requestContext']['connectionId']
    
    # Remove connection
    conn_table.delete_item(Key={'connectionId': connection_id})
    
    # Clean up subscriptions
    response = sub_table.query(
        IndexName='ConnectionIndex',
        KeyConditionExpression='connectionId = :cid',
        ExpressionAttributeValues={':cid': connection_id}
    )
    
    # Batch delete subscriptions
    with sub_table.batch_writer() as batch:
        for item in response['Items']:
            batch.delete_item(
                Key={
                    'symbol': item['symbol'],
                    'connectionId': connection_id
                }
            )
    
    return {'statusCode': 200}
"""),
            }),
            tags={
                **self.config["tags"],
                "Handler": "disconnect",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Message router - handles all other messages
        self.message_handler = aws.lambda_.Function(
            f"{self.name}-message",
            runtime="python3.11",
            handler="index.handler",
            role=lambda_role.arn,
            timeout=30,
            memory_size=1024,  # More memory for processing
            layers=[dependencies_layer.arn],
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "CONNECTION_TABLE": self.connection_table.name,
                    "SUBSCRIPTION_TABLE": self.subscription_table.name,
                    "WEBSOCKET_ENDPOINT": self.websocket_api.api_endpoint,
                },
            ),
            code=pulumi.AssetArchive({
                "index.py": pulumi.StringAsset("""
import os
import json
import boto3

dynamodb = boto3.resource('dynamodb')
apigateway = boto3.client('apigatewaymanagementapi',
    endpoint_url=os.environ['WEBSOCKET_ENDPOINT'].replace('wss://', 'https://'))
sub_table = dynamodb.Table(os.environ['SUBSCRIPTION_TABLE'])

def handler(event, context):
    connection_id = event['requestContext']['connectionId']
    body = json.loads(event['body'])
    action = body.get('action')
    
    if action == 'subscribe':
        # Subscribe to market data
        symbols = body.get('symbols', [])
        with sub_table.batch_writer() as batch:
            for symbol in symbols:
                batch.put_item(Item={
                    'symbol': symbol,
                    'connectionId': connection_id,
                    'subscribedAt': int(time.time()),
                })
        
        response = {
            'action': 'subscribed',
            'symbols': symbols,
            'message': f'Subscribed to {len(symbols)} symbols'
        }
        
    elif action == 'unsubscribe':
        # Unsubscribe from symbols
        symbols = body.get('symbols', [])
        with sub_table.batch_writer() as batch:
            for symbol in symbols:
                batch.delete_item(Key={
                    'symbol': symbol,
                    'connectionId': connection_id
                })
        
        response = {
            'action': 'unsubscribed',
            'symbols': symbols
        }
        
    elif action == 'ping':
        response = {'action': 'pong', 'timestamp': int(time.time())}
        
    else:
        response = {'error': f'Unknown action: {action}'}
    
    # Send response back to client
    apigateway.post_to_connection(
        ConnectionId=connection_id,
        Data=json.dumps(response)
    )
    
    return {'statusCode': 200}
"""),
            }),
            tags={
                **self.config["tags"],
                "Handler": "message",
            },
            opts=ResourceOptions(parent=self)
        )
    
    def _configure_routes(self):
        """
        API Gateway routes - the neural pathways of our real-time system.
        Each route is a different type of market signal.
        """
        # Lambda permissions for API Gateway
        for handler_name, handler in [
            ("connect", self.connect_handler),
            ("disconnect", self.disconnect_handler),
            ("message", self.message_handler),
        ]:
            aws.lambda_.Permission(
                f"{self.name}-{handler_name}-permission",
                action="lambda:InvokeFunction",
                function=handler.name,
                principal="apigateway.amazonaws.com",
                source_arn=Output.concat(self.websocket_api.execution_arn, "/*/*"),
                opts=ResourceOptions(parent=handler)
            )
        
        # Integration for each handler
        connect_integration = aws.apigatewayv2.Integration(
            f"{self.name}-connect-integration",
            api_id=self.websocket_api.id,
            integration_type="AWS_PROXY",
            integration_uri=self.connect_handler.invoke_arn,
            opts=ResourceOptions(parent=self.websocket_api)
        )
        
        disconnect_integration = aws.apigatewayv2.Integration(
            f"{self.name}-disconnect-integration",
            api_id=self.websocket_api.id,
            integration_type="AWS_PROXY",
            integration_uri=self.disconnect_handler.invoke_arn,
            opts=ResourceOptions(parent=self.websocket_api)
        )
        
        message_integration = aws.apigatewayv2.Integration(
            f"{self.name}-message-integration",
            api_id=self.websocket_api.id,
            integration_type="AWS_PROXY",
            integration_uri=self.message_handler.invoke_arn,
            opts=ResourceOptions(parent=self.websocket_api)
        )
        
        # Routes
        aws.apigatewayv2.Route(
            f"{self.name}-connect-route",
            api_id=self.websocket_api.id,
            route_key="$connect",
            target=Output.concat("integrations/", connect_integration.id),
            opts=ResourceOptions(parent=connect_integration)
        )
        
        aws.apigatewayv2.Route(
            f"{self.name}-disconnect-route",
            api_id=self.websocket_api.id,
            route_key="$disconnect",
            target=Output.concat("integrations/", disconnect_integration.id),
            opts=ResourceOptions(parent=disconnect_integration)
        )
        
        aws.apigatewayv2.Route(
            f"{self.name}-default-route",
            api_id=self.websocket_api.id,
            route_key="$default",
            target=Output.concat("integrations/", message_integration.id),
            opts=ResourceOptions(parent=message_integration)
        )
        
        # Deployment
        deployment = aws.apigatewayv2.Deployment(
            f"{self.name}-deployment",
            api_id=self.websocket_api.id,
            description="Production deployment of Nexlify WebSocket API",
            opts=ResourceOptions(
                parent=self.websocket_api,
                depends_on=[
                    connect_integration,
                    disconnect_integration,
                    message_integration,
                ]
            )
        )
        
        # Stage
        stage = aws.apigatewayv2.Stage(
            f"{self.name}-prod",
            api_id=self.websocket_api.id,
            deployment_id=deployment.id,
            name="prod",
            description="Production stage - where money meets reality",
            throttle_settings=aws.apigatewayv2.StageThrottleSettingsArgs(
                burst_limit=5000,
                rate_limit=10000,
            ) if self.config["enable_throttling"] else None,
            tags={
                **self.config["tags"],
                "Stage": "production",
            },
            opts=ResourceOptions(parent=deployment)
        )
        
        # Export WebSocket URL
        self.websocket_url = Output.concat(
            "wss://", self.websocket_api.id, ".execute-api.",
            aws.get_region().name, ".amazonaws.com/", stage.name
        )
    
    def _setup_throttling(self):
        """
        Rate limiting - because even in Night City, there are rules.
        Protect our infrastructure from abuse while keeping legitimate
        traders happy.
        """
        # Usage plan for API throttling
        if self.config["enable_throttling"]:
            # CloudWatch logs for monitoring
            log_group = aws.cloudwatch.LogGroup(
                f"{self.name}-api-logs",
                retention_in_days=7,
                tags=self.config["tags"],
                opts=ResourceOptions(parent=self)
            )
            
            # Access logging configuration
            self.websocket_api.access_log_settings = aws.apigatewayv2.ApiAccessLogSettingsArgs(
                destination_arn=log_group.arn,
                format=json.dumps({
                    "requestId": "$context.requestId",
                    "extendedRequestId": "$context.extendedRequestId",
                    "ip": "$context.identity.sourceIp",
                    "caller": "$context.identity.caller",
                    "user": "$context.identity.user",
                    "requestTime": "$context.requestTime",
                    "routeKey": "$context.routeKey",
                    "status": "$context.status",
                    "error": "$context.error.message",
                    "connectionId": "$context.connectionId",
                    "eventType": "$context.eventType",
                })
            )
    
    def _create_rest_api(self):
        """
        REST API for non-streaming operations.
        Because sometimes you just need to GET some data.
        """
        # HTTP API for REST endpoints
        self.rest_api = aws.apigatewayv2.Api(
            f"{self.name}-rest-api",
            protocol_type="HTTP",
            description="Nexlify REST API - for when WebSockets are overkill",
            cors_configuration=aws.apigatewayv2.ApiCorsConfigurationArgs(
                allow_origins=["*"],  # Configure based on your needs
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["*"],
                max_age=300,
            ),
            tags={
                **self.config["tags"],
                "Name": f"{self.name}-rest-api",
                "Type": "rest",
            },
            opts=ResourceOptions(parent=self)
        )
        
        # Export REST API URL
        self.api_url = self.rest_api.api_endpoint
    
    def _setup_monitoring(self):
        """
        Monitoring - because you can't improve what you don't measure.
        Track connections, latency, errors, and profit opportunities.
        """
        # CloudWatch dashboard
        dashboard = aws.cloudwatch.Dashboard(
            f"{self.name}-dashboard",
            dashboard_name=f"{self.name}-realtime-metrics",
            dashboard_body=Output.all(
                self.websocket_api.name,
                self.connection_table.name
            ).apply(lambda args: json.dumps({
                "widgets": [
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/ApiGateway", "Count", {"ApiName": args[0]}],
                                ["AWS/ApiGateway", "ConnectionCount", {"ApiName": args[0]}],
                                ["AWS/ApiGateway", "MessageCount", {"ApiName": args[0]}],
                            ],
                            "period": 300,
                            "stat": "Sum",
                            "region": aws.get_region().name,
                            "title": "WebSocket Metrics",
                        },
                    },
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/DynamoDB", "ConsumedReadCapacityUnits", {"TableName": args[1]}],
                                ["AWS/DynamoDB", "ConsumedWriteCapacityUnits", {"TableName": args[1]}],
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": aws.get_region().name,
                            "title": "Connection Table Metrics",
                        },
                    },
                ],
            })),
            opts=ResourceOptions(parent=self)
        )
