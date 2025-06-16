# nexlify/testing/cyber_test_suite.py
"""
Nexlify Cyber Test Suite - Battle-testing your code in the digital dojo
Because in Night City, untested code is dead code
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import random
import string
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Test fixtures and utilities
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis
from fakeredis import FakeRedis
import factory
from faker import Faker

# Our modules - the chrome we're testing
from config.config_manager import NexlifyConfig, get_config
from security.auth_manager import NexlifyAuthManager, User
from database.models import Base, Market, Symbol, Candle, Portfolio, Position, Order
from database.migration_manager import DataEvolutionEngine
from monitoring.sentinel import NexlifySentinel, Alert, AlertSeverity, MetricType

# Initialize faker for realistic test data
fake = Faker(['en_US', 'ja_JP'])  # Multi-cultural like Night City

@dataclass
class TestContext:
    """Test context - your loadout for the testing battlefield"""
    db_session: Session
    redis_client: redis.Redis
    config: NexlifyConfig
    auth_manager: NexlifyAuthManager
    sentinel: NexlifySentinel
    temp_dir: Path

class DatabaseFactory:
    """Factory for test database - spawn test environments like a braindance"""
    
    @staticmethod
    def create_test_engine():
        """Create in-memory SQLite for tests - fast as a Sandevistan"""
        return create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
    
    @staticmethod
    def create_test_session(engine):
        """Create test session - isolated like a Faraday cage"""
        TestingSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        return TestingSessionLocal()

class UserFactory(factory.Factory):
    """Generate test users - digital personas for our tests"""
    class Meta:
        model = User
    
    id = factory.LazyFunction(lambda: fake.uuid4())
    username = factory.LazyFunction(lambda: f"netrunner_{fake.user_name()}")
    email = factory.LazyFunction(fake.email)
    password_hash = "$2b$12$test_hash"  # Pre-hashed for speed
    pin_hash = "test_pin_hash"
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    two_fa_enabled = True
    totp_secret = factory.LazyFunction(lambda: fake.pystr(32, 32))

class MarketFactory(factory.Factory):
    """Generate test markets - trading grounds for our battles"""
    class Meta:
        model = Market
    
    id = factory.LazyFunction(lambda: fake.uuid4())
    name = factory.LazyFunction(lambda: f"{fake.company()} Exchange")
    exchange = factory.LazyFunction(lambda: random.choice(['binance', 'kraken', 'nexus']))
    api_url = factory.LazyFunction(lambda: f"https://api.{fake.domain_name()}")
    websocket_url = factory.LazyFunction(lambda: f"wss://stream.{fake.domain_name()}")
    maker_fee = factory.LazyFunction(lambda: Decimal(str(random.uniform(0.0001, 0.002))))
    taker_fee = factory.LazyFunction(lambda: Decimal(str(random.uniform(0.0002, 0.003))))

class CandleFactory(factory.Factory):
    """Generate test candles - market data for our simulations"""
    class Meta:
        model = Candle
    
    id = factory.LazyFunction(lambda: fake.uuid4())
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    interval = "1h"
    open = factory.LazyFunction(lambda: Decimal(str(random.uniform(30000, 40000))))
    high = factory.LazyAttribute(lambda obj: obj.open * Decimal("1.02"))
    low = factory.LazyAttribute(lambda obj: obj.open * Decimal("0.98"))
    close = factory.LazyFunction(lambda: Decimal(str(random.uniform(30000, 40000))))
    volume = factory.LazyFunction(lambda: Decimal(str(random.uniform(100, 10000))))

@pytest.fixture(scope="function")
async def test_context():
    """
    Main test context fixture - your complete testing rig
    Fresh for each test, clean as a corpo lab
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock config
        config = NexlifyConfig(config_path=temp_path / "test_config.yaml")
        config.database.db_type = "sqlite"
        config.database.database = ":memory:"
        
        # Create test database
        engine = DatabaseFactory.create_test_engine()
        Base.metadata.create_all(engine)
        session = DatabaseFactory.create_test_session(engine)
        
        # Create fake Redis
        redis_client = FakeRedis(decode_responses=True)
        
        # Initialize managers
        auth_manager = NexlifyAuthManager(session, redis_client)
        sentinel = NexlifySentinel()
        
        context = TestContext(
            db_session=session,
            redis_client=redis_client,
            config=config,
            auth_manager=auth_manager,
            sentinel=sentinel,
            temp_dir=temp_path
        )
        
        yield context
        
        # Cleanup
        session.close()
        engine.dispose()

class TestConfigurationManager:
    """Test the configuration system - the neural core"""
    
    def test_config_initialization(self, test_context: TestContext):
        """Test config loads properly - basic neural handshake"""
        config = test_context.config
        
        assert config.security.pin_length >= 4
        assert config.security.enable_2fa is True
        assert config.database.pool_size > 0
        assert config.performance.worker_threads > 0
        assert config.ml.model_architecture == "transformer"
    
    def test_pin_generation(self, test_context: TestContext):
        """Test PIN generation - no more hardcoded 2077"""
        config = test_context.config
        
        # Generate multiple PINs
        pins = set()
        for _ in range(100):
            pin, pin_hash = config.generate_pin()
            assert len(pin) == config.security.pin_length
            assert pin.isdigit()
            assert pin not in pins  # Should be unique
            pins.add(pin)
            
            # Verify hash works
            assert config.verify_pin(pin, pin_hash)
            assert not config.verify_pin("wrong_pin", pin_hash)
    
    def test_config_encryption(self, test_context: TestContext):
        """Test sensitive data encryption - keep the secrets safe"""
        config = test_context.config
        
        # Set sensitive data
        config.database.password = "super_secret_password"
        config.security.master_key = "ultra_secret_key"
        
        # Save config
        config.save_config()
        
        # Load into new instance
        new_config = NexlifyConfig(config_path=config.config_path)
        
        # Verify decryption worked
        assert new_config.database.password == "super_secret_password"
        assert new_config.security.master_key == "ultra_secret_key"
    
    @pytest.mark.parametrize("env_var,expected", [
        ("NEXLIFY_SECURITY_RATE_LIMIT_REQUESTS", "200"),
        ("NEXLIFY_DB_HOST", "cyber-postgres.local"),
        ("NEXLIFY_PERF_WORKER_THREADS", "16"),
    ])
    def test_env_override(self, test_context: TestContext, monkeypatch, env_var, expected):
        """Test environment variable overrides - adapt to any environment"""
        monkeypatch.setenv(env_var, expected)
        
        # Reload config
        config = NexlifyConfig()
        
        # Verify override worked
        if "RATE_LIMIT" in env_var:
            assert config.security.rate_limit_requests == int(expected)
        elif "DB_HOST" in env_var:
            assert config.database.host == expected
        elif "WORKER_THREADS" in env_var:
            assert config.performance.worker_threads == int(expected)

class TestAuthenticationSystem:
    """Test the authentication system - the digital bouncer"""
    
    @pytest.mark.asyncio
    async def test_user_registration(self, test_context: TestContext):
        """Test user registration - welcome to the grid"""
        auth = test_context.auth_manager
        
        # Register new user
        user, pin, totp_uri = await auth.register_user(
            username="v_silverhand",
            email="v@samurai.nc",
            password="WakeUpSamurai2077",
            enable_2fa=True
        )
        
        assert user.username == "v_silverhand"
        assert user.email == "v@samurai.nc"
        assert len(pin) == test_context.config.security.pin_length
        assert totp_uri is not None
        assert user.two_fa_enabled is True
        assert len(user.backup_codes) == 10
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, test_context: TestContext):
        """Test full auth flow - the gauntlet"""
        auth = test_context.auth_manager
        
        # Register user
        user, pin, totp_uri = await auth.register_user(
            username="jackie_welles",
            email="jackie@heywood.nc",
            password="ToTheMajorLeagues",
            enable_2fa=False  # Simpler for testing
        )
        
        # Successful auth
        tokens = await auth.authenticate(
            username="jackie_welles",
            password="ToTheMajorLeagues",
            pin=pin,
            ip_address="192.168.1.100"
        )
        
        assert 'access_token' in tokens
        assert 'refresh_token' in tokens
        assert tokens['token_type'] == 'bearer'
        
        # Verify token
        payload = await auth.verify_token(tokens['access_token'])
        assert payload['user_id'] == user.id
    
    @pytest.mark.asyncio
    async def test_failed_authentication(self, test_context: TestContext):
        """Test auth failures - when things go wrong"""
        auth = test_context.auth_manager
        
        # Register user
        user, pin, _ = await auth.register_user(
            username="adam_smasher",
            email="smasher@arasaka.corp",
            password="CrushTheMeat",
            enable_2fa=False
        )
        
        # Wrong password
        with pytest.raises(Exception) as exc_info:
            await auth.authenticate(
                username="adam_smasher",
                password="wrong_password",
                pin=pin
            )
        
        # Wrong PIN
        with pytest.raises(Exception) as exc_info:
            await auth.authenticate(
                username="adam_smasher",
                password="CrushTheMeat",
                pin="000000"
            )
        
        # Check failed attempts recorded
        test_context.db_session.refresh(user)
        assert user.failed_attempts > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_context: TestContext):
        """Test rate limiting - stop the script kiddies"""
        auth = test_context.auth_manager
        ip = "192.168.1.50"
        
        # Register user
        user, pin, _ = await auth.register_user(
            username="rate_test",
            email="rate@test.nc",
            password="test123",
            enable_2fa=False
        )
        
        # Hammer the auth endpoint
        max_requests = test_context.config.security.rate_limit_requests
        
        for i in range(max_requests + 5):
            try:
                await auth.authenticate(
                    username="rate_test",
                    password="wrong_pass",
                    pin="000000",
                    ip_address=ip
                )
            except:
                pass  # Expected to fail
        
        # Next request should be rate limited
        with pytest.raises(Exception) as exc_info:
            await auth.authenticate(
                username="rate_test",
                password="test123",
                pin=pin,
                ip_address=ip
            )
        
        assert "429" in str(exc_info.value) or "rate limit" in str(exc_info.value).lower()

class TestDatabaseMigration:
    """Test database migration - evolving the data consciousness"""
    
    @pytest.mark.asyncio
    async def test_sqlite_to_postgres_migration(self, test_context: TestContext):
        """Test SQLite to PostgreSQL migration - the great evolution"""
        # Create SQLite database with test data
        sqlite_path = test_context.temp_dir / "old_nexlify.db"
        sqlite_conn = sqlite3.connect(sqlite_path)
        
        # Create old schema
        sqlite_conn.execute("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                created_at TEXT
            )
        """)
        
        # Insert test data
        sqlite_conn.execute("""
            INSERT INTO users (id, username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("test-id", "old_user", "old@user.com", "hash123", "2024-01-01T00:00:00Z"))
        
        sqlite_conn.commit()
        sqlite_conn.close()
        
        # Mock migration engine
        engine = DataEvolutionEngine()
        engine.pg_engine = test_context.db_session.bind
        engine.SessionLocal = lambda: test_context.db_session
        
        # Run migration
        results = await engine.migrate_from_sqlite(sqlite_path)
        
        assert results['success'] is True
        assert 'users' in results['tables_migrated']
        assert results['records_migrated'] > 0
    
    def test_data_transformation(self, test_context: TestContext):
        """Test data transformation - upgrading the bits"""
        engine = DataEvolutionEngine()
        
        # Test user transformation
        old_user = {
            'id': 'user-123',
            'username': 'netrunner',
            'email': 'runner@net.nc',
            'password_hash': 'old_hash',
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        transformed = engine._transform_user(old_user)
        
        assert transformed['id'] == 'user-123'
        assert transformed['username'] == 'netrunner'
        assert transformed['pin_hash'] == ''  # Should handle missing PIN
        assert isinstance(transformed['created_at'], datetime)

class TestMonitoringSystem:
    """Test the monitoring system - the all-seeing eye"""
    
    @pytest.mark.asyncio
    async def test_sentinel_initialization(self, test_context: TestContext):
        """Test Sentinel starts up properly - eyes online"""
        sentinel = test_context.sentinel
        
        # Start monitoring
        await sentinel.start_monitoring()
        assert sentinel.is_running is True
        
        # Check metrics are initialized
        assert sentinel.cpu_usage._value.get() >= 0
        assert len(sentinel.monitoring_tasks) > 0
        
        # Stop monitoring
        await sentinel.stop_monitoring()
        assert sentinel.is_running is False
    
    @pytest.mark.asyncio
    async def test_alert_system(self, test_context: TestContext):
        """Test alert system - the alarm bells"""
        sentinel = test_context.sentinel
        
        # Raise an alert
        await sentinel.raise_alert(
            severity=AlertSeverity.WARNING,
            metric_type=MetricType.SYSTEM,
            title="Test Alert",
            description="This is a test of the emergency broadcast system",
            metric_value=90.0,
            threshold=80.0,
            component="test_component"
        )
        
        assert len(sentinel.active_alerts) == 1
        
        # Get the alert
        alert = list(sentinel.active_alerts.values())[0]
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value == 90.0
        
        # Resolve the alert
        sentinel.resolve_alert(alert.id)
        assert len(sentinel.active_alerts) == 0
    
    def test_metric_recording(self, test_context: TestContext):
        """Test metric recording - logging the data streams"""
        sentinel = test_context.sentinel
        
        # Record various metrics
        sentinel.record_metric(
            MetricType.TRADING,
            "test_metric",
            {"value": 42, "status": "online"}
        )
        
        # Check buffer
        key = f"{MetricType.TRADING.value}.test_metric"
        assert key in sentinel.metrics_buffer
        assert len(sentinel.metrics_buffer[key]) > 0
        
        # Verify metric data
        metric = sentinel.metrics_buffer[key][-1]
        assert metric['value']['value'] == 42
        assert metric['value']['status'] == "online"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, test_context: TestContext):
        """Test anomaly detection - catching the glitches"""
        sentinel = test_context.sentinel
        
        # Populate baseline with normal values
        for i in range(60):
            sentinel.performance_baselines['cpu_usage'].append(50.0 + random.uniform(-5, 5))
        
        # Add anomalous value
        sentinel.performance_baselines['cpu_usage'].append(95.0)
        
        # Run anomaly detection
        await sentinel._detect_anomalies()
        
        # Should have raised an alert
        assert len(sentinel.active_alerts) > 0

class TestTradingComponents:
    """Test trading components - where the eddies flow"""
    
    def test_market_model(self, test_context: TestContext):
        """Test market model - the trading grounds"""
        market = MarketFactory.build()
        
        assert market.name is not None
        assert market.api_url.startswith("https://")
        assert market.maker_fee < market.taker_fee  # Usually true
        assert market.is_active is True
    
    def test_candle_validation(self, test_context: TestContext):
        """Test candle data validation - market truth"""
        candle = CandleFactory.build()
        
        # OHLC relationships should be valid
        assert candle.high >= candle.open
        assert candle.high >= candle.close
        assert candle.high >= candle.low
        assert candle.low <= candle.open
        assert candle.low <= candle.close
    
    def test_portfolio_calculations(self, test_context: TestContext):
        """Test portfolio calculations - counting the eddies"""
        portfolio = Portfolio(
            user_id="test-user",
            name="Test Portfolio",
            initial_balance=Decimal("10000"),
            total_trades=100,
            winning_trades=60
        )
        
        # Test win rate calculation
        assert portfolio.win_rate == 0.6  # 60%

class TestIntegration:
    """Integration tests - when all the chrome works together"""
    
    @pytest.mark.asyncio
    async def test_full_auth_monitoring_flow(self, test_context: TestContext):
        """Test auth with monitoring - the full package"""
        auth = test_context.auth_manager
        sentinel = test_context.sentinel
        
        # Add alert handler to capture alerts
        alerts_captured = []
        async def capture_alert(alert):
            alerts_captured.append(alert)
        
        sentinel.add_alert_handler(capture_alert)
        
        # Register user
        user, pin, _ = await auth.register_user(
            username="full_test_user",
            email="full@test.nc",
            password="FullTest123",
            enable_2fa=False
        )
        
        # Simulate multiple failed auth attempts
        for i in range(10):
            try:
                await auth.authenticate(
                    username="full_test_user",
                    password="wrong_password",
                    pin="000000",
                    ip_address="192.168.1.100"
                )
            except:
                pass
        
        # Check if security events were logged
        security_events = test_context.db_session.query(
            auth.SecurityEvent
        ).filter_by(user_id=user.id).all()
        
        assert len(security_events) > 0
        assert any(event.event_type == "failed_password" for event in security_events)

class PerformanceTests:
    """Performance tests - pushing the chrome to its limits"""
    
    @pytest.mark.performance
    def test_config_load_performance(self, test_context: TestContext, benchmark):
        """Test config loading performance - how fast can we boot?"""
        def load_config():
            return NexlifyConfig(config_path=test_context.temp_dir / "perf_test.yaml")
        
        # Benchmark config loading
        result = benchmark(load_config)
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_auth_performance(self, test_context: TestContext, benchmark):
        """Test authentication performance - speed of access"""
        auth = test_context.auth_manager
        
        # Setup user
        user, pin, _ = await auth.register_user(
            username="perf_user",
            email="perf@test.nc",
            password="PerfTest123",
            enable_2fa=False
        )
        
        # Benchmark authentication
        async def authenticate():
            return await auth.authenticate(
                username="perf_user",
                password="PerfTest123",
                pin=pin
            )
        
        result = await benchmark(authenticate)
        assert 'access_token' in result

# Test runner configuration
if __name__ == "__main__":
    # Run with different configs for different scenarios
    pytest.main([
        "-v",  # Verbose
        "-s",  # Show print statements
        "--tb=short",  # Short traceback
        "--cov=nexlify",  # Coverage
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "-m", "not performance",  # Skip performance tests by default
        __file__
    ])
