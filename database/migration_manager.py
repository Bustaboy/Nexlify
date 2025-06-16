# nexlify/database/migration_manager.py
"""
Database Migration Manager - Evolving data like a techno-shaman
Handles PostgreSQL migrations, data transformation, and keeps your bits intact
"""

import os
import json
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import logging

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.migration import MigrationContext
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

from database.models import Base, Market, Symbol, Candle, Portfolio, Position, Order
from config.config_manager import get_config
from security.auth_manager import User

class DataEvolutionEngine:
    """
    Handles database migrations like a street doc handles cyberware upgrades
    Smooth, clean, and won't leave you braindead
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("nexlify.migrations")
        
        # PostgreSQL engine - the new neural cortex
        self.pg_engine = create_engine(
            self.config.database.connection_string,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow,
            pool_pre_ping=True,  # Keep connections alive
            echo=False
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.pg_engine
        )
        
        # Alembic config for schema versioning
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", "nexlify/database/alembic")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.config.database.connection_string)
        
        # Migration status tracking
        self.migration_status = {
            'in_progress': False,
            'last_migration': None,
            'errors': []
        }
    
    async def initialize_database(self) -> bool:
        """
        Initialize fresh PostgreSQL database - birth of a new data consciousness
        """
        try:
            self.logger.info("Initializing PostgreSQL database...")
            
            # Create database if it doesn't exist
            await self._ensure_database_exists()
            
            # Create all tables
            Base.metadata.create_all(bind=self.pg_engine)
            
            # Initialize Alembic
            command.init(self.alembic_cfg, "nexlify/database/alembic")
            command.stamp(self.alembic_cfg, "head")
            
            # Create default market configurations
            await self._create_default_markets()
            
            self.logger.info("Database initialization complete - we're online, choom!")
            return True
            
        except Exception as e:
            self.logger.error(f"Database init failed: {e}")
            self.migration_status['errors'].append(str(e))
            return False
    
    async def _ensure_database_exists(self):
        """Create database if it doesn't exist - like claiming new turf in the Net"""
        conn_params = {
            'host': self.config.database.host,
            'port': self.config.database.port,
            'user': self.config.database.username,
            'password': self.config.database.password
        }
        
        try:
            # Connect to default postgres database
            conn = psycopg2.connect(database='postgres', **conn_params)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if our database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config.database.database,)
            )
            
            if not cursor.fetchone():
                # Create it - new territory in the digital frontier
                cursor.execute(f"CREATE DATABASE {self.config.database.database}")
                self.logger.info(f"Created database: {self.config.database.database}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database creation error: {e}")
            raise
    
    async def migrate_from_sqlite(
        self, 
        sqlite_path: Path,
        batch_size: int = 1000,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Migrate from SQLite to PostgreSQL - upgrading from street chrome to corpo grade
        """
        if self.migration_status['in_progress']:
            return {
                'success': False,
                'error': 'Migration already in progress'
            }
        
        self.migration_status['in_progress'] = True
        self.migration_status['errors'] = []
        start_time = datetime.now(timezone.utc)
        
        results = {
            'success': False,
            'tables_migrated': [],
            'records_migrated': 0,
            'duration': 0,
            'errors': []
        }
        
        try:
            # Connect to SQLite - the old brain
            sqlite_conn = sqlite3.connect(sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            
            # Get table mappings
            table_mappings = self._get_table_mappings()
            
            with self.SessionLocal() as session:
                for old_table, (new_table, transformer) in table_mappings.items():
                    try:
                        records = await self._migrate_table(
                            sqlite_conn,
                            session,
                            old_table,
                            new_table,
                            transformer,
                            batch_size,
                            progress_callback
                        )
                        
                        results['tables_migrated'].append(old_table)
                        results['records_migrated'] += records
                        
                    except Exception as e:
                        error_msg = f"Failed to migrate {old_table}: {e}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # Commit all changes
                session.commit()
            
            sqlite_conn.close()
            
            # Update sequences for auto-increment fields
            await self._update_sequences()
            
            results['success'] = len(results['errors']) == 0
            results['duration'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self.logger.info(
                f"Migration complete: {results['records_migrated']} records in {results['duration']:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            results['errors'].append(str(e))
        
        finally:
            self.migration_status['in_progress'] = False
            self.migration_status['last_migration'] = datetime.now(timezone.utc)
        
        return results
    
    def _get_table_mappings(self) -> Dict[str, Tuple[Any, callable]]:
        """
        Map old SQLite tables to new PostgreSQL models
        Like a fixer matching street deals to corpo contracts
        """
        return {
            'users': (User, self._transform_user),
            'markets': (Market, self._transform_market),
            'symbols': (Symbol, self._transform_symbol),
            'candles': (Candle, self._transform_candle),
            'portfolios': (Portfolio, self._transform_portfolio),
            'positions': (Position, self._transform_position),
            'orders': (Order, self._transform_order),
        }
    
    async def _migrate_table(
        self,
        sqlite_conn: sqlite3.Connection,
        pg_session: Session,
        old_table: str,
        new_model: Any,
        transformer: callable,
        batch_size: int,
        progress_callback: Optional[callable]
    ) -> int:
        """Migrate a single table - one neural pathway at a time"""
        cursor = sqlite_conn.cursor()
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM {old_table}")
        total_records = cursor.fetchone()[0]
        
        if total_records == 0:
            return 0
        
        # Migrate in batches - don't overload the neural net
        migrated = 0
        offset = 0
        
        while offset < total_records:
            cursor.execute(
                f"SELECT * FROM {old_table} LIMIT {batch_size} OFFSET {offset}"
            )
            rows = cursor.fetchall()
            
            if not rows:
                break
            
            # Transform and insert
            new_records = []
            for row in rows:
                try:
                    transformed = transformer(dict(row))
                    if transformed:
                        new_records.append(new_model(**transformed))
                except Exception as e:
                    self.logger.warning(f"Failed to transform record: {e}")
            
            if new_records:
                pg_session.bulk_save_objects(new_records)
                pg_session.flush()  # Flush but don't commit yet
            
            migrated += len(new_records)
            offset += batch_size
            
            # Progress update
            if progress_callback:
                progress = (migrated / total_records) * 100
                await progress_callback(old_table, progress, migrated, total_records)
        
        self.logger.info(f"Migrated {migrated}/{total_records} records from {old_table}")
        return migrated
    
    def _transform_user(self, row: Dict) -> Optional[Dict]:
        """Transform user data - upgrade their digital identity"""
        try:
            return {
                'id': row.get('id'),
                'username': row.get('username'),
                'email': row.get('email'),
                'password_hash': row.get('password_hash'),
                'pin_hash': row.get('pin_hash', ''),  # Generate new if missing
                'created_at': datetime.fromisoformat(row.get('created_at', '').replace('Z', '+00:00'))
                if row.get('created_at') else datetime.now(timezone.utc),
                'two_fa_enabled': bool(row.get('two_fa_enabled', 0))
            }
        except Exception as e:
            self.logger.error(f"User transform error: {e}")
            return None
    
    def _transform_market(self, row: Dict) -> Optional[Dict]:
        """Transform market data - new trading grounds"""
        try:
            return {
                'name': row.get('name'),
                'exchange': row.get('exchange', 'unknown'),
                'api_url': row.get('api_url', ''),
                'websocket_url': row.get('websocket_url'),
                'maker_fee': Decimal(str(row.get('maker_fee', 0.001))),
                'taker_fee': Decimal(str(row.get('taker_fee', 0.001))),
                'is_active': bool(row.get('is_active', 1))
            }
        except Exception as e:
            self.logger.error(f"Market transform error: {e}")
            return None
    
    def _transform_symbol(self, row: Dict) -> Optional[Dict]:
        """Transform symbol data - the digital assets"""
        try:
            symbol = row.get('symbol', '')
            base, quote = symbol.split('/') if '/' in symbol else (symbol, 'USDT')
            
            return {
                'market_id': row.get('market_id'),
                'symbol': symbol,
                'base_asset': base,
                'quote_asset': quote,
                'min_quantity': Decimal(str(row.get('min_quantity', 0.001))),
                'max_quantity': Decimal(str(row.get('max_quantity', 1000000))),
                'step_size': Decimal(str(row.get('step_size', 0.001))),
                'min_price': Decimal(str(row.get('min_price', 0.00001))),
                'max_price': Decimal(str(row.get('max_price', 1000000))),
                'tick_size': Decimal(str(row.get('tick_size', 0.01))),
                'is_trading': bool(row.get('is_trading', 1))
            }
        except Exception as e:
            self.logger.error(f"Symbol transform error: {e}")
            return None
    
    def _transform_candle(self, row: Dict) -> Optional[Dict]:
        """Transform candle data - market memories"""
        try:
            return {
                'symbol_id': row.get('symbol_id'),
                'timestamp': datetime.fromisoformat(row.get('timestamp', '').replace('Z', '+00:00'))
                if row.get('timestamp') else None,
                'interval': row.get('interval', '1h'),
                'open': Decimal(str(row.get('open', 0))),
                'high': Decimal(str(row.get('high', 0))),
                'low': Decimal(str(row.get('low', 0))),
                'close': Decimal(str(row.get('close', 0))),
                'volume': Decimal(str(row.get('volume', 0)))
            }
        except Exception as e:
            self.logger.error(f"Candle transform error: {e}")
            return None
    
    def _transform_portfolio(self, row: Dict) -> Optional[Dict]:
        """Transform portfolio data - digital wealth upgrade"""
        try:
            return {
                'user_id': row.get('user_id'),
                'name': row.get('name', 'Default Portfolio'),
                'is_active': bool(row.get('is_active', 1)),
                'is_paper_trading': bool(row.get('is_paper_trading', 0)),
                'initial_balance': Decimal(str(row.get('initial_balance', 10000))),
                'total_trades': int(row.get('total_trades', 0)),
                'winning_trades': int(row.get('winning_trades', 0)),
                'total_pnl': Decimal(str(row.get('total_pnl', 0)))
            }
        except Exception as e:
            self.logger.error(f"Portfolio transform error: {e}")
            return None
    
    def _transform_position(self, row: Dict) -> Optional[Dict]:
        """Transform position data - active market stakes"""
        try:
            return {
                'portfolio_id': row.get('portfolio_id'),
                'symbol_id': row.get('symbol_id'),
                'side': row.get('side', 'long'),
                'quantity': Decimal(str(row.get('quantity', 0))),
                'entry_price': Decimal(str(row.get('entry_price', 0))),
                'current_price': Decimal(str(row.get('current_price', 0))) if row.get('current_price') else None,
                'stop_loss': Decimal(str(row.get('stop_loss', 0))) if row.get('stop_loss') else None,
                'take_profit': Decimal(str(row.get('take_profit', 0))) if row.get('take_profit') else None,
                'realized_pnl': Decimal(str(row.get('realized_pnl', 0))),
                'unrealized_pnl': Decimal(str(row.get('unrealized_pnl', 0))),
                'is_open': bool(row.get('is_open', 1))
            }
        except Exception as e:
            self.logger.error(f"Position transform error: {e}")
            return None
    
    def _transform_order(self, row: Dict) -> Optional[Dict]:
        """Transform order data - market commands"""
        try:
            return {
                'portfolio_id': row.get('portfolio_id'),
                'symbol_id': row.get('symbol_id'),
                'position_id': row.get('position_id'),
                'exchange_order_id': row.get('exchange_order_id'),
                'type': row.get('type', 'market'),
                'side': row.get('side', 'buy'),
                'status': row.get('status', 'pending'),
                'quantity': Decimal(str(row.get('quantity', 0))),
                'filled_quantity': Decimal(str(row.get('filled_quantity', 0))),
                'price': Decimal(str(row.get('price', 0))) if row.get('price') else None,
                'average_fill_price': Decimal(str(row.get('average_fill_price', 0))) 
                if row.get('average_fill_price') else None,
                'fee': Decimal(str(row.get('fee', 0)))
            }
        except Exception as e:
            self.logger.error(f"Order transform error: {e}")
            return None
    
    async def _update_sequences(self):
        """Update PostgreSQL sequences - sync the neural counters"""
        with self.pg_engine.connect() as conn:
            # Get all tables with serial columns
            result = conn.execute(text("""
                SELECT table_name, column_name 
                FROM information_schema.columns 
                WHERE column_default LIKE 'nextval%'
                AND table_schema = 'public'
            """))
            
            for table, column in result:
                # Update sequence to max value + 1
                conn.execute(text(f"""
                    SELECT setval(
                        pg_get_serial_sequence('{table}', '{column}'),
                        COALESCE((SELECT MAX({column}) FROM {table}), 0) + 1,
                        false
                    )
                """))
                conn.commit()
    
    async def _create_default_markets(self):
        """Create default market configurations - the trading playgrounds"""
        default_markets = [
            {
                'name': 'Binance Spot',
                'exchange': 'binance',
                'api_url': 'https://api.binance.com',
                'websocket_url': 'wss://stream.binance.com:9443/ws',
                'maker_fee': Decimal('0.001'),
                'taker_fee': Decimal('0.001')
            },
            {
                'name': 'Kraken Spot',
                'exchange': 'kraken',
                'api_url': 'https://api.kraken.com',
                'websocket_url': 'wss://ws.kraken.com',
                'maker_fee': Decimal('0.0016'),
                'taker_fee': Decimal('0.0026')
            }
        ]
        
        with self.SessionLocal() as session:
            for market_data in default_markets:
                market = Market(**market_data)
                session.add(market)
            session.commit()
    
    async def backup_database(self, backup_path: Optional[Path] = None) -> Path:
        """
        Backup PostgreSQL database - always have an escape route
        """
        if not backup_path:
            backup_dir = Path('./backups')
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f'nexlify_backup_{timestamp}.sql'
        
        try:
            # Use pg_dump for backup
            import subprocess
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.database.password
            
            cmd = [
                'pg_dump',
                '-h', self.config.database.host,
                '-p', str(self.config.database.port),
                '-U', self.config.database.username,
                '-d', self.config.database.database,
                '-f', str(backup_path),
                '--verbose',
                '--no-owner',
                '--no-privileges'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Backup failed: {result.stderr}")
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise
    
    async def restore_database(self, backup_path: Path) -> bool:
        """
        Restore database from backup - resurrection protocol
        """
        try:
            # Drop and recreate database
            await self._ensure_database_exists()
            
            # Restore from backup
            import subprocess
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.database.password
            
            cmd = [
                'psql',
                '-h', self.config.database.host,
                '-p', str(self.config.database.port),
                '-U', self.config.database.username,
                '-d', self.config.database.database,
                '-f', str(backup_path)
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")
            
            self.logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status - system diagnostics"""
        return {
            **self.migration_status,
            'database_online': self._check_database_connection(),
            'schema_version': self._get_schema_version()
        }
    
    def _check_database_connection(self) -> bool:
        """Check if we can jack into the database"""
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def _get_schema_version(self) -> Optional[str]:
        """Get current schema version from Alembic"""
        try:
            with self.pg_engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except:
            return None

# Initialize the engine
_migration_engine: Optional[DataEvolutionEngine] = None

def get_migration_engine() -> DataEvolutionEngine:
    """Get or create migration engine instance"""
    global _migration_engine
    if _migration_engine is None:
        _migration_engine = DataEvolutionEngine()
    return _migration_engine
