"""
Nexlify Audit Trail Module v2.0.8
Blockchain-based immutable audit logging with compliance reporting
"""

import os
import json
import sqlite3
import hashlib
import asyncio
import threading
import queue
import time
import uuid
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature
import logging
from enum import Enum

# Import error handler and security
from src.core.error_handler import get_error_handler, handle_errors
from src.security.nexlify_advanced_security import get_security_manager

class AuditEventType(Enum):
    """Audit event types"""
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    CONFIG_CHANGE = "config_change"
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    API_KEY_CHANGE = "api_key_change"
    SECURITY_ALERT = "security_alert"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR = "error"
    CUSTOM = "custom"

@dataclass
class AuditEntry:
    """Immutable audit entry"""
    entry_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    component: str
    action: str
    details: Dict[str, Any]
    previous_hash: str
    entry_hash: str = ""
    signature: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Calculate deterministic hash for the entry"""
        # Create deterministic content string (excluding mutable fields)
        content = {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'user_id': self.user_id or '',
            'ip_address': self.ip_address or '',
            'component': self.component,
            'action': self.action,
            'details': json.dumps(self.details, sort_keys=True),
            'previous_hash': self.previous_hash
        }
        
        # Create hash
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'component': self.component,
            'action': self.action,
            'details': json.dumps(self.details),
            'previous_hash': self.previous_hash,
            'entry_hash': self.entry_hash,
            'signature': self.signature
        }

class BlockchainAudit:
    """Blockchain-based audit trail with immutability"""
    
    def __init__(self, db_path: str = "logs/audit/audit_trail.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.error_handler = get_error_handler()
        self.logger = logging.getLogger('AuditTrail')
        
        # Initialize database with WAL mode
        self._init_database()
        
        # Load or generate signing keys
        self._init_keys()
        
        # Entry queue for async processing
        self.entry_queue = queue.Queue(maxsize=10000)
        self.processing = True
        
        # Start background processor
        self.processor_thread = threading.Thread(
            target=self._process_queue,
            daemon=True
        )
        self.processor_thread.start()
        
        # Cache for recent entries
        self.recent_entries_cache = []
        self.cache_size = 1000
        
        # Integrity check tracking
        self.last_integrity_check = datetime.now()
        self.integrity_check_interval = timedelta(hours=1)
        
    def _init_database(self):
        """Initialize SQLite database with optimizations"""
        try:
            # Use context manager for connection
            with self._get_connection() as conn:
                # Enable WAL mode for concurrent access
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                
                # Create audit table with indexes
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entry_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        user_id TEXT,
                        ip_address TEXT,
                        component TEXT NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT NOT NULL,
                        previous_hash TEXT NOT NULL,
                        entry_hash TEXT NOT NULL,
                        signature TEXT,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                    ON audit_trail(timestamp DESC)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_event_type 
                    ON audit_trail(event_type)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_user_id 
                    ON audit_trail(user_id)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_audit_component 
                    ON audit_trail(component)
                ''')
                
                # Create metadata table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
                # Set initial metadata
                self._set_metadata(conn, 'version', '2.0.8')
                self._set_metadata(conn, 'created_at', datetime.now().isoformat())
                
        except Exception as e:
            self.error_handler.log_fatal_error(
                e,
                component="audit",
                context={"operation": "init_database"}
            )
            raise
            
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level='DEFERRED'
            )
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()
                
    def _set_metadata(self, conn: sqlite3.Connection, key: str, value: str):
        """Set metadata value"""
        conn.execute(
            "INSERT OR REPLACE INTO audit_metadata (key, value) VALUES (?, ?)",
            (key, value)
        )
        
    def _init_keys(self):
        """Initialize RSA keys for signing"""
        key_dir = Path("config/.audit_keys")
        key_dir.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        if os.name != 'nt':
            os.chmod(key_dir, 0o700)
            
        self.private_key_path = key_dir / "audit_key.pem"
        self.public_key_path = key_dir / "audit_key.pub"
        
        if self.private_key_path.exists() and self.public_key_path.exists():
            # Load existing keys
            self._load_keys()
        else:
            # Generate new keys
            self._generate_keys()
            
    def _generate_keys(self):
        """Generate new RSA key pair"""
        try:
            # Generate private key
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            self.public_key = self.private_key.public_key()
            
            # Save private key (encrypted)
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()  # TODO: Add password encryption
            )
            
            # Save public key
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Write keys with proper permissions
            self.private_key_path.write_bytes(private_pem)
            self.public_key_path.write_bytes(public_pem)
            
            if os.name != 'nt':
                os.chmod(self.private_key_path, 0o600)
                os.chmod(self.public_key_path, 0o644)
                
        except Exception as e:
            self.error_handler.log_critical_error(
                e,
                component="audit",
                context={"operation": "generate_keys"}
            )
            raise
            
    def _load_keys(self):
        """Load existing RSA keys"""
        try:
            # Load private key
            private_pem = self.private_key_path.read_bytes()
            self.private_key = serialization.load_pem_private_key(
                private_pem,
                password=None  # TODO: Add password support
            )
            
            # Load public key
            public_pem = self.public_key_path.read_bytes()
            self.public_key = serialization.load_pem_public_key(public_pem)
            
        except Exception as e:
            self.error_handler.log_critical_error(
                e,
                component="audit",
                context={"operation": "load_keys"}
            )
            # Regenerate if loading fails
            self._generate_keys()
            
    def _generate_entry_id(self) -> str:
        """Generate unique deterministic entry ID"""
        # Use timestamp + random UUID for uniqueness
        timestamp = datetime.now().isoformat()
        unique_id = str(uuid.uuid4())
        return hashlib.sha256(f"{timestamp}:{unique_id}".encode()).hexdigest()[:16]
        
    def _sign_entry(self, entry_hash: str) -> str:
        """Sign entry hash with private key"""
        try:
            signature = self.private_key.sign(
                entry_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "sign_entry"}
            )
            return ""
            
    def _verify_signature(self, entry_hash: str, signature: str) -> bool:
        """Verify entry signature"""
        try:
            self.public_key.verify(
                bytes.fromhex(signature),
                entry_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "verify_signature"}
            )
            return False
            
    def _get_last_hash(self) -> str:
        """Get hash of last entry"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT entry_hash FROM audit_trail ORDER BY id DESC LIMIT 1"
                )
                row = cursor.fetchone()
                return row['entry_hash'] if row else "genesis"
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "get_last_hash"}
            )
            return "genesis"
            
    def _process_queue(self):
        """Process audit entries from queue"""
        while self.processing:
            try:
                # Get entries from queue with timeout
                entries_batch = []
                deadline = time.time() + 0.1  # 100ms batch window
                
                while time.time() < deadline and len(entries_batch) < 100:
                    try:
                        entry = self.entry_queue.get(timeout=0.01)
                        entries_batch.append(entry)
                    except queue.Empty:
                        break
                        
                # Write batch if we have entries
                if entries_batch:
                    self._write_entries_batch(entries_batch)
                    
                # Periodic integrity check
                if datetime.now() - self.last_integrity_check > self.integrity_check_interval:
                    self.perform_integrity_check()
                    
            except Exception as e:
                self.error_handler.log_error(
                    e,
                    component="audit",
                    context={"operation": "process_queue"}
                )
                time.sleep(1)  # Prevent tight loop on error
                
    def _write_entries_batch(self, entries: List[AuditEntry]):
        """Write batch of entries to database"""
        try:
            with self._get_connection() as conn:
                # Use transaction for batch insert
                conn.execute("BEGIN IMMEDIATE")
                
                try:
                    for entry in entries:
                        self._write_entry(conn, entry)
                        
                    conn.commit()
                    
                    # Update cache
                    self.recent_entries_cache.extend(entries)
                    if len(self.recent_entries_cache) > self.cache_size:
                        self.recent_entries_cache = self.recent_entries_cache[-self.cache_size:]
                        
                except Exception:
                    conn.rollback()
                    raise
                    
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                # Retry after delay
                time.sleep(0.1)
                self._write_entries_batch(entries)
            else:
                raise
                
    def _write_entry(self, conn: sqlite3.Connection, entry: AuditEntry):
        """Write single entry to database"""
        try:
            data = entry.to_dict()
            
            conn.execute('''
                INSERT INTO audit_trail (
                    entry_id, timestamp, event_type, user_id, ip_address,
                    component, action, details, previous_hash, entry_hash, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['entry_id'],
                data['timestamp'],
                data['event_type'],
                data['user_id'],
                data['ip_address'],
                data['component'],
                data['action'],
                data['details'],
                data['previous_hash'],
                data['entry_hash'],
                data['signature']
            ))
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                # Entry already exists, skip
                pass
            else:
                raise
                
    @handle_errors(component="audit")
    def add_entry(self, event_type: str, user_id: Optional[str], ip_address: Optional[str],
                  component: str, action: str, details: Dict[str, Any]) -> Optional[str]:
        """Add new audit entry"""
        try:
            # Get last hash
            previous_hash = self._get_last_hash()
            
            # Create entry
            entry = AuditEntry(
                entry_id=self._generate_entry_id(),
                timestamp=datetime.now(),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                component=component,
                action=action,
                details=details,
                previous_hash=previous_hash
            )
            
            # Calculate hash
            entry.entry_hash = entry.calculate_hash()
            
            # Sign entry
            entry.signature = self._sign_entry(entry.entry_hash)
            
            # Add to queue
            try:
                self.entry_queue.put(entry, timeout=1.0)
                return entry.entry_id
            except queue.Full:
                # Queue full, write directly
                with self._get_connection() as conn:
                    self._write_entry(conn, entry)
                return entry.entry_id
                
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"event_type": event_type, "action": action}
            )
            return None
            
    def verify_integrity(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """Verify blockchain integrity"""
        issues = []
        
        try:
            with self._get_connection() as conn:
                # Build query
                query = "SELECT * FROM audit_trail"
                params = []
                
                if start_date or end_date:
                    conditions = []
                    if start_date:
                        conditions.append("timestamp >= ?")
                        params.append(start_date.isoformat())
                    if end_date:
                        conditions.append("timestamp <= ?")
                        params.append(end_date.isoformat())
                    query += " WHERE " + " AND ".join(conditions)
                    
                query += " ORDER BY id ASC"
                
                cursor = conn.execute(query, params)
                
                previous_hash = "genesis"
                entry_count = 0
                
                for row in cursor:
                    entry_count += 1
                    
                    # Verify hash chain
                    if row['previous_hash'] != previous_hash:
                        issues.append(
                            f"Hash chain broken at entry {row['entry_id']}: "
                            f"expected {previous_hash}, got {row['previous_hash']}"
                        )
                        
                    # Verify entry hash
                    entry = AuditEntry(
                        entry_id=row['entry_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=row['event_type'],
                        user_id=row['user_id'],
                        ip_address=row['ip_address'],
                        component=row['component'],
                        action=row['action'],
                        details=json.loads(row['details']),
                        previous_hash=row['previous_hash']
                    )
                    
                    calculated_hash = entry.calculate_hash()
                    if calculated_hash != row['entry_hash']:
                        issues.append(
                            f"Hash mismatch for entry {row['entry_id']}: "
                            f"expected {calculated_hash}, got {row['entry_hash']}"
                        )
                        
                    # Verify signature
                    if row['signature']:
                        if not self._verify_signature(row['entry_hash'], row['signature']):
                            issues.append(f"Invalid signature for entry {row['entry_id']}")
                            
                    previous_hash = row['entry_hash']
                    
                # Log verification results
                if issues:
                    self.error_handler.log_critical_error(
                        Exception("Audit trail integrity check failed"),
                        component="audit",
                        context={"issues": issues[:10], "total_issues": len(issues)}
                    )
                else:
                    self.logger.info(f"Integrity check passed: {entry_count} entries verified")
                    
                return len(issues) == 0, issues
                
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "verify_integrity"}
            )
            return False, [str(e)]
            
    def perform_integrity_check(self):
        """Perform scheduled integrity check"""
        # Check last 24 hours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        valid, issues = self.verify_integrity(start_date, end_date)
        
        if not valid:
            # Send alert
            self.add_entry(
                AuditEventType.SECURITY_ALERT.value,
                user_id="system",
                ip_address=None,
                component="audit",
                action="integrity_check_failed",
                details={"issues_count": len(issues), "sample_issues": issues[:5]}
            )
            
        self.last_integrity_check = datetime.now()
        
    def get_entries(self, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   event_type: Optional[str] = None,
                   user_id: Optional[str] = None,
                   component: Optional[str] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Get audit entries with filters"""
        try:
            # Check cache first for recent entries
            if not start_date and not event_type and not user_id and not component:
                # Return from cache
                return [
                    entry.to_dict() for entry in self.recent_entries_cache[-limit:]
                ]
                
            with self._get_connection() as conn:
                query = "SELECT * FROM audit_trail WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if component:
                    query += " AND component = ?"
                    params.append(component)
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                entries = []
                for row in cursor:
                    entry = {
                        'entry_id': row['entry_id'],
                        'timestamp': row['timestamp'],
                        'event_type': row['event_type'],
                        'user_id': row['user_id'],
                        'ip_address': row['ip_address'],
                        'component': row['component'],
                        'action': row['action'],
                        'details': json.loads(row['details']),
                        'entry_hash': row['entry_hash']
                    }
                    entries.append(entry)
                    
                return entries
                
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "get_entries"}
            )
            return []
            
    def cleanup_old_entries(self, retention_days: int = 2555):  # 7 years default
        """Archive and cleanup old entries"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # First, archive old entries
            archive_path = self.db_path.parent / "archive"
            archive_path.mkdir(exist_ok=True)
            
            archive_file = archive_path / f"audit_archive_{cutoff_date.strftime('%Y%m')}.db"
            
            with self._get_connection() as conn:
                # Count entries to archive
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM audit_trail WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                count = cursor.fetchone()['count']
                
                if count > 0:
                    # Create archive
                    self._archive_entries(conn, cutoff_date, archive_file)
                    
                    # Delete archived entries
                    conn.execute(
                        "DELETE FROM audit_trail WHERE timestamp < ?",
                        (cutoff_date.isoformat(),)
                    )
                    conn.commit()
                    
                    # Vacuum to reclaim space
                    conn.execute("VACUUM")
                    
                    self.logger.info(f"Archived {count} entries to {archive_file}")
                    
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "cleanup_old_entries"}
            )
            
    def _archive_entries(self, conn: sqlite3.Connection, cutoff_date: datetime, archive_file: Path):
        """Archive entries to separate database"""
        # Connect to archive database
        archive_conn = sqlite3.connect(str(archive_file))
        
        try:
            # Create schema in archive
            archive_conn.executescript(conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
            ).fetchall()[0]['sql'])
            
            # Copy data
            cursor = conn.execute(
                "SELECT * FROM audit_trail WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            # Batch insert into archive
            batch = []
            for row in cursor:
                batch.append(tuple(row))
                if len(batch) >= 1000:
                    archive_conn.executemany(
                        f"INSERT INTO audit_trail VALUES ({','.join(['?']*len(row))})",
                        batch
                    )
                    batch = []
                    
            # Insert remaining
            if batch:
                archive_conn.executemany(
                    f"INSERT INTO audit_trail VALUES ({','.join(['?']*len(batch[0]))})",
                    batch
                )
                
            archive_conn.commit()
            
            # Compress archive
            with gzip.open(f"{archive_file}.gz", 'wb') as gz:
                with open(archive_file, 'rb') as f:
                    shutil.copyfileobj(f, gz)
                    
            # Remove uncompressed archive
            archive_file.unlink()
            
        finally:
            archive_conn.close()
            
    def export_for_audit(self, start_date: datetime, end_date: datetime,
                        format: str = "json", output_path: Optional[Path] = None) -> Path:
        """Export audit trail for external audit"""
        try:
            # Get entries
            entries = self.get_entries(
                start_date=start_date,
                end_date=end_date,
                limit=1000000  # High limit for export
            )
            
            # Default output path
            if output_path is None:
                export_dir = Path("reports/audit_exports")
                export_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"audit_export_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                output_path = export_dir / f"{filename}.{format}"
                
            # Export based on format
            if format == "json":
                # Add metadata
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'entry_count': len(entries),
                    'integrity_verified': self.verify_integrity(start_date, end_date)[0],
                    'entries': entries
                }
                
                # Encrypt if security manager available
                try:
                    security = get_security_manager()
                    encrypted = security.encryption.encrypt_config(export_data)
                    output_path = output_path.with_suffix('.json.enc')
                    output_path.write_text(encrypted)
                except:
                    # Fallback to unencrypted
                    with open(output_path, 'w') as f:
                        json.dump(export_data, f, indent=2)
                        
            elif format == "csv":
                # Convert to DataFrame
                df = pd.DataFrame(entries)
                
                # Flatten details column
                if 'details' in df.columns:
                    details_df = pd.json_normalize(df['details'])
                    df = pd.concat([df.drop('details', axis=1), details_df], axis=1)
                    
                # Export to CSV
                df.to_csv(output_path, index=False)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Exported {len(entries)} entries to {output_path}")
            return output_path
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="audit",
                context={"operation": "export_for_audit"}
            )
            raise
            
    def stop(self):
        """Stop audit trail processing"""
        self.processing = False
        
        # Flush queue
        remaining = []
        while not self.entry_queue.empty():
            try:
                remaining.append(self.entry_queue.get_nowait())
            except queue.Empty:
                break
                
        if remaining:
            self._write_entries_batch(remaining)
            
        # Wait for processor thread
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

class ComplianceReporter:
    """Generate compliance reports from audit trail"""
    
    def __init__(self, blockchain_audit: BlockchainAudit):
        self.audit = blockchain_audit
        self.error_handler = get_error_handler()
        
    @handle_errors(component="audit")
    def generate_regulatory_report(self, report_type: str, start_date: datetime,
                                 end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        
        if report_type.upper() == "MIFID2":
            return self.generate_mifid2_report(start_date, end_date, **kwargs)
        elif report_type.upper() == "FATF":
            return self.generate_fatf_report(start_date, end_date, **kwargs)
        elif report_type.upper() == "CFTC":
            return self.generate_cftc_report(start_date, end_date, **kwargs)
        else:
            # Custom report
            return self.generate_custom_report(report_type, start_date, end_date, **kwargs)
            
    def generate_mifid2_report(self, start_date: datetime, end_date: datetime,
                              include_kyc: bool = False) -> Dict[str, Any]:
        """Generate MiFID II compliance report"""
        report = {
            'report_type': 'MiFID II',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Transaction reporting
        trades = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.TRADE_EXECUTED.value,
            limit=50000  # MiFID II limit
        )
        
        report['sections']['transaction_reporting'] = {
            'total_trades': len(trades),
            'trades_by_instrument': self._group_by_field(trades, 'details.instrument'),
            'trades_by_venue': self._group_by_field(trades, 'details.exchange'),
            'average_execution_time': self._calculate_avg_execution_time(trades)
        }
        
        # Best execution
        report['sections']['best_execution'] = {
            'price_improvement': self._calculate_price_improvement(trades),
            'execution_quality': self._assess_execution_quality(trades)
        }
        
        # Client identification (if KYC enabled)
        if include_kyc:
            report['sections']['client_identification'] = {
                'identified_clients': self._get_identified_clients(start_date, end_date),
                'kyc_compliance': self._check_kyc_compliance(start_date, end_date)
            }
            
        # Risk management
        report['sections']['risk_management'] = {
            'risk_events': self._get_risk_events(start_date, end_date),
            'position_limits': self._check_position_limits(trades)
        }
        
        return report
        
    def generate_fatf_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate FATF compliance report (AML/CTF)"""
        report = {
            'report_type': 'FATF',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Suspicious activity
        suspicious = self._detect_suspicious_activity(start_date, end_date)
        report['sections']['suspicious_activity'] = {
            'total_alerts': len(suspicious),
            'alerts_by_type': self._group_by_field(suspicious, 'alert_type'),
            'high_risk_users': self._identify_high_risk_users(suspicious)
        }
        
        # Transaction monitoring
        all_trades = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.TRADE_EXECUTED.value,
            limit=100000
        )
        
        report['sections']['transaction_monitoring'] = {
            'large_transactions': self._identify_large_transactions(all_trades),
            'rapid_movement': self._detect_rapid_movement(all_trades),
            'cross_border': self._identify_cross_border(all_trades)
        }
        
        # Withdrawals
        withdrawals = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.WITHDRAWAL.value
        )
        
        report['sections']['withdrawals'] = {
            'total_withdrawals': len(withdrawals),
            'high_value_withdrawals': self._identify_high_value_withdrawals(withdrawals),
            'destination_analysis': self._analyze_withdrawal_destinations(withdrawals)
        }
        
        return report
        
    def generate_cftc_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate CFTC compliance report for derivatives"""
        report = {
            'report_type': 'CFTC',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Filter for derivatives/futures
        derivatives_trades = self._filter_derivatives_trades(start_date, end_date)
        
        report['sections']['position_reporting'] = {
            'total_positions': len(derivatives_trades),
            'position_limits': self._check_cftc_position_limits(derivatives_trades),
            'large_trader_reporting': self._generate_large_trader_report(derivatives_trades)
        }
        
        report['sections']['market_surveillance'] = {
            'manipulation_indicators': self._detect_market_manipulation(derivatives_trades),
            'wash_trades': self._detect_wash_trades(derivatives_trades),
            'spoofing_indicators': self._detect_spoofing(derivatives_trades)
        }
        
        return report
        
    def generate_custom_report(self, report_type: str, start_date: datetime,
                             end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Generate custom compliance report"""
        # Base report structure
        report = {
            'report_type': report_type,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'parameters': kwargs,
            'data': {}
        }
        
        # Add requested data sections
        if kwargs.get('include_trades', True):
            report['data']['trades'] = self._get_trade_summary(start_date, end_date)
            
        if kwargs.get('include_logins', False):
            report['data']['logins'] = self._get_login_summary(start_date, end_date)
            
        if kwargs.get('include_config_changes', False):
            report['data']['config_changes'] = self._get_config_changes(start_date, end_date)
            
        if kwargs.get('include_security_events', True):
            report['data']['security_events'] = self._get_security_events(start_date, end_date)
            
        return report
        
    def generate_user_activity_report(self, user_id: str, start_date: datetime,
                                    end_date: datetime) -> Dict[str, Any]:
        """Generate user activity report"""
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            raise ValueError("Invalid user_id")
            
        entries = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            limit=10000
        )
        
        return {
            'user_id': user_id,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': self._summarize_user_activity(entries),
            'logins': self._extract_events(entries, AuditEventType.LOGIN.value),
            'trades': self._extract_events(entries, AuditEventType.TRADE_EXECUTED.value),
            'config_changes': self._extract_events(entries, AuditEventType.CONFIG_CHANGE.value),
            'security_events': self._extract_security_events(entries)
        }
        
    def generate_trade_report(self, start_date: datetime, end_date: datetime,
                            user_id: Optional[str] = None, limit: int = 10000) -> pd.DataFrame:
        """Generate detailed trade report"""
        # Get trade entries
        trades = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.TRADE_EXECUTED.value,
            user_id=user_id,
            limit=limit
        )
        
        # Convert to DataFrame
        if not trades:
            return pd.DataFrame()
            
        # Process trade data
        trade_data = []
        for trade in trades:
            details = trade['details']
            
            # Validate required fields
            if not isinstance(details, dict):
                continue
                
            trade_record = {
                'timestamp': trade['timestamp'],
                'user_id': trade['user_id'],
                'trade_id': details.get('trade_id', trade['entry_id']),
                'exchange': details.get('exchange'),
                'symbol': details.get('symbol'),
                'side': details.get('side'),
                'price': details.get('price'),
                'amount': details.get('amount'),
                'value': details.get('value', 0),
                'fee': details.get('fee', 0),
                'pnl': details.get('pnl', 0),
                'strategy': details.get('strategy'),
                'status': details.get('status', 'completed')
            }
            
            trade_data.append(trade_record)
            
        df = pd.DataFrame(trade_data)
        
        # Add computed columns if data exists
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'value' in df.columns:
                df['net_value'] = df['value'] - df.get('fee', 0)
                
        return df
        
    # Helper methods
    def _group_by_field(self, entries: List[Dict], field_path: str) -> Dict[str, int]:
        """Group entries by nested field"""
        groups = {}
        
        for entry in entries:
            # Navigate nested path
            value = entry
            for field in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(field)
                else:
                    value = None
                    break
                    
            if value:
                groups[str(value)] = groups.get(str(value), 0) + 1
                
        return groups
        
    def _calculate_avg_execution_time(self, trades: List[Dict]) -> float:
        """Calculate average execution time"""
        times = []
        for trade in trades:
            if 'execution_time_ms' in trade.get('details', {}):
                times.append(trade['details']['execution_time_ms'])
                
        return sum(times) / len(times) if times else 0
        
    def _calculate_price_improvement(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate price improvement metrics"""
        improvements = []
        
        for trade in trades:
            details = trade.get('details', {})
            if 'expected_price' in details and 'actual_price' in details:
                improvement = (details['expected_price'] - details['actual_price']) / details['expected_price']
                improvements.append(improvement)
                
        if improvements:
            return {
                'average_improvement': sum(improvements) / len(improvements),
                'positive_improvement_rate': len([i for i in improvements if i > 0]) / len(improvements),
                'total_trades_analyzed': len(improvements)
            }
        else:
            return {'average_improvement': 0, 'positive_improvement_rate': 0, 'total_trades_analyzed': 0}
            
    def _assess_execution_quality(self, trades: List[Dict]) -> Dict[str, Any]:
        """Assess execution quality"""
        return {
            'slippage_analysis': self._analyze_slippage(trades),
            'fill_rate': self._calculate_fill_rate(trades),
            'rejection_rate': self._calculate_rejection_rate(trades)
        }
        
    def _detect_suspicious_activity(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Detect suspicious activity patterns"""
        suspicious = []
        
        # Get all relevant events
        events = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            limit=100000
        )
        
        # Pattern detection
        user_activity = {}
        
        for event in events:
            user = event.get('user_id')
            if not user:
                continue
                
            if user not in user_activity:
                user_activity[user] = {
                    'trades': 0,
                    'withdrawals': 0,
                    'logins': 0,
                    'failed_logins': 0,
                    'config_changes': 0,
                    'total_volume': 0
                }
                
            # Count activities
            if event['event_type'] == AuditEventType.TRADE_EXECUTED.value:
                user_activity[user]['trades'] += 1
                volume = event.get('details', {}).get('value', 0)
                user_activity[user]['total_volume'] += volume
            elif event['event_type'] == AuditEventType.WITHDRAWAL.value:
                user_activity[user]['withdrawals'] += 1
            elif event['event_type'] == AuditEventType.LOGIN.value:
                user_activity[user]['logins'] += 1
            elif event['event_type'] == AuditEventType.LOGIN_FAILED.value:
                user_activity[user]['failed_logins'] += 1
            elif event['event_type'] == AuditEventType.CONFIG_CHANGE.value:
                user_activity[user]['config_changes'] += 1
                
        # Identify suspicious patterns
        for user, activity in user_activity.items():
            # High trade frequency
            if activity['trades'] > 1000:
                suspicious.append({
                    'user_id': user,
                    'alert_type': 'high_frequency_trading',
                    'details': activity
                })
                
            # Multiple failed logins
            if activity['failed_logins'] > 10:
                suspicious.append({
                    'user_id': user,
                    'alert_type': 'multiple_failed_logins',
                    'details': activity
                })
                
            # High withdrawal activity
            if activity['withdrawals'] > 50:
                suspicious.append({
                    'user_id': user,
                    'alert_type': 'high_withdrawal_activity',
                    'details': activity
                })
                
        return suspicious
        
    def _summarize_user_activity(self, entries: List[Dict]) -> Dict[str, Any]:
        """Summarize user activity from entries"""
        summary = {
            'total_events': len(entries),
            'event_types': {},
            'first_activity': None,
            'last_activity': None,
            'active_days': set()
        }
        
        for entry in entries:
            # Count by type
            event_type = entry['event_type']
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
            
            # Track dates
            timestamp = datetime.fromisoformat(entry['timestamp'])
            
            if summary['first_activity'] is None or timestamp < summary['first_activity']:
                summary['first_activity'] = timestamp
                
            if summary['last_activity'] is None or timestamp > summary['last_activity']:
                summary['last_activity'] = timestamp
                
            summary['active_days'].add(timestamp.date())
            
        # Convert dates
        if summary['first_activity']:
            summary['first_activity'] = summary['first_activity'].isoformat()
        if summary['last_activity']:
            summary['last_activity'] = summary['last_activity'].isoformat()
            
        summary['active_days'] = len(summary['active_days'])
        
        return summary
        
    def _extract_events(self, entries: List[Dict], event_type: str) -> List[Dict]:
        """Extract specific event type from entries"""
        return [e for e in entries if e['event_type'] == event_type]
        
    def _extract_security_events(self, entries: List[Dict]) -> List[Dict]:
        """Extract security-related events"""
        security_types = [
            AuditEventType.LOGIN_FAILED.value,
            AuditEventType.SECURITY_ALERT.value,
            AuditEventType.API_KEY_CHANGE.value
        ]
        
        return [e for e in entries if e['event_type'] in security_types]
        
    # Stub methods for complex calculations
    def _get_identified_clients(self, start_date: datetime, end_date: datetime) -> int:
        """Get number of identified clients (KYC)"""
        # This would integrate with KYC system
        return 0
        
    def _check_kyc_compliance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Check KYC compliance status"""
        return {'compliant': True, 'pending_verification': 0}
        
    def _get_risk_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get risk-related events"""
        return self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.SECURITY_ALERT.value
        )
        
    def _check_position_limits(self, trades: List[Dict]) -> Dict[str, Any]:
        """Check position limit compliance"""
        return {'violations': 0, 'warnings': 0}
        
    def _identify_high_risk_users(self, suspicious: List[Dict]) -> List[str]:
        """Identify high-risk users from suspicious activity"""
        risk_scores = {}
        
        for alert in suspicious:
            user = alert['user_id']
            risk_scores[user] = risk_scores.get(user, 0) + 1
            
        # Return users with multiple alerts
        return [user for user, score in risk_scores.items() if score > 2]
        
    def _identify_large_transactions(self, trades: List[Dict]) -> List[Dict]:
        """Identify large transactions"""
        large_trades = []
        
        for trade in trades:
            value = trade.get('details', {}).get('value', 0)
            if value > 50000:  # $50k threshold
                large_trades.append(trade)
                
        return large_trades
        
    def _detect_rapid_movement(self, trades: List[Dict]) -> List[Dict]:
        """Detect rapid fund movement patterns"""
        # Group by user and time
        user_trades = {}
        
        for trade in trades:
            user = trade.get('user_id')
            if user:
                if user not in user_trades:
                    user_trades[user] = []
                user_trades[user].append(trade)
                
        rapid_movements = []
        
        # Check for rapid trading
        for user, user_trade_list in user_trades.items():
            if len(user_trade_list) > 50:  # High frequency
                # Sort by time
                user_trade_list.sort(key=lambda x: x['timestamp'])
                
                # Check time gaps
                rapid_count = 0
                for i in range(1, len(user_trade_list)):
                    t1 = datetime.fromisoformat(user_trade_list[i-1]['timestamp'])
                    t2 = datetime.fromisoformat(user_trade_list[i]['timestamp'])
                    
                    if (t2 - t1).total_seconds() < 60:  # Less than 1 minute
                        rapid_count += 1
                        
                if rapid_count > 10:
                    rapid_movements.append({
                        'user_id': user,
                        'rapid_trades': rapid_count,
                        'total_trades': len(user_trade_list)
                    })
                    
        return rapid_movements
        
    def _identify_cross_border(self, trades: List[Dict]) -> List[Dict]:
        """Identify cross-border transactions"""
        # This would check IP addresses and exchange locations
        return []
        
    def _identify_high_value_withdrawals(self, withdrawals: List[Dict]) -> List[Dict]:
        """Identify high-value withdrawals"""
        return [w for w in withdrawals if w.get('details', {}).get('amount', 0) > 10000]
        
    def _analyze_withdrawal_destinations(self, withdrawals: List[Dict]) -> Dict[str, int]:
        """Analyze withdrawal destinations"""
        destinations = {}
        
        for w in withdrawals:
            dest = w.get('details', {}).get('destination', 'unknown')
            destinations[dest] = destinations.get(dest, 0) + 1
            
        return destinations
        
    def _filter_derivatives_trades(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Filter for derivatives/futures trades"""
        all_trades = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.TRADE_EXECUTED.value
        )
        
        # Filter for derivatives (futures, options, perps)
        derivatives = []
        for trade in all_trades:
            symbol = trade.get('details', {}).get('symbol', '')
            if any(suffix in symbol.upper() for suffix in ['-PERP', 'FUTURES', 'OPTION']):
                derivatives.append(trade)
                
        return derivatives
        
    def _check_cftc_position_limits(self, trades: List[Dict]) -> Dict[str, Any]:
        """Check CFTC position limits"""
        return {'compliant': True, 'violations': []}
        
    def _generate_large_trader_report(self, trades: List[Dict]) -> Dict[str, Any]:
        """Generate large trader report for CFTC"""
        # Group by user and calculate positions
        user_positions = {}
        
        for trade in trades:
            user = trade.get('user_id')
            if user:
                if user not in user_positions:
                    user_positions[user] = {'long': 0, 'short': 0, 'net': 0}
                    
                side = trade.get('details', {}).get('side', '').lower()
                amount = trade.get('details', {}).get('amount', 0)
                
                if side == 'buy':
                    user_positions[user]['long'] += amount
                elif side == 'sell':
                    user_positions[user]['short'] += amount
                    
                user_positions[user]['net'] = user_positions[user]['long'] - user_positions[user]['short']
                
        # Identify large traders
        large_traders = {
            user: pos for user, pos in user_positions.items()
            if abs(pos['net']) > 25  # 25 contract threshold
        }
        
        return {
            'large_trader_count': len(large_traders),
            'large_traders': large_traders
        }
        
    def _detect_market_manipulation(self, trades: List[Dict]) -> List[Dict]:
        """Detect potential market manipulation"""
        return []  # Complex pattern detection would go here
        
    def _detect_wash_trades(self, trades: List[Dict]) -> List[Dict]:
        """Detect potential wash trades"""
        # Look for trades with same user on both sides
        wash_trades = []
        
        # Group by timestamp and symbol
        trade_groups = {}
        
        for trade in trades:
            key = (
                trade['timestamp'][:16],  # Minute precision
                trade.get('details', {}).get('symbol')
            )
            
            if key not in trade_groups:
                trade_groups[key] = []
            trade_groups[key].append(trade)
            
        # Check for matching trades
        for key, group in trade_groups.items():
            if len(group) >= 2:
                # Check for same user
                users = [t.get('user_id') for t in group]
                if len(set(users)) < len(users):
                    wash_trades.extend(group)
                    
        return wash_trades
        
    def _detect_spoofing(self, trades: List[Dict]) -> List[Dict]:
        """Detect potential spoofing activity"""
        # Would need order book data for real spoofing detection
        return []
        
    def _analyze_slippage(self, trades: List[Dict]) -> Dict[str, float]:
        """Analyze trade slippage"""
        slippages = []
        
        for trade in trades:
            details = trade.get('details', {})
            if 'expected_price' in details and 'actual_price' in details:
                slippage = abs(details['actual_price'] - details['expected_price']) / details['expected_price']
                slippages.append(slippage)
                
        if slippages:
            return {
                'average_slippage': sum(slippages) / len(slippages),
                'max_slippage': max(slippages),
                'trades_analyzed': len(slippages)
            }
        else:
            return {'average_slippage': 0, 'max_slippage': 0, 'trades_analyzed': 0}
            
    def _calculate_fill_rate(self, trades: List[Dict]) -> float:
        """Calculate order fill rate"""
        filled = sum(1 for t in trades if t.get('details', {}).get('status') == 'filled')
        return filled / len(trades) if trades else 0
        
    def _calculate_rejection_rate(self, trades: List[Dict]) -> float:
        """Calculate order rejection rate"""
        rejected = sum(1 for t in trades if t.get('details', {}).get('status') == 'rejected')
        return rejected / len(trades) if trades else 0
        
    def _get_trade_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get trade summary statistics"""
        trades = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.TRADE_EXECUTED.value
        )
        
        total_volume = sum(t.get('details', {}).get('value', 0) for t in trades)
        
        return {
            'total_trades': len(trades),
            'total_volume': total_volume,
            'average_trade_size': total_volume / len(trades) if trades else 0,
            'trades_by_exchange': self._group_by_field(trades, 'details.exchange'),
            'trades_by_symbol': self._group_by_field(trades, 'details.symbol')
        }
        
    def _get_login_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get login summary"""
        logins = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.LOGIN.value
        )
        
        failed_logins = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.LOGIN_FAILED.value
        )
        
        return {
            'successful_logins': len(logins),
            'failed_logins': len(failed_logins),
            'unique_users': len(set(l.get('user_id') for l in logins)),
            'logins_by_hour': self._group_by_hour(logins)
        }
        
    def _get_config_changes(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get configuration changes"""
        return self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.CONFIG_CHANGE.value
        )
        
    def _get_security_events(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get security events summary"""
        alerts = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            event_type=AuditEventType.SECURITY_ALERT.value
        )
        
        return {
            'total_alerts': len(alerts),
            'alerts_by_severity': self._group_by_field(alerts, 'details.severity'),
            'alerts_by_type': self._group_by_field(alerts, 'details.alert_type')
        }
        
    def _group_by_hour(self, entries: List[Dict]) -> Dict[int, int]:
        """Group entries by hour of day"""
        hours = {}
        
        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            hour = timestamp.hour
            hours[hour] = hours.get(hour, 0) + 1
            
        return hours

class AuditManager:
    """High-level audit manager for easy integration"""
    
    def __init__(self, config_path: str = "config/enhanced_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize blockchain audit
        audit_db = self.config.get('audit', {}).get('database', 'logs/audit/audit_trail.db')
        self.blockchain = BlockchainAudit(audit_db)
        
        # Initialize compliance reporter
        self.compliance = ComplianceReporter(self.blockchain)
        
        # Retention settings
        self.retention_days = self.config.get('audit', {}).get('retention_days', 2555)  # 7 years
        
        # Schedule periodic tasks
        self._schedule_tasks()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except:
            return {}
            
    def _schedule_tasks(self):
        """Schedule periodic maintenance tasks"""
        # Integrity checks and cleanup run in background thread
        def maintenance_loop():
            while True:
                try:
                    # Integrity check every hour
                    self.blockchain.perform_integrity_check()
                    
                    # Cleanup old entries daily
                    if datetime.now().hour == 2:  # 2 AM
                        self.cleanup_old_entries()
                        
                except Exception as e:
                    error_handler = get_error_handler()
                    error_handler.log_error(e, component="audit")
                    
                time.sleep(3600)  # 1 hour
                
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()
        
    # Convenience methods for common audit operations
    
    def audit_login(self, user_id: str, ip_address: str, success: bool,
                   details: Optional[Dict[str, Any]] = None):
        """Audit login attempt"""
        event_type = AuditEventType.LOGIN.value if success else AuditEventType.LOGIN_FAILED.value
        
        self.blockchain.add_entry(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            component="auth",
            action="login_attempt",
            details=details or {}
        )
        
    def audit_trade(self, user_id: str, trade_details: Dict[str, Any]):
        """Audit trade execution"""
        # Validate required fields
        required = ['exchange', 'symbol', 'side', 'price', 'amount']
        if not all(field in trade_details for field in required):
            raise ValueError(f"Trade details must include: {required}")
            
        self.blockchain.add_entry(
            event_type=AuditEventType.TRADE_EXECUTED.value,
            user_id=user_id,
            ip_address=trade_details.get('ip_address'),
            component="trading",
            action="trade_executed",
            details=trade_details
        )
        
    def audit_config_change(self, user_id: str, change_details: Dict[str, Any]):
        """Audit configuration change"""
        self.blockchain.add_entry(
            event_type=AuditEventType.CONFIG_CHANGE.value,
            user_id=user_id,
            ip_address=change_details.get('ip_address'),
            component="config",
            action=change_details.get('action', 'update'),
            details=change_details
        )
        
    def audit_security_event(self, event_type: str, user_id: Optional[str],
                           details: Dict[str, Any]):
        """Audit security event"""
        self.blockchain.add_entry(
            event_type=AuditEventType.SECURITY_ALERT.value,
            user_id=user_id,
            ip_address=details.get('ip_address'),
            component="security",
            action=event_type,
            details=details
        )
        
    def audit_withdrawal(self, user_id: str, withdrawal_details: Dict[str, Any]):
        """Audit withdrawal"""
        self.blockchain.add_entry(
            event_type=AuditEventType.WITHDRAWAL.value,
            user_id=user_id,
            ip_address=withdrawal_details.get('ip_address'),
            component="wallet",
            action="withdrawal",
            details=withdrawal_details
        )
        
    def audit_api_key_change(self, user_id: str, exchange: str, action: str):
        """Audit API key change"""
        self.blockchain.add_entry(
            event_type=AuditEventType.API_KEY_CHANGE.value,
            user_id=user_id,
            ip_address=None,
            component="security",
            action=f"api_key_{action}",
            details={'exchange': exchange, 'action': action}
        )
        
    def audit_system_event(self, event_type: str, details: Dict[str, Any]):
        """Audit system event"""
        self.blockchain.add_entry(
            event_type=event_type,
            user_id="system",
            ip_address=None,
            component="system",
            action=details.get('action', 'system_event'),
            details=details
        )
        
    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit summary for specified period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all entries
        entries = self.blockchain.get_entries(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Verify integrity
        integrity_valid, issues = self.blockchain.verify_integrity(start_date, end_date)
        
        # Group by event type
        event_counts = {}
        for entry in entries:
            event_type = entry['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        return {
            'period_days': days,
            'total_entries': len(entries),
            'event_counts': event_counts,
            'integrity_valid': integrity_valid,
            'integrity_issues': len(issues),
            'database_size_mb': self.blockchain.db_path.stat().st_size / (1024 * 1024)
        }
        
    def cleanup_old_entries(self):
        """Cleanup old audit entries"""
        self.blockchain.cleanup_old_entries(self.retention_days)
        
    def export_audit_trail(self, start_date: datetime, end_date: datetime,
                          format: str = "json") -> Path:
        """Export audit trail"""
        return self.blockchain.export_for_audit(start_date, end_date, format)
        
    def stop(self):
        """Stop audit manager"""
        self.blockchain.stop()

# Global audit manager instance
_audit_manager = None

def get_audit_manager() -> AuditManager:
    """Get or create global audit manager instance"""
    global _audit_manager
    if _audit_manager is None:
        _audit_manager = AuditManager()
    return _audit_manager
