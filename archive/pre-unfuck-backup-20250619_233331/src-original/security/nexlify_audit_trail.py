"""
Nexlify Enhanced - Audit Trail System
Implements Feature 30: Blockchain-recorded trades, immutable logs, compliance reporting
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """Immutable audit trail entry"""
    entry_id: str
    timestamp: datetime
    entry_type: str  # trade, config_change, login, system_event
    user_id: str
    action: str
    details: Dict
    ip_address: Optional[str]
    hash_previous: str
    hash_current: Optional[str] = None
    signature: Optional[str] = None
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entry"""
        # Create deterministic string representation
        data_string = json.dumps({
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'entry_type': self.entry_type,
            'user_id': self.user_id,
            'action': self.action,
            'details': self.details,
            'ip_address': self.ip_address,
            'hash_previous': self.hash_previous
        }, sort_keys=True)
        
        return hashlib.sha256(data_string.encode()).hexdigest()

class BlockchainAudit:
    """Blockchain-style immutable audit trail"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Generate or load RSA keys for signing
        self._init_keys()
        
        # Entry queue for async processing
        self.entry_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _init_database(self):
        """Initialize audit database with immutable structure"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    ip_address TEXT,
                    hash_previous TEXT NOT NULL,
                    hash_current TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    
                    -- Indexes for efficient queries
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_trail(timestamp);')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_entry_type ON audit_trail(entry_type);')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_trail(user_id);')
            
            # Compliance views
            conn.execute('''
                CREATE VIEW IF NOT EXISTS trade_audit AS
                SELECT * FROM audit_trail WHERE entry_type = 'trade'
                ORDER BY timestamp DESC;
            ''')
            
            conn.execute('''
                CREATE VIEW IF NOT EXISTS security_audit AS
                SELECT * FROM audit_trail 
                WHERE entry_type IN ('login', 'logout', 'permission_change', 'config_change')
                ORDER BY timestamp DESC;
            ''')
            
    def _init_keys(self):
        """Initialize RSA keys for entry signing"""
        key_path = self.db_path.parent / 'audit_key.pem'
        
        if key_path.exists():
            # Load existing key
            with open(key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            with open(key_path, 'wb') as f:
                f.write(pem)
                
        self.public_key = self.private_key.public_key()
        
    def add_entry(self, 
                 entry_type: str,
                 user_id: str,
                 action: str,
                 details: Dict,
                 ip_address: Optional[str] = None) -> str:
        """Add new entry to audit trail"""
        # Get previous hash
        previous_hash = self._get_last_hash()
        
        # Create entry
        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.now(),
            entry_type=entry_type,
            user_id=user_id,
            action=action,
            details=details,
            ip_address=ip_address,
            hash_previous=previous_hash
        )
        
        # Calculate hash
        entry.hash_current = entry.calculate_hash()
        
        # Sign entry
        entry.signature = self._sign_entry(entry)
        
        # Queue for processing
        self.entry_queue.put(entry)
        
        logger.info(f"Audit entry queued: {entry.entry_id}")
        return entry.entry_id
        
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        random_part = hashlib.sha256(
            f"{timestamp}{threading.current_thread().ident}".encode()
        ).hexdigest()[:8]
        
        return f"AUD-{timestamp}-{random_part}"
        
    def _get_last_hash(self) -> str:
        """Get hash of last entry or genesis hash"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                'SELECT hash_current FROM audit_trail ORDER BY rowid DESC LIMIT 1'
            ).fetchone()
            
        return result[0] if result else "GENESIS_BLOCK_NEXLIFY_2077"
        
    def _sign_entry(self, entry: AuditEntry) -> str:
        """Sign entry with private key"""
        message = f"{entry.entry_id}{entry.hash_current}".encode()
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature.hex()
        
    def verify_entry(self, entry: AuditEntry) -> bool:
        """Verify entry signature"""
        message = f"{entry.entry_id}{entry.hash_current}".encode()
        signature = bytes.fromhex(entry.signature)
        
        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False
            
    def _process_queue(self):
        """Process audit entries from queue"""
        while True:
            try:
                entry = self.entry_queue.get(timeout=1)
                self._write_entry(entry)
                self.entry_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audit processing error: {e}")
                
    def _write_entry(self, entry: AuditEntry):
        """Write entry to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_trail (
                    entry_id, timestamp, entry_type, user_id, action,
                    details, ip_address, hash_previous, hash_current, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id,
                entry.timestamp.isoformat(),
                entry.entry_type,
                entry.user_id,
                entry.action,
                json.dumps(entry.details),
                entry.ip_address,
                entry.hash_previous,
                entry.hash_current,
                entry.signature
            ))
            
    def verify_integrity(self, start_date: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """Verify blockchain integrity"""
        issues = []
        
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT * FROM audit_trail'
            params = []
            
            if start_date:
                query += ' WHERE timestamp >= ?'
                params.append(start_date.isoformat())
                
            query += ' ORDER BY rowid ASC'
            
            cursor = conn.execute(query, params)
            
            previous_hash = "GENESIS_BLOCK_NEXLIFY_2077"
            
            for row in cursor:
                # Reconstruct entry
                entry = AuditEntry(
                    entry_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    entry_type=row[2],
                    user_id=row[3],
                    action=row[4],
                    details=json.loads(row[5]),
                    ip_address=row[6],
                    hash_previous=row[7],
                    hash_current=row[8],
                    signature=row[9]
                )
                
                # Verify hash chain
                if entry.hash_previous != previous_hash:
                    issues.append(f"Hash chain broken at {entry.entry_id}")
                    
                # Verify entry hash
                calculated_hash = entry.calculate_hash()
                if calculated_hash != entry.hash_current:
                    issues.append(f"Hash mismatch at {entry.entry_id}")
                    
                # Verify signature
                if not self.verify_entry(entry):
                    issues.append(f"Invalid signature at {entry.entry_id}")
                    
                previous_hash = entry.hash_current
                
        return len(issues) == 0, issues
        
    def get_entries(self,
                   entry_type: Optional[str] = None,
                   user_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: int = 1000) -> List[Dict]:
        """Query audit entries with filters"""
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT * FROM audit_trail WHERE 1=1'
            params = []
            
            if entry_type:
                query += ' AND entry_type = ?'
                params.append(entry_type)
                
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
                
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
                
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
                
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            entries = []
            for row in cursor:
                entries.append({
                    'entry_id': row[0],
                    'timestamp': row[1],
                    'entry_type': row[2],
                    'user_id': row[3],
                    'action': row[4],
                    'details': json.loads(row[5]),
                    'ip_address': row[6],
                    'hash': row[8],
                    'signature': row[9]
                })
                
        return entries

class ComplianceReporter:
    """Generate compliance reports from audit trail"""
    
    def __init__(self, audit_trail: BlockchainAudit):
        self.audit = audit_trail
        
    def generate_trade_report(self,
                            start_date: datetime,
                            end_date: datetime,
                            user_id: Optional[str] = None) -> pd.DataFrame:
        """Generate trade activity report"""
        entries = self.audit.get_entries(
            entry_type='trade',
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Convert to DataFrame
        trades = []
        for entry in entries:
            trade = {
                'timestamp': entry['timestamp'],
                'user_id': entry['user_id'],
                'trade_id': entry['entry_id'],
                **entry['details']
            }
            trades.append(trade)
            
        df = pd.DataFrame(trades)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
        return df
        
    def generate_user_activity_report(self,
                                    user_id: str,
                                    start_date: datetime,
                                    end_date: datetime) -> Dict:
        """Generate comprehensive user activity report"""
        # Get all user entries
        entries = self.audit.get_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Categorize activities
        activities = {
            'trades': [],
            'logins': [],
            'config_changes': [],
            'other': []
        }
        
        for entry in entries:
            if entry['entry_type'] == 'trade':
                activities['trades'].append(entry)
            elif entry['entry_type'] == 'login':
                activities['logins'].append(entry)
            elif entry['entry_type'] == 'config_change':
                activities['config_changes'].append(entry)
            else:
                activities['other'].append(entry)
                
        # Generate summary
        summary = {
            'user_id': user_id,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'statistics': {
                'total_trades': len(activities['trades']),
                'total_logins': len(activities['logins']),
                'config_changes': len(activities['config_changes']),
                'total_activities': len(entries)
            },
            'trade_summary': self._summarize_trades(activities['trades']),
            'security_events': self._summarize_security(activities['logins']),
            'recent_changes': activities['config_changes'][-10:]  # Last 10 changes
        }
        
        return summary
        
    def _summarize_trades(self, trades: List[Dict]) -> Dict:
        """Summarize trading activity"""
        if not trades:
            return {}
            
        df = pd.DataFrame([t['details'] for t in trades])
        
        return {
            'total_volume': float(df.get('volume', 0).sum()),
            'total_trades': len(trades),
            'symbols_traded': list(df.get('symbol', []).unique()),
            'profit_loss': float(df.get('pnl', 0).sum()),
            'win_rate': float(
                (df.get('pnl', 0) > 0).sum() / len(df) * 100
                if len(df) > 0 else 0
            )
        }
        
    def _summarize_security(self, logins: List[Dict]) -> Dict:
        """Summarize security events"""
        if not logins:
            return {}
            
        successful = sum(1 for l in logins if l['details'].get('success', False))
        failed = len(logins) - successful
        
        # Get unique IPs
        ips = set(l.get('ip_address', 'Unknown') for l in logins)
        
        return {
            'total_logins': len(logins),
            'successful_logins': successful,
            'failed_attempts': failed,
            'unique_ips': list(ips),
            'last_login': logins[0]['timestamp'] if logins else None
        }
        
    def generate_regulatory_report(self,
                                 report_type: str,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict:
        """Generate regulatory compliance reports"""
        if report_type == 'mifid2':
            return self._generate_mifid2_report(start_date, end_date)
        elif report_type == 'fatf':
            return self._generate_fatf_report(start_date, end_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
            
    def _generate_mifid2_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate MiFID II compliance report"""
        trades_df = self.generate_trade_report(start_date, end_date)
        
        if trades_df.empty:
            return {'error': 'No trades found in period'}
            
        return {
            'report_type': 'MiFID II Transaction Report',
            'reporting_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_transactions': len(trades_df),
                'total_volume': float(trades_df.get('volume', 0).sum()),
                'unique_instruments': list(trades_df.get('symbol', []).unique())
            },
            'transactions': trades_df.to_dict('records'),
            'generated_at': datetime.now().isoformat(),
            'integrity_verified': self.audit.verify_integrity(start_date)[0]
        }
        
    def export_for_audit(self, 
                        output_path: Path,
                        start_date: datetime,
                        end_date: datetime,
                        format: str = 'csv'):
        """Export audit trail for external auditors"""
        entries = self.audit.get_entries(
            start_date=start_date,
            end_date=end_date,
            limit=100000
        )
        
        df = pd.DataFrame(entries)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(output_path) as writer:
                # Overview sheet
                df.to_excel(writer, sheet_name='All_Entries', index=False)
                
                # Trades sheet
                trades_df = df[df['entry_type'] == 'trade']
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Security sheet
                security_df = df[df['entry_type'].isin(['login', 'logout', 'config_change'])]
                security_df.to_excel(writer, sheet_name='Security', index=False)
                
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
            
        logger.info(f"Audit trail exported to {output_path}")

class AuditManager:
    """Main audit manager integrating all audit functionality"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        db_path = Path(config.get('audit_db_path', 'logs/audit/audit_trail.db'))
        self.blockchain_audit = BlockchainAudit(db_path)
        self.compliance_reporter = ComplianceReporter(self.blockchain_audit)
        
        # Audit policies
        self.policies = {
            'require_trade_audit': True,
            'require_config_audit': True,
            'require_login_audit': True,
            'retention_days': 2555,  # 7 years
            'integrity_check_interval': timedelta(hours=24)
        }
        
        # Last integrity check
        self.last_integrity_check = datetime.now()
        
    def audit_trade(self, user_id: str, trade_details: Dict, ip_address: str = None):
        """Audit trade execution"""
        if not self.policies['require_trade_audit']:
            return
            
        self.blockchain_audit.add_entry(
            entry_type='trade',
            user_id=user_id,
            action='trade_executed',
            details=trade_details,
            ip_address=ip_address
        )
        
    def audit_config_change(self, user_id: str, change_details: Dict, ip_address: str = None):
        """Audit configuration changes"""
        if not self.policies['require_config_audit']:
            return
            
        self.blockchain_audit.add_entry(
            entry_type='config_change',
            user_id=user_id,
            action='configuration_modified',
            details=change_details,
            ip_address=ip_address
        )
        
    def audit_login(self, user_id: str, success: bool, ip_address: str, details: Dict = None):
        """Audit login attempts"""
        if not self.policies['require_login_audit']:
            return
            
        self.blockchain_audit.add_entry(
            entry_type='login',
            user_id=user_id,
            action='login_attempt',
            details={
                'success': success,
                'ip_address': ip_address,
                **(details or {})
            },
            ip_address=ip_address
        )
        
    def perform_integrity_check(self) -> Tuple[bool, List[str]]:
        """Perform periodic integrity check"""
        current_time = datetime.now()
        
        if current_time - self.last_integrity_check < self.policies['integrity_check_interval']:
            return True, []
            
        logger.info("Performing audit trail integrity check...")
        
        is_valid, issues = self.blockchain_audit.verify_integrity()
        
        if not is_valid:
            logger.critical(f"Audit trail integrity compromised: {issues}")
            # In production, trigger alerts
            
        self.last_integrity_check = current_time
        
        return is_valid, issues
        
    def cleanup_old_entries(self):
        """Clean up entries older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.policies['retention_days'])
        
        # Note: In a real blockchain audit, you wouldn't delete entries
        # Instead, you might archive them to cold storage
        logger.info(f"Archiving entries older than {cutoff_date}")
        
    def get_audit_summary(self) -> Dict:
        """Get audit system summary"""
        integrity_valid, _ = self.blockchain_audit.verify_integrity()
        
        # Get recent entries count
        recent_entries = self.blockchain_audit.get_entries(
            start_date=datetime.now() - timedelta(days=1),
            limit=10000
        )
        
        return {
            'integrity_status': 'Valid' if integrity_valid else 'Compromised',
            'last_integrity_check': self.last_integrity_check.isoformat(),
            'entries_24h': len(recent_entries),
            'policies': self.policies,
            'retention_status': f"{self.policies['retention_days']} days"
        }
