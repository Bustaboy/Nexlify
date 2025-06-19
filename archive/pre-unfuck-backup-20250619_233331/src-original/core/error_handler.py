"""
Night City Error Handler - Enhanced for v2.0.8
Handles all system errors with proper logging, crash reporting, and notifications
"""

import sys
import logging
import traceback
import json
import asyncio
import psutil
import threading
import gc
import signal
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from functools import wraps
import time
import aiohttp
from asyncio import iscoroutinefunction
from sklearn.cluster import DBSCAN
import numpy as np

@dataclass
class ErrorInstance:
    """Represents a single error occurrence"""
    timestamp: datetime
    error_type: str
    message: str
    traceback: str
    component: str
    severity: str
    context: Dict[str, Any]
    hash: str

@dataclass
class ErrorPattern:
    """Represents a pattern of related errors"""
    pattern_id: str
    error_hashes: Set[str]
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    components: Set[str]
    impact_score: float

class ErrorCorrelationAnalyzer:
    """Analyzes error patterns and correlations"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.error_vectors = {}
        self.patterns = {}
        
    def vectorize_error(self, error: ErrorInstance) -> np.ndarray:
        """Convert error to vector for clustering"""
        # Simple vectorization based on error features
        features = [
            hash(error.error_type) % 1000,
            hash(error.component) % 1000,
            len(error.message),
            len(error.traceback),
            error.timestamp.timestamp() % 3600,  # Time of day component
        ]
        return np.array(features)
    
    def find_correlations(self, errors: List[ErrorInstance]) -> List[ErrorPattern]:
        """Find correlated error patterns using clustering"""
        if len(errors) < 2:
            return []
            
        # Vectorize errors
        vectors = np.array([self.vectorize_error(e) for e in errors])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=100, min_samples=2).fit(vectors)
        
        # Group errors by cluster
        patterns = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise
                patterns[label].append(errors[idx])
        
        # Create ErrorPattern objects
        result = []
        for pattern_id, error_group in patterns.items():
            pattern = ErrorPattern(
                pattern_id=f"pattern_{pattern_id}_{int(time.time())}",
                error_hashes={e.hash for e in error_group},
                occurrences=len(error_group),
                first_seen=min(e.timestamp for e in error_group),
                last_seen=max(e.timestamp for e in error_group),
                components={e.component for e in error_group},
                impact_score=self._calculate_impact_score(error_group)
            )
            result.append(pattern)
            
        return result
    
    def _calculate_impact_score(self, errors: List[ErrorInstance]) -> float:
        """Calculate impact score for error pattern"""
        severity_weights = {
            'fatal': 10.0,
            'critical': 5.0,
            'error': 2.0,
            'warning': 1.0
        }
        
        total_score = sum(severity_weights.get(e.severity, 1.0) for e in errors)
        component_multiplier = len({e.component for e in errors})  # More components = higher impact
        frequency_factor = min(len(errors) / 10, 2.0)  # Cap at 2x
        
        return total_score * component_multiplier * frequency_factor

class ErrorRecoveryManager:
    """Manages error recovery workflows"""
    
    def __init__(self):
        self.recovery_strategies = {
            'telegram_notification': self._retry_telegram,
            'audit_write': self._retry_audit_write,
            'api_connection': self._retry_api_connection,
            'file_write': self._retry_file_write,
            'database_operation': self._retry_database,
        }
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        
    async def attempt_recovery(self, error_type: str, operation: callable, *args, **kwargs):
        """Attempt to recover from error with exponential backoff"""
        strategy = self.recovery_strategies.get(error_type)
        if not strategy:
            return None
            
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                return await strategy(operation, *args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                continue
                
    async def _retry_telegram(self, operation, *args, **kwargs):
        """Retry Telegram notification with rate limit handling"""
        try:
            return await operation(*args, **kwargs)
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Rate limited
                retry_after = int(e.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                return await operation(*args, **kwargs)
            raise
            
    async def _retry_audit_write(self, operation, *args, **kwargs):
        """Retry audit write with transaction handling"""
        # Implementation depends on audit system
        return await operation(*args, **kwargs)
        
    async def _retry_api_connection(self, operation, *args, **kwargs):
        """Retry API connection with backoff"""
        return await operation(*args, **kwargs)
        
    async def _retry_file_write(self, operation, *args, **kwargs):
        """Retry file write with disk space check"""
        # Check disk space first
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 95:
            raise Exception("Insufficient disk space")
        return await operation(*args, **kwargs)
        
    async def _retry_database(self, operation, *args, **kwargs):
        """Retry database operation with connection pooling"""
        return await operation(*args, **kwargs)

class NightCityErrorHandler:
    def __init__(self, config_path: str = "config/enhanced_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize paths
        self.error_log_path = Path(self.config.get('logging', {}).get('error_log', 'logs/errors.log'))
        self.crash_report_path = Path(self.config.get('logging', {}).get('crash_reports', 'logs/crash_reports'))
        
        # Create directories
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.crash_report_path.mkdir(parents=True, exist_ok=True)
        
        # Error tracking with deduplication
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=10000)  # Increased buffer
        self.error_patterns = {}
        self.suppressed_components = set()
        
        # Initialize components
        self.correlation_analyzer = ErrorCorrelationAnalyzer()
        self.recovery_manager = ErrorRecoveryManager()
        
        # Setup
        self._setup_error_logger()
        self._install_exception_hooks()
        
        # Emergency stop threshold
        self.error_rate_threshold = self.config.get('error_handling', {}).get('emergency_stop_threshold', 100)
        self.error_window = timedelta(minutes=5)
        
    def _load_config(self) -> dict:
        """Load configuration from enhanced_config.json with migration support"""
        try:
            # Try enhanced config first
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            
            # Fall back to old neural_config.json
            old_config_path = Path("config/neural_config.json")
            if old_config_path.exists():
                with open(old_config_path, 'r') as f:
                    old_config = json.load(f)
                    # Migrate to new structure
                    return self._migrate_config(old_config)
                    
            # Return defaults if no config found
            return self._get_default_config()
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _migrate_config(self, old_config: dict) -> dict:
        """Migrate from neural_config.json to enhanced_config.json structure"""
        return {
            'version': '2.0.8',
            'error_handling': {
                'telegram_notifications': old_config.get('telegram_notifications', False),
                'telegram_bot_token': old_config.get('telegram_bot_token', ''),
                'telegram_chat_id': old_config.get('telegram_chat_id', ''),
                'email_notifications': old_config.get('email_notifications', False),
                'emergency_contact': old_config.get('emergency_contact', ''),
                'emergency_stop_threshold': 100,
                'error_deduplication_window': 300,  # 5 minutes
                'suppress_non_critical': True
            },
            'logging': {
                'level': old_config.get('log_level', 'INFO'),
                'error_log': 'logs/errors.log',
                'crash_reports': 'logs/crash_reports',
                'max_log_size': 100,  # MB
                'log_rotation_count': 10
            }
        }
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'version': '2.0.8',
            'error_handling': {
                'telegram_notifications': False,
                'telegram_bot_token': '',
                'telegram_chat_id': '',
                'email_notifications': False,
                'emergency_contact': '',
                'emergency_stop_threshold': 100,
                'error_deduplication_window': 300,
                'suppress_non_critical': True
            },
            'logging': {
                'level': 'INFO',
                'error_log': 'logs/errors.log',
                'crash_reports': 'logs/crash_reports',
                'max_log_size': 100,
                'log_rotation_count': 10
            }
        }
    
    def _setup_error_logger(self):
        """Setup error logger with rotation and proper formatting"""
        from logging.handlers import RotatingFileHandler
        
        self.logger = logging.getLogger('NightCityErrors')
        self.logger.setLevel(self.config.get('logging', {}).get('level', 'INFO'))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create rotating file handler
        max_bytes = self.config.get('logging', {}).get('max_log_size', 100) * 1024 * 1024
        backup_count = self.config.get('logging', {}).get('log_rotation_count', 10)
        
        handler = RotatingFileHandler(
            self.error_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _install_exception_hooks(self):
        """Install exception hooks for unhandled errors"""
        # Preserve existing hooks
        self._original_excepthook = sys.excepthook
        self._original_thread_excepthook = threading.excepthook if hasattr(threading, 'excepthook') else None
        
        sys.excepthook = self._handle_exception
        if hasattr(threading, 'excepthook'):
            threading.excepthook = self._handle_thread_exception
    
    def _generate_error_hash(self, error_type: str, message: str, traceback_str: str) -> str:
        """Generate hash for error deduplication"""
        # Hash based on error type and key parts of message/traceback
        key_parts = [
            error_type,
            # Extract key parts of message (remove variable parts like IDs, timestamps)
            ''.join(c for c in message if c.isalpha() or c.isspace())[:100],
            # Extract file and line info from traceback
            ''.join(line for line in traceback_str.split('\n') if 'File' in line)[:200]
        ]
        
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()
    
    def _should_deduplicate(self, error_hash: str) -> bool:
        """Check if error should be deduplicated"""
        window = self.config.get('error_handling', {}).get('error_deduplication_window', 300)
        cutoff_time = datetime.now() - timedelta(seconds=window)
        
        # Check recent errors
        for error in self.error_history:
            if error.hash == error_hash and error.timestamp > cutoff_time:
                return True
        return False
    
    def _calculate_error_impact(self, error: ErrorInstance) -> float:
        """Calculate impact score for single error"""
        component_weights = {
            'trading': 10.0,
            'neural_net': 10.0,
            'exchange': 8.0,
            'security': 8.0,
            'audit': 6.0,
            'prediction': 5.0,
            'strategy': 5.0,
            'dex': 5.0,
            'mobile': 3.0,
            'gui': 2.0,
            'animation': 1.0,
            'sound': 0.5
        }
        
        severity_weights = {
            'fatal': 10.0,
            'critical': 5.0,
            'error': 2.0,
            'warning': 1.0
        }
        
        # Get base weights
        component_weight = component_weights.get(error.component.lower(), 2.0)
        severity_weight = severity_weights.get(error.severity.lower(), 1.0)
        
        # Check for specific high-impact keywords
        high_impact_keywords = ['trade', 'fund', 'security', 'breach', 'loss', 'crash', 'fail']
        keyword_multiplier = 1.5 if any(kw in error.message.lower() for kw in high_impact_keywords) else 1.0
        
        return component_weight * severity_weight * keyword_multiplier
    
    def _check_error_rate_threshold(self):
        """Check if error rate exceeds emergency stop threshold"""
        if not self.error_rate_threshold:
            return
            
        cutoff_time = datetime.now() - self.error_window
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Count critical errors only
        critical_count = sum(1 for e in recent_errors if e.severity in ['fatal', 'critical'])
        
        if critical_count >= self.error_rate_threshold:
            # Create emergency stop file
            stop_file = Path("EMERGENCY_STOP_ACTIVE")
            stop_file.write_text(f"Emergency stop triggered at {datetime.now()} due to {critical_count} critical errors")
            
            # Log fatal error
            self.log_fatal_error(
                Exception("Emergency stop triggered"),
                context={'critical_errors': critical_count, 'threshold': self.error_rate_threshold}
            )
    
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        # Log the error
        self.log_fatal_error(
            exc_value,
            context={
                'exc_type': str(exc_type),
                'uncaught': True,
                'thread': threading.current_thread().name
            }
        )
        
        # Call original hook if exists
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _handle_thread_exception(self, args):
        """Handle uncaught thread exceptions with full context"""
        # Extract thread context
        thread_context = {
            'thread_name': args.thread.name if hasattr(args, 'thread') else 'Unknown',
            'thread_id': args.thread.ident if hasattr(args, 'thread') else None,
            'thread_daemon': args.thread.daemon if hasattr(args, 'thread') else None,
            'exc_type': str(args.exc_type),
            'uncaught_thread': True
        }
        
        # Capture thread-local variables if possible
        try:
            if hasattr(args.thread, '__dict__'):
                thread_locals = {k: str(v)[:100] for k, v in args.thread.__dict__.items() 
                               if not k.startswith('_')}
                thread_context['thread_locals'] = thread_locals
        except:
            pass
        
        self.log_fatal_error(args.exc_value, context=thread_context)
        
        # Call original hook if exists
        if self._original_thread_excepthook:
            self._original_thread_excepthook(args)
    
    def log_error(self, error: Exception, component: str = "unknown", 
                  severity: str = "error", context: Optional[Dict[str, Any]] = None):
        """Log error with deduplication and impact scoring"""
        context = context or {}
        
        # Create error instance
        error_instance = ErrorInstance(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            component=component,
            severity=severity,
            context=context,
            hash=self._generate_error_hash(
                type(error).__name__,
                str(error),
                traceback.format_exc()
            )
        )
        
        # Check deduplication
        if self._should_deduplicate(error_instance.hash):
            self.error_counts[error_instance.hash] += 1
            return
        
        # Check if component is suppressed for non-critical errors
        if (self.config.get('error_handling', {}).get('suppress_non_critical', True) and
            severity in ['warning'] and 
            component.lower() in self.suppressed_components):
            return
        
        # Add to history
        self.error_history.append(error_instance)
        self.error_counts[error_instance.hash] += 1
        
        # Calculate impact
        impact_score = self._calculate_error_impact(error_instance)
        
        # Log with proper context
        extra = {
            'component': component,
            'impact_score': impact_score,
            'occurrence_count': self.error_counts[error_instance.hash]
        }
        
        if severity == 'warning':
            self.logger.warning(f"{error}", extra=extra)
        elif severity == 'critical':
            self.logger.critical(f"{error}", extra=extra)
        elif severity == 'fatal':
            self.logger.critical(f"FATAL: {error}", extra=extra)
        else:
            self.logger.error(f"{error}", extra=extra)
        
        # Check error patterns
        self._analyze_error_patterns()
        
        # Check threshold
        self._check_error_rate_threshold()
        
        # Send notifications for high-impact errors
        if impact_score > 5.0 and severity in ['critical', 'fatal']:
            asyncio.create_task(self._notify_critical_error(error_instance))
    
    def _analyze_error_patterns(self):
        """Analyze errors for patterns and correlations"""
        recent_errors = list(self.error_history)[-100:]  # Last 100 errors
        if len(recent_errors) < 5:
            return
            
        patterns = self.correlation_analyzer.find_correlations(recent_errors)
        
        # Log significant patterns
        for pattern in patterns:
            if pattern.impact_score > 10.0:
                self.logger.warning(
                    f"Error pattern detected: {pattern.pattern_id} "
                    f"affecting {pattern.components} with impact score {pattern.impact_score:.2f}"
                )
        
        self.error_patterns = {p.pattern_id: p for p in patterns}
    
    def log_warning(self, message: str, component: str = "unknown", 
                    context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.log_error(
            Exception(message),
            component=component,
            severity='warning',
            context=context
        )
    
    def log_critical_error(self, error: Exception, component: str = "unknown",
                          context: Optional[Dict[str, Any]] = None):
        """Log critical error"""
        self.log_error(error, component, 'critical', context)
        
    def log_fatal_error(self, error: Exception, component: str = "unknown",
                       context: Optional[Dict[str, Any]] = None):
        """Log fatal error and generate crash report"""
        self.log_error(error, component, 'fatal', context)
        
        # Generate crash report
        crash_report = self._generate_crash_report(error, component, context)
        
        # Save crash report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.crash_report_path / f"crash_report_{timestamp}.json"
        
        try:
            # Check disk space first
            disk_usage = psutil.disk_usage(str(self.crash_report_path))
            if disk_usage.percent > 95:
                self.logger.error("Insufficient disk space for crash report")
                return
                
            with open(report_file, 'w') as f:
                json.dump(crash_report, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save crash report: {e}")
        
        # Notify
        asyncio.create_task(self._notify_fatal_crash(error, report_file))
    
    def _generate_crash_report(self, error: Exception, component: str,
                              context: Optional[Dict[str, Any]] = None) -> dict:
        """Generate comprehensive crash report with all diagnostics"""
        context = context or {}
        
        # System information
        system_info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': {str(p.mountpoint): p.percent 
                          for p in psutil.disk_partitions() if p.fstype},
        }
        
        # Add load averages
        try:
            system_info['load_average'] = psutil.getloadavg()
        except:
            system_info['load_average'] = [0, 0, 0]
        
        # CPU stats
        try:
            system_info['cpu_percent'] = psutil.cpu_percent(interval=0.1, percpu=True)
            system_info['cpu_stats'] = psutil.cpu_stats()._asdict()
        except:
            pass
        
        # Disk I/O stats
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                system_info['disk_io'] = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
        except:
            pass
        
        # Network interface status
        try:
            net_stats = {}
            for interface, stats in psutil.net_if_stats().items():
                net_stats[interface] = {
                    'is_up': stats.isup,
                    'speed': stats.speed,
                    'mtu': stats.mtu
                }
            system_info['network_interfaces'] = net_stats
        except:
            pass
        
        # Process information
        process_info = {
            'pid': os.getpid(),
            'create_time': datetime.fromtimestamp(psutil.Process().create_time()),
            'num_threads': threading.active_count(),
            'threads': [t.name for t in threading.enumerate()],
        }
        
        # Add open file descriptors
        try:
            process = psutil.Process()
            process_info['open_files'] = len(process.open_files())
            process_info['connections'] = len(process.connections())
            process_info['num_fds'] = process.num_fds() if hasattr(process, 'num_fds') else None
        except:
            pass
        
        # Thread pool executor stats
        try:
            executor_stats = {}
            for name, obj in globals().items():
                if isinstance(obj, ThreadPoolExecutor):
                    executor_stats[name] = {
                        '_threads': len(obj._threads),
                        '_shutdown': obj._shutdown
                    }
            if executor_stats:
                process_info['thread_pools'] = executor_stats
        except:
            pass
        
        # Python garbage collector state
        gc_info = {
            'counts': gc.get_count(),
            'thresholds': gc.get_threshold(),
            'is_enabled': gc.isenabled()
        }
        
        # Signal handler states
        signal_info = {}
        for sig in ['SIGINT', 'SIGTERM', 'SIGUSR1', 'SIGUSR2']:
            if hasattr(signal, sig):
                sig_num = getattr(signal, sig)
                try:
                    handler = signal.getsignal(sig_num)
                    signal_info[sig] = str(handler)
                except:
                    pass
        
        # Error information
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'component': component,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Recent errors
        recent_errors = []
        for err in list(self.error_history)[-20:]:
            recent_errors.append({
                'timestamp': err.timestamp.isoformat(),
                'type': err.error_type,
                'message': err.message[:200],
                'component': err.component,
                'severity': err.severity
            })
        
        # Error patterns
        pattern_summary = {}
        for pattern_id, pattern in self.error_patterns.items():
            pattern_summary[pattern_id] = {
                'occurrences': pattern.occurrences,
                'components': list(pattern.components),
                'impact_score': pattern.impact_score,
                'time_span': (pattern.last_seen - pattern.first_seen).total_seconds()
            }
        
        # Configuration (sanitized)
        safe_config = self._get_safe_config()
        
        return {
            'crash_id': hashlib.md5(f"{error}{datetime.now()}".encode()).hexdigest(),
            'error': error_info,
            'context': context,
            'system': system_info,
            'process': process_info,
            'gc_state': gc_info,
            'signal_handlers': signal_info,
            'recent_errors': recent_errors,
            'error_patterns': pattern_summary,
            'error_summary': self.get_error_summary(),
            'config': safe_config,
        }
    
    def _get_safe_config(self) -> dict:
        """Get sanitized config without sensitive data"""
        safe_config = self.config.copy()
        
        # List of sensitive keys to redact
        sensitive_keys = [
            'telegram_bot_token',
            'telegram_chat_id',
            'api_key',
            'secret',
            'password',
            'private_key',
            'token'
        ]
        
        def sanitize_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    sanitize_dict(value)
                elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                    d[key] = "***REDACTED***"
                    
        sanitize_dict(safe_config)
        return safe_config
    
    def _get_recent_errors(self, limit: int = 200) -> List[str]:
        """Get recent errors from log with improved parsing"""
        recent_errors = []
        
        try:
            if not self.error_log_path.exists():
                return []
                
            # Use file locking to prevent race conditions
            import fcntl
            
            with open(self.error_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Try to acquire shared lock
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except:
                    # If can't lock, proceed anyway
                    pass
                
                # Read from end of file
                f.seek(0, 2)  # Go to end
                file_size = f.tell()
                
                # Read last ~50KB
                read_size = min(file_size, 50000)
                f.seek(max(0, file_size - read_size))
                
                lines = f.readlines()
                recent_errors = lines[-limit:]
                
                # Release lock
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Failed to read recent errors: {e}")
            
        return recent_errors
    
    async def _send_telegram_notification(self, message: str, retry_count: int = 0):
        """Send telegram notification with improved error handling"""
        config = self.config.get('error_handling', {})
        
        if not config.get('telegram_notifications'):
            return
            
        bot_token = config.get('telegram_bot_token')
        chat_id = config.get('telegram_chat_id')
        
        if not bot_token or not chat_id:
            return
        
        # Truncate message to Telegram limit
        max_length = 4000  # Leave room for formatting
        if len(message) > max_length:
            message = message[:max_length] + "\n... (truncated)"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        'chat_id': chat_id,
                        'text': message,
                        'parse_mode': 'HTML'
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        if retry_count < 3:
                            await asyncio.sleep(retry_after)
                            await self._send_telegram_notification(message, retry_count + 1)
                    elif response.status != 200:
                        text = await response.text()
                        self.logger.error(f"Telegram API error {response.status}: {text}")
                        
        except asyncio.TimeoutError:
            self.logger.error("Telegram notification timeout")
        except aiohttp.ClientError as e:
            self.logger.error(f"Telegram notification error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected Telegram error: {e}")
    
    async def _send_email_notification(self, subject: str, body: str):
        """Send email notification (placeholder for implementation)"""
        config = self.config.get('error_handling', {})
        
        if not config.get('email_notifications'):
            return
            
        emergency_contact = config.get('emergency_contact')
        if not emergency_contact:
            return
        
        # TODO: Implement email sending via SMTP
        # This would integrate with email settings from config
        self.logger.info(f"Email notification queued: {subject}")
    
    async def _notify_critical_error(self, error_instance: ErrorInstance):
        """Notify about critical errors with queuing"""
        message = (
            f"🚨 <b>CRITICAL ERROR</b> 🚨\n\n"
            f"<b>Component:</b> {error_instance.component}\n"
            f"<b>Error:</b> {error_instance.error_type}\n"
            f"<b>Message:</b> {error_instance.message[:200]}\n"
            f"<b>Impact Score:</b> {self._calculate_error_impact(error_instance):.2f}\n"
            f"<b>Occurrences:</b> {self.error_counts.get(error_instance.hash, 1)}\n"
            f"<b>Time:</b> {error_instance.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Try recovery manager for resilient sending
        try:
            await self.recovery_manager.attempt_recovery(
                'telegram_notification',
                self._send_telegram_notification,
                message
            )
        except Exception as e:
            self.logger.error(f"Failed to send critical notification after retries: {e}")
    
    async def _notify_fatal_crash(self, error: Exception, report_file: Path):
        """Notify about fatal crashes with report location"""
        message = (
            f"💀 <b>FATAL CRASH</b> 💀\n\n"
            f"<b>Error:</b> {type(error).__name__}\n"
            f"<b>Message:</b> {str(error)[:200]}\n"
            f"<b>Report:</b> {report_file.name}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"System has generated a crash report. Check logs for details."
        )
        
        await self._send_telegram_notification(message)
        await self._send_email_notification(
            "NEXLIFY FATAL CRASH",
            f"Fatal crash occurred. Report saved to: {report_file}"
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary with patterns and rates"""
        total_errors = sum(self.error_counts.values())
        
        # Calculate error rate
        cutoff_time = datetime.now() - self.error_window
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        error_rate = len(recent_errors) / (self.error_window.total_seconds() / 60)  # per minute
        
        # Get top errors
        top_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get last error time
        last_error_time = self.error_history[-1].timestamp if self.error_history else None
        
        # Component breakdown
        component_errors = defaultdict(int)
        severity_breakdown = defaultdict(int)
        
        for error in self.error_history:
            component_errors[error.component] += 1
            severity_breakdown[error.severity] += 1
        
        return {
            'total_errors': total_errors,
            'error_rate_per_minute': round(error_rate, 2),
            'unique_errors': len(self.error_counts),
            'last_error_time': last_error_time.isoformat() if last_error_time else None,
            'top_errors': [
                {'hash': h, 'count': c} for h, c in top_errors
            ],
            'component_breakdown': dict(component_errors),
            'severity_breakdown': dict(severity_breakdown),
            'active_patterns': len(self.error_patterns),
            'pattern_summary': [
                {
                    'id': p.pattern_id,
                    'impact': p.impact_score,
                    'components': list(p.components)
                }
                for p in self.error_patterns.values()
            ][:5]  # Top 5 patterns
        }
    
    def add_suppressed_component(self, component: str):
        """Add component to suppression list for non-critical errors"""
        self.suppressed_components.add(component.lower())
    
    def remove_suppressed_component(self, component: str):
        """Remove component from suppression list"""
        self.suppressed_components.discard(component.lower())
    
    def clear_error_history(self):
        """Clear error history and counts"""
        self.error_history.clear()
        self.error_counts.clear()
        self.error_patterns.clear()

class ErrorContext:
    """Context manager for error handling with recovery"""
    
    def __init__(self, component: str, operation: str = "operation",
                 reraise: bool = True, severity: str = "error"):
        self.component = component
        self.operation = operation
        self.reraise = reraise
        self.severity = severity
        self.error_handler = get_error_handler()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Log the error
            self.error_handler.log_error(
                exc_val,
                component=self.component,
                severity=self.severity,
                context={'operation': self.operation}
            )
            
            # Optionally suppress the exception
            if not self.reraise:
                return True
        return False

def handle_errors(component: str = "unknown", severity: str = "error", 
                 reraise: bool = False):
    """Decorator for error handling with support for async functions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_handler.log_error(
                    e,
                    component=component,
                    severity=severity,
                    context={
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                )
                if reraise:
                    raise
                return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_handler.log_error(
                    e,
                    component=component,
                    severity=severity,
                    context={
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                )
                if reraise:
                    raise
                return None
        
        if iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

# Global error handler instance
_error_handler = None

def get_error_handler() -> NightCityErrorHandler:
    """Get or create global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = NightCityErrorHandler()
    return _error_handler

# Import guard for os
import os
