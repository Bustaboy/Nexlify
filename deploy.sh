#!/bin/bash
# deploy.sh - Nexlify Deployment Script
# Deploy like a pro netrunner - smooth, fast, and leaving no trace

set -e  # Exit on error

# Colors for that cyberpunk aesthetic
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Banner
echo -e "${CYAN}"
cat << "EOF"
    _   _           _ _  __       
   | \ | |         | (_)/ _|      
   |  \| | _____  _| |_| |_ _   _ 
   | . ` |/ _ \ \/ / | |  _| | | |
   | |\  |  __/>  <| | | | | |_| |
   |_| \_|\___/_/\_\_|_|_|  \__, |
                             __/ |
   DEPLOYMENT SYSTEM v2.0   |___/ 
EOF
echo -e "${NC}"

# Configuration
DEPLOY_ENV=${1:-production}
DEPLOY_DIR="/opt/nexlify"
BACKUP_DIR="/var/backups/nexlify"
LOG_FILE="/var/log/nexlify-deploy.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# Check prerequisites
check_requirements() {
    log "Checking system requirements..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    # Check required commands
    commands=("docker" "docker-compose" "git" "curl" "psql" "redis-cli")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            error "$cmd is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check disk space (minimum 10GB)
    available_space=$(df -BG /opt | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        error "Insufficient disk space. At least 10GB required"
    fi
    
    log "All requirements satisfied"
}

# Backup existing deployment
backup_current() {
    if [[ -d "$DEPLOY_DIR" ]]; then
        log "Backing up current deployment..."
        
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_path="$BACKUP_DIR/nexlify_backup_$timestamp"
        
        mkdir -p "$BACKUP_DIR"
        
        # Backup database
        log "Backing up database..."
        docker exec nexlify-postgres pg_dump -U nexlify_user nexlify_trading > "$backup_path.sql"
        
        # Backup volumes
        log "Backing up Docker volumes..."
        docker run --rm \
            -v nexlify_postgres_data:/data/postgres \
            -v nexlify_redis_data:/data/redis \
            -v $backup_path:/backup \
            alpine tar -czf /backup/volumes.tar.gz /data
        
        # Backup configs
        cp -r "$DEPLOY_DIR/config" "$backup_path.config"
        
        log "Backup completed: $backup_path"
    fi
}

# Deploy application
deploy_app() {
    log "Starting deployment for environment: $DEPLOY_ENV"
    
    # Create deployment directory
    mkdir -p "$DEPLOY_DIR"
    cd "$DEPLOY_DIR"
    
    # Clone or update repository
    if [[ -d ".git" ]]; then
        log "Updating existing repository..."
        git pull origin main
    else
        log "Cloning repository..."
        git clone https://github.com/nexlify/nexlify.git .
    fi
    
    # Set up environment
    if [[ ! -f ".env" ]]; then
        log "Creating environment configuration..."
        cp .env.example .env
        
        # Generate secure passwords
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        REDIS_PASSWORD=$(openssl rand -base64 32)
        MASTER_KEY=$(openssl rand -base64 32)
        
        # Update .env file
        sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
        sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$REDIS_PASSWORD/" .env
        sed -i "s/MASTER_KEY=.*/MASTER_KEY=$MASTER_KEY/" .env
        
        warning "Generated passwords saved in .env - KEEP THESE SECURE!"
    fi
    
    # Build and start services
    log "Building Docker images..."
    docker-compose build --parallel
    
    log "Starting services..."
    if [[ "$DEPLOY_ENV" == "production" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 10
    
    # Run migrations
    log "Running database migrations..."
    docker exec nexlify-api alembic upgrade head
    
    # Initialize default data
    log "Initializing default data..."
    docker exec nexlify-api python scripts/init_data.py
    
    # Set up SSL certificates (production only)
    if [[ "$DEPLOY_ENV" == "production" ]]; then
        setup_ssl
    fi
}

# Setup SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."
    
    # Check if certificates exist
    if [[ ! -f "/etc/letsencrypt/live/nexlify.io/fullchain.pem" ]]; then
        # Install certbot if not present
        if ! command -v certbot &> /dev/null; then
            apt-get update && apt-get install -y certbot
        fi
        
        # Get certificate
        certbot certonly --standalone -d nexlify.io -d www.nexlify.io \
            --non-interactive --agree-tos -m admin@nexlify.io
    fi
    
    # Link certificates to nginx
    mkdir -p "$DEPLOY_DIR/nginx/ssl"
    ln -sf /etc/letsencrypt/live/nexlify.io/fullchain.pem "$DEPLOY_DIR/nginx/ssl/cert.pem"
    ln -sf /etc/letsencrypt/live/nexlify.io/privkey.pem "$DEPLOY_DIR/nginx/ssl/key.pem"
    
    # Reload nginx
    docker exec nexlify-nginx nginx -s reload
}

# Health check
health_check() {
    log "Running health checks..."
    
    services=("nexlify-api" "nexlify-frontend" "nexlify-postgres" "nexlify-redis")
    all_healthy=true
    
    for service in "${services[@]}"; do
        if docker ps | grep -q $service; then
            echo -e "${GREEN}✓${NC} $service is running"
        else
            echo -e "${RED}✗${NC} $service is not running"
            all_healthy=false
        fi
    done
    
    # Check API endpoint
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "${GREEN}✓${NC} API is responding"
    else
        echo -e "${RED}✗${NC} API is not responding"
        all_healthy=false
    fi
    
    if $all_healthy; then
        log "All systems operational!"
    else
        error "Some services are not healthy"
    fi
}

# Post-deployment tasks
post_deploy() {
    log "Running post-deployment tasks..."
    
    # Set up cron jobs
    setup_cron_jobs
    
    # Configure monitoring alerts
    setup_monitoring
    
    # Send deployment notification
    send_notification "Deployment completed successfully"
    
    # Display access information
    echo -e "\n${CYAN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${CYAN}════════════════════════════════════════${NC}"
    echo -e "\nAccess your Nexlify instance at:"
    echo -e "  Web UI: ${BLUE}https://nexlify.io${NC}"
    echo -e "  API: ${BLUE}https://nexlify.io/api${NC}"
    echo -e "  Monitoring: ${BLUE}https://nexlify.io:3001${NC}"
    echo -e "\nDefault credentials are in: ${YELLOW}$DEPLOY_DIR/.env${NC}"
    echo -e "${CYAN}════════════════════════════════════════${NC}\n"
}

# Setup cron jobs
setup_cron_jobs() {
    log "Setting up cron jobs..."
    
    # Backup job
    echo "0 2 * * * /opt/nexlify/scripts/backup.sh" | crontab -
    
    # Certificate renewal
    echo "0 3 * * 1 certbot renew --quiet && docker exec nexlify-nginx nginx -s reload" | crontab -
    
    # Cleanup old logs
    echo "0 4 * * * find /var/log/nexlify -name '*.log' -mtime +30 -delete" | crontab -
}

# Setup monitoring
setup_monitoring() {
    log "Configuring monitoring..."
    
    # Create Grafana datasource
    cat > /tmp/prometheus-datasource.json << EOF
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://prometheus:9090",
  "access": "proxy",
  "isDefault": true
}
EOF
    
    # Add datasource to Grafana
    curl -X POST http://admin:n3xl1fy_gr4f4n4@localhost:3001/api/datasources \
        -H "Content-Type: application/json" \
        -d @/tmp/prometheus-datasource.json
    
    rm /tmp/prometheus-datasource.json
}

# Send notification
send_notification() {
    message=$1
    
    # If webhook URL is configured, send notification
    if [[ -n "$DEPLOY_WEBHOOK_URL" ]]; then
        curl -X POST "$DEPLOY_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"Nexlify Deployment: $message\"}"
    fi
}

# Rollback function
rollback() {
    error "Deployment failed! Rolling back..."
    
    # Stop current containers
    docker-compose down
    
    # Restore from latest backup
    latest_backup=$(ls -t $BACKUP_DIR/nexlify_backup_*.sql | head -1)
    if [[ -n "$latest_backup" ]]; then
        log "Restoring from backup: $latest_backup"
        # Restore database
        docker exec -i nexlify-postgres psql -U nexlify_user nexlify_trading < "$latest_backup"
    fi
    
    error "Rollback completed. Please check logs and retry deployment"
}

# Main execution
main() {
    # Trap errors for rollback
    trap rollback ERR
    
    # Start deployment
    log "Starting Nexlify deployment process..."
    
    check_requirements
    backup_current
    deploy_app
    health_check
    post_deploy
    
    log "Deployment completed successfully!"
}

# Run main function
main

---

# scripts/backup.sh
#!/bin/bash
# Automated backup script for Nexlify

set -e

# Configuration
BACKUP_DIR="/var/backups/nexlify"
RETENTION_DAYS=30
S3_BUCKET="nexlify-backups"  # Optional S3 backup

# Create backup
timestamp=$(date +%Y%m%d_%H%M%S)
backup_name="nexlify_backup_$timestamp"

echo "[$(date)] Starting backup: $backup_name"

# Create backup directory
mkdir -p "$BACKUP_DIR/$backup_name"

# Backup PostgreSQL
docker exec nexlify-postgres pg_dump -U nexlify_user nexlify_trading | \
    gzip > "$BACKUP_DIR/$backup_name/database.sql.gz"

# Backup Redis
docker exec nexlify-redis redis-cli --rdb /data/backup.rdb
docker cp nexlify-redis:/data/backup.rdb "$BACKUP_DIR/$backup_name/redis.rdb"

# Backup configuration
cp -r /opt/nexlify/config "$BACKUP_DIR/$backup_name/"

# Backup models
cp -r /opt/nexlify/models "$BACKUP_DIR/$backup_name/"

# Create archive
cd "$BACKUP_DIR"
tar -czf "$backup_name.tar.gz" "$backup_name"
rm -rf "$backup_name"

# Upload to S3 (if configured)
if command -v aws &> /dev/null && [[ -n "$S3_BUCKET" ]]; then
    aws s3 cp "$backup_name.tar.gz" "s3://$S3_BUCKET/backups/"
    echo "[$(date)] Backup uploaded to S3"
fi

# Clean old backups
find "$BACKUP_DIR" -name "nexlify_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "[$(date)] Backup completed: $backup_name.tar.gz"

---

# scripts/performance_tune.sh
#!/bin/bash
# Performance tuning script for Nexlify

echo "Applying performance optimizations..."

# PostgreSQL tuning
cat > /tmp/postgresql_tune.sql << EOF
-- Increase shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '4GB';

-- Increase work memory
ALTER SYSTEM SET work_mem = '256MB';

-- Optimize for SSD
ALTER SYSTEM SET random_page_cost = 1.1;

-- Increase checkpoint segments
ALTER SYSTEM SET checkpoint_segments = 32;

-- Enable parallel queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
EOF

docker exec -i nexlify-postgres psql -U postgres < /tmp/postgresql_tune.sql
docker restart nexlify-postgres

# Redis tuning
docker exec nexlify-redis redis-cli CONFIG SET maxmemory 2gb
docker exec nexlify-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
docker exec nexlify-redis redis-cli CONFIG REWRITE

# System tuning
# Increase file descriptors
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# Network tuning
cat >> /etc/sysctl.conf << EOF
# Network performance tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
EOF

sysctl -p

echo "Performance optimizations applied!"

---

# scripts/health_monitor.py
#!/usr/bin/env python3
"""
Health monitoring script - Continuous system health checks
Sends alerts when issues are detected
"""

import asyncio
import aiohttp
import psutil
import json
from datetime import datetime
import logging
from typing import Dict, List, Any

# Configuration
API_URL = "http://localhost:8000"
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")
CHECK_INTERVAL = 60  # seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("health_monitor")

class HealthMonitor:
    def __init__(self):
        self.checks = {
            "api": self.check_api,
            "database": self.check_database,
            "redis": self.check_redis,
            "disk_space": self.check_disk_space,
            "memory": self.check_memory,
            "docker": self.check_docker_containers
        }
        self.alert_history = {}
    
    async def check_api(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/health", timeout=5) as resp:
                    return {
                        "status": "healthy" if resp.status == 200 else "unhealthy",
                        "response_time": resp.headers.get("X-Response-Time", "N/A"),
                        "details": await resp.json() if resp.status == 200 else None
                    }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="nexlify_trading",
                user="nexlify_user",
                password=os.environ.get("POSTGRES_PASSWORD")
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return {"status": "healthy", "connections": "active"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            info = r.info()
            return {
                "status": "healthy",
                "memory_used": f"{info['used_memory_human']}",
                "connected_clients": info['connected_clients']
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        usage = psutil.disk_usage('/')
        status = "healthy" if usage.percent < 80 else "warning" if usage.percent < 90 else "critical"
        
        return {
            "status": status,
            "usage_percent": usage.percent,
            "free_gb": usage.free / (1024**3)
        }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        status = "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
        
        return {
            "status": status,
            "usage_percent": memory.percent,
            "available_gb": memory.available / (1024**3)
        }
    
    async def check_docker_containers(self) -> Dict[str, Any]:
        """Check Docker containers"""
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    container = json.loads(line)
                    containers.append({
                        "name": container.get("Names"),
                        "status": container.get("Status"),
                        "state": container.get("State")
                    })
            
            unhealthy = [c for c in containers if c["state"] != "running"]
            
            return {
                "status": "healthy" if not unhealthy else "unhealthy",
                "total": len(containers),
                "running": len(containers) - len(unhealthy),
                "unhealthy": unhealthy
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def send_alert(self, check_name: str, result: Dict[str, Any]):
        """Send alert via webhook"""
        if not WEBHOOK_URL:
            return
        
        # Avoid alert spam
        alert_key = f"{check_name}:{result['status']}"
        if alert_key in self.alert_history:
            last_alert = self.alert_history[alert_key]
            if (datetime.now() - last_alert).seconds < 3600:  # 1 hour cooldown
                return
        
        payload = {
            "text": f"🚨 Nexlify Health Alert",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Check:* {check_name}\n*Status:* {result['status']}\n*Time:* {datetime.now().isoformat()}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{json.dumps(result, indent=2)}```"
                    }
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(WEBHOOK_URL, json=payload)
            
            self.alert_history[alert_key] = datetime.now()
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def run_checks(self):
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
                
                # Send alert if unhealthy
                if result.get("status") not in ["healthy", "normal"]:
                    await self.send_alert(name, result)
                    
            except Exception as e:
                logger.error(f"Check {name} failed: {e}")
                results[name] = {"status": "error", "error": str(e)}
        
        return results
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting health monitoring...")
        
        while True:
            try:
                results = await self.run_checks()
                
                # Log summary
                healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
                logger.info(f"Health check complete: {healthy_count}/{len(results)} healthy")
                
                # Save results
                with open("/var/log/nexlify/health.json", "w") as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "results": results
                    }, f, indent=2)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor = HealthMonitor()
    asyncio.run(monitor.monitor_loop())
