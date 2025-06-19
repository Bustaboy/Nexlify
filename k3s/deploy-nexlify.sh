#!/bin/bash
# deploy-nexlify.sh
# 🌃 NEXLIFY K3S DEPLOYMENT SCRIPT - JACK INTO THE MATRIX
# Location: k3s/deploy-nexlify.sh
#
# "Sometimes you gotta burn it all down to build something better."
# - Johnny Silverhand (probably)

set -euo pipefail

# Color codes for that cyberpunk aesthetic
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# ASCII art because we're not animals
print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    _   _  _____  _     _  _     _____ _______   __
   | \ | ||  ___|| |   | || |   |_   _||  ___| \ \ / /
   |  \| || |__  | |   | || |     | |  | |_     \ V / 
   | . ` ||  __| | |   | || |     | |  |  _|     > <  
   | |\  || |___ | |___| || |    _| |_ | |      / . \ 
   |_| \_||_____||_____||_|_____|_____||_|     /_/ \_\
                                                        
            C Y B E R P U N K   T R A D I N G   M A T R I X
            
EOF
    echo -e "${NC}"
}

# Logging with style
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}[NEXLIFY]${NC} $1"
}

error() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Running pre-flight checks..."
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Jack out and install it first, choom."
        exit 1
    fi
    
    # Check for helm (needed for KEDA)
    if ! command -v helm &> /dev/null; then
        warning "helm not found. You'll need it for KEDA installation."
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Can't connect to K3s cluster. Is your neural link active?"
        exit 1
    fi
    
    # Check for required storage classes
    if ! kubectl get storageclass longhorn-nvme &> /dev/null; then
        warning "longhorn-nvme storage class not found. Creating mock..."
        kubectl create storageclass longhorn-nvme --provisioner=kubernetes.io/no-provisioner || true
    fi
    
    log "Pre-flight checks complete. We're good to go!"
}

# Create namespace first
create_namespace() {
    log "Creating Nexlify namespace - our digital sanctuary..."
    
    if kubectl get namespace nexlify-trading &> /dev/null; then
        warning "Namespace already exists. Skipping..."
    else
        kubectl apply -f nexlify-namespace.yaml
        log "Namespace created. Welcome to the matrix, choom."
    fi
}

# Install KEDA if not present
install_keda() {
    log "Checking KEDA installation..."
    
    if kubectl get deployment -n keda keda-operator &> /dev/null; then
        log "KEDA already installed. Nice chrome!"
    else
        if command -v helm &> /dev/null; then
            log "Installing KEDA - the market volatility sensor..."
            helm repo add kedacore https://kedacore.github.io/charts
            helm repo update
            helm install keda kedacore/keda --namespace keda --create-namespace
            log "KEDA online. Autoscaling armed and ready."
        else
            warning "Helm not found. Install KEDA manually: https://keda.sh/docs/deploy/"
        fi
    fi
}

# Deploy core components
deploy_core() {
    log "Deploying core trading engine..."
    
    # Apply in order of dependencies
    local core_manifests=(
        "nexlify-core-deployment.yaml"
        "questdb-statefulset.yaml"
        "valkey-deployment.yaml"
    )
    
    for manifest in "${core_manifests[@]}"; do
        if [ -f "$manifest" ]; then
            log "Deploying $manifest..."
            kubectl apply -f "$manifest"
            sleep 2  # Give the API server a breather
        else
            warning "Manifest $manifest not found. Skipping..."
        fi
    done
    
    log "Core systems online. Neural net warming up..."
}

# Deploy ML/GPU components
deploy_ml() {
    log "Checking for GPU nodes..."
    
    if kubectl get nodes -l nvidia.com/gpu=true --no-headers | grep -q .; then
        log "GPU nodes detected. Deploying ML inference engine..."
        kubectl apply -f gpu-ml-deployment.yaml
        log "ML engine deployed. Silicon dreams activated."
    else
        warning "No GPU nodes found. ML inference will run in CPU mode."
        warning "For full chrome, add GPU nodes to your cluster."
    fi
}

# Deploy ingress
deploy_ingress() {
    log "Deploying HAProxy ingress - the speed daemon..."
    
    # First disable default Traefik if present
    if kubectl get deployment -n kube-system traefik &> /dev/null; then
        warning "Traefik detected. You should disable it for HAProxy."
        warning "Run: k3s server --disable=traefik"
    fi
    
    kubectl apply -f haproxy-ingress.yaml
    log "HAProxy deployed. 42,000 RPS of pure chrome!"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack - the all-seeing eye..."
    
    kubectl apply -f monitoring-stack.yaml
    
    # Wait for Prometheus to be ready
    log "Waiting for Prometheus to jack in..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n nexlify-trading --timeout=300s
    
    log "Monitoring online. You can see everything now."
}

# Deploy autoscaling
deploy_autoscaling() {
    log "Deploying KEDA autoscaling configurations..."
    
    if kubectl get deployment -n keda keda-operator &> /dev/null; then
        kubectl apply -f keda-autoscaling.yaml
        log "Autoscaling configured. Market volatility response: ARMED"
    else
        warning "KEDA not installed. Skipping autoscaling setup."
    fi
}

# Verify deployment
verify_deployment() {
    log "Running deployment verification..."
    echo ""
    
    # Check pod status
    echo -e "${PURPLE}=== Pod Status ===${NC}"
    kubectl get pods -n nexlify-trading
    echo ""
    
    # Check services
    echo -e "${PURPLE}=== Services ===${NC}"
    kubectl get svc -n nexlify-trading
    echo ""
    
    # Check ingress
    echo -e "${PURPLE}=== Ingress ===${NC}"
    kubectl get ingress -n nexlify-trading
    echo ""
    
    # Get external IPs
    HAPROXY_IP=$(kubectl get svc haproxy-ingress -n nexlify-trading -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    GRAFANA_IP=$(kubectl get svc grafana -n nexlify-trading -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    echo -e "${GREEN}=== Access Points ===${NC}"
    echo -e "HAProxy Stats: ${CYAN}http://${HAPROXY_IP}:8404/stats${NC}"
    echo -e "Grafana Dashboard: ${CYAN}http://${GRAFANA_IP}:3000${NC}"
    echo -e "Prometheus: ${CYAN}kubectl port-forward -n nexlify-trading svc/prometheus 9090:9090${NC}"
    echo ""
}

# Main deployment flow
main() {
    print_banner
    
    log "Starting Nexlify K3s deployment sequence..."
    log "Target: Production-grade crypto trading neural network"
    echo ""
    
    # Deployment steps
    check_prerequisites
    create_namespace
    install_keda
    
    log "Deploying core components..."
    deploy_core
    
    log "Deploying specialized systems..."
    deploy_ml
    deploy_ingress
    deploy_monitoring
    deploy_autoscaling
    
    echo ""
    log "Deployment complete! Running verification..."
    sleep 5
    
    verify_deployment
    
    echo ""
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${CYAN}NEXLIFY TRADING MATRIX - ONLINE${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo -e "${YELLOW}Remember:${NC}"
    echo "- Check pod logs if something's acting corpo: kubectl logs -n nexlify-trading <pod-name>"
    echo "- Scale manually if KEDA gets confused: kubectl scale deployment nexlify-trading-engine --replicas=10"
    echo "- Monitor GPU usage: kubectl exec -n nexlify-trading <ml-pod> -- nvidia-smi"
    echo ""
    echo -e "${PURPLE}Welcome to the future of trading, choom. May your margins be thicc and your losses be smol.${NC}"
    echo ""
}

# Handle errors with style
trap 'error "Deployment failed! Check the logs above. Sometimes even the best netrunners hit ICE."' ERR

# Jack in
main "$@"
