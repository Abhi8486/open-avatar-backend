#!/usr/bin/env bash

# Build script for CUDA 12.8 Dockerfile with dynamic versioning
# Usage: ./build_cuda128.sh [--tag TAG] [--no-cache] [--push REGISTRY]

set -e

# Default values
IMAGE_TAG=""
NO_CACHE=""
PUSH_REGISTRY=""
CONFIG_PATH="config/chat_with_openai_compatible_bailian_cosyvoice.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get version from git
get_version() {
    if [ -d .git ]; then
        # # Try to get version from git tag first
        # local git_tag=$(git describe --tags --exact-match 2>/dev/null)
        # if [ $? -eq 0 ] && [ -n "$git_tag" ]; then
        #     echo "$git_tag"
        #     return
        # fi
        
        # # Try to get version from nearest tag + commits
        # local desc_tag=$(git describe --tags --always 2>/dev/null)
        # if [ $? -eq 0 ] && [ -n "$desc_tag" ] && [[ "$desc_tag" == *"-"* ]]; then
        #     # Replace slash with underscore for Docker compatibility
        #     echo "$desc_tag" | sed 's/\//_/g'
        #     return
        # fi
        
        # Fallback: Generate version from branch and short commit
        local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
        local commit=$(git rev-parse --short HEAD 2>/dev/null)
        
        if [ -n "$branch" ] && [ -n "$commit" ]; then
            # Replace slash with underscore for Docker compatibility  
            local clean_branch=$(echo "$branch" | sed 's/\//_/g')
            echo "${clean_branch}-${commit}"
        else
            echo "unknown-$(date +%Y%m%d)"
        fi
    else
        echo "dev-$(date +%Y%m%d)"
    fi
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag )
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache )
            NO_CACHE="--no-cache"
            shift
            ;;
        --push )
            PUSH_REGISTRY="$2"
            shift 2
            ;;
        --config )
            CONFIG_PATH="$2"
            shift 2
            ;;
        -h | --help )
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --tag TAG         Docker image tag (default: open-avatar-chat:VERSION)"
            echo "  --no-cache        Do not use Docker build cache"
            echo "  --push REGISTRY   Push to specified registry after building"
            echo "  --config PATH     Configuration file path used at runtime"
            echo "  -h, --help        Show help information"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Basic build"
            echo "  $0 --tag myapp:latest                       # Custom tag"
            echo "  $0 --push ghcr.io/user/repo                # Build and push"
            echo "  $0 --no-cache --push ghcr.io/user/repo     # Force rebuild and push"
            exit 0
            ;;
        * )
            log_error "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker not found, please install Docker first"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    log_error "Dockerfile not found"
    exit 1
fi

# Set default image tag if not provided
if [ -z "$IMAGE_TAG" ]; then
    IMAGE_TAG="open-avatar-chat:latest"
fi

# Get build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
BUILD_VERSION=$(get_version)

log_info "=== Build Information ==="
log_info "Image Tag: ${IMAGE_TAG}"
log_info "Build Version: ${BUILD_VERSION} (embedded in image tags)"
log_info "Build Date: ${BUILD_DATE} (embedded in image tags)"
log_info "Dockerfile: Dockerfile"

#Build command
BUILD_CMD="docker build ${NO_CACHE} \
    --build-arg BUILD_VERSION=\"${BUILD_VERSION}\" \
    --build-arg BUILD_DATE=\"${BUILD_DATE}\" \
    -t \"${IMAGE_TAG}\" ."

log_info "=== Starting Build ==="
log_info "Executing command: ${BUILD_CMD}"

# Execute build
if eval $BUILD_CMD; then
    log_success "Image built successfully: ${IMAGE_TAG}"
    
    # Show image info
    log_info "=== Image Information ==="
    docker images "${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # Push if requested
    if [ -n "$PUSH_REGISTRY" ]; then
        PUSH_TAG="${PUSH_REGISTRY}:${BUILD_VERSION}"
        log_info "=== Pushing Image ==="
        log_info "Tagging image: ${PUSH_TAG}"
        
        if docker tag "${IMAGE_TAG}" "${PUSH_TAG}"; then
            log_info "Pushing image: ${PUSH_TAG}"
            if docker push "${PUSH_TAG}"; then
                log_success "Image pushed successfully: ${PUSH_TAG}"
            else
                log_error "Failed to push image"
                exit 1
            fi
        else
            log_error "Failed to tag image"
            exit 1
        fi
    fi
    
    # Provide usage instructions
    log_info "=== Usage Instructions ==="
    echo ""
    echo "View commit info in image:"
    echo "docker inspect \"${IMAGE_TAG}\" | grep -A 10 'Labels'"
    echo ""
    echo "Start container:"
    echo "docker run --rm --gpus all -it \\"
    echo "  --network=host \\"
    echo "  -v \$(pwd)/build:/root/open-avatar-chat/build \\"
    echo "  -v \$(pwd)/models:/root/open-avatar-chat/models \\"
    echo "  -v \$(pwd)/ssl_certs:/root/open-avatar-chat/ssl_certs \\"
    echo "  -v \$(pwd)/config:/root/open-avatar-chat/config \\"
    echo "  -v \$(pwd)/models/musetalk/s3fd-619a316812/:/root/.cache/torch/hub/checkpoints/ \\"
    echo "  -p 8282:8282 \\"
    echo "  \"${IMAGE_TAG}\" \\"
    echo "  --config \"${CONFIG_PATH}\""
    echo ""
    echo "View version info inside container:"
    echo "docker run --rm \"${IMAGE_TAG}\" printenv | grep -E '^(APP_VERSION|BUILD_DATE|BUILD_COMMIT)='"
    echo ""
    
else
    log_error "Image build failed"
    exit 1
fi
