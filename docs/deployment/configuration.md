# ProteinFlex Configuration Guide

## Configuration Overview

ProteinFlex uses a hierarchical configuration system that allows fine-tuning of all system components through YAML configuration files and environment variables.

## Configuration Files

### Main Configuration
```yaml
# config/main.yaml
model:
  architecture: transformer
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  batch_size: 32

optimization:
  memory_management:
    enabled: true
    max_memory: 16GB
    cache_size: 4GB

  adaptive_processing:
    enabled: true
    optimization_level: O2
    target_latency: 100ms

  performance_monitoring:
    enabled: true
    collection_interval: 60s
    retention_period: 7d

visualization:
  engine: py3dmol
  quality: high
  interactive: true
```

### Hardware Configuration
```yaml
# config/hardware.yaml
gpu:
  enabled: true
  devices: [0, 1]
  memory_limit: 8GB

cpu:
  num_threads: 8
  affinity: performance

memory:
  system_reserve: 4GB
  swap_limit: 8GB
```

## Environment Variables

### System Variables
```bash
# Required
PROTEINFLEX_HOME=/path/to/installation
PROTEINFLEX_CONFIG=/path/to/config

# Optional
PROTEINFLEX_LOG_LEVEL=INFO
PROTEINFLEX_CACHE_DIR=/path/to/cache
PROTEINFLEX_DATA_DIR=/path/to/data
```

### Performance Variables
```bash
# Hardware
CUDA_VISIBLE_DEVICES=0,1
CUDA_CACHE_PATH=/path/to/cache
OMP_NUM_THREADS=8

# Memory
MAX_MEMORY_ALLOCATION=16GB
CACHE_SIZE_LIMIT=4GB
```

## Feature Configuration

### Model Settings
```yaml
model_config:
  generation:
    max_length: 1000
    temperature: 0.8
    top_p: 0.9

  analysis:
    binding_site_threshold: 0.8
    fold_confidence: 0.9
    structure_resolution: high
```

### Optimization Settings
```yaml
optimization_config:
  memory:
    checkpoint_interval: 1000
    cache_cleanup_threshold: 0.8

  processing:
    batch_optimization: true
    mixed_precision: true
    kernel_fusion: true
```

## Security Configuration

### Access Control
```yaml
security:
  authentication:
    enabled: true
    method: jwt
    token_expiry: 24h

  authorization:
    role_based: true
    default_role: user
```

### Resource Limits
```yaml
resource_limits:
  max_sequence_length: 2000
  max_batch_size: 64
  max_concurrent_jobs: 10
  rate_limit: 100/hour
```

## Monitoring Configuration

### Metrics Collection
```yaml
monitoring:
  metrics:
    collection_interval: 60s
    retention_period: 30d
    export_format: prometheus

  alerts:
    cpu_threshold: 80%
    memory_threshold: 90%
    latency_threshold: 1s
```

### Logging Configuration
```yaml
logging:
  level: INFO
  format: json
  output: file
  rotation: daily
  retention: 30d
```

## Integration Configuration

### External Services
```yaml
integrations:
  database:
    type: postgresql
    host: localhost
    port: 5432

  cache:
    type: redis
    host: localhost
    port: 6379
```

### API Configuration
```yaml
api:
  host: 0.0.0.0
  port: 8080
  workers: 4
  timeout: 30s
```

## Development Configuration

### Debug Settings
```yaml
debug:
  enabled: false
  verbose_logging: true
  profiling: true
  trace_calls: false
```

### Test Configuration
```yaml
testing:
  mock_services: true
  test_data_path: /path/to/test/data
  performance_benchmarks: true
```
