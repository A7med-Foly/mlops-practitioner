variable "network_name" {
  description = "Docker network name for inter-container communication"
  type        = string
  default     = "prodml-network"
}

# -----------------------------------------------------------------------------
# PostgreSQL Configuration
# -----------------------------------------------------------------------------

variable "postgres_image" {
  description = "Docker image for MLflow PostgreSQL backend store"
  type        = string
  default     = "postgres:16-alpine"
}

variable "postgres_container_name" {
  description = "Container name for PostgreSQL"
  type        = string
  default     = "mlflow-postgres"
}

variable "postgres_user" {
  description = "PostgreSQL user for MLflow"
  type        = string
  default     = "mlflow"
}

variable "postgres_password" {
  description = "PostgreSQL password for MLflow"
  type        = string
  default     = "mlflow"
  sensitive   = true
}

variable "postgres_db" {
  description = "PostgreSQL database name for MLflow"
  type        = string
  default     = "mlflow"
}

variable "postgres_host_port" {
  description = "Host port mapped to PostgreSQL (5432)"
  type        = number
  default     = 5433
}

variable "postgres_volume_name" {
  description = "Named volume for PostgreSQL persistence"
  type        = string
  default     = "postgres_data"
}

# -----------------------------------------------------------------------------
# MinIO Configuration
# -----------------------------------------------------------------------------

variable "minio_image" {
  description = "Docker image for MinIO object storage"
  type        = string
  default     = "quay.io/minio/minio:latest"
}

variable "minio_container_name" {
  description = "Container name for MinIO"
  type        = string
  default     = "mlflow-minio"
}

variable "minio_root_user" {
  description = "MinIO root / admin username"
  type        = string
  default     = "minioadmin"
}

variable "minio_root_password" {
  description = "MinIO root / admin password"
  type        = string
  default     = "minioadmin"
  sensitive   = true
}

variable "minio_api_port" {
  description = "Host port mapped to MinIO S3 API (9000)"
  type        = number
  default     = 9000
}

variable "minio_console_port" {
  description = "Host port mapped to MinIO web console (9001)"
  type        = number
  default     = 9001
}

variable "minio_volume_name" {
  description = "Named volume for MinIO persistence"
  type        = string
  default     = "minio_data"
}

variable "mc_image" {
  description = "Docker image for MinIO Client CLI"
  type        = string
  default     = "quay.io/minio/mc:latest"
}

variable "minio_buckets" {
  description = "List of MinIO S3 buckets to provision"
  type        = list(string)
  default     = ["mlflow", "prodml-dvc"]
}

# -----------------------------------------------------------------------------
# MLflow Configuration
# -----------------------------------------------------------------------------

variable "mlflow_image" {
  description = "Docker image tag for the MLflow tracking server"
  type        = string
  default     = "prodml-mlflow:latest"
}

variable "mlflow_container_name" {
  description = "Container name for MLflow tracking server"
  type        = string
  default     = "mlflow-server"
}

variable "mlflow_port" {
  description = "Host port mapped to MLflow tracking server (5000)"
  type        = number
  default     = 5000
}

variable "aws_region" {
  description = "AWS default region for S3 compatibility"
  type        = string
  default     = "us-east-1"
}
