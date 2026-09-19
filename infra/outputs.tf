output "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  value       = "http://localhost:${var.mlflow_port}"
}

output "minio_endpoint_url" {
  description = "MinIO S3 API endpoint URL"
  value       = "http://localhost:${var.minio_api_port}"
}

output "minio_console_url" {
  description = "MinIO web console URL"
  value       = "http://localhost:${var.minio_console_port}"
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://${var.postgres_user}:${var.postgres_password}@localhost:${var.postgres_host_port}/${var.postgres_db}"
  sensitive   = true
}

output "created_buckets" {
  description = "Provisioned MinIO S3 buckets"
  value       = var.minio_buckets
}
