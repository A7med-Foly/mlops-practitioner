# -----------------------------------------------------------------------------
# Networks & Storage Volumes
# -----------------------------------------------------------------------------

resource "docker_network" "prodml_network" {
  name = var.network_name
}

resource "docker_volume" "postgres_data" {
  name = var.postgres_volume_name
}

resource "docker_volume" "minio_data" {
  name = var.minio_volume_name
}

# -----------------------------------------------------------------------------
# Docker Images
# -----------------------------------------------------------------------------

resource "docker_image" "postgres" {
  name         = var.postgres_image
  keep_locally = true
}

resource "docker_image" "minio" {
  name         = var.minio_image
  keep_locally = true
}

resource "docker_image" "mc" {
  name         = var.mc_image
  keep_locally = true
}

resource "docker_image" "mlflow" {
  name         = var.mlflow_image
  keep_locally = true
  build {
    context    = "${path.module}/.."
    dockerfile = "docker/Dockerfile.mlflow"
  }
}

# -----------------------------------------------------------------------------
# PostgreSQL Service (Backend Metadata Store)
# -----------------------------------------------------------------------------

resource "docker_container" "postgres" {
  name  = var.postgres_container_name
  image = docker_image.postgres.image_id

  restart = "unless-stopped"

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}",
  ]

  networks_advanced {
    name = docker_network.prodml_network.name
  }

  ports {
    internal = 5432
    external = var.postgres_host_port
  }

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U ${var.postgres_user} -d ${var.postgres_db}"]
    interval = "5s"
    timeout  = "5s"
    retries  = 5
  }
}

# -----------------------------------------------------------------------------
# MinIO Service (S3-compatible Object Storage for MLflow & DVC)
# -----------------------------------------------------------------------------

resource "docker_container" "minio" {
  name  = var.minio_container_name
  image = docker_image.minio.image_id

  restart = "unless-stopped"
  command = ["server", "/data", "--console-address", ":${var.minio_console_port}"]

  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}",
  ]

  networks_advanced {
    name = docker_network.prodml_network.name
  }

  ports {
    internal = 9000
    external = var.minio_api_port
  }

  ports {
    internal = 9001
    external = var.minio_console_port
  }

  volumes {
    volume_name    = docker_volume.minio_data.name
    container_path = "/data"
  }

  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    interval = "5s"
    timeout  = "5s"
    retries  = 5
  }
}

# -----------------------------------------------------------------------------
# MinIO Bucket Provisioner (Automated bucket creation)
# -----------------------------------------------------------------------------

resource "docker_container" "minio_create_buckets" {
  name     = "mlflow-minio-create-buckets"
  image    = docker_image.mc.image_id
  restart  = "no"
  must_run = false

  depends_on = [docker_container.minio]

  networks_advanced {
    name = docker_network.prodml_network.name
  }

  entrypoint = [
    "/bin/sh",
    "-c",
    <<-EOT
    until /usr/bin/mc alias set local http://${var.minio_container_name}:9000 ${var.minio_root_user} ${var.minio_root_password}; do
      echo "Waiting for MinIO..."
      sleep 1
    done
    for b in ${join(" ", var.minio_buckets)}; do
      /usr/bin/mc mb --ignore-existing local/$b
    done
    exit 0
    EOT
  ]
}

# -----------------------------------------------------------------------------
# MLflow Tracking Server Service
# -----------------------------------------------------------------------------

resource "docker_container" "mlflow" {
  name  = var.mlflow_container_name
  image = docker_image.mlflow.image_id

  restart = "unless-stopped"

  depends_on = [
    docker_container.postgres,
    docker_container.minio,
    docker_container.minio_create_buckets,
  ]

  networks_advanced {
    name = docker_network.prodml_network.name
  }

  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://${var.minio_container_name}:9000",
    "AWS_ACCESS_KEY_ID=${var.minio_root_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_root_password}",
    "AWS_DEFAULT_REGION=${var.aws_region}",
    "MLFLOW_SERVER_ALLOWED_HOSTS=*",
  ]

  command = [
    "mlflow",
    "server",
    "--backend-store-uri",
    "postgresql://${var.postgres_user}:${var.postgres_password}@${var.postgres_container_name}:5432/${var.postgres_db}",
    "--default-artifact-root",
    "s3://${var.minio_buckets[0]}/",
    "--host",
    "0.0.0.0",
    "--port",
    "${var.mlflow_port}",
  ]

  ports {
    internal = var.mlflow_port
    external = var.mlflow_port
  }
}
