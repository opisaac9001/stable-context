# Running YAWL with Docker

This document provides instructions on how to build and run the YAWL application using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.
- [Docker Compose](https://docs.docker.com/compose/install/) installed (often included with Docker Desktop).

## Build the Docker Images

Navigate to the project root directory (`yawl/`) where the `docker-compose.yml` file is located, and run:

```bash
docker-compose build
```
This command will build the Docker images for both the backend API and the frontend GUI services as defined in their respective Dockerfiles.

## Run the Application

Once the images are built, you can start the application services using:

```bash
docker-compose up
```
To run the services in detached mode (in the background), use:

```bash
docker-compose up -d
```

## Accessing the Services

After the services are up and running:

-   **Backend API Documentation (Swagger UI):** Open your web browser and go to `http://localhost:8000/docs`
-   **Frontend GUI:** Open your web browser and go to `http://localhost:5173`

## Stopping the Application

To stop the running services, navigate to the project root directory and run:

```bash
docker-compose down
```
This command will stop and remove the containers. To also remove the volumes (if you defined named volumes and want to clear them), you can use `docker-compose down -v`.

## Notes

-   **Initial Build Time:** The first time you build the images, it might take a few minutes, especially for downloading base images and installing dependencies.
-   **Models Directory:** The current Docker setup does not automatically include models from your local `models/` directory inside the backend container unless you uncomment and configure the volume mounts in `docker-compose.yml`. For production, models would typically be managed as part of the image or a dedicated volume strategy.
-   **Development:** For active development, you might want to uncomment the volume mounts in `docker-compose.yml` to enable live code reloading for the backend. Remember that changes to `requirements.txt` or frontend dependencies (`package.json`) will still require a rebuild of the respective image (`docker-compose build backend` or `docker-compose build frontend`).
