#!/bin/bash
echo "Stopping containers..."
docker compose down

echo "Removing images..."
docker compose down --rmi local

echo "Removing dangling images and build cache..."
docker system prune -f

echo "Done. All space recovered."
