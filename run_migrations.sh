#!/bin/bash

echo "========================================"
echo "Running Database Migrations"
echo "========================================"
echo ""

python -m app.database.migrations

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Migrations completed successfully!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Migrations failed! Check the error above."
    echo "========================================"
    exit 1
fi
