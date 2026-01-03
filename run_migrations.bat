@echo off
echo ========================================
echo Running Database Migrations
echo ========================================
echo.

python -m app.database.migrations

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Migrations completed successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Migrations failed! Check the error above.
    echo ========================================
)

pause
