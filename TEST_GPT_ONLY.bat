@echo off
echo ========================================
echo TEST GPT-ONLY CONFIGURATION
echo ========================================
echo.

echo [1/4] Starting server...
echo Press Ctrl+C after server starts, then run this script again
start cmd /k "cd /d %~dp0 && uvicorn app.main:app --reload"
timeout /t 5 /nobreak >nul

echo.
echo [2/4] Testing FREE user (3 features)...
echo.
curl -X POST http://localhost:8000/api/v1/process ^
  -F "text=run walk eat lunch drink water" ^
  -F "user_id=testuser" ^
  -F "note_id=test_free_1" ^
  -F "content_type=checklist" ^
  -F "checked_vocab_items=[\"run\",\"walk\",\"eat lunch\"]"

echo.
echo.
echo [3/4] Getting account benefits...
echo.
curl http://localhost:8000/api/auth/account-benefits

echo.
echo.
echo [4/4] Check server logs for:
echo   - "Vocab bundle for free: 3 features enabled"
echo   - "Running 3 API calls in parallel"
echo   - "Using GPT-4o-mini for FREE account"
echo.
echo ========================================
echo TEST COMPLETE
echo ========================================
echo.
echo To test PRO user:
echo 1. Upgrade user: UPDATE users SET account_type = 'PRO' WHERE username = 'testuser';
echo 2. Run: curl -X POST http://localhost:8000/api/v1/process -F "text=..." -F "user_id=testuser" ...
echo.
pause
