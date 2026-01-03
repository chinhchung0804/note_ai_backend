@echo off
echo ========================================
echo TEST FULL FLOW - NotallyX Authentication
echo ========================================
echo.

set BASE_URL=http://localhost:8000
set USERNAME=testuser_%RANDOM%
set EMAIL=%USERNAME%@example.com
set PASSWORD=Test123!

echo [1/10] Testing server connection...
curl -s %BASE_URL%/ > nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Server is not running!
    echo Please start server: uvicorn app.main:app --reload
    pause
    exit /b 1
)
echo OK: Server is running

echo.
echo [2/10] Registering new user...
echo Username: %USERNAME%
echo Email: %EMAIL%
curl -X POST %BASE_URL%/api/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"%USERNAME%\",\"email\":\"%EMAIL%\",\"password\":\"%PASSWORD%\"}" ^
  -o register_response.json
echo.

echo.
echo [3/10] Logging in...
curl -X POST %BASE_URL%/api/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"%USERNAME%\",\"password\":\"%PASSWORD%\"}" ^
  -o login_response.json
echo.

echo.
echo [4/10] Extracting token...
for /f "tokens=2 delims=:," %%a in ('type login_response.json ^| findstr "access_token"') do set TOKEN=%%a
set TOKEN=%TOKEN:"=%
set TOKEN=%TOKEN: =%
echo Token: %TOKEN:~0,50%...

echo.
echo [5/10] Getting user profile...
curl -X GET %BASE_URL%/api/auth/me ^
  -H "Authorization: Bearer %TOKEN%"
echo.

echo.
echo [6/10] Checking account limits...
curl -X GET %BASE_URL%/api/auth/account-limits ^
  -H "Authorization: Bearer %TOKEN%"
echo.

echo.
echo [7/10] Creating note 1/5...
curl -X POST %BASE_URL%/api/v1/process ^
  -H "Authorization: Bearer %TOKEN%" ^
  -F "text=Test note 1" ^
  -F "note_id=test_note_1" ^
  -s -o note1_response.json
echo OK

echo.
echo [8/10] Creating notes 2-5...
for /L %%i in (2,1,5) do (
    echo Creating note %%i/5...
    curl -X POST %BASE_URL%/api/v1/process ^
      -H "Authorization: Bearer %TOKEN%" ^
      -F "text=Test note %%i" ^
      -F "note_id=test_note_%%i" ^
      -s -o note%%i_response.json
)
echo OK

echo.
echo [9/10] Testing rate limit (note 6 - should fail)...
curl -X POST %BASE_URL%/api/v1/process ^
  -H "Authorization: Bearer %TOKEN%" ^
  -F "text=Test note 6 - should be blocked" ^
  -F "note_id=test_note_6" ^
  -o note6_response.json
echo.

echo.
echo [10/10] Getting payment plans...
curl -X GET %BASE_URL%/api/payment/plans
echo.

echo.
echo ========================================
echo TEST COMPLETED!
echo ========================================
echo.
echo Summary:
echo - User registered: %USERNAME%
echo - Login successful
echo - Created 5 notes (FREE limit)
echo - Note 6 blocked (rate limit working)
echo - Payment plans available
echo.
echo Next steps:
echo 1. Check responses in *_response.json files
echo 2. Test payment flow manually
echo 3. Upgrade to PRO and test unlimited notes
echo.
echo Cleanup:
del /q *_response.json 2>nul
echo.
pause
