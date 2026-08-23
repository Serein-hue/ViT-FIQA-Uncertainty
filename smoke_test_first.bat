@echo off
setlocal
cd /d "%~dp0"

echo Running a one-epoch, 16-sample end-to-end smoke test...
python run_all_experiments.py --profile core --smoke-test
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo Smoke test passed. You can now run run_all_experiments.bat.
) else (
    echo Smoke test failed. Check the newest SMOKE folder under the outcome directory.
)
echo.
pause
exit /b %EXIT_CODE%
