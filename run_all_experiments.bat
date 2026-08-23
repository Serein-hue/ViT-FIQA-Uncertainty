@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ViT-FIQA-Uncertainty pipeline - STANDARD profile
echo Results are written to .\outcome
echo ============================================================
echo.

python run_all_experiments.py --profile standard
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo Pipeline completed successfully.
) else (
    echo Pipeline failed. Open the newest pipeline.log and PIPELINE_FAILED.json
    echo under .\outcome for the exact reason.
)
echo.
pause
exit /b %EXIT_CODE%
