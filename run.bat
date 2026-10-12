@echo off
cd /d "%~dp0"

echo ─────────────────────────────────────────────────────────
echo   Smart Waste Management System  ^|  Government of India
echo ─────────────────────────────────────────────────────────

:: Activate the virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Launch the Streamlit app
echo Starting app...
python -m streamlit run app.py

pause
