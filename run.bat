@echo off
<<<<<<< HEAD
echo Starting Waste Classification System...
python -m streamlit run app.py
=======
cd /d "%~dp0"

echo ─────────────────────────────────────────────────────────
echo   Smart Waste Management System  ^|  Government of India
echo ─────────────────────────────────────────────────────────

:: Activate the virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Launch the Streamlit app
echo Starting app...
python -m streamlit run app.py

>>>>>>> d10834e (Add YOLO+ResNet50 two-stage pipeline, training scripts, evaluation reports)
pause
