@echo off
echo ========================================
echo Starting AgentMed Frontend Server
echo ========================================
echo.
echo This will start the web interface for AgentMed.
echo.
echo The server will run on: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server.
echo.
cd frontend
python app.py
pause


