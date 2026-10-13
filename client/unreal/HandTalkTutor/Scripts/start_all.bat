@echo off
rem Start the whole handTalk live tutor stack on this computer:
rem   1) handTalk server  http://localhost:8000/app/   2) Pixel Streaming signalling  3) Unreal MetaHuman tutor (off-screen)
set ROOT=%~dp0..\..\..\..
start "handTalk server" cmd /k "cd /d %ROOT%\server && .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
start "signalling" cmd /k "%~dp0run_signalling.bat"
timeout /t 5 /nobreak >nul
start "live tutor" /min cmd /c "%~dp0run_live_tutor.bat"
timeout /t 3 /nobreak >nul
start "" http://localhost:8000/app/
