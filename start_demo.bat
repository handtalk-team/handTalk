@echo off
rem handTalk demo: install on first run, then start the server and open the browser
powershell -ExecutionPolicy Bypass -File "%~dp0tools\install_demo.ps1" %*
pause
