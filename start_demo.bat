@echo off
rem handTalk 시연: 처음이면 설치하고, 서버를 켠 뒤 브라우저를 연다
powershell -ExecutionPolicy Bypass -File "%~dp0tools\install_demo.ps1" %*
pause
