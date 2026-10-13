@echo off
rem Pixel Streaming signalling + player web server (Epic PixelStreamingInfrastructure UE5.7).
rem First time: run get_ps_servers.bat from Engine\Plugins\Media\PixelStreaming2\Resources\WebServers in an empty folder,
rem then: npm install --ignore-scripts && npm run build:all:cjs
if "%PS_INFRA%"=="" set PS_INFRA=C:\handtalk\ps-infra
cd /d "%PS_INFRA%\SignallingWebServer"
node dist\index.js --serve --http_root "%PS_INFRA%\SignallingWebServer\www" --player_port 8080 --streamer_port 8888 --homepage uiless.html
