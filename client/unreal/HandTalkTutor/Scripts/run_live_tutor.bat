@echo off
rem handTalk live tutor: render the MetaHuman tutor off-screen and send it to the web app via Pixel Streaming.
rem Prerequisites: handTalk server (port 8000) and the Pixel Streaming signalling server (port 8888 for UE, 8080 for browsers).
set UE=C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe
set PROJ=%~dp0..\HandTalkTutor.uproject
"%UE%" "%PROJ%" /Game/HandTalk/Live/TutorLive -game -RenderOffScreen -ResX=720 -ResY=1280 -ForceRes -PixelStreamingURL=ws://127.0.0.1:8888 -PixelStreamingEncoderCodec=H264 -AudioMixer -log
