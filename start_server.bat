@echo off
setlocal enabledelayedexpansion

:: -------------------------------
:: CONFIGURATION
:: -------------------------------
:: Path to your llama.cpp binaries
echo LLAMA_DIR: %LLAMA_DIR% 
:: Default model directory
echo MODEL_DIR: %MODEL_DIR% 

:: -------------------------------
:: SELECT WORKING DIRECTORY
:: -------------------------------
echo Current working directory: %CD%
echo.
cd /d "%LLAMA_DIR%"

echo.
echo Working directory is now: %CD%
echo.

:: -------------------------------
:: SELECT MODEL FILE
:: -------------------------------
echo Available GGUF models in %MODEL_DIR%:
echo.

for %%f in ("%MODEL_DIR%\*.gguf") do (
    echo   %%~nxf
)

echo.
set MODEL=Qwen3.5-9B-Q8_0.gguf

if not exist "%MODEL_DIR%\%MODEL%" (
    echo Model not found: %MODEL_DIR%\%MODEL%
    pause
    exit /b
)

echo.
echo Selected model: %MODEL_DIR%\%MODEL%
echo.

:: -------------------------------
:: RUN llama-server
:: -------------------------------
:run_server
echo.
set /p PORT=Enter server port (default 8080): 
if "%PORT%"=="" set PORT=8080

echo.
echo Starting llama-server on port %PORT%...
echo.

"%LLAMA_DIR%\llama-b8680-bin-win-cuda-12.4-x64\llama-server.exe" ^
    --model "%MODEL_DIR%\%MODEL%" ^
    --ctx-size 8192 ^
    --temp 0.6 ^
    --top-p 0.95 ^
    --top-k 20 ^
    --min-p 0.00 ^
    --alias "Qwen3.5-9B-GGUF" ^
    --port %PORT% ^
    --gpu-layers 999 ^
    --chat-template-kwargs "{\"enable_thinking\":true}"

pause
exit /b