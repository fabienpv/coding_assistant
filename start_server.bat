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

echo Choose a model number:
echo: 1 = Qwen3.5-9B-Q8_0.gguf
echo: 2 = Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
echo: 3 = gemma-4-26B-A4B-it-UD-Q6_K.gguf

set /p MODEL_NB=Select a model: 

if "%MODEL_NB%"=="1" set MODEL=Qwen3.5-9B-GGUF\Qwen3.5-9B-Q8_0.gguf
if "%MODEL_NB%"=="1" set MMPROJ=Qwen3.5-9B-GGUF\mmproj-F16.gguf
if "%MODEL_NB%"=="2" set MODEL=Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
if "%MODEL_NB%"=="3" set MODEL=gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q6_K.gguf

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

if "%MODEL_NB%"=="1" goto run_qwen35
if "%MODEL_NB%"=="2" goto run_qwen36
if "%MODEL_NB%"=="3" goto run_gemma4

:: -------------------------------
:: RUN Qwen 3.5 9B
:: -------------------------------
:run_qwen35
"%LLAMA_DIR%\llama-b8680-bin-win-cuda-12.4-x64\llama-server.exe" ^
    --model "%MODEL_DIR%\%MODEL%" ^
    --mmproj "%MODEL_DIR%\%MMPROJ%"
    --ctx-size 8192 ^
    --temp 0.6 ^
    --top-p 0.95 ^
    --top-k 20 ^
    --min-p 0.00 ^
    --alias "Qwen3.5-9B-Q8" ^
    --port %PORT% ^
    --gpu-layers 999 ^
    --chat-template-kwargs "{\"enable_thinking\":true}"

:: -------------------------------
:: RUN Qwen 3.6 35B
:: -------------------------------
:run_qwen36
"%LLAMA_DIR%\llama-b8680-bin-win-cuda-12.4-x64\llama-server.exe" ^
    --model "%MODEL_DIR%\%MODEL%" ^
    --ctx-size 16384 ^
    --temp 0.6 ^
    --top-p 0.95 ^
    --top-k 20 ^
    --min-p 0.00 ^
    --alias "Qwen3.6-35B-Q6" ^
    --port %PORT% ^
    --gpu-layers 18 ^
    --chat-template-kwargs "{\"enable_thinking\":true}"

:: -------------------------------
:: RUN Gemma 4B 26B
:: -------------------------------
:run_gemma4
"%LLAMA_DIR%\llama-b8680-bin-win-cuda-12.4-x64\llama-server.exe" ^
    --model "%MODEL_DIR%\%MODEL%" ^
    --ctx-size 16384 ^
    --temp 0.6 ^
    --top-p 0.95 ^
    --top-k 20 ^
    --min-p 0.00 ^
    --alias "Gemma4-26B-Q6" ^
    --port %PORT% ^
    --gpu-layers 18 ^
    --chat-template-kwargs "{\"enable_thinking\":true}"

pause
exit /b