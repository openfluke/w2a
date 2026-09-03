@echo off
REM Windows amd64 C-ABI (run on Windows with mingw or from WSL with mingw cross).
cd /d "%~dp0"
go run builder.go -os windows -arch amd64 %*
if exist copy_to_python.sh bash copy_to_python.sh
