@echo off
REM Run full Go↔C++ comparison + all-train determinism suite; overwrite logs\windowsx86.txt
REM (also logs\windows.txt and optionally copy to %%GIT_ROOT%%\windowsx86.txt).
REM
REM Usage (from w2a or anywhere):
REM   scripts\run_go_v_cpp_windows_snapshot.cmd
REM   set WELVET_CPP_ROOT=C:\path\to\welvet.cpp && scripts\run_go_v_cpp_windows_snapshot.cmd
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "W2A_ROOT=%%~fI"
if not defined WELVET_CPP_ROOT set "WELVET_CPP_ROOT=%W2A_ROOT%\..\welvet.cpp"
for %%I in ("%WELVET_CPP_ROOT%") do set "WELVET_CPP_ROOT=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%WELVET_CPP_ROOT%\build"
set "LOG_DIR=%W2A_ROOT%\logs"
set "OUT_X86=%LOG_DIR%\windowsx86.txt"
set "OUT_WIN=%LOG_DIR%\windows.txt"
for %%I in ("%W2A_ROOT%\..") do set "GIT_ROOT=%%~fI"
set "OUT_GIT=%GIT_ROOT%\windowsx86.txt"

if not defined GOTMPDIR set "GOTMPDIR=%TEMP%\gotmp-w2a-win-snapshot"
if not exist "%GOTMPDIR%" mkdir "%GOTMPDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%WELVET_CPP_ROOT%\CMakeLists.txt" (
  echo error: welvet.cpp not found at %WELVET_CPP_ROOT% ^(set WELVET_CPP_ROOT^) 1>&2
  exit /b 1
)

REM Prefer llvm-mingw on PATH when present (same toolchain as windows_arm64 ship).
if exist "C:\llvm-mingw\bin\clang++.exe" set "PATH=C:\llvm-mingw\bin;%PATH%"

echo ==^> building welvet.cpp -^> %BUILD_DIR%
where cmake >nul 2>&1
if errorlevel 1 (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
    set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
  )
)
where clang++ >nul 2>&1
if errorlevel 1 (
  echo error: need clang++ ^(llvm-mingw^) on PATH to build windows amd64 lib+tool 1>&2
  exit /b 1
)
if not exist "%BUILD_DIR%\CMakeCache.txt" (
  cmake -S "%WELVET_CPP_ROOT%" -B "%BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ || exit /b 1
)
cmake --build "%BUILD_DIR%" -j %NUMBER_OF_PROCESSORS% || exit /b 1

set "TOOL=%BUILD_DIR%\welvet_dense_fwd.exe"
if not exist "%TOOL%" (
  echo error: missing tool %TOOL% 1>&2
  exit /b 1
)

REM DLL next to tool + on PATH for go test / nativecpp
set "PATH=%BUILD_DIR%;%W2A_ROOT%\..\welvet\native\cpp\windows_amd64;%PATH%"
set "WELVET_DENSE_FWD=%TOOL%"
set "CGO_ENABLED=1"
if exist "C:\llvm-mingw\bin\gcc.exe" set "CC=C:\llvm-mingw\bin\gcc.exe"
if exist "C:\llvm-mingw\bin\g++.exe" set "CXX=C:\llvm-mingw\bin\g++.exe"

for /f "delims=" %%V in ('"%TOOL%" --version') do set "VER=%%V"
for /f "delims=" %%D in ('powershell -NoProfile -Command "[DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')"') do set "START_UTC=%%D"

(
  echo === welvet.cpp ^<-^> Go w2a windows snapshot ===
  echo date:    %START_UTC%
  echo host:    windows/amd64
  echo tool:    %VER%
  echo WELVET_DENSE_FWD=%WELVET_DENSE_FWD%
  echo W2A_ROOT=%W2A_ROOT%
  echo.
) > "%OUT_X86%"

set "rc_fwd=0"
set "rc_train=0"

echo ==^> PART 1/2: TestGoVCpp ^(full Go^<-^>C++ comparison + metrics tables^)
>> "%OUT_X86%" echo ==^> PART 1/2: TestGoVCpp ^(full Go^<-^>C++ comparison + metrics tables^)
pushd "%W2A_ROOT%"
go test ./tests/go_v_cpp/ -run "^TestGoVCpp$" -v -count=1 -timeout 120m >> "%OUT_X86%" 2>&1
if errorlevel 1 set "rc_fwd=1"
popd

>> "%OUT_X86%" echo.
echo ==^> PART 2/2: TestGoVCppAllTrain ^(section timing table + totals^)
>> "%OUT_X86%" echo ==^> PART 2/2: TestGoVCppAllTrain ^(section timing table + totals^)
pushd "%W2A_ROOT%"
go test ./tests/go_v_cpp/ -run "^TestGoVCppAllTrain$" -v -count=1 -timeout 180m >> "%OUT_X86%" 2>&1
if errorlevel 1 set "rc_train=1"
popd

for /f "delims=" %%D in ('powershell -NoProfile -Command "[DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')"') do set "END_UTC=%%D"
>> "%OUT_X86%" echo.
>> "%OUT_X86%" echo === snapshot complete %END_UTC% rc_go_v_cpp=%rc_fwd% rc_all_train=%rc_train% ===

copy /Y "%OUT_X86%" "%OUT_WIN%" >nul
copy /Y "%OUT_X86%" "%OUT_GIT%" >nul
echo ==^> wrote %OUT_X86% ^(+ %OUT_WIN%, %OUT_GIT%^)

if not "%rc_fwd%"=="0" goto :fail
if not "%rc_train%"=="0" goto :fail
echo ==^> ok
exit /b 0

:fail
echo error: snapshot tests failed ^(rc_fwd=%rc_fwd% rc_train=%rc_train%^); log retained 1>&2
exit /b 1
