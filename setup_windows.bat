@echo off
cd /d "%~dp0"
py -3.11 portable.py setup
if errorlevel 1 pause
