@echo off
rem D_10 daily batch wrapper.
rem ASCII ONLY in this file: cmd.exe reads .bat bytes in the OEM codepage
rem (CP950 on zh-TW Windows), so any UTF-8 Chinese here becomes mojibake and
rem broke the scheduled run on 2026-07-17 (cd to a garbled path -> exit 1).
rem The repo path contains Chinese, so resolve it at runtime via %~dp0
rem (this .bat lives at <repo>\d10_trading\scripts\).
cd /d "%~dp0..\.."
if not exist d10_trading\var\logs mkdir d10_trading\var\logs
C:\Users\User\anaconda3\envs\ML_env\python.exe -X utf8 -m d10_trading.main >> d10_trading\var\logs\scheduler.log 2>&1
