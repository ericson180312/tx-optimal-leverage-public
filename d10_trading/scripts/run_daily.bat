@echo off
rem D_10 日批次包裝：工作排程器每平日 13:20 呼叫本檔
rem （非交易日/已執行/KILL_SWITCH 由 d10_trading.main 自行判斷退出）
cd /d C:\Users\User\Desktop\python_台指最佳槓桿
if not exist d10_trading\var\logs mkdir d10_trading\var\logs
C:\Users\User\anaconda3\envs\ML_env\python.exe -X utf8 -m d10_trading.main >> d10_trading\var\logs\scheduler.log 2>&1
