@echo off
REM YOLOv11 Test Runner Script
REM 使用conda yolov11环境运行测试

REM 激活conda的yolov11环境并运行测试
D:\miniconda\Scripts\conda.exe run -n yolov11 python test\test_yolov11_inference.py

pause
