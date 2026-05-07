@echo off
REM change to project folder (optional, for relative paths)
cd /d "C:\New Plant Diseases Dataset(Augmented)"
REM run Streamlit using the env python (keeps it in that env)
"C:\Users\bibek\.conda\envs\plant_ai\python.exe" -m streamlit run "C:\New Plant Diseases Dataset(Augmented)\main.py"
pause
