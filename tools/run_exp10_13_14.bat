@echo off
REM Runs the pre-registered experiments one after another (about 4-5 hours in total).
REM Start from the repository root:  tools\run_exp10_13_14.bat
REM Each step writes a log to results\logs\. exp10 resumes where it stopped if interrupted.
if not exist results\logs mkdir results\logs
python tools\exp14_pharma_ets.py > results\logs\exp14.log 2>&1
python tools\exp13_pharma_lstm_wide.py > results\logs\exp13.log 2>&1
python tools\exp10_controlled_perturbation.py --part pharma > results\logs\exp10_pharma.log 2>&1
python tools\exp10_controlled_perturbation.py --part rossmann > results\logs\exp10_rossmann.log 2>&1
python tools\exp10_controlled_perturbation.py --part summary > results\logs\exp10_summary.log 2>&1
echo Selesai. Lihat results\logs\ dan results\exp1*.csv
