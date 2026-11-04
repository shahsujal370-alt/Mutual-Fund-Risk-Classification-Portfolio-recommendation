MUTUAL FUND RISK CLASSIFICATION - STREAMLIT DEPLOYMENT

IMPORTANT: Keep all these files in the ROOT of your GitHub repository (not inside another folder):
- app.py
- requirements.txt
- risk_classifier.joblib
- mutual_funds_clean.csv

1. Rename any existing app(1).py to app.py (or upload this package's app.py).
2. Ensure requirements.txt is in the SAME ROOT folder as app.py.
3. Ensure risk_classifier.joblib and mutual_funds_clean.csv are also in that folder.
4. Commit and push all files to GitHub.
5. In Streamlit Community Cloud, choose the repository and set Main file path to app.py.
6. If the app was already deployed, open Manage app > Reboot after pushing the corrected files.

The requirements file explicitly installs joblib, which provides `import joblib`.
If dependency installation still appears stale, use Manage app > Reboot, or delete and redeploy the app after confirming requirements.txt is committed to the selected branch.
