# EcoSort AI Dashboard

The original `main.py` remains unchanged. To use the dashboard and spoken bin
instructions, install the project's existing dependencies and run this
additional launcher from the project root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smart_sorter_dashboard.py
```

The dashboard reuses the camera, model, statistics, and history logic from
`main.py`. On Windows, it uses the built-in speech synthesizer to say the
recommended bin after an item is recognized and counted. Voice guidance can be
turned off in **Settings**.
