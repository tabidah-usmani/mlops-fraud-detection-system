"""
Webhook server to receive alerts and trigger retraining
"""
from fastapi import FastAPI, Request
import requests
import json

app = FastAPI()

GITHUB_TOKEN = "your_github_token"
REPO_OWNER = "your-username"
REPO_NAME = "fraud-detection"

@app.post("/alerts")
async def receive_alert(request: Request):
    alert_data = await request.json()
    
    print(f"Received alert: {alert_data}")
    
    # Determine alert type
    for alert in alert_data.get('alerts', []):
        alertname = alert.get('labels', {}).get('alertname')
        
        if alertname == 'ModelRecallDrop':
            print("🚨 Model recall drop detected - triggering retraining")
            trigger_retraining("model_performance_drop", alert)
        
        elif alertname == 'DataDriftDetected':
            print("📊 Data drift detected - triggering retraining")
            trigger_retraining("data_drift_detected", alert)
    
    return {"status": "received"}

def trigger_retraining(event_type, alert_data):
    """Trigger GitHub Actions workflow"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
    
    payload = {
        "event_type": event_type,
        "client_payload": {
            "alert": alert_data,
            "timestamp": alert_data.get('startsAt')
        }
    }
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 204:
        print(f"✓ Triggered {event_type} retraining")
    else:
        print(f"✗ Failed to trigger: {response.status_code}")