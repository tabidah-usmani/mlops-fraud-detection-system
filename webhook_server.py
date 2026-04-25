"""
Webhook Receiver for Prometheus Alerts - COMPLETE VERSION
Run: python webhook_server.py
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import datetime
import os
import subprocess
import threading
from contextlib import asynccontextmanager

# Global state
retraining_in_progress = False

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "tabidah-usmani/mlops-fraud-detection-system")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "tabidah-usmani")

def trigger_local_retraining(alert_name, alert_value):
    """Run retraining locally"""
    global retraining_in_progress
    
    if retraining_in_progress:
        print("   ⏳ Retraining already in progress, skipping...")
        return False
    
    retraining_in_progress = True
    
    print(f"   🚀 Starting LOCAL retraining...")
    print(f"   Reason: {alert_name} = {alert_value}")
    
    try:
        result = subprocess.run(
            ["python", "fraud_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(f"   ✅ Local retraining completed successfully!")
            return True
        else:
            print(f"   ❌ Local retraining failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ❌ Retraining timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        retraining_in_progress = False

def trigger_github_retraining(alert_name, alert_value, description=""):
    """Trigger GitHub Actions workflow"""
    
    if not GITHUB_TOKEN:
        print("   ⚠️ No GitHub token, using local retraining")
        return trigger_local_retraining(alert_name, alert_value)
    
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    payload = {
        "event_type": "retrain-model",
        "client_payload": {
            "alert_name": alert_name,
            "alert_value": alert_value,
            "description": description,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    
    try:
        import requests
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 204:
            print(f"   ✅ GitHub workflow triggered")
            print(f"   🔗 https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions")
            return True
        else:
            print(f"   ⚠️ GitHub trigger failed ({response.status_code})")
            return trigger_local_retraining(alert_name, alert_value)
    except Exception as e:
        print(f"   ⚠️ GitHub error: {e}")
        return trigger_local_retraining(alert_name, alert_value)

# Create FastAPI app
app = FastAPI(title="Alert Webhook Receiver")

@app.get("/")
async def root():
    return {
        "service": "Alert Webhook Receiver",
        "version": "2.0",
        "endpoints": {
            "POST /alerts": "Receive alerts from Alertmanager",
            "POST /retrain": "Manually trigger retraining",
            "GET /health": "Health check",
            "GET /status": "Check retraining status"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "github_configured": bool(GITHUB_TOKEN),
        "repo": f"{GITHUB_OWNER}/{GITHUB_REPO}" if GITHUB_TOKEN else "Not configured",
        "retraining_in_progress": retraining_in_progress
    }

@app.get("/status")
async def get_status():
    return {
        "retraining_in_progress": retraining_in_progress,
        "github_configured": bool(GITHUB_TOKEN),
        "timestamp": str(datetime.datetime.now())
    }

@app.post("/retrain")
async def manual_retrain():
    """Manually trigger retraining"""
    global retraining_in_progress
    
    print("\n" + "="*60)
    print(f"🔧 MANUAL RETRAINING TRIGGERED")
    print("="*60)
    
    if retraining_in_progress:
        return JSONResponse(
            status_code=409,
            content={"status": "already_running", "message": "Retraining already in progress"}
        )
    
    # Start retraining in background
    thread = threading.Thread(
        target=trigger_github_retraining,
        args=("manual", "manual", "Manual trigger")
    )
    thread.start()
    
    return {
        "status": "started",
        "message": "Retraining triggered",
        "repo": f"{GITHUB_OWNER}/{GITHUB_REPO}"
    }

@app.post("/alerts")
async def receive_alert(request: Request):
    """Receive alerts from Alertmanager"""
    try:
        alert_data = await request.json()
    except:
        alert_data = {}
    
    print("\n" + "="*60)
    print(f"🔔 ALERT RECEIVED at {datetime.datetime.now()}")
    print("="*60)
    
    triggered = False
    
    for alert in alert_data.get('alerts', []):
        status = alert.get('status', 'unknown')
        alertname = alert.get('labels', {}).get('alertname', 'unknown')
        
        # Extract value from description
        description = alert.get('annotations', {}).get('description', '')
        import re
        value_match = re.search(r'([0-9.]+)', description)
        alert_value = value_match.group(1) if value_match else 'unknown'
        
        print(f"\n📋 Alert: {alertname}")
        print(f"   Status: {status}")
        print(f"   Description: {description}")
        
        # Trigger only on firing alerts
        if status == 'firing' and alertname in ['ModelRecallDrop', 'ModelAucDrop', 'DataDriftDetected']:
            print(f"\n   🚀 TRIGGERING RETRAINING for: {alertname}")
            
            thread = threading.Thread(
                target=trigger_github_retraining,
                args=(alertname, alert_value, description)
            )
            thread.start()
            triggered = True
    
    print("="*60)
    
    return {"status": "received", "triggered": triggered}

if __name__ == "__main__":
    print("="*60)
    print("  ALERT WEBHOOK RECEIVER v2.0")
    print("="*60)
    print("  Server: http://localhost:8080")
    print("  POST /retrain - Manual retraining")
    print("  POST /alerts - Receive alerts")
    print("  GET /health - Health check")
    print("  GET /status - Check status")
    print("="*60)
    
    if GITHUB_TOKEN:
        print(f"✅ GitHub configured: {GITHUB_OWNER}/{GITHUB_REPO}")
    else:
        print("⚠️ GitHub not configured - will use local retraining")
    
    print("\n🚀 Starting server...\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)