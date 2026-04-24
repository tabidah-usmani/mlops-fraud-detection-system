"""
Webhook Receiver for Prometheus Alerts
Receives alerts from Alertmanager and triggers actions
Run: python webhook_server.py
"""

from fastapi import FastAPI, Request
import uvicorn
import json
import datetime

app = FastAPI(title="Alert Webhook Receiver")

@app.post("/alerts")
async def receive_alert(request: Request):
    """Receive alerts from Alertmanager"""
    alert_data = await request.json()
    
    print("="*60)
    print(f"🔔 ALERT RECEIVED at {datetime.datetime.now()}")
    print("="*60)
    
    for alert in alert_data.get('alerts', []):
        status = alert.get('status', 'unknown')
        alertname = alert.get('labels', {}).get('alertname', 'unknown')
        
        # Color coding for different statuses
        if status == 'firing':
            status_icon = "🔥 FIRING"
        elif status == 'resolved':
            status_icon = "✅ RESOLVED"
        else:
            status_icon = f"⚠️ {status.upper()}"
        
        print(f"\n📋 Alert: {alertname}")
        print(f"   Status: {status_icon}")
        print(f"   Severity: {alert.get('labels', {}).get('severity', 'N/A')}")
        print(f"   Summary: {alert.get('annotations', {}).get('summary', 'N/A')}")
        print(f"   Description: {alert.get('annotations', {}).get('description', 'N/A')}")
        
        # Extract value from description if present
        desc = alert.get('annotations', {}).get('description', '')
        if 'Current recall' in desc:
            print(f"   📊 Metric value: {desc}")
        
        # Log for CI/CD trigger
        if status == 'firing' and alertname == 'ModelRecallDrop':
            print(f"\n   🚀 ACTION TRIGGERED: Retraining pipeline should start")
            print(f"   📡 Sending webhook to CI/CD system...")
            # Here you would call GitHub API to trigger retraining
    
    print("\n" + "="*60)
    
    return {"status": "received", "message": "Alert processed"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "webhook-receiver"}

@app.get("/")
async def root():
    return {
        "service": "Alert Webhook Receiver",
        "endpoints": {
            "POST /alerts": "Receive alerts from Alertmanager",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    print("="*60)
    print("  ALERT WEBHOOK RECEIVER")
    print("="*60)
    print("  Listening for alerts on: http://0.0.0.0:8080")
    print("  Alert endpoint: POST http://localhost:8080/alerts")
    print("="*60)
    print("\n⚠️  Make sure Alertmanager is configured to send alerts here")
    print("   alertmanager.yml should have webhook URL:")
    print("   - url: 'http://host.docker.internal:8080/alerts'\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)