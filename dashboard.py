import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from main import Player
from window_util import get_windows

# Disable uvicorn access logs for less noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Player MK1 Dashboard")

# Global player instance
player = Player(config_path="config.yaml")

def recreate_player():
    global player
    current_region = player.vision_thread.region if hasattr(player, "vision_thread") else None
    player = Player(config_path="config.yaml")
    if current_region:
        player.vision_thread.set_region(current_region)

class WindowSelect(BaseModel):
    window_id: str

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("dashboard.html", "r") as f:
        return f.read()

@app.get("/api/windows")
async def list_windows():
    windows = get_windows()
    return {"windows": windows}

@app.post("/api/select_window")
async def select_window(selection: WindowSelect):
    wid = selection.window_id
    if not wid:
        # Reset to full screen
        player.vision_thread.set_region(None)
        return {"message": "Reset to full screen capture"}

    windows = get_windows()
    for w in windows:
        if w["id"] == wid:
            x, y, width, height = w["geometry"]
            # Important: mss expects monitor dict or tuple, handled in vision/_capture
            player.vision_thread.set_region((x, y, width, height))
            return {"message": f"Region updated to {w['title']} at {x},{y} {width}x{height}"}

    return {"message": "Window not found", "error": True}

@app.post("/api/start")
async def start_player():
    global player
    if not player._running:
        if player.vision_thread._sct is not None and not player.vision_thread._running:
            # Player was stopped previously, we must recreate to re-initialize threads
            recreate_player()
        # Start player in background thread so it doesn't block the API
        threading.Thread(target=player.run, daemon=True).start()
    return {"status": "started"}

@app.post("/api/stop")
async def stop_player():
    if player._running:
        player.stop()
    return {"status": "stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.1) # 10fps stream update rate for UI

            payload = {}

            # Fetch latest frame
            frame = player.frame_buffer.get_latest()
            if frame:
                # Downsample or lower quality for dashboard performance
                payload["image"] = frame.to_base64(format="jpeg", quality=50)

            # Fetch status
            active = player.tactical_layer.active_goal
            payload["status"] = {
                "running": player._running,
                "fps": f"{player.vision_thread.actual_fps:.1f}",
                "active_goal": active.id if active else "None",
                "pending_goals": [g.id for g in player.tactical_layer.pending_goals]
            }

            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")

if __name__ == "__main__":
    print("Starting dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
