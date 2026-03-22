import subprocess
import sys

def get_windows():
    """Returns a list of dicts: [{'id': '...', 'title': '...', 'geometry': [x, y, w, h]}]"""
    windows = []

    if sys.platform == "linux" or sys.platform == "linux2":
        try:
            # Requires xdotool
            output = subprocess.check_output(["xdotool", "search", "--name", ".*"]).decode("utf-8")
            window_ids = output.strip().split("\n")

            for wid in window_ids:
                if not wid: continue
                try:
                    name_out = subprocess.check_output(["xdotool", "getwindowname", wid], stderr=subprocess.DEVNULL).decode("utf-8").strip()
                    geom_out = subprocess.check_output(["xdotool", "getwindowgeometry", wid], stderr=subprocess.DEVNULL).decode("utf-8")

                    if not name_out: continue

                    # Parse geometry
                    # Example geom_out:
                    # Window 62914565
                    #   Position: 0,33 (server)
                    #   Geometry: 1920x1047
                    lines = geom_out.split("\n")
                    pos = lines[1].split(":")[1].split("(")[0].strip()
                    geom = lines[2].split(":")[1].strip()

                    x, y = map(int, pos.split(","))
                    w, h = map(int, geom.split("x"))

                    windows.append({
                        "id": wid,
                        "title": name_out,
                        "geometry": [x, y, w, h]
                    })
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Error getting windows on linux (is xdotool installed?): {e}")

    elif sys.platform == "win32":
        try:
            import pygetwindow as gw
            for w in gw.getAllWindows():
                if w.title:
                    windows.append({
                        "id": str(w._hWnd),
                        "title": w.title,
                        "geometry": [w.left, w.top, w.width, w.height]
                    })
        except Exception as e:
             print(f"Error getting windows on Windows: {e}")
    else:
        print(f"Platform {sys.platform} not supported for window capture.")

    return windows
