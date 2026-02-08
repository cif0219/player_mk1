"""
Player MK1 - Vision Module
Asynchronous screen capture with shared frame buffer
"""

import io
import time
import threading
from typing import Optional, Tuple, Callable
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import mss
import mss.tools


@dataclass
class Frame:
    """A captured frame with metadata."""
    image: np.ndarray          # RGB numpy array (H, W, 3)
    timestamp: float           # Capture time
    index: int                 # Frame counter
    
    @property
    def pil(self) -> Image.Image:
        return Image.fromarray(self.image)
    
    def to_base64(self, format: str = "jpeg", quality: int = 85) -> str:
        import base64
        buffer = io.BytesIO()
        img = self.pil
        if format == "png":
            img.save(buffer, format="PNG")
        else:
            img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SharedFrameBuffer:
    """
    Thread-safe frame buffer.
    Vision writes, all layers read.
    """
    
    def __init__(self, max_frames: int = 10):
        self.max_frames = max_frames
        self._frames: list[Frame] = []
        self._lock = threading.RLock()
        self._new_frame = threading.Event()
        self._frame_count = 0
    
    def push(self, image: np.ndarray) -> Frame:
        """Push a new frame (called by vision thread)."""
        with self._lock:
            self._frame_count += 1
            frame = Frame(
                image=image,
                timestamp=time.perf_counter(),
                index=self._frame_count
            )
            self._frames.append(frame)
            if len(self._frames) > self.max_frames:
                self._frames.pop(0)
        
        self._new_frame.set()
        return frame
    
    def get_latest(self) -> Optional[Frame]:
        """Get the most recent frame."""
        with self._lock:
            return self._frames[-1] if self._frames else None
    
    def get_recent(self, n: int = 1) -> list[Frame]:
        """Get the N most recent frames."""
        with self._lock:
            return list(self._frames[-n:])
    
    def wait_for_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Wait for a new frame."""
        self._new_frame.wait(timeout=timeout)
        self._new_frame.clear()
        return self.get_latest()
    
    @property
    def frame_count(self) -> int:
        return self._frame_count


class VisionThread(threading.Thread):
    """
    Async vision capture thread.
    Captures frames at target FPS and pushes to shared buffer.
    """
    
    def __init__(
        self,
        buffer: SharedFrameBuffer,
        region: Optional[Tuple[int, int, int, int]] = None,
        resize: Optional[Tuple[int, int]] = None,
        target_fps: int = 60
    ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.region = region
        self.resize = resize
        self.target_fps = target_fps
        
        self._running = False
        self._sct: Optional[mss.mss] = None
        
        # Stats
        self.actual_fps = 0.0
        self._fps_samples: list[float] = []
    
    def run(self):
        self._sct = mss.mss()
        self._running = True
        
        interval = 1.0 / self.target_fps
        last_time = time.perf_counter()
        
        while self._running:
            start = time.perf_counter()
            
            # Capture
            frame = self._capture()
            self.buffer.push(frame)
            
            # FPS tracking
            now = time.perf_counter()
            self._fps_samples.append(1.0 / (now - last_time) if now > last_time else 0)
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
            self.actual_fps = sum(self._fps_samples) / len(self._fps_samples)
            last_time = now
            
            # Sleep to maintain target FPS
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def _capture(self) -> np.ndarray:
        """Capture a single frame."""
        if self.region:
            x, y, w, h = self.region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = self._sct.monitors[1]  # Primary monitor
        
        screenshot = self._sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        if self.resize:
            img = img.resize(self.resize, Image.Resampling.LANCZOS)
        
        return np.array(img)
    
    def stop(self):
        self._running = False


# Legacy compatibility
class Vision:
    """Synchronous vision (for simple use cases)."""
    
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        resize: Optional[Tuple[int, int]] = None,
        format: str = "jpeg",
        quality: int = 85
    ):
        self.region = region
        self.resize = resize
        self.format = format.lower()
        self.quality = quality
        self._sct = mss.mss()
    
    def capture(self) -> Image.Image:
        if self.region:
            x, y, w, h = self.region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = self._sct.monitors[1]
        
        screenshot = self._sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        if self.resize:
            img = img.resize(self.resize, Image.Resampling.LANCZOS)
        
        return img
    
    def capture_bytes(self) -> bytes:
        img = self.capture()
        buffer = io.BytesIO()
        if self.format == "png":
            img.save(buffer, format="PNG")
        else:
            img.save(buffer, format="JPEG", quality=self.quality)
        return buffer.getvalue()
    
    def capture_base64(self) -> str:
        import base64
        return base64.b64encode(self.capture_bytes()).decode("utf-8")
    
    @property
    def mime_type(self) -> str:
        return f"image/{self.format}"


class FrameBuffer:
    """Legacy simple frame buffer."""
    
    def __init__(self, max_frames: int = 5):
        self.max_frames = max_frames
        self._frames: list[Tuple[float, Image.Image]] = []
    
    def push(self, frame: Image.Image):
        self._frames.append((time.time(), frame))
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)
    
    def get_recent(self, n: int = 1) -> list[Image.Image]:
        return [f[1] for f in self._frames[-n:]]
    
    def clear(self):
        self._frames.clear()


if __name__ == "__main__":
    # Test async vision
    buffer = SharedFrameBuffer()
    vision = VisionThread(buffer, resize=(1280, 720), target_fps=60)
    vision.start()
    
    print("Capturing for 3 seconds...")
    time.sleep(3)
    vision.stop()
    
    print(f"Captured {buffer.frame_count} frames")
    print(f"Actual FPS: {vision.actual_fps:.1f}")
    
    frame = buffer.get_latest()
    if frame:
        print(f"Latest frame: {frame.image.shape} @ {frame.timestamp:.3f}")
