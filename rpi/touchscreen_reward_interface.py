import pygame
import sys
import time
from datetime import datetime
from gpiozero import LED
from signal import pause

# --- Constants ---
TOUCHSCREEN_WIDTH = 800
TOUCHSCREEN_HEIGHT = 480
BLUE = (0, 0, 255)
RELAY_DURATION = 0.5  # seconds
RELAY_PIN = 17  # GPIO17 (BCM)

# --- GPIO Setup (gpiozero) ---
relay = LED(RELAY_PIN)
relay.on()  # Make sure it's off initially (I know it says 'on' here, but that's because of some wiring quirks)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Touchscreen Reward System")
screen.fill(BLUE)
pygame.display.update()


def trigger_reward():
    print(f"[{datetime.now().isoformat()}] ✅ Triggering reward...")
    relay.off()
    time.sleep(RELAY_DURATION)
    relay.on()
    print(f"[{datetime.now().isoformat()}] ✅ Reward delivered.")


print("🟦 Touchscreen reward interface is now running.")
print("Touch to trigger reward. ESC or close to exit.")

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
            elif event.type == pygame.MOUSEBUTTONDOWN:
                try:
                    with open("/tmp/last_detection.txt", "r") as f:
                        last_detection_time = float(f.read())
                    if time.time() - last_detection_time <= 10:
                        trigger_reward()
                    else:
                        print("[INFO] Touch occurred, but no recent detection.")
                except FileNotFoundError:
                    print("[INFO] Detection log not found — no reward triggered.")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] Exiting program.")

finally:
    relay.on()
    pygame.quit()
