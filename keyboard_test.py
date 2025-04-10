import time
from pynput import keyboard

# A set to track currently pressed keys
pressed_keys = set()
def on_press(key):
    pressed_keys.add(key.char if hasattr(key, 'char') else str(key))
def on_release(key):
    pressed_keys.discard(key.char if hasattr(key, 'char') else str(key))
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
while True:
    if 'w' in pressed_keys:
        print("W is pressed")
    else:
        print("W is not pressed")
    print(pressed_keys)
    time.sleep(0.1)