from pynput import keyboard

# Monitor key presses
pressed_keys = set()
toggles = {}
def is_pressed(key):
    return key in pressed_keys
def is_toggled(key):
    if key not in toggles:
        toggles[key] = False
    return toggles.get(key, False)
def on_press(key):
    key_repr = key.char if hasattr(key, 'char') else str(key)
    pressed_keys.add(key_repr)
    if key_repr in toggles:
        toggles[key_repr] = not toggles[key_repr]
def on_release(key):
    pressed_keys.discard(key.char if hasattr(key, 'char') else str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()
