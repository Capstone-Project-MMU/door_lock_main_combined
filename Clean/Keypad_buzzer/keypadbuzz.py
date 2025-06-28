from gpiozero import DigitalOutputDevice, DigitalInputDevice, Buzzer
import time
import requests
from solenoid_test import yes

YOUR_PI_IP = "192.168.10.92"

# GPIO pin setup (BCM numbering)
ROW_PINS = [5, 6, 13, 19]     # 4 rows
COL_PINS = [12, 16, 20]       # 3 columns

# Define keypad layout
KEYS = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['*', '0', '#']
]

# Initialize buzzer
buzzer = Buzzer(17)

# Define passkey
PASSKEY = "1234"

# Setup rows as outputs (initially low)
rows = [DigitalOutputDevice(pin, initial_value=False) for pin in ROW_PINS]

# Setup columns as inputs with pull-down resistors
cols = [DigitalInputDevice(pin, pull_up=False) for pin in COL_PINS]

def scan_keypad():
    for row_idx, row in enumerate(rows):
        row.on()
        for col_idx, col in enumerate(cols):
            if col.value:
                row.off()
                return KEYS[row_idx][col_idx]
        row.off()
    return None

def incorrect_buzz():
    print("Access Denied")
    end_time = time.time() + 3
    while time.time() < end_time:
        buzzer.on()
        time.sleep(0.2)
        buzzer.off()
        time.sleep(0.2)

def correct_buzz():
    print("Access Granted")
    buzzer.on()
    time.sleep(1)
    yes()
    buzzer.off()

def main():
    print("Enter passkey and press #: ")
    entered = ""
    try:
        while True:
            key = scan_keypad()
            if key:
                print(f"Key Pressed: {key}")
                if key == "#":
                    if entered == PASSKEY:
                        correct_buzz()
                        requests.post(f"http://{YOUR_PI_IP}:8000/send-word", json={"word": "Numpad Access: True"})
                    else:
                        incorrect_buzz()
                        requests.post(f"http://{YOUR_PI_IP}:8000/send-word", json={"word": "Numpad Access: False"})
                    entered = ""  # reset input
                elif key == "*":
                    entered = ""
                    print("Input cleared")
                else:
                    entered += key
                time.sleep(0.3)  # debounce
    except KeyboardInterrupt:
        print("\nProgram stopped.")

if __name__ == "__main__":
    main()
