import gpiozero 
import time
import sys
RELAY_PIN=17
# active_high=False ? relay is ON when GPIO is LOW
relay = gpiozero.OutputDevice(RELAY_PIN, active_high=False, initial_value=False)

while True:
    
    relay.on()   # GPIO LOW ? turns relay ON (active low)
    print(relay.value)
    time.sleep(1)
    relay.off()  # GPIO HIGH ? turns relay OFF
    print(relay.value)
    time.sleep(1)
