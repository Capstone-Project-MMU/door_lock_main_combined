import gpiozero 
import time
import sys
RELAY_PIN=27 #17
# active_high=False ? relay is ON when GPIO is LOW
relay = gpiozero.OutputDevice(RELAY_PIN, active_high=True, initial_value=False)

def yes():    
    relay.on()   # GPIO LOW ? turns relay ON (active low)
    time.sleep(3)
    relay.off()  # GPIO HIGH ? turns relay OFF
