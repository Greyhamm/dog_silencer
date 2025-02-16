import RPi.GPIO as GPIO
import time

class HCSR04:
    def __init__(self, trigger_pin=22, echo_pin=17):  # Updated pin numbers
        # Use BCM GPIO references instead of physical pin numbers
        GPIO.setmode(GPIO.BCM)
        
        # Define GPIO pins
        self.GPIO_TRIGGER = trigger_pin
        self.GPIO_ECHO = echo_pin
        
        # Set pins as output and input
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)  # Trigger
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)      # Echo
        
        # Set trigger to False (Low) initially
        GPIO.output(self.GPIO_TRIGGER, False)
        
        # Allow module to settle
        time.sleep(0.5)
        
    def measure_distance(self):
        """
        Measure distance using the ultrasonic sensor
        Returns distance in centimeters
        """
        # Send 10us pulse to trigger
        GPIO.output(self.GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, False)
        
        start_time = time.time()
        stop_time = time.time()
        
        # Save start time
        while GPIO.input(self.GPIO_ECHO) == 0:
            start_time = time.time()
            # Add timeout to avoid infinite loop
            if time.time() - stop_time > 0.1:
                return None
        
        # Save time of arrival
        while GPIO.input(self.GPIO_ECHO) == 1:
            stop_time = time.time()
            # Add timeout to avoid infinite loop
            if stop_time - start_time > 0.1:
                return None
        
        # Time difference between start and arrival
        time_elapsed = stop_time - start_time
        
        # Multiply by the sonic speed (34300 cm/s) and divide by 2,
        # because there and back
        distance = (time_elapsed * 34300) / 2
        
        return distance
    
    def cleanup(self):
        """
        Cleanup GPIO pins
        """
        GPIO.cleanup()

def main():
    try:
        sensor = HCSR04()  # Using default pins (TRIGGER=22, ECHO=17)
        print("HC-SR04 Distance Measurement Test")
        print("Using GPIO22 for TRIGGER and GPIO17 for ECHO")
        print("Press CTRL+C to exit")
        
        while True:
            distance = sensor.measure_distance()
            
            if distance is not None:
                print(f"Distance: {distance:.1f} cm")
            else:
                print("Measurement timed out")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nMeasurement stopped by user")
    finally:
        sensor.cleanup()
        print("GPIO cleaned up")

if __name__ == "__main__":
    main()