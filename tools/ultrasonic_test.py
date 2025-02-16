import RPi.GPIO as GPIO
import time
import threading
from queue import Queue

class UltrasonicDevice:
    def __init__(self, trigger_pin=22, echo_pin=17):
        GPIO.setmode(GPIO.BCM)
        
        self.GPIO_TRIGGER = trigger_pin
        self.GPIO_ECHO = echo_pin
        
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)
        
        GPIO.output(self.GPIO_TRIGGER, False)
        time.sleep(0.5)
        
        self.is_pulsing = False
        self.current_frequency = 0
        self.distance_queue = Queue()

    def generate_pulses(self, frequency):
        """Generate pulses at specified frequency"""
        self.current_frequency = frequency
        delay = 1.0 / (frequency * 2)
        
        while self.is_pulsing:
            GPIO.output(self.GPIO_TRIGGER, True)
            time.sleep(delay)
            GPIO.output(self.GPIO_TRIGGER, False)
            time.sleep(delay)

    def measure_distance(self):
        """Measure distance and return in cm"""
        # Ensure trigger is off before measuring
        GPIO.output(self.GPIO_TRIGGER, False)
        time.sleep(0.05)  # Brief pause
        
        # Send trigger pulse
        GPIO.output(self.GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, False)

        start_time = time.time()
        stop_time = time.time()

        # Get start time
        timeout = start_time + 0.1  # 100ms timeout
        while GPIO.input(self.GPIO_ECHO) == 0:
            start_time = time.time()
            if start_time > timeout:
                return None

        # Get time of arrival
        while GPIO.input(self.GPIO_ECHO) == 1:
            stop_time = time.time()
            if stop_time > timeout:
                return None

        # Calculate distance
        elapsed = stop_time - start_time
        distance = (elapsed * 34300) / 2  # Speed of sound = 34300 cm/s
        return distance

    def distance_monitor(self):
        """Continuously monitor distance"""
        while self.is_pulsing:
            # Briefly pause pulsing to take measurement
            self.is_pulsing = False
            time.sleep(0.1)  # Let pulses settle
            
            distance = self.measure_distance()
            self.distance_queue.put(distance)
            
            self.is_pulsing = True
            time.sleep(0.9)  # Wait before next measurement

    def cleanup(self):
        self.is_pulsing = False
        time.sleep(0.1)
        GPIO.cleanup()

def main():
    device = UltrasonicDevice()
    
    try:
        while True:
            print("\nUltrasonic Pulse and Distance Test")
            print("1. Test 25kHz")
            print("2. Test 30kHz")
            print("3. Test 40kHz")
            print("4. Custom frequency")
            print("5. Exit")
            
            choice = input("Select an option (1-5): ")
            
            choice = input("Select an option (1-5): ")

            if choice == '5':
                return  # or break, depending on your loop structure

            if choice == '1':
                frequency = 25000
            elif choice == '2':
                frequency = 30000
            elif choice == '3':
                frequency = 40000
            elif choice == '4':
                frequency = int(input("Enter frequency in Hz (1000-40000): "))
            else:
                print("Invalid choice")
                continue  # or handle the error appropriately

          
                
            print(f"\nStarting {frequency}Hz pulses with distance monitoring")
            print("Press CTRL+C to stop")
            
            # Start pulsing
            device.is_pulsing = True
            
            # Create and start threads
            pulse_thread = threading.Thread(target=device.generate_pulses, args=(frequency,))
            monitor_thread = threading.Thread(target=device.distance_monitor)
            
            pulse_thread.start()
            monitor_thread.start()
            
            # Display distance measurements
            try:
                while True:
                    if not device.distance_queue.empty():
                        distance = device.distance_queue.get()
                        if distance is not None:
                            print(f"\rFrequency: {frequency}Hz  Distance: {distance:.1f} cm    ", end='')
                        else:
                            print("\rFrequency: {frequency}Hz  Distance: No reading    ", end='')
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                device.is_pulsing = False
                pulse_thread.join()
                monitor_thread.join()
                print("\nStopped")
            
            time.sleep(0.5)  # Brief pause between tests
            
    finally:
        device.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    main()