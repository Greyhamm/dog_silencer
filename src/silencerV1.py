import numpy as np
import pyaudio
from msclap import CLAP
import soundfile as sf
import os
import joblib
import warnings
import RPi.GPIO as GPIO
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import threading
warnings.filterwarnings("ignore")

class DogDeterrentSystem:
    def __init__(self, trigger_pin=22, echo_pin=17):
        # Initialize Firebase
        cred = credentials.Certificate('firebase_config/firebaseconfig.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://dogsilencer-19cad-default-rtdb.firebaseio.com/'
        })
        
        # Initialize system state in Firebase
        self.ref = db.reference('dog_deterrent')
        self.ref.set({
            'system_enabled': True,
            'manual_pulse': False,
            'is_pulsing': False,
            'last_detection': None,
            'current_distance': None
        })
        
        # Initialize hardware
        self.initialize_hardware(trigger_pin, echo_pin)
        
        # Initialize detection models
        self.initialize_models()
        
        # Start the manual pulse monitoring thread
        self.pulse_thread = threading.Thread(target=self.monitor_manual_pulse)
        self.pulse_thread.daemon = True
        self.pulse_thread.start()

    def initialize_hardware(self, trigger_pin, echo_pin):
        GPIO.setmode(GPIO.BCM)
        self.GPIO_TRIGGER = trigger_pin
        self.GPIO_ECHO = echo_pin
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)
        GPIO.output(self.GPIO_TRIGGER, False)
        time.sleep(0.5)

    def initialize_models(self):
        self.clap_model = CLAP(version='2023', use_cuda=False)
        self.clf = joblib.load('dog_bark_classifier.pkl')
        
        # Audio parameters
        self.samplerate = 22050
        self.channels = 1
        self.chunk = 1024
        self.duration = 1
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.samplerate,
            input=True,
            input_device_index=3,
            frames_per_buffer=self.chunk
        )

    def measure_distance(self):
        GPIO.output(self.GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_TRIGGER, False)
        
        start_time = time.time()
        stop_time = time.time()
        
        while GPIO.input(self.GPIO_ECHO) == 0:
            start_time = time.time()
            if time.time() - stop_time > 0.1:
                return None
                
        while GPIO.input(self.GPIO_ECHO) == 1:
            stop_time = time.time()
            if stop_time - start_time > 0.1:
                return None
                
        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2
        return distance

    def activate_pulse(self, duration=1):
        self.ref.child('is_pulsing').set(True)
        start_time = time.time()
        while time.time() - start_time < duration:
            distance = self.measure_distance()
            if distance is not None:
                self.ref.child('current_distance').set(float(distance))
            # Reduced sleep to 5ms to simulate nearly continuous emission
            time.sleep(0.00025)
        self.ref.child('is_pulsing').set(False)


    def monitor_manual_pulse(self):
        while True:
            try:
                manual_pulse = self.ref.child('manual_pulse').get()
                if manual_pulse:
                    self.activate_pulse()
                    self.ref.child('manual_pulse').set(False)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in manual pulse monitoring: {e}")
                time.sleep(1)

    def run(self):
        print("Starting dog bark detection system...")
        temp_filename = 'temp_audio.wav'
        
        try:
            while True:
                # Check if system is enabled
                if not self.ref.child('system_enabled').get():
                    time.sleep(1)
                    continue

                frames = []
                num_frames = int(self.samplerate / self.chunk * self.duration)
                
                # Collect audio data
                for _ in range(num_frames):
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                data = np.hstack(frames)
                
                # Ensure correct audio length
                expected_length = int(self.samplerate * self.duration)
                if len(data) < expected_length:
                    data = np.pad(data, (0, int(expected_length - len(data))), 'constant')
                elif len(data) > expected_length:
                    data = data[:expected_length]
                
                # Save to temporary file
                sf.write(temp_filename, data, self.samplerate)
                
                # Extract audio embedding
                audio_embedding = self.clap_model.get_audio_embeddings([temp_filename])
                audio_embedding = audio_embedding.detach().numpy()
                
                # Make prediction
                prediction = self.clf.predict(audio_embedding)
                
                if prediction[0] == 1:
                    print("Dog bark detected!")
                    self.ref.child('last_detection').set(str(time.time()))
                    self.activate_pulse()
                
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup(temp_filename)

    def cleanup(self, temp_filename):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        GPIO.cleanup()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print("Cleanup completed")

if __name__ == "__main__":
    system = DogDeterrentSystem()
    system.run()