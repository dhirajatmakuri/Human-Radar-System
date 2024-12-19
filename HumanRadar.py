from picamera2 import Picamera2
import time
import numpy as np
import RPi.GPIO as GPIO
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit
from twilio.rest import Client
import requests
import logging
from datetime import datetime
import serial
import mediapipe as mp
import cv2
import adafruit_bh1750
import os

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize I2C bus for the PCA9685 PWM Controller and BH1750 Light sensor
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50
kit = ServoKit(channels=16)

# Camera servo constants
PAN_CHANNEL = 1
TILT_CHANNEL = 2
SERVO_MIN = 0
SERVO_MAX = 180
CENTER_X = 90
CENTER_Y = 90
MIN_TILT_ANGLE = 110  # For fall detection, it won't tilt down below this angle

# Steering constants
STEERING_CENTER = 90
STEERING_MAX_RIGHT = 45
STEERING_MAX_LEFT = 135
TURN_THRESHOLD = 5
MAX_TURN_ANGLE = 60

# Optimal, fixed motor speed
MOTOR_SPEED = 70

# Scanning parameters
SCAN_SPEED = 30
SCAN_RANGE_MIN = 0
SCAN_RANGE_MAX = 180

# Fall Detection Constants
FALL_DELAY = 5  # seconds
Y_THRESHOLD_PERCENT = 0.7

# Environment variables for Github
ACCOUNT_SID = os.environ['ACCOUNT_SID']
AUTH_TOKEN = os.environ['AUTH_TOKEN']
PHONE_NUMBER = os.environ['PHONE_NUMBER']

# Car servo and motor setup
in1 = 23
in2 = 24
en = 25
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
p_motor = GPIO.PWM(en, 1000)
p_motor.start(0)

class LocationWhatsAppSender:
    def __init__(self, account_sid, auth_token):
        self.client = Client(account_sid, auth_token)
        self.from_number = 'whatsapp:+14155238886'
        
        # Logging
        logging.basicConfig(
            filename='fall_detection_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def get_ip_location(self, ip_address=None):
        '''
        To emulate the original GPS functionality we planned. Basically gets the geographical location of the current public IP
        '''
        try:
            if not ip_address:
                public_ip = requests.get('https://api.ipify.org').text
                ip_address = public_ip
                
            response = requests.get(f'https://ipapi.co/{ip_address}/json/')
            data = response.json()
            
            if response.status_code != 200 or 'error' in data:
                return f"Error getting location data: {data.get('reason', 'Unknown error')}"
                
            location_info = {
                'ip': data.get('ip'),
                'city': data.get('city'),
                'region': data.get('region'),
                'country': data.get('country_name'),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'isp': data.get('org'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return location_info
            
        except Exception as e:
            error_msg = f"Error getting location: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def format_location_message(self, location_info, custom_message=""):
        '''
        Format location details into something more nice and readable for the actual text message
        '''
        if isinstance(location_info, str):
            return f"{custom_message}\nError: {location_info}"

        message = f"{custom_message}\n\nLocation Details:\n"
        message += f"City: {location_info['city']}\n"
        message += f"Region: {location_info['region']}\n"
        message += f"Country: {location_info['country']}\n"
        message += f"Coordinates: {location_info['latitude']}, {location_info['longitude']}\n"
        message += f"Timestamp: {location_info['timestamp']}\n"
        message += f"\nGoogle Maps Link: https://www.google.com/maps?q={location_info['latitude']},{location_info['longitude']}"
        
        return message

    def send_fall_alert(self, to_number):
        '''
        Sends the whatsaspp SMS alert when a fall is detected
        '''
        try:
            location_info = self.get_ip_location()
            full_message = self.format_location_message(location_info, "ALERT: Fall detected!")
            
            formatted_to = f'whatsapp:{to_number}'
            
            message = self.client.messages.create(
                body=full_message,
                from_=self.from_number,
                to=formatted_to
            )
            
            logging.info(f"Fall alert sent successfully to {to_number}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to send fall alert to {to_number}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            return False

class PersonFollower:
    '''
    This is going to be the main class we want to coordinate all the subsystems.
    Pseudo-code:
    1) Initialize all the hardware components
    1.1) Configure servo motor parameters
    1.2) Initializes vision and sensor systems 
    1.3) Establish communication channels

    2) In the main loop,
    2.1) Capture and process camera frame
    2.2) Detect person pose with mediapipe
    2.3) Update servos to track person
    2.4) Control movement of RC car based on lidar
    2.5) Monitor for a fall
    2.6) Handle a person disappearing from frame and starting a scan
    2.7) Camera switching based on the current light level
    '''
    def __init__(self, whatsapp_sender, alert_number):
        self.whatsapp_sender = whatsapp_sender
        self.alert_number = alert_number

        # Initializing the subsytems
        self.setup_camera()
        self.setup_servos()
        self.setup_tracking()
        self.setup_scanning()
        self.setup_car_control()
        self.setup_lidar()
        self.last_known_position = CENTER_X

        # Fall detection variables
        self.fall_potential = False
        self.fall_time = None
        self.last_alert_time = 0
        self.alert_cooldown = 60  # 1 minute cooldown between alerts to prevent spam

        # Initialize CLAHE for adaptive histogram equalization; this image processing will make low light scenarios be brighter
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Initialize light sensor
        self.init_light_sensor()
        self.last_light_check_time = time.time()
        self.current_camera_num = 1  # Let's just start with camera 1 (normal one)

    def init_light_sensor(self):
        # Initialize the BH1750 light sensor
        try:
            self.i2c_bus = busio.I2C(board.SCL, board.SDA)
            self.light_sensor = adafruit_bh1750.BH1750(self.i2c_bus)
            print("Light sensor initialized")
        except Exception as e:
            print(f"Failed to initialize light sensor: {e}")
            self.light_sensor = None

    def setup_camera(self):
        self.picam2 = Picamera2(camera_num=1)  # Specify camera 1
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (820, 616), "format": "RGB888"},
            buffer_count=4
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        self.current_camera_num = 1
        print("Camera initialized successfully with camera 1")

    def setup_servos(self):
        '''
        Defaults to middle position on program start
        '''
        self.current_pan = CENTER_X
        self.current_tilt = CENTER_Y
        kit.servo[PAN_CHANNEL].angle = self.current_pan
        kit.servo[TILT_CHANNEL].angle = self.current_tilt
        print("Camera servos initialized")

    def setup_tracking(self):
        self.Kp = 0.3
        self.Ki = 0.01
        self.Kd = 0.05
        self.pan_error_sum = 0
        self.tilt_error_sum = 0
        self.last_pan_error = 0
        self.last_tilt_error = 0
        self.smooth_factor = 0.7

    def setup_scanning(self):
        self.scanning = True
        self.scan_direction = 1
        self.last_scan_time = time.time()
        self.person_lost_time = None
        print("Starting in scan mode")

    def setup_car_control(self):
        self.desired_distance = 2  # Distance we want to be from the person in meters
        self.distance_tolerance = 0.2  # Tolerance of the distance we want to be from the person in meters
        self.center_threshold = 0.2
        self.last_movement_time = time.time()
        self.movement_cooldown = 0.1

    def setup_lidar(self):
        # Configure the serial port for lidar
        self.ser = serial.Serial(
            port='/dev/ttyAMA0',  # UART port is here
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0  # Non-blocking mode
        )
        self.last_valid_distance = None
        print("Lidar initialized")

    def read_lidar_distance(self):
        count = self.ser.in_waiting
        if count >= 9:
            recv = self.ser.read(9)
            self.ser.reset_input_buffer()
            
            if recv[0] == 0x59 and recv[1] == 0x59:
                distance = recv[2] + recv[3] * 256
                meters_distance = distance / 100.0
                self.last_valid_distance = meters_distance
                return meters_distance
        # If no new data, return last valid distance
        return self.last_valid_distance

    def start_scanning(self):
        '''
        We'll want this for person re-acquisition if someone got lost from frame; like a radar, this will move 180 degrees to the last direction we saw someone, turn and move 180 degrees again, and repeat
        '''
        print("Starting scan from last known position:", self.last_known_position)
        self.scanning = True
        self.last_scan_time = time.time()
        
        if self.last_known_position > SCAN_RANGE_MAX:
            self.scan_direction = -1
            self.current_pan = SCAN_RANGE_MAX
        elif self.last_known_position < SCAN_RANGE_MIN:
            self.scan_direction = 1
            self.current_pan = SCAN_RANGE_MIN
        else:
            self.current_pan = self.last_known_position
            if self.last_known_position > CENTER_X:
                self.scan_direction = -1
            else:
                self.scan_direction = 1
                
        print(f"Starting scan from position {self.current_pan} with direction {self.scan_direction}")
        kit.servo[PAN_CHANNEL].angle = self.current_pan

    def update_scan(self):
        if not self.scanning:
            return

        current_time = time.time()
        elapsed_time = current_time - self.last_scan_time

        movement = SCAN_SPEED * elapsed_time * self.scan_direction
        new_pan = self.current_pan + movement

        if new_pan >= SCAN_RANGE_MAX:
            self.scan_direction = -1
            new_pan = SCAN_RANGE_MAX
        elif new_pan <= SCAN_RANGE_MIN:
            self.scan_direction = 1
            new_pan = SCAN_RANGE_MIN
            
        self.current_pan = new_pan
        kit.servo[PAN_CHANNEL].angle = self.current_pan
        self.last_scan_time = current_time

    def move_forward(self, speed=MOTOR_SPEED):
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        p_motor.ChangeDutyCycle(speed)

    def move_backward(self, speed=MOTOR_SPEED):
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        p_motor.ChangeDutyCycle(speed)

    def stop_motors(self):
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        p_motor.ChangeDutyCycle(0)

    def apply_post_processing(self, frame):
        # Ensure the frame has 3 channels (BGR)
        if frame.shape[2] == 4:  # If it has an alpha channel
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert to grayscale for brightness equalization
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization
        equalized_frame = self.clahe.apply(gray_frame)

        # Convert back to BGR
        equalized_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

        # Detect and handle overexposed areas
        mask = self.detect_overexposure(gray_frame)
        frame[mask] = equalized_frame[mask]

        # Additional low-light enhancement
        frame = self.enhance_low_light(frame)

        return frame

    def detect_overexposure(self, gray_frame):
        # Define a threshold for detecting overexposed regions
        overexposed_threshold = 240
        mask = gray_frame > overexposed_threshold
        return mask

    def enhance_low_light(self, frame):
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        l_channel = self.clahe.apply(l_channel)

        # Merge channels
        lab = cv2.merge((l_channel, a, b))

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Adjust gamma for better visibility in low light; We probably want a hardware fix instead of software
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)

        return enhanced

    def update_camera_servos(self, target_x, target_y, frame_width, frame_height):
        '''
        Uses PID control for smooth, accurate camera movements tracking the target.
        '''
        self.scanning = False
        self.person_lost_time = None
        
        self.last_known_position = self.current_pan
        
        error_x = (target_x - frame_width/2) / (frame_width/2)
        error_y = (target_y - frame_height/2) / (frame_height/2)

        self.pan_error_sum = self.pan_error_sum * 0.8 + error_x
        pan_derivative = error_x - self.last_pan_error
        pan_output = (self.Kp * error_x) + (self.Ki * self.pan_error_sum) + (self.Kd * pan_derivative)
        self.last_pan_error = error_x

        self.tilt_error_sum = self.tilt_error_sum * 0.8 + error_y
        tilt_derivative = error_y - self.last_tilt_error
        tilt_output = (self.Kp * error_y) + (self.Ki * self.tilt_error_sum) + (self.Kd * tilt_derivative)
        self.last_tilt_error = error_y

        target_pan = self.current_pan - pan_output * 60
        target_tilt = self.current_tilt - tilt_output * 60

        self.current_pan = self.current_pan * (1 - self.smooth_factor) + target_pan * self.smooth_factor
        self.current_tilt = self.current_tilt * (1 - self.smooth_factor) + target_tilt * self.smooth_factor

        self.current_pan = max(SCAN_RANGE_MIN, min(SCAN_RANGE_MAX, self.current_pan))
        self.current_tilt = max(MIN_TILT_ANGLE, min(SERVO_MAX, self.current_tilt))

        kit.servo[PAN_CHANNEL].angle = self.current_pan
        kit.servo[TILT_CHANNEL].angle = self.current_tilt

        return self.current_pan

    def get_steering_angle(self, pan_angle):
        angle_diff = pan_angle - CENTER_X
        
        if abs(angle_diff) < TURN_THRESHOLD:
            return STEERING_CENTER
        
        steering_factor = (angle_diff / MAX_TURN_ANGLE) ** 1.5
        steering_factor = min(abs(steering_factor), 1.0)
        
        if angle_diff > 0:
            steering_angle = STEERING_CENTER - (MAX_TURN_ANGLE * steering_factor)
        else:
            steering_angle = STEERING_CENTER + (MAX_TURN_ANGLE * steering_factor)
            
        return int(steering_angle)

    def control_car(self, pose_landmarks, frame_width, pan_angle, distance):
        '''
        Coordinates robot movement based on visual tracking and LIDAR data.
        '''
        steering_angle = self.get_steering_angle(pan_angle)
        kit.servo[0].angle = steering_angle

        if distance is not None:
            if distance > self.desired_distance + self.distance_tolerance:
                print(f"Moving forward - Distance: {distance:.2f}m - Steering: {steering_angle}")
                self.move_forward()
            elif distance < self.desired_distance - self.distance_tolerance:
                print(f"Moving backward - Distance: {distance:.2f}m - Steering: {steering_angle}")
                self.move_backward()
            else:
                print(f"Good distance - Distance: {distance:.2f}m - Steering: {steering_angle}")
                self.stop_motors()
        else:
            print("Distance data not available, stopping motors")
            self.stop_motors()

    def handle_fall_detection(self, nose_y, frame_height, current_time):
        '''
        Monitors person's vertical position and triggers alerts when falls are detected.
        '''
        y_threshold = frame_height * Y_THRESHOLD_PERCENT
        if nose_y > y_threshold:
            if not self.fall_potential:
                self.fall_potential = True
                self.fall_time = current_time
                print("Potential fall detected, starting timer...")
        else:
            self.fall_potential = False
            self.fall_time = None

        if self.fall_potential and (current_time - self.fall_time >= FALL_DELAY):
            print("Fall detected!")
            if current_time - self.last_alert_time >= self.alert_cooldown:
                if self.whatsapp_sender.send_fall_alert(self.alert_number):
                    print("Fall alert sent successfully")
                    self.last_alert_time = current_time
                else:
                    print("Failed to send fall alert")
            else:
                print("Alert cooldown active, skipping notification")
            self.fall_potential = False
            self.fall_time = None

    def switch_camera(self, camera_num):
        print(f"Switching to camera {camera_num}")
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception as e:
            print(f"Error stopping camera: {e}")
        self.picam2 = Picamera2(camera_num=camera_num)
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (820, 616), "format": "RGB888"},
            buffer_count=4
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        self.current_camera_num = camera_num
        print(f"Camera switched to {camera_num}")

    def run(self):
        try:
            print("Starting person following and fall detection system.")
            frames_without_person = 0
            last_process_time = time.time()
            
            while True:
                current_time = time.time()

                # Check light level every 20 seconds
                if self.light_sensor and (current_time - self.last_light_check_time >= 5):
                    light_level = self.light_sensor.lux
                    print(f"Light Level: {light_level:.2f} lux")
                    self.last_light_check_time = current_time
                    if light_level < 20 and self.current_camera_num != 0:
                        # Switch to night vision camera 0
                        self.switch_camera(0)
                    elif light_level >= 20 and self.current_camera_num != 1:
                        # Switch back to day camera 1
                        self.switch_camera(1)

                frame = self.picam2.capture_array()

                # Apply post-processing for better low-light performance
                frame = self.apply_post_processing(frame)

                if current_time - last_process_time < 0.1:  # Limit to ~10 FPS
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Read lidar distance
                    distance = self.read_lidar_distance()
                    
                    if results.pose_landmarks:
                        frames_without_person = 0
                        
                        mp_draw.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
                        
                        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        nose_x = int(nose.x * frame_width)
                        nose_y = int(nose.y * frame_height)
                        
                        current_pan = self.update_camera_servos(nose_x, nose_y, frame_width, frame_height)
                        
                        self.control_car(results.pose_landmarks, frame_width, current_pan, distance)

                        # Fall detection
                        self.handle_fall_detection(nose_y, frame_height, current_time)
                        
                        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, "Following", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    else:
                        frames_without_person += 1
                        if frames_without_person == 10:
                            print(f"Person lost at position {self.last_known_position}, starting scan")
                            self.start_scanning()
                        
                        if frames_without_person >= 10:
                            self.stop_motors()
                            kit.servo[0].angle = 90
                            self.update_scan()
                            cv2.putText(frame, f"Scanning... ({int(self.current_pan)})", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # If fall was being detected when person is lost
                        if self.fall_potential:
                            self.handle_fall_detection(nose_y, frame_height, current_time)
                    
                    # Helpful values
                    cv2.putText(frame, f"Mode: {'Scanning' if self.scanning else 'Tracking'}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Pan: {int(self.current_pan)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Last Known: {int(self.last_known_position)}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if distance is not None:
                        cv2.putText(frame, f"Distance: {distance:.2f}m", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow('Person Following and Fall Detection', frame)
                
                last_process_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
        cv2.destroyAllWindows()
        self.stop_motors()

        # Re-center everything
        kit.servo[0].angle = 90
        kit.servo[PAN_CHANNEL].angle = CENTER_X
        kit.servo[TILT_CHANNEL].angle = CENTER_Y

        p_motor.stop()
        GPIO.cleanup()
        pose.close()
        self.ser.close()
        print("Cleanup complete")

if __name__ == "__main__":

    # SMS Messaging details
    whatsapp_sender = LocationWhatsAppSender(ACCOUNT_SID, AUTH_TOKEN)

    # Start the code    
    follower = PersonFollower(
        whatsapp_sender=whatsapp_sender,
        alert_number=PHONE_NUMBER  # Phone number as an env var like +11234567890
    )
    follower.run()