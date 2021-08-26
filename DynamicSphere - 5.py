# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:38:11 2021
Changes and modification Version 1 - V-1.0.0

@author: Munish
Edited by: Hemant Singh D, Dated 02-Aug-2021
DynamicSphere Team 

"""
import tkinter as tk
from time import strftime
import joblib as jbl
import statistics as stats
import pandas as pd
from sklearn.model_selection import train_test_split as split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import time as t
import skimage.restoration as dw
import numpy as np
import time
from adafruit_servokit import ServoKit
from ADCDifferentialPi import ADCDifferentialPi
import RPi.GPIO as GPIO 
import threading
import math as m


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self,bg='orchid')

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, EMG_ML, EMG_Signal, Hand_control, ML_Data_preparation):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent,bg='orchid')
        label = tk.Label(self, text="Dynamic Sphere", bg='sky blue', fg='white', font=('Impact',30,'bold'))
        #label.place(x=10,y=10)
        label.place(x=self.winfo_screenwidth()/2 ,y=10)

        button = tk.Button(self, text="EMG\n Machine Learning\n model", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(EMG_ML))
        button.place(x=100, y=100)

        button2 = tk.Button(self, text="EMG\n Signal Acquisition", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(EMG_Signal))
        button2.place(x=400, y=100)

        button3 = tk.Button(self, text="Hand Control", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(Hand_control))
        button3.place(x=750, y=100)

        button4 = tk.Button(self, text="ML Data\n preparation", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(ML_Data_preparation))
        button4.place(x=1050, y=100)


class EMG_ML(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='brown')
        label = tk.Label(self, text="EMG Machine Learningmodel", fg='white',bg="goldenrod",bd=0,font=('times new roman', 30))
        label.pack(pady=10,padx=10)

        b1_on = tk.Button(self, text = "RUN EMG Machinelerning model", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.EMG_Main) 
        b1_on.place(x=400, y=200)
        #b1_off = tk.Button(self, text = "OFF", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #b1_off.place(x=400, y=300)
        #reset = tk.Button(self, text = "RESET", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #reset.place(x=400, y=400)
        button1 = tk.Button(self, text="Back to Home", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)

        self.start_time = time.time()

        # Load signals data
        self.ml_data = jbl.load('ML_Data_5.joblib')
        self.class1 = self.ml_data[0] # Activation
        self.class2 = self.ml_data[1] # Idle Contact
        self.class3 = self.ml_data[2] # Noise
        self.class4 = self.ml_data[3] # No Contact

        # Calculate features
    def calc_features(self, class_data, lag_number):
        self.class_features = []
        for i in range(len(class_data)):
            self.x = class_data[i]
            # Variance
            self.V = stats.variance(self.x)
            # Autocorrelation function
            self.series = pd.Series(self.x)
            self.R = [self.series.autocorr(lag = r) for r in range(lag_number+1)]
            self.class_features.append([self.V]+self.R[1:])
        return self.class_features

    # Maching learning  
    def train_classifier(self, features, responses, testPercentage):
        self.X_train, self.X_test, self.Y_train, self.Y_test = split(self.features, self.responses, test_size = self.testPercentage)
        self.accuracy = 0
        while self.accuracy < 0.99:
            # Train KNN classifier
            self.knn = KNeighborsClassifier(n_neighbors=3)
            self.knn.fit(self.X_train, self.Y_train)
            # Predict response for test data
            self.Y_pred = self.knn.predict(self.X_test)
            # Check accuracy
            self.accuracy = metrics.accuracy_score(self.Y_test, self.Y_pred)
            
        # Confusion matrix and Classification report
        #class_names = ['Activation','Idle','Noise','No Contact']
        #cReport = metrics.classification_report(Y_test, Y_pred, target_names=class_names)
        self.cMatrix = metrics.confusion_matrix(self.Y_test, self.Y_pred, normalize='true')
        #print(cReport)
        #print(cMatrix)
        return(self.knn, self.cMatrix, self.accuracy)
        #return(accuracy)

    def EMG_Main(self):
        # Main Code
        self.m = 2  # Lag number for autocorrelation coefficients
        self.responses = [1]*len(self.class1) + [2]*len(self.class2) + [3]*len(self.class3) + [4]*len(self.class4)
        self.accuracy_list = []

        for m in range(10,11):
            self.features = self.calc_features(self.class1, self.m) + self.calc_features(self.class2, self.m) + self.calc_features(self.class3, self.m) + self.calc_features(self.class4, self.m)
            [self.knn, self.cMatrix, self.accuracy] = train_classifier(self.features, self.responses, 0.5)
            #accuracy = train_classifier(features, responses, 0.25)
            #accuracy_list.append(accuracy)
            print(self.m, ',', self.accuracy*100)

        # Classifier 4 is working fine. 5 is the newer version    
        jbl.dump(knn, 'EMG_Classifier_4.joblib')
        jbl.dump(self.knn, 'EMG_Classifier_5.joblib')


        self.end_time = time.time()
        self.total_time_elapsed = self.end_time - self.start_time

class EMG_Signal(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='brown')
        label = tk.Label(self, text="EMG Signal Acquisition", fg='white',bg="goldenrod",bd=0,font=('times new roman', 30))
        label.pack(pady=10,padx=10)

        b1_on = tk.Button(self, text = "RUN Signal Acquisition", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.signal_acquisition) 
        b1_on.place(x=400, y=200)
        #b1_off = tk.Button(self, text = "OFF", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #b1_off.place(x=400, y=300)
        #reset = tk.Button(self, text = "RESET", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #reset.place(x=400, y=400)

        button1 = tk.Button(self, text="Back to Home", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)



    def signal_acquisition(self):
        # Initialize
        self.adc = ADCDifferentialPi(0x68, 0x69, 12)
        self.signalData = {}
        #signalIndices = {}

        # Signal acquisition
        print('Acquiring signal...')
        self.signalWindow = 1.5 # time in seconds
        self.numWindows = 5
        t.sleep(1)
        for j in range(self.numWindows):
            t.sleep(3)
            self.values = []
            #indices = []
            self.t0 = t.time()
            #i=0
            print('Start - ', j+1)
            t.sleep(0.5)
            while (t.time()-self.t0)<=self.signalWindow:
                self.emgVal = self.adc.read_raw(1)
                #i = i+1
                self.values.append(self.emgVal)
                #indices.append(i) 

            t.sleep(0.5)
            print('Stop')
            self.signalData[j] = self.values
            #signalIndices[j] = indices

        # Combining and saving signal data
        print('Signal data collected. Saving to file...')
        self.signalDataArray = []
        for i in range(len(self.signalData)):
            self.signalDataArray += self.signalData[i]

        #f = open('EMG_Data', 'wb')
        #pkl.dump(signalDataArray)
        #f.close()
        jbl.dump(self.signalDataArray, 'Test_Data.joblib')

        # Showing result
        print('Signal data saved. Showing saved data now.')
        #f = open('EMG_Data','rb')
        #data = pkl.loads(signalDataArray, f)
        #f.close()
        self.data = jbl.load('Test_Data.joblib')
        plt.plot(self.data)
        plt.show()

        print('Done')
    
    
    
class Hand_control(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='brown')
        label = tk.Label(self, text="Hand control", fg='white',bg="goldenrod",bd=0,font=('times new roman', 30))
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",  fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)

        b1_on = tk.Button(self, text = "RUN Hand Control", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.Hand_main) 
        b1_on.place(x=400, y=200)
        #b1_off = tk.Button(self, text = "OFF", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #b1_off.place(x=400, y=300)
        #reset = tk.Button(self, text = "RESET", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #reset.place(x=400, y=400)

        button1 = tk.Button(self, text="Back to Home", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)

        # ADC
        self.adc = ADCDifferentialPi(0x68, 0x69, 12)
        # Servo bonnet
        self.kit = ServoKit(channels=16)

        # GPIO
        self.emgPin = 1
        self.vdPin = 3
        self.relayPin = 4
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relayPin, GPIO.OUT)
        GPIO.output(self.relayPin, GPIO.HIGH)
        self.relayPinLevel = 1
        self.current_pos = 'open'
        self.motor_delay = 0.5
        # ML model
        #print('Loading machine learning model...')
        self.knn = jbl.load('EMG_Classifier_5.joblib')
        self.previous_prediction = 0

    # Signal preprocessing and feature calculation
    def calc_features(self, pro_data, lag_number):
        # Calculate features
        # Variance
        self.V = stats.variance(self.pro_data)
        # Autocorrelation function
        self.series = pd.Series(self.pro_data)
        self.R = [self.series.autocorr(lag = r) for r in range(lag_number+1)]
        # Compile features
        self.signal_features = []
        self.signal_features.append([self.V]+self.R[1:])
        return self.signal_features

    def check_voltage(self):
        self.relayPinLevel
        # Read voltage levels from battery and boost converter
        self.voltage = self.adc.read_voltage(self.vdPin)
        # Auto-cut for relay
        if self.voltage>2.5:
            GPIO.output(self.relayPin, GPIO.LOW)
            print('Battery voltage upper limit exceeded - ', self.voltage)
            self.relayPinLevel = 0
            time.sleep(1)
        elif self.voltage<2:
            GPIO.output(self.relayPin, GPIO.LOW)
            print('Battery voltage lower limit exceeded - ', self.voltage)
            self.relayPinLevel = 0
            time.sleep(1)
        else:
            if self.relayPinLevel == 0:
                 GPIO.output(self.relayPin, GPIO.HIGH)
                 self.relayPinLevel = 1 

    def fist_close(self):
        # Index flexion
        self.kit.servo[2].angle = 30
        time.sleep(self.motor_delay)
        # Middle flexion
        self.kit.servo[3].angle = 5
        time.sleep(self.motor_delay)
        # Ring flexion
        self.kit.servo[4].angle = 5
        time.sleep(self.motor_delay)
        # Little flexion
        self.kit.servo[5].angle = 5
        time.sleep(self.motor_delay)
        # Thumb abduction
        self.kit.servo[0].angle = 5
        time.sleep(self.motor_delay)
        # Thumb flexion
        self.kit.servo[1].angle = 90
        time.sleep(self.motor_delay)
        
    def fist_open(self):
        # Thumb extension
        self.kit.servo[1].angle = 155
        time.sleep(self.motor_delay)
        # Thumb adduction
        self.kit.servo[0].angle = 110
        time.sleep(self.motor_delay)
        # Index extension
        self.kit.servo[2].angle = 175
        time.sleep(self.motor_delay)
        # Middle extension
        self.kit.servo[3].angle = 145
        time.sleep(self.motor_delay)
        # Ring extension
        self.kit.servo[4].angle = 140
        time.sleep(self.motor_delay)
        # Little extension
        self.kit.servo[5].angle = 175
        time.sleep(self.motor_delay)
  
    def Hand_main(self):
        # 3. Main loop
        print('Initializing threads...')
        self.fclose_thread = threading.Thread(target = self.fist_close)
        self.fopen_thread = threading.Thread(target = self.fist_open)

        print('Control system online...')
        while True:
            
            # 4. Collect EMG window
            self.emg_data = []
            for i in range(100):
                self.emg_data.append(self.adc.read_raw(self.emgPin))   

            # Pre-processing - Apply wavelet transform
            self.pro_array = np.array(self.emg_data)
            self.decomposed_pro_array = 100000000*dw.denoise_wavelet(self.pro_array, method='BayesShrink', wavelet_levels=3, wavelet='db1')
            self.pro_data = self.decomposed_pro_array.tolist()
            
            # 5. Extract features
            self.emg_features = self.calc_features(self.pro_data, 10)
            
            # 6. Classify features
            self.prediction = self.knn.predict(self.emg_features)
            self.prediction = self.prediction.tolist()[0]
            print('Prediction = ', self.prediction)
            
            # 7. Motor command
            self.check_voltage()
            if self.prediction == 1 and self.previous_prediction != 1:
                print('Motor on')
                if self.current_pos == 'open':
                    print('Flexing fingers')
                    if self.fclose_thread.is_alive()==False and self.fopen_thread.is_alive()==False:
                        self.fclose_thread = threading.Thread(target=self.fist_close)
                        self.fclose_thread.start()
                        self.current_pos = 'closed'
                    
                elif self.current_pos == 'closed':
                    print('Extending fingers')
                    if self.fclose_thread.is_alive()==False and self.fopen_thread.is_alive()==False:
                        self.fopen_thread = threading.Thread(target=self.fist_open)
                        self.fopen_thread.start()
                        self.current_pos = 'open'
                        
                else:
                    print('Extending fingers')
                    if self.fclose_thread.is_alive()==False and self.fopen_thread.is_alive()==False:
                        self.fopen_thread = self.threading.Thread(target=self.fist_open)
                        self.fopen_thread.start()
                        self.current_pos = 'open'

            # Updata previous prediction 
            self.previous_prediction = self.prediction
    


class ML_Data_preparation(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='brown')
        label = tk.Label(self, text="ML Data preparation", fg='white',bg="goldenrod",bd=0,font=('times new roman', 30))
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)

        b1_on = tk.Button(self, text = "RUN ML_Data_prep_main", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.ML_Data_prep_main) 
        b1_on.place(x=400, y=200)
        #b1_off = tk.Button(self, text = "OFF", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #b1_off.place(x=400, y=300)
        #reset = tk.Button(self, text = "RESET", fg='white',bg="purple",bd=0,font=('times new roman', 20), command=self.off) 
        #reset.place(x=400, y=400)

        button1 = tk.Button(self, text="Back to Home", fg='white',bg="goldenrod",bd=0,font=('times new roman', 20), command=lambda: controller.show_frame(StartPage))
        button1.place(x=1100, y=550)

    def load_class1(self):
        self.class1 = jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_01.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_02.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_03.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_04.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_05.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_06.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_07.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_08.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_09.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_10.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_11.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_12.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_13.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_14.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_15.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_16.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_17.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_18.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_19.joblib')
        self.class1 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Activation_Data_2_20.joblib')
        
        return self.class1

    def load_class2(self):
        self.class2 = jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_01.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_02.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_03.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_04.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_05.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_06.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_07.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_08.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_09.joblib')
        self.class2 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Idle_Data_2_10.joblib')

        return self.class2

    def load_class3(self):
        self.class3 = jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_1.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_2.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_3.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_4.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_5.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_6.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_7.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_8.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_9.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_10.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_11.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_12.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_13.joblib')
        self.class3 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/Noise_Data_14.joblib')
        return self.class3

    def load_class4(self):
        self.self.class4 = jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_01.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_02.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_03.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_04.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_05.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_06.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_07.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_08.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_09.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_10.joblib')
        self.class4 += jbl.load('C:/Users/Munish Kumar/OneDrive/My Desk/Python/EMG Data (Joblib)/NoContact_Data_2_11.joblib')

        return self.class4

    def segregate_class_data(self, class_data, num_samples):
        self.sData = []
        self.num_windows = m.floor(len(self.class_data)/self.num_samples)
        for i in range(self.num_windows):
            self.window = self.class_data[i:i+self.num_samples]
            #t1 = time.time()
            self.window_array = np.array(self.window)
            self.decomposed_window_array = 100000000*dw.denoise_wavelet(self.window_array, method='BayesShrink', wavelet_levels=3, wavelet='db1')
            self.window = self.decomposed_window_array.tolist()
            #t2 = time.time()
            self.sData.append(self.window)
            #print(t2-t1)
        return self.sData

    def ML_Data_prep_main(self):
        # Main code
        self.class1 = self.load_class1()
        self.class2 = self.load_class2()
        self.class3 = self.load_class3()
        self.class4 = self.load_class4()

        self.num_samples = 100
        self.class1_sData = segregate_class_data(self.class1, self.num_samples)
        self.class2_sData = segregate_class_data(self.class2, self.num_samples)
        self.class3_sData = segregate_class_data(self.class3, self.num_samples)
        self.class4_sData = segregate_class_data(self.class4, self.num_samples)

        self.ml_data = {'class1':self.class1_sData, 'class2':self.class2_sData, 'class3':self.class2_sData, 'class4':self.class2_sData}
        self.ml_data = (self.class1_sData, self.class2_sData, self.class3_sData, self.class4_sData)

        # Save data
        jbl.dump(self.ml_data, 'ML_Data_5.joblib')
           


if __name__ == "__main__":
    app = App()
    
    #app.overrideredirect(True)
    app.title('HEMANT')
    app.geometry("{0}x{1}+0+0".format(app.winfo_screenwidth(), app.winfo_screenheight()))
    app.focus_set()  
    #app.geometry('600x500')


    app.mainloop()

