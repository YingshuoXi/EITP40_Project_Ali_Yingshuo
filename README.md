# EITP40_Project_Ali_Yingshuo
EITP40 Final Project

This project aims to build a number detective model trained by the Forward-Forward algorithm on the MNIST dataset.
The final outcome is expected to be deployed on the Arduino Nano 33 BLE Sense with the peripheral camera.

The model will be trained by FF with TensorFlow, while the MicroTFLite will do the inference on the Arduino.
A bonus would be a 3-D printing frame to hold the cards (with handwritten numbers) and the camera.

# How to use the current code
1. Please check the user manual of NUCLEO-WB55RG https://www.st.com/resource/en/user_manual/um2819-stm32wb-nucleo64-board-mb1355-stmicroelectronics.pdf

physically remove the jump wires [13,14] [15,16]
![alt text](image.png)
connet the ST-link (pin 14, 16) to LPUART (pin A0, A1) as:
14 - A1, 16 - A0

2. Deploy the STM32 code to the board and run the python code (in any order).

# Indication
LED1 (BLUE) toggle = DATA is being received
LED2 (Green) on = pause
LED3 (RED) on = the program is running

SW1 press = shut down the python progress
SW2 press = pause
Sw3 press = continue from pause

Please note that once you press SW1, the python will be terminated and you should manully run it again.