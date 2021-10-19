#include <Wire.h>

void setup() {
  Wire.begin();
  Serial.begin(9600);
  
}

byte x = 0;
byte y = 12;
byte z = 30;

void loop() {
  Wire.beginTransmission(8);  
  Wire.write("x is ");    // 5 byte send
  Serial.println(x);
  Wire.write(x);// sends one byte
  Serial.println(y);
  Wire.write(y);
  Serial.println(z);
  Wire.write(z);
  Wire.endTransmission();    // stop transmitting
  delay(500);
}