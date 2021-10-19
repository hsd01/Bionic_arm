#include <Wire.h>

int x = 0;
void setup() {
  Wire.begin(8);                
  Wire.onReceive(receiveEvent);
  Serial.begin(9600);           
}

void loop() {
  delay(500);
  Serial.print("value of x : ");
  Serial.print(x);
  Serial.println("***");
}

void receiveEvent(int howMany) 
{
  while (1 < Wire.available()) 
  { 
    char c = Wire.read(); 
    //Serial.print(" c :");
    Serial.print(c);
    //Serial.println();
  }
  //x = Wire.read();    
 // Serial.println(x);         
}