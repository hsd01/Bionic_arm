void setup()
{
  pinMode(7,OUTPUT);
  digitalWrite(7,LOW);
  Serial.begin(9600);
}

void loop()
{
  
  while(Serial.available()>0)
  {
  char i = Serial.read();
  Serial.print(i);
  int x = Serial.parseInt();
  //Serial.print(i);
  if(x >= 30)
  {
   digitalWrite(7,HIGH);
    //delay(10);
  }
  else
  {
      digitalWrite(7,LOW);
  }
  Serial.print("The temperature is:");
  Serial.print(x);
  Serial.println();
  }
  
}