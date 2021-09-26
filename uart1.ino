void setup()
{
  pinMode(A1, INPUT);
  Serial.begin(9600);
}

void loop()
{
 int a = analogRead(A1);
 int x = map(a,0,1023,0,5000);
 int c=x-500;
 int temp = c/10;
 Serial.print("The temperature is:");
 Serial.print(temp);
 Serial.println();
 
 delay(10);
}