#include <Arduino.h>

const int LED_FACE  = 14;
const int LED_EYES  = 12;
const int LED_SMILE = 13;

void setup() {
  Serial.begin(115200);
  pinMode(LED_FACE, OUTPUT);
  pinMode(LED_EYES, OUTPUT);
  pinMode(LED_SMILE, OUTPUT);
  digitalWrite(LED_FACE, LOW);
  digitalWrite(LED_EYES, LOW);
  digitalWrite(LED_SMILE, LOW);
  Serial.println("ESP32 lista para detección múltiple.");
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    int face  = msg.substring(msg.indexOf("face=") + 5, msg.indexOf(" eyes")).toInt();
    int eyes  = msg.substring(msg.indexOf("eyes=") + 5, msg.indexOf(" smile")).toInt();
    int smile = msg.substring(msg.indexOf("smile=") + 6).toInt();

    digitalWrite(LED_FACE,  face  ? HIGH : LOW);
    digitalWrite(LED_EYES,  eyes  ? HIGH : LOW);
    digitalWrite(LED_SMILE, smile ? HIGH : LOW);

    Serial.printf("FACE:%d  EYES:%d  SMILE:%d\n", face, eyes, smile);
  }
}
