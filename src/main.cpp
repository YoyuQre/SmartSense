// ============================================================
//  Smart Occupancy IoT System  -  Phase 2B (local build)
//  ESP32 DevKit C V4 + PIR + LDR + DS18B20
//
//  This is the single canonical firmware source for the project.
//  It is kept in sync with the Wokwi copy at ../sketch.ino so the
//  same logic runs in Wokwi and in the local PlatformIO build.
//
//  PHASE SCOPE: hardware sanity check only.
//  No machine learning here. Occupancy is currently the raw PIR
//  motion signal; the ML Decision Tree comes in a later phase.
// ============================================================

#include <OneWire.h>
#include <DallasTemperature.h>

// ---------------- PIN DEFINITIONS ----------------
// These match wokwi/diagram.json exactly. Do not change one
// without changing the other.
#define PIR_PIN       27   // PIR motion sensor, digital OUT
#define LDR_PIN       34   // LDR module, ANALOG OUT (ADC1_CH6)
#define DS18B20_PIN    4   // DS18B20 1-Wire data line

#define LIGHT_LED     18   // Room lighting indicator LED
#define HVAC_LED      19   // HVAC status indicator LED

// ---------------- TEMPERATURE SENSOR ----------------
OneWire oneWire(DS18B20_PIN);
DallasTemperature temperatureSensor(&oneWire);

// ---------------- CONFIGURATION ----------------
// All tunables live here as named constants instead of
// magic numbers buried in the logic.
const int   LIGHT_THRESHOLD      = 300;   // normalised 0-1000 light scale
const float HVAC_COOLING_THRESHOLD = 26.0; // degrees Celsius

const unsigned long SENSOR_INTERVAL = 5000; // ms between readings

// ESP32 ADC is 12-bit by default: raw range 0-4095.
const int ADC_RAW_MAX = 4095;
const int LIGHT_SCALE_MAX = 1000;

// A DS18B20 reports -127.0 on CRC error / no device, and 85.0 as a
// power-on-reset placeholder. Anything outside this physical band is
// treated as an invalid reading.
const float TEMP_PLAUSIBLE_MIN = -55.0;
const float TEMP_PLAUSIBLE_MAX = 125.0;

// ---------------- STATE ----------------
unsigned long lastReading = 0;

void setup() {
  Serial.begin(115200);

  pinMode(PIR_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);          // GPIO 34 is input-only, no pullup

  pinMode(LIGHT_LED, OUTPUT);
  pinMode(HVAC_LED, OUTPUT);

  digitalWrite(LIGHT_LED, LOW);
  digitalWrite(HVAC_LED, LOW);

  // DallasTemperature::begin() returns void, so presence is confirmed
  // by counting the devices found on the 1-Wire bus.
  temperatureSensor.begin();

  uint8_t deviceCount = temperatureSensor.getDeviceCount();
  if (deviceCount > 0) {
    Serial.print("[INIT] DS18B20 found on GPIO 4, devices=");
    Serial.println(deviceCount);
  } else {
    Serial.println("[INIT] ERROR: no DS18B20 detected on GPIO 4");
  }

  Serial.println();
  Serial.println("======================================");
  Serial.println(" Smart Occupancy IoT System");
  Serial.println(" ESP32 + PIR + LDR + DS18B20");
  Serial.println("======================================");
  Serial.println("light_level is a NORMALISED 0-1000 scale, not calibrated lux.");
  Serial.println("An invalid DS18B20 reading is reported as -127.00 and also");
  Serial.println("logged on its own ERR line. It is never used to drive HVAC.");
  Serial.println();
  Serial.println("timestamp,motion,light_level,temperature,light_status,hvac_status");
}

void loop() {

  // Non-blocking timing: the loop stays responsive, no delay() is used.
  if (millis() - lastReading < SENSOR_INTERVAL) {
    return;
  }
  lastReading = millis();

  // ---------------- READ SENSORS ----------------

  int motion = digitalRead(PIR_PIN);

  int lightRaw = analogRead(LDR_PIN);

  // Raw ADC -> normalised 0-1000 light scale, then clamp for safety.
  long scaled = map(lightRaw, 0, ADC_RAW_MAX, 0, LIGHT_SCALE_MAX);
  if (scaled < 0) scaled = 0;
  if (scaled > LIGHT_SCALE_MAX) scaled = LIGHT_SCALE_MAX;
  int lightLevel = (int)scaled;

  bool temperatureValid = temperatureSensor.requestTemperatures();
  float temperature = temperatureValid ? temperatureSensor.getTempCByIndex(0)
                                      : DEVICE_DISCONNECTED_C;

  if (temperature < TEMP_PLAUSIBLE_MIN || temperature > TEMP_PLAUSIBLE_MAX) {
    temperatureValid = false;
  }

  if (!temperatureValid) {
    Serial.print("ERR,TEMP_INVALID,raw=");
    Serial.print(temperature, 2);
    Serial.println(",action=HVAC_FORCED_OFF");
    temperature = DEVICE_DISCONNECTED_C;
  }

  // ---------------- CONTROL LOGIC (hardware sanity check) ----------------

  bool lightOn = false;
  bool hvacOn  = false;

  // Lighting: occupied + not enough ambient light.
  if (motion == 1 && lightLevel < LIGHT_THRESHOLD) {
    lightOn = true;
  }

  // HVAC: occupied + temperature above the cooling threshold.
  // Fail-safe: an invalid temperature can never switch HVAC on.
  if (motion == 1 && temperatureValid && temperature > HVAC_COOLING_THRESHOLD) {
    hvacOn = true;
  }

  digitalWrite(LIGHT_LED, lightOn ? HIGH : LOW);
  digitalWrite(HVAC_LED,  hvacOn  ? HIGH : LOW);

  // ---------------- SERIAL OUTPUT ----------------
  // Format is fixed for the later Python ingestion step:
  // DATA,timestamp,motion,light_level,temperature,light_status,hvac_status
  Serial.print("DATA,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(motion);
  Serial.print(",");
  Serial.print(lightLevel);
  Serial.print(",");
  Serial.print(temperature, 2);
  Serial.print(",");
  Serial.print(lightOn ? 1 : 0);
  Serial.print(",");
  Serial.println(hvacOn ? 1 : 0);
}
