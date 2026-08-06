#include <Servo.h>
#include <SoftwareSerial.h>

#include <XBee.h>

XBee xbee;
SoftwareSerial serXBee(2, 3);
Rx16Response rx16 = Rx16Response();
Tx16Request tx16 = Tx16Request();
uint8_t offboardTxPayload[8] = {};
uint8_t onboardRxPayload[16] = {};
uint8_t onboardTxPayload[14] = {};

Servo throttle;
Servo steering;

const unsigned long commsTimeout_ms = 200;
const bool enableOffboardTx = true;

enum class Mode : uint8_t {
  ESTOP,
  RC_ARMED,
  RC_ACTIVE,
  AUTO_ARMED,
  AUTO_ACTIVE
};

// Serialization helpers for comma-delimited ascii messages

bool deserialize(bool &val, const uint8_t *&data, const uint8_t *end) {
  if (data == end) {
    return true;
  }

  bool error = false;
  switch (*data) {
  case '0':
    val = false;
    break;
  case '1':
    val = true;
    break;
  default:
    error = true;
  }
  while (data != end && *data != '\n' && *data != ',') {
    ++data;
  }
  if (data != end && *data == ',') {
    ++data;
  }
  return error;
}

bool deserialize(uint8_t &val, const uint8_t *&data, const uint8_t *end) {
  if (data == end || *data == '\n' || *data == ',') {
    return true;
  }

  uint32_t tempVal = 0;
  bool error = false;
  while (data != end && *data != '\n' && *data != ',') {
    switch (*data) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      tempVal = tempVal * 10 + (*data - '0');
      break;
    default:
      error = true;
    }
    ++data;
  }
  if (tempVal > 255) {
    error = true;
    val = 0;
  } else {
    val = static_cast<uint8_t>(tempVal);
  }
  if (data != end && *data == ',') {
    ++data;
  }
  return error;
}

bool deserialize(Mode &val, const uint8_t *&data, const uint8_t *end) {
  uint8_t intMode;
  bool retVal = deserialize(intMode, data, end);
  val = static_cast<Mode>(intMode);
  return retVal;
}

bool serialize(const uint8_t &val, uint8_t **buffer) {
  uint8_t digits = 1;
  uint8_t tempVal = val;
  if (val >= 100) {
    digits = 3;
  } else if (val >= 10) {
    digits = 2;
  }
  while (digits > 0) {
    --digits;
    uint8_t digitMultiplier;
    switch (digits) {
    case 0:
      digitMultiplier = 1;
      break;
    case 1:
      digitMultiplier = 10;
      break;
    case 2:
      digitMultiplier = 100;
      break;
    }
    uint8_t newDigit =
        (tempVal - (tempVal % digitMultiplier)) / digitMultiplier;
    tempVal -= (newDigit * digitMultiplier);
    **buffer = '0' + newDigit;
    ++(*buffer);
  }
  **buffer = ',';
  ++(*buffer);
  return true;
}

bool serialize(Mode &val, uint8_t **buffer) {
  uint8_t intMode = static_cast<uint8_t>(val);
  return serialize(intMode, buffer);
}

bool serialize(bool &val, uint8_t **buffer) {
  **buffer = val ? '1' : '0';
  ++(*buffer);
  **buffer = ',';
  ++(*buffer);
  return true;
}

// Onboard and offboard messages

typedef struct {
  bool ESTOP{false};
  bool AUTO_ARM{false};
  bool MANUAL_START{false};
  bool RC_PRESENT{false};
  uint8_t AXIS_LX{127};
  uint8_t AXIS_LY{127};
  uint8_t AXIS_RX{127};
  uint8_t AXIS_RY{127};
  bool BUTTON_X{false};
  bool BUTTON_O{false};
  bool BUTTON_SQUARE{false};
  bool BUTTON_TRIANGLE{false};
  bool BUTTON_L1{false};
  bool BUTTON_R1{false};
  bool BUTTON_L2{false};
  bool BUTTON_R2{false};
  bool BUTTON_L3{false};
  bool BUTTON_R3{false};
  bool BUTTON_SELECT{false};
  bool BUTTON_START{false};
  bool BUTTON_PS{false};
  bool BUTTON_UP{false};
  bool BUTTON_RIGHT{false};
  bool BUTTON_DOWN{false};
  bool BUTTON_LEFT{false};

  bool deSerialize(uint8_t *data, uint8_t dataLength) {
    bool error = false;

    // Variable length, but must have at least one character per field including
    // comma delimiters and integer fields can be no longer than 3 characters
    // each.  Optional trailing comma delimiter.
    if (dataLength < 49 || dataLength > 58) {
      return false;
    }

    const uint8_t *cursor = data;
    const uint8_t *end = data + dataLength;

    error = deserialize(ESTOP, cursor, end) || error;
    error = deserialize(AUTO_ARM, cursor, end) || error;
    error = deserialize(MANUAL_START, cursor, end) || error;
    error = deserialize(RC_PRESENT, cursor, end) || error;
    error = deserialize(AXIS_LX, cursor, end) || error;
    error = deserialize(AXIS_LY, cursor, end) || error;
    error = deserialize(AXIS_RX, cursor, end) || error;
    error = deserialize(AXIS_RY, cursor, end) || error;
    error = deserialize(BUTTON_X, cursor, end) || error;
    error = deserialize(BUTTON_O, cursor, end) || error;
    error = deserialize(BUTTON_SQUARE, cursor, end) || error;
    error = deserialize(BUTTON_TRIANGLE, cursor, end) || error;
    error = deserialize(BUTTON_L1, cursor, end) || error;
    error = deserialize(BUTTON_R1, cursor, end) || error;
    error = deserialize(BUTTON_L2, cursor, end) || error;
    error = deserialize(BUTTON_R2, cursor, end) || error;
    error = deserialize(BUTTON_L3, cursor, end) || error;
    error = deserialize(BUTTON_R3, cursor, end) || error;
    error = deserialize(BUTTON_SELECT, cursor, end) || error;
    error = deserialize(BUTTON_START, cursor, end) || error;
    error = deserialize(BUTTON_PS, cursor, end) || error;
    error = deserialize(BUTTON_UP, cursor, end) || error;
    error = deserialize(BUTTON_RIGHT, cursor, end) || error;
    error = deserialize(BUTTON_DOWN, cursor, end) || error;
    error = deserialize(BUTTON_LEFT, cursor, end) || error;

    return !error;
  }

} FromOffboard;

typedef struct {
  Mode MODE{Mode::ESTOP};
  uint8_t BATTERY_LEVEL{255};

  bool serialize(uint8_t *buffer, uint8_t bufferSize) {
    // Need enough space for max size values with comma delimiter and trailing
    // '\n' and '\0'
    if (bufferSize < 8) {
      return false;
    }

    ::serialize(MODE, &buffer);
    ::serialize(BATTERY_LEVEL, &buffer);
    *buffer = '\n';
    ++(buffer);
    *buffer = '\0';
    ++(buffer);
    return true;
  }

} ToOffboard;

typedef struct {
  bool AUTO_READY{false};
  uint8_t CMD_STEERING{127};
  uint8_t CMD_THROTTLE{127};

  bool deSerialize(uint8_t *data, uint8_t dataLength) {
    bool error = false;

    // Variable length, but must have at least one character per field including
    // comma delimiters and integer fields can be no longer than 3
    // characters each.  Optional trailing comma delimiter.
    if (dataLength < 6 || dataLength > 11) {
      return false;
    }

    const uint8_t *cursor = data;
    const uint8_t *end = data + dataLength;

    error = deserialize(AUTO_READY, cursor, end) || error;
    error = deserialize(CMD_STEERING, cursor, end) || error;
    error = deserialize(CMD_THROTTLE, cursor, end) || error;

    return !error;
  }

} FromJetson;

typedef struct {
  bool ESTOP{false};
  bool AUTO_ARM{false};
  bool MANUAL_START{false};
  Mode MODE{Mode::ESTOP};
  uint8_t BATTERY_LEVEL{255};

  bool serialize(uint8_t *buffer, uint8_t bufferSize) {
    // Need enough space for max size values with comma delimiter and trailing
    // '\n' and '\0'
    if (bufferSize < 14) {
      return false;
    }

    ::serialize(ESTOP, &buffer);
    ::serialize(AUTO_ARM, &buffer);
    ::serialize(MANUAL_START, &buffer);
    ::serialize(MODE, &buffer);
    ::serialize(BATTERY_LEVEL, &buffer);
    *buffer = '\n';
    ++(buffer);
    *buffer = '\0';
    ++(buffer);
    return true;
  }

} ToJetson;

// Helpers

bool IsTimedOut(unsigned long latestUpdate, unsigned long sampleTime,
                unsigned long timeout) {
  // Handle 32-bit rollover
  if (latestUpdate > sampleTime) {
    return (0xFFFFFFFFUL - latestUpdate + sampleTime) >= timeout;
  }
  return (sampleTime - latestUpdate) >= timeout;
}

uint8_t GetPacketSize(uint8_t *buffer, uint8_t bufferSize,
                      char endChar = '\n') {
  uint8_t length = 0;
  while (length < bufferSize) {
    if (buffer[length] == endChar) {
      return length + 1;
    }
    ++length;
  }
  return length;
}

bool IsNearlyCenter(uint8_t value, uint8_t centerValue = 127,
                    uint8_t tolerance = 5) {
  if (centerValue < tolerance) {
    return value <= (centerValue + tolerance);
  } else if ((255 - centerValue) < tolerance) {
    return value >= (centerValue - tolerance);
  }
  return (value >= (centerValue - tolerance)) &&
         (value <= (centerValue + tolerance));
}

// Deadband around center [125,130].  Overall pulse range [1000us,2000us]
int PctToPulseLength(uint8_t throttle, bool deadband = true) {
  if (deadband) {
    if (throttle > 130) {
      return 1500 + ((throttle - 130) * 4);
    } else if (throttle < 125) {
      return 1000 + ((throttle + 1) * 4);
    }
    return 1500;
  } else {
    // Some saturation at min and max values
    return constrain(((static_cast<int>(throttle) - 127) * 4) + 1500, 1000,
                     2000);
  }
}

// Main Code

void setup() {
  xbee = XBee();

  // Setup USB serial
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect
  }
  Serial.setTimeout(1);

  serXBee.begin(38400);
  while (!serXBee) {
    ; // wait for serial port to connect
  }
  xbee.setSerial(serXBee);
  tx16.setFrameId(NO_RESPONSE_FRAME_ID);

  // Throttle to FL(10), Steering to FR(5)
  throttle.attach(10, -114,
                  100); // min=1000us (544-(-114*4)), max=2000us (2400-(100*4))
  steering.attach(5, -114,
                  100); // min=1000us (544-(-114*4)), max=2000us (2400-(100*4))
}

void loop() {
  static uint8_t debugStage = 0;
  static uint8_t debugLoops = 0;
  const bool printDebugStages = false; // debugLoops < 10;
  static FromOffboard offboardState;
  static FromJetson jetsonState;
  static unsigned long latestOffboardUpdate = 0;
  static unsigned long latestJetsonUpdate = 0;
  static bool offboardTimedOut = true;
  static bool jetsonTimedOut = true;
  static Mode autoMode = Mode::ESTOP;

  const auto now = millis();
  static auto lastTx = now;
  static uint16_t xbeeFrames = 0;
  static uint16_t xbeeRx16Frames = 0;
  static uint16_t xbeeValidMessages = 0;
  static uint16_t xbeeInvalidMessages = 0;
  static uint16_t xbeeErrors = 0;
  static uint8_t lastXBeeApiId = 0;
  static uint8_t lastXBeeError = 0;

  // Offboard Rx
  xbee.readPacket();
  if (xbee.getResponse().isAvailable()) {
    ++xbeeFrames;
    lastXBeeApiId = xbee.getResponse().getApiId();
    if (xbee.getResponse().getApiId() == RX_16_RESPONSE) {
      ++xbeeRx16Frames;
      latestOffboardUpdate = now;
      xbee.getResponse().getRx16Response(rx16);
      if (offboardState.deSerialize(rx16.getData(), rx16.getDataLength())) {
        ++xbeeValidMessages;
        latestOffboardUpdate = now;

      } else {
        ++xbeeInvalidMessages;
      }
    }
  } else if (xbee.getResponse().isError()) {
    ++xbeeErrors;
    lastXBeeError = xbee.getResponse().getErrorCode();
  }

  // Onboard Rx
  while (true) {
    auto retval =
        Serial.readBytesUntil('\n', onboardRxPayload, sizeof(onboardRxPayload));
    if (retval > 0) {
      if (jetsonState.deSerialize(onboardRxPayload, retval)) {
        latestJetsonUpdate = now;
      }
    } else {
      break;
    }
  }

  if (IsTimedOut(latestOffboardUpdate, now, commsTimeout_ms)) {
    offboardState = FromOffboard();
    offboardTimedOut = true;
  } else {
    offboardTimedOut = false;
  }

  if (IsTimedOut(latestJetsonUpdate, now, commsTimeout_ms)) {
    jetsonState = FromJetson();
    jetsonTimedOut = true;
  } else {
    jetsonTimedOut = false;
  }

  // State management
  if (offboardState.ESTOP || offboardTimedOut) {
    autoMode = Mode::ESTOP;
  } else if (offboardState.AUTO_ARM == false && autoMode != Mode::RC_ACTIVE) {
    autoMode = Mode::RC_ARMED;
  } else if (offboardState.AUTO_ARM && autoMode != Mode::AUTO_ACTIVE) {
    autoMode = Mode::AUTO_ARMED;
  }

  switch (autoMode) {
  case Mode::ESTOP:
    break;
  case Mode::RC_ARMED:
    if (offboardState.RC_PRESENT && IsNearlyCenter(offboardState.AXIS_LY) &&
        IsNearlyCenter(offboardState.AXIS_LX)) {
      autoMode = Mode::RC_ACTIVE;
    }
    break;
  case Mode::RC_ACTIVE:
    break;
  case Mode::AUTO_ARMED:
    if (jetsonState.AUTO_READY && IsNearlyCenter(jetsonState.CMD_STEERING) &&
        IsNearlyCenter(jetsonState.CMD_THROTTLE)) {
      autoMode = Mode::AUTO_ACTIVE;
    }
    break;
  case Mode::AUTO_ACTIVE:
    if (jetsonTimedOut) {
      autoMode = Mode::AUTO_ARMED;
    }
    break;
  }

  // Car control
  auto throttleCmd = PctToPulseLength(127);
  auto steeringCmd = PctToPulseLength(127);

  switch (autoMode) {
  case Mode::RC_ACTIVE:
    throttleCmd = PctToPulseLength(offboardState.AXIS_LY);
    steeringCmd = PctToPulseLength(offboardState.AXIS_RX);
    break;
  case Mode::AUTO_ACTIVE:
    throttleCmd = PctToPulseLength(jetsonState.CMD_THROTTLE, false);
    steeringCmd = PctToPulseLength(jetsonState.CMD_STEERING, false);
    break;
  default:
    throttleCmd = PctToPulseLength(127);
    steeringCmd = PctToPulseLength(127);
  }

  throttle.writeMicroseconds(throttleCmd);
  steering.writeMicroseconds(steeringCmd);

  /// @todo Read battery level
  uint8_t batteryLevel = 255;

  ToJetson onboardFeedback;
  onboardFeedback.ESTOP = offboardState.ESTOP;
  onboardFeedback.AUTO_ARM = offboardState.AUTO_ARM;
  onboardFeedback.MANUAL_START = offboardState.MANUAL_START;
  onboardFeedback.MODE = autoMode;
  onboardFeedback.BATTERY_LEVEL = batteryLevel;
  onboardFeedback.serialize((uint8_t *)onboardTxPayload,
                            sizeof(onboardTxPayload));

  ToOffboard offboardFeedback;
  offboardFeedback.MODE = autoMode;
  offboardFeedback.BATTERY_LEVEL = batteryLevel;
  offboardFeedback.serialize((uint8_t *)offboardTxPayload,
                             sizeof(offboardTxPayload));

  static int count = 0;
  // Limit to 20Hz updates
  if (now - lastTx > 50) {
    // Offboard Tx
    // tx16.setAddress16(rx16.getRemoteAddress16());
    tx16.setAddress16(0);
    tx16.setPayload(
        offboardTxPayload,
        GetPacketSize(offboardTxPayload, sizeof(offboardTxPayload)));
    if (enableOffboardTx) {
      xbee.send(tx16);
    }
    // Onboard Tx
    Serial.write(onboardTxPayload,
                 GetPacketSize(onboardTxPayload, sizeof(onboardTxPayload)));
    lastTx = now;
  }
}
