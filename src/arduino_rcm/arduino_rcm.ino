#include <SoftwareSerial.h>

#include <XBee.h>

XBee xbee;
SoftwareSerial serXBee(2, 3);
Rx16Response rx16 = Rx16Response();
Tx16Request tx16 = Tx16Request();
uint8_t offboardTxPayload[8] = {};
uint8_t onboardRxPayload[16] = {};
uint8_t onboardTxPayload[10] = {};

const unsigned long commsTimeout_ms = 200;

enum class Mode : uint8_t {
  ESTOP,
  RC_ARMED,
  RC_ACTIVE,
  AUTO_ARMED,
  AUTO_ACTIVE
};

// Serialization helpers for comma-delimited ascii messages

bool deserialize(bool &val, uint8_t **data) {
  bool error = false;
  switch (**data) {
  case '0':
    val = false;
    break;
  case '1':
    val = true;
    break;
  default:
    error = true;
  }
  while (**data != '\0' && **data != '\n' && **data != ',') {
    ++(*data);
  }
  if (**data == ',') {
    ++(*data);
  }
  return error;
}

bool deserialize(uint8_t &val, uint8_t **data) {
  uint32_t tempVal = 0;
  bool error = false;
  while (**data != '\0' && **data != '\n' && **data != ',') {
    switch (**data) {
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
      tempVal = tempVal * 10 + (**data - '0');
      break;
    default:
      error = true;
    }
  }
  if (tempVal > 255) {
    error = true;
    val = 0;
  } else {
    val = static_cast<uint8_t>(tempVal);
  }
  if (**data == ',') {
    ++(*data);
  }
  return error;
}

bool deserialize(Mode &val, uint8_t **data) {
  uint8_t intMode;
  bool retVal = deserialize(intMode, data);
  val = static_cast<Mode>(intMode);
  return retVal;
}

bool serialize(uint8_t &val, uint8_t **buffer) {
  uint8_t digits = 1;
  uint8_t tempVal = val;
  if (val >= 100) {
    digits = 3;
  } else if (val >= 10) {
    digits = 2;
  }
  while (digits > 0) {
    --digits;
    const uint8_t digitMultiplier = pow(10, digits);
    uint8_t newDigit =
        (tempVal - (tempVal % digitMultiplier)) / digitMultiplier;
    tempVal -= (newDigit * digitMultiplier);
    **buffer = '0' + newDigit;
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

    error = error || deserialize(ESTOP, &data);
    error = error || deserialize(AUTO_ARM, &data);
    error = error || deserialize(MANUAL_START, &data);
    error = error || deserialize(RC_PRESENT, &data);
    error = error || deserialize(AXIS_LX, &data);
    error = error || deserialize(AXIS_LY, &data);
    error = error || deserialize(AXIS_RX, &data);
    error = error || deserialize(AXIS_RY, &data);
    error = error || deserialize(BUTTON_X, &data);
    error = error || deserialize(BUTTON_O, &data);
    error = error || deserialize(BUTTON_SQUARE, &data);
    error = error || deserialize(BUTTON_TRIANGLE, &data);
    error = error || deserialize(BUTTON_L1, &data);
    error = error || deserialize(BUTTON_R1, &data);
    error = error || deserialize(BUTTON_L2, &data);
    error = error || deserialize(BUTTON_R2, &data);
    error = error || deserialize(BUTTON_L3, &data);
    error = error || deserialize(BUTTON_R3, &data);
    error = error || deserialize(BUTTON_SELECT, &data);
    error = error || deserialize(BUTTON_START, &data);
    error = error || deserialize(BUTTON_PS, &data);
    error = error || deserialize(BUTTON_UP, &data);
    error = error || deserialize(BUTTON_RIGHT, &data);
    error = error || deserialize(BUTTON_DOWN, &data);
    error = error || deserialize(BUTTON_LEFT, &data);

    return error;
  }

} FromOffboard;

typedef struct {
  Mode MODE{Mode::ESTOP};
  uint8_t BATTERY_LEVEL{255};

  bool serialize(uint8_t **buffer, uint8_t bufferSize) {
    // Need enough space for max size values with comma delimiter and trailing
    // '\n' and '\0'
    if (bufferSize < 8) {
      return false;
    }

    ::serialize(MODE, buffer);
    ::serialize(BATTERY_LEVEL, buffer);
    **buffer = '\n';
    ++(*buffer);
    **buffer = '\0';
    ++(*buffer);
  }

} ToOffboard;

typedef struct {
  bool AUTO_READY{false};
  uint8_t CMD_STEERING{127};
  uint8_t CMD_THROTTLE{127};

  bool deSerialize(uint8_t *data, uint8_t dataLength) {
    bool error = false;

    // Variable length, but must have at least one character per field including
    // comma delimiters and integer fields can be no longer than 3 characters
    // each.  Optional trailing comma delimiter.
    if (dataLength < 6 || dataLength > 11) {
      return false;
    }

    error = error || deserialize(AUTO_READY, &data);
    error = error || deserialize(CMD_STEERING, &data);
    error = error || deserialize(CMD_THROTTLE, &data);

    return error;
  }

} FromJetson;

typedef struct {
  bool ESTOP{false};
  bool AUTO_ARM{false};
  bool MANUAL_START{false};
  Mode MODE{Mode::ESTOP};

  bool serialize(uint8_t **buffer, uint8_t bufferSize) {
    // Need enough space for max size values with comma delimiter and trailing
    // '\n' and '\0'
    if (bufferSize < 10) {
      return false;
    }

    ::serialize(ESTOP, buffer);
    ::serialize(AUTO_ARM, buffer);
    ::serialize(MANUAL_START, buffer);
    ::serialize(MODE, buffer);
    **buffer = '\n';
    ++(*buffer);
    **buffer = '\0';
    ++(*buffer);
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

// Main Code

void setup() {
  xbee = XBee();

  // Setup USB serial
  Serial.begin(112500);
  while (!Serial) {
    ; // wait for serial port to connect
  }
  Serial.setTimeout(1);

  serXBee.begin(9600);
  xbee.setSerial(serXBee);
}

void loop() {
  static FromOffboard offboardState;
  static FromJetson jetsonState;
  static unsigned long latestOffboardUpdate = 0;
  static unsigned long latestJetsonUpdate = 0;
  static bool offboardTimedOut = true;
  static bool jetsonTimedOut = true;
  static Mode autoMode = Mode::ESTOP;

  const auto now = millis();

  // Offboard Rx
  xbee.readPacket();
  if (xbee.getResponse().isAvailable()) {
    if (xbee.getResponse().getApiId() == RX_16_RESPONSE) {
      latestOffboardUpdate = now;
      xbee.getResponse().getRx16Response(rx16);
      offboardState.deSerialize(rx16.getFrameData() + rx16.getDataOffset(),
                                rx16.getDataLength());
    }
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

  if (offboardState.ESTOP) {
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
    if (offboardTimedOut) {
      autoMode = Mode::RC_ARMED;
    }
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

  // Offboard Tx
  tx16.setAddress16(rx16.getRemoteAddress16());
  tx16.setPayload(offboardTxPayload,
                  GetPacketSize(offboardTxPayload, sizeof(offboardTxPayload)));
  xbee.send(tx16);

  // Onboard Tx
  Serial.write(onboardTxPayload,
               GetPacketSize(onboardTxPayload, sizeof(onboardTxPayload)));
}