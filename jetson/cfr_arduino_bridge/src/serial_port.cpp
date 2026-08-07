#include "cfr_arduino_bridge/serial_port.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace cfr_arduino_bridge {

SerialPort::~SerialPort() { Close(); }

bool SerialPort::BaudToSpeed(unsigned baud, unsigned &speed) {
  switch (baud) {
  case 9600:
    speed = B9600;
    return true;
  case 19200:
    speed = B19200;
    return true;
  case 38400:
    speed = B38400;
    return true;
  case 57600:
    speed = B57600;
    return true;
  case 115200:
    speed = B115200;
    return true;
  case 230400:
    speed = B230400;
    return true;
  default:
    return false;
  }
}

bool SerialPort::Open(const std::string &device, unsigned baud) {
  Close();

  unsigned speed = 0;
  if (!BaudToSpeed(baud, speed)) {
    last_error_ = "unsupported baud rate " + std::to_string(baud);
    return false;
  }

  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    last_error_ = "open(" + device + ") failed: " + std::strerror(errno);
    return false;
  }

  struct termios tty {};
  if (::tcgetattr(fd_, &tty) != 0) {
    last_error_ = std::string("tcgetattr failed: ") + std::strerror(errno);
    Close();
    return false;
  }

  ::cfmakeraw(&tty);
  ::cfsetispeed(&tty, static_cast<speed_t>(speed));
  ::cfsetospeed(&tty, static_cast<speed_t>(speed));

  tty.c_cflag |=
      (CLOCAL | CREAD);    // ignore modem control lines, enable receiver
  tty.c_cflag &= ~CRTSCTS; // no hardware flow control
  tty.c_cflag &=
      ~HUPCL; // do not drop DTR on close, which would reset the Arduino
  tty.c_cflag &= ~CSTOPB; // one stop bit
  tty.c_cc[VMIN] = 0;     // fully non-blocking reads
  tty.c_cc[VTIME] = 0;

  if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
    last_error_ = std::string("tcsetattr failed: ") + std::strerror(errno);
    Close();
    return false;
  }

  ::tcflush(fd_, TCIOFLUSH);
  rx_buffer_.clear();
  last_error_.clear();
  return true;
}

void SerialPort::Close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  rx_buffer_.clear();
}

bool SerialPort::Write(const std::string &payload) {
  if (!IsOpen()) {
    last_error_ = "port is not open";
    return false;
  }

  size_t written = 0;
  while (written < payload.size()) {
    const ssize_t result =
        ::write(fd_, payload.data() + written, payload.size() - written);
    if (result > 0) {
      written += static_cast<size_t>(result);
      continue;
    }
    if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      // The kernel buffer is full.  Dropping the rest of a stale command is
      // better than blocking the executor; the next cycle sends a fresh one.
      last_error_ = "write would block, command truncated";
      return false;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    last_error_ = std::string("write failed: ") + std::strerror(errno);
    return false;
  }
  return true;
}

bool SerialPort::ReadLines(std::vector<std::string> &lines) {
  if (!IsOpen()) {
    last_error_ = "port is not open";
    return false;
  }

  std::array<char, 256> chunk{};
  while (true) {
    const ssize_t result = ::read(fd_, chunk.data(), chunk.size());
    if (result > 0) {
      rx_buffer_.append(chunk.data(), static_cast<size_t>(result));
      continue;
    }
    if (result == 0) {
      break; // nothing more to read right now
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    last_error_ = std::string("read failed: ") + std::strerror(errno);
    return false;
  }

  size_t newline = rx_buffer_.find('\n');
  while (newline != std::string::npos) {
    std::string line = rx_buffer_.substr(0, newline);
    rx_buffer_.erase(0, newline + 1);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      lines.push_back(std::move(line));
    }
    newline = rx_buffer_.find('\n');
  }

  if (rx_buffer_.size() > kMaxBufferedBytes) {
    rx_buffer_.clear();
    last_error_ = "receive buffer overflowed without a newline, discarded";
  }
  return true;
}

} // namespace cfr_arduino_bridge
