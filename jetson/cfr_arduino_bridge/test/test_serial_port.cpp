// Exercises SerialPort against a pseudo terminal, so the framing logic is
// covered without an Arduino attached.

#include <gtest/gtest.h>
#include <pty.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "cfr_arduino_bridge/serial_port.hpp"

using cfr_arduino_bridge::SerialPort;

namespace {

class SerialPortTest : public ::testing::Test {
protected:
  void SetUp() override {
    char name[256] = {};
    ASSERT_EQ(::openpty(&master_fd_, &slave_fd_, name, nullptr, nullptr), 0);
    slave_name_ = name;
    // The port reopens the slave by name; the fd from openpty is not needed.
    ::close(slave_fd_);
    slave_fd_ = -1;
    ASSERT_TRUE(port_.Open(slave_name_, 115200)) << port_.LastError();
  }

  void TearDown() override {
    port_.Close();
    if (master_fd_ >= 0) {
      ::close(master_fd_);
    }
  }

  /// Write to the far end of the link, as the Arduino would.
  void SendFromPeer(const std::string &data) {
    ASSERT_EQ(::write(master_fd_, data.data(), data.size()),
              static_cast<ssize_t>(data.size()));
    // A pty round trip is not instantaneous.
    ::usleep(20000);
  }

  std::string ReceiveAtPeer() {
    ::usleep(20000);
    std::string received;
    char buffer[256];
    const ssize_t count = ::read(master_fd_, buffer, sizeof(buffer));
    if (count > 0) {
      received.assign(buffer, static_cast<size_t>(count));
    }
    return received;
  }

  int master_fd_ = -1;
  int slave_fd_ = -1;
  std::string slave_name_;
  SerialPort port_;
};

TEST_F(SerialPortTest, OpenAndClose) {
  EXPECT_TRUE(port_.IsOpen());
  port_.Close();
  EXPECT_FALSE(port_.IsOpen());
}

TEST_F(SerialPortTest, RejectsUnsupportedBaud) {
  SerialPort other;
  EXPECT_FALSE(other.Open(slave_name_, 12345));
  EXPECT_FALSE(other.LastError().empty());
}

TEST_F(SerialPortTest, ReadReturnsNoLinesWhenIdle) {
  std::vector<std::string> lines;
  EXPECT_TRUE(port_.ReadLines(lines));
  EXPECT_TRUE(lines.empty());
}

TEST_F(SerialPortTest, SplitsCompleteLines) {
  SendFromPeer("0,0,0,1,255,\n1,0,0,4,200,\n");
  std::vector<std::string> lines;
  ASSERT_TRUE(port_.ReadLines(lines));
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "0,0,0,1,255,");
  EXPECT_EQ(lines[1], "1,0,0,4,200,");
}

TEST_F(SerialPortTest, HoldsPartialLineUntilTerminated) {
  SendFromPeer("0,0,0,1,2");
  std::vector<std::string> lines;
  ASSERT_TRUE(port_.ReadLines(lines));
  EXPECT_TRUE(lines.empty()) << "a partial frame must not be delivered";

  SendFromPeer("55,\n");
  ASSERT_TRUE(port_.ReadLines(lines));
  ASSERT_EQ(lines.size(), 1U);
  EXPECT_EQ(lines[0], "0,0,0,1,255,");
}

TEST_F(SerialPortTest, StripsCarriageReturnAndEmptyLines) {
  SendFromPeer("0,0,0,1,255,\r\n\n1,1,1,4,10,\r\n");
  std::vector<std::string> lines;
  ASSERT_TRUE(port_.ReadLines(lines));
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "0,0,0,1,255,");
  EXPECT_EQ(lines[1], "1,1,1,4,10,");
}

// A peer that never sends a newline must not grow the buffer without bound.
TEST_F(SerialPortTest, DiscardsOverlongUnterminatedInput) {
  std::vector<std::string> lines;
  for (int i = 0; i < 8; ++i) {
    SendFromPeer(std::string(128, 'x'));
    ASSERT_TRUE(port_.ReadLines(lines));
  }
  EXPECT_TRUE(lines.empty());

  // The port stays usable once real frames resume.
  SendFromPeer("\n0,0,0,1,255,\n");
  ASSERT_TRUE(port_.ReadLines(lines));
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(lines.back(), "0,0,0,1,255,");
}

TEST_F(SerialPortTest, WriteReachesPeer) {
  ASSERT_TRUE(port_.Write("1,127,127\n")) << port_.LastError();
  EXPECT_EQ(ReceiveAtPeer(), "1,127,127\n");
}

TEST_F(SerialPortTest, WriteAndReadFailWhenClosed) {
  port_.Close();
  std::vector<std::string> lines;
  EXPECT_FALSE(port_.Write("1,127,127\n"));
  EXPECT_FALSE(port_.ReadLines(lines));
}

} // namespace
