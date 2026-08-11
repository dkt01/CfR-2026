// Minimal non-blocking POSIX serial port wrapper.
//
// Deliberately dependency free (plain termios) so the bridge builds on a stock
// JetPack image without pulling in a serial library.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cfr_arduino_bridge {

  class SerialPort {
   public:
    SerialPort() = default;
    ~SerialPort();

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    /// Open @p device in raw, non-blocking mode at @p baud.
    /// Returns false on failure; LastError() describes what went wrong.
    bool Open(const std::string& device, unsigned baud);

    void Close();

    bool IsOpen() const { return fd_ >= 0; }

    /// Write the whole payload.  Returns false on a hard error, in which case
    /// the caller should Close() and reconnect.
    bool Write(const std::string& payload);

    /// Drain everything currently readable and append every complete,
    /// newline terminated line to @p lines (newline stripped).  Returns false
    /// on a hard error.  Returning true with no new lines is normal.
    bool ReadLines(std::vector<std::string>& lines);

    const std::string& LastError() const { return last_error_; }

   private:
    static bool BaudToSpeed(unsigned baud, unsigned& speed);

    /// Guard against a peer that never sends a newline.
    static constexpr size_t kMaxBufferedBytes = 512;

    int fd_ = -1;
    std::string rx_buffer_;
    std::string last_error_;
  };

}  // namespace cfr_arduino_bridge
