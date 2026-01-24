#pragma once

#include <chrono>
#include <cstdint>
#include <string>

/*
 * Timer
 *
 * A lightweight utility class for measuring elapsed time.
 * - Uses std::chrono::steady_clock (monotonic, not affected by system clock changes)
 * - Supports manual start/stop and RAII-style scope timing
 */
class Timer
{
public:
    using clock = std::chrono::steady_clock;

    // Construct a timer without starting it
    Timer(const std::string& name = "name");

    // Construct and start immediately
    explicit Timer(bool start_now, const std::string& name = "name");

    ~Timer();
    // Start or restart the timer
    void start();

    // Stop the timer (records end time)
    void stop();

    // Reset timer to initial state (not running)
    void reset();

    // Check whether the timer is currently running
    bool running() const;

    // Elapsed time queries
    double elapsed_ms() const;   // milliseconds
    double elapsed_us() const;   // microseconds

private:
    std::string name_;
    clock::time_point start_;
    clock::time_point end_;
    bool running_ = false;
};
