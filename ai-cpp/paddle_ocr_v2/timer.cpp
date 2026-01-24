#include "Timer.h"

Timer::Timer(const std::string& name)
    : name_(name)
    , running_(false)
{
}

Timer::Timer(bool start_now, const std::string& name)
    : name_(name)
    , running_(false)
{
    if (start_now) {
        start();
    }
}

Timer::~Timer()
{
    stop();
}

void Timer::start()
{
    running_ = true;
    start_ = clock::now();
}

void Timer::stop()
{
    if (running_) {
        end_ = clock::now();
        running_ = false;
        printf("%s Cost %.2f ms\n", name_.c_str(), elapsed_ms());
    }
}

void Timer::reset()
{
    running_ = false;
}

bool Timer::running() const
{
    return running_;
}

double Timer::elapsed_ms() const
{
    clock::time_point end_time = running_ ? clock::now() : end_;
    return std::chrono::duration<double, std::milli>(end_time - start_).count();
}

double Timer::elapsed_us() const
{
    clock::time_point end_time = running_ ? clock::now() : end_;
    return std::chrono::duration<double, std::micro>(end_time - start_).count();
}
