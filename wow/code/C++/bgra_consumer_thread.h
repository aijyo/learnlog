#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "utils.h"

class BgraFrameConsumerThread
{
public:
    using FrameCallback = std::function<void(BgraFrame&&)>;

    explicit BgraFrameConsumerThread(FrameCallback cb = nullptr)
        : cb_(std::move(cb))
    {
    }

    ~BgraFrameConsumerThread()
    {
        Stop();
    }

    BgraFrameConsumerThread(const BgraFrameConsumerThread&) = delete;
    BgraFrameConsumerThread& operator=(const BgraFrameConsumerThread&) = delete;

    void Start()
    {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true))
            return;

        worker_ = std::thread([this]() { ThreadMain(); });
    }

    void Stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false))
            return;

        {
            std::lock_guard<std::mutex> lk(mtx_);
            // Signal the worker to exit
            has_frame_ = true;
        }
        cv_.notify_one();

        if (worker_.joinable())
            worker_.join();
    }

    // Producer pushes a new frame. This overwrites the previous pending frame.
    void Submit(BgraFrame&& frame)
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            latest_ = std::move(frame);
            has_frame_ = true;
        }
        cv_.notify_one();
    }

private:
    void ThreadMain()
    {
        while (true)
        {
            BgraFrame local;

            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]() { return has_frame_; });

                // If Stop() was called, running_ is false; we exit after waking up.
                if (!running_.load())
                    break;

                // Move the latest frame out, and reset the flag.
                local = std::move(latest_);
                has_frame_ = false;
            }

            // Consume outside the lock to minimize contention.
            if (cb_)
                cb_(std::move(local));
        }
    }

private:
    FrameCallback cb_;

    std::atomic<bool> running_{ false };
    std::thread worker_;

    std::mutex mtx_;
    std::condition_variable cv_;

    bool has_frame_ = false;
    BgraFrame latest_;
};
