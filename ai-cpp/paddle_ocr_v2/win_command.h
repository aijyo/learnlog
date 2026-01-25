#pragma once
#include <string>

namespace wowapp {

    // Start the embedded pipeline using a UTF-8 JSON configuration string.
    // Returns false if already running or configuration is invalid.
    bool Start(const std::string& json_config_utf8);

    // Stop the embedded pipeline. 'reason' is user-defined for logging/telemetry.
    void Stop(int reason);

    // Query running state.
    bool IsRunning();

} // namespace wowapp
