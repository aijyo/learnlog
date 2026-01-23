#pragma once
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "utils/absl_shim.h"

#include "frame_def.h"
// ---------------------------
// Handler output (parsed UI)
// ---------------------------
class WowTextHandler {
public:
    struct ParsedCooldown {
        // Example input: "1.2s/30.0s" or "0"
        float remain_sec = 0.0f;
        float duration_sec = 0.0f;
        bool on_cd = false;
    };

    struct ParsedCharges {
        // Example input: "1/2(3.4s)" or "-"
        int cur = -1;
        int max = -1;
        float recover_remain_sec = 0.0f;
        bool valid = false;
    };

    struct ParsedCast {
        // playerCast format (lua):
        // "NONE"
        // or "CAST:Fireball(123):1.2s THIS"
        // or "CHAN:xxx(456):0.8s OTHER"
        std::string mode;          // "NONE" / "CAST" / "CHAN"
        std::string name;          // cast spell name
        int spell_id = 0;
        float remain_sec = 0.0f;
        bool match_this_spell = false; // THIS / OTHER (only for player line)
        bool active = false;
    };

    struct ParsedTargetCast {
        // target cast format (lua targetLine):
        // "cast=NONE" or "cast=CAST:xxx(123):1.2s"
        std::string mode;     // "NONE"/"CAST"/"CHAN"
        std::string name;
        int spell_id = 0;
        float remain_sec = 0.0f;
        bool active = false;
    };

    struct ParsedTarget {
        // target line (lua):
        // "target:Name  HP=88%  d=1  atk=Y enemy=Y elite  aura=1/2  cast=CAST:xxx(123):1.2s"
        bool exists = false;
        std::string name;
        int hp_pct = -1;            // 0..100, -1 unknown
        std::string dist_tier;      // e.g. "d=1" / "d=2" / "-" ...
        bool can_attack = false;    // atk=Y/N
        bool is_enemy = false;      // enemy=Y/N
        std::string classification; // "normal/elite/rare/rareelite/worldboss/..."
        int buffs = -1;
        int debuffs = -1;
        ParsedTargetCast cast;
    };

    struct ParsedSilence {
        // l5: "silence=0" or "silence=1.2s Kick"
        bool active = false;
        float remain_sec = 0.0f;
        std::string name;
    };

    struct ParsedUi {
        // Raw reconstructed lines (top -> down)
        std::string l1, l2, l3, l4, l5;

        // l1:
        int action_slot = -1;
        std::string action_type;  // e.g. "spell/macro/..."
        std::string action_id;    // string form inside (...) in lua
        int spell_id = 0;         // 0 if "-"
        std::string spell_name;   // trailing part

        // l2:
        ParsedCooldown cd;
        ParsedCharges charges;
        bool usable = false;      // usable=Y/N
        bool not_enough = false;  // mana?=Y/N (lua prints notEnough as "mana?=")
        std::string range;        // "in/out/-"
        bool is_current = false;  // cur=Y/N
        bool is_attack = false;   // atk=Y/N
        bool is_auto_repeat = false; // rep=Y/N

        // l3:
        ParsedCooldown gcd;
        ParsedCast player_cast;
        bool in_combat = false;
        bool is_mounted = false;
        bool is_dead = false;

        // l4:
        ParsedTarget target;

        // l5:
        ParsedSilence silence;
    };

    struct Config {
        // English:
        // When grouping OCR instances into lines, this controls how tolerant we are.
        // Larger -> more likely to merge into same line.
        float line_merge_y_tolerance_mul = 0.6f;

        // English:
        // Only accept OCR tokens with rec_score >= this threshold (0..1).
        float min_rec_score = 0.0f;

        // English:
        // Max lines to take from top (UI has 5 lines l1~l5).
        int max_ui_lines = 5;
    };

    using Callback = std::function<void(const ParsedUi&)>;

public:
    WowTextHandler() = default;
    explicit WowTextHandler(const Config& cfg) : cfg_(cfg) {}

    void SetCallback(Callback cb);

    // Main entry required by you.
    absl::Status ProcessByTextHander(OcrFrameResult& frame) const;

private:
    // Reconstruct lines from OCR instances, top-down.
    static std::vector<std::string> ReconstructLinesTopDown(
        const std::vector<OcrInstance>& items,
        const Config& cfg);

    // Normalize spaces/punctuations to make parsing stable.
    static std::string NormalizeLine(const std::string& s);

    // Parsing helpers (match your lua formatting).
    static bool ParseLine1(const std::string& l1, ParsedUi& out);
    static bool ParseLine2(const std::string& l2, ParsedUi& out);
    static bool ParseLine3(const std::string& l3, ParsedUi& out);
    static bool ParseLine4Target(const std::string& l4, ParsedUi& out);
    static bool ParseLine5Silence(const std::string& l5, ParsedUi& out);

    static float ParseTimeSeconds(const std::string& token);
    static ParsedCooldown ParseCooldownToken(const std::string& token);
    static ParsedCharges ParseChargesToken(const std::string& token);

    static bool ParseYN(const std::string& v, bool& out);
    static std::string Trim(const std::string& s);

private:
    Config cfg_;
    Callback cb_;
    mutable std::mutex mu_;
};
