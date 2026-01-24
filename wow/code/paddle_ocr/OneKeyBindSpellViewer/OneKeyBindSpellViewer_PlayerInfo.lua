--[[
English:
PlayerInfo module for OneKeyBindSpellViewer (WoW 12.0 compatible)
Fix: Target info in 12.0 may be "Soft Target" rather than hard "target".
- Keep exported interfaces unchanged:
  M.GetPlayerIdentityLine()
  M.GetStatusLine()
  M.GetTargetLine()
  M.GetTargetCastLine()
- Defensive coding: all API calls are guarded via SafeCall.
]]--

OneKeyBindSpellViewer_PlayerInfo = OneKeyBindSpellViewer_PlayerInfo or {}
local M = OneKeyBindSpellViewer_PlayerInfo

-- =========================
-- Safe helpers
-- =========================

local function SafeCall(fn, ...)
    -- Call WoW API safely. Returns nil if fn missing or errors.
    if type(fn) ~= "function" then return nil end
    local ok, a, b, c, d, e, f, g, h, i, j = pcall(fn, ...)
    if not ok then return nil end
    return a, b, c, d, e, f, g, h, i, j
end

local function Now()
    return SafeCall(GetTime) or 0
end

local function FormatSec(sec)
    if not sec or sec <= 0 then return "0.00" end
    return string.format("%.2f", sec)
end

local function Clamp01(x)
    if not x then return 0 end
    if x < 0 then return 0 end
    if x > 1 then return 1 end
    return x
end

local function FormatPct(cur, maxv)
    if not cur or not maxv or maxv <= 0 then return "0%" end
    return string.format("%d%%", math.floor(Clamp01(cur / maxv) * 100 + 0.5))
end

local function GetUnitFullName(unit)
    -- Returns "Name-Realm" when realm exists; otherwise "Name".
    local name, realm = SafeCall(UnitName, unit)
    if not name or name == "" then return "-" end
    if realm and realm ~= "" then
        return tostring(name) .. "-" .. tostring(realm)
    end
    return tostring(name)
end

-- =========================
-- Player basic info
-- =========================

function M.GetPlayerIdentityLine()
    -- NOTE: You asked to use player name + realm (NOT GUID).
    local fullName = GetUnitFullName("player")

    local className = "-"
    local raceName = "-"
    local specName = "-"

    do
        local localizedClass = SafeCall(UnitClass, "player")
        if type(localizedClass) == "string" and localizedClass ~= "" then
            className = localizedClass
        end
    end

    do
        local localizedRace = SafeCall(UnitRace, "player")
        if type(localizedRace) == "string" and localizedRace ~= "" then
            raceName = localizedRace
        end
    end

    -- Spec (player only; safe if unavailable)
    do
        local specIndex = SafeCall(GetSpecialization)
        if type(specIndex) == "number" and specIndex > 0 and type(GetSpecializationInfo) == "function" then
            local _, specNameLocal = SafeCall(GetSpecializationInfo, specIndex)
            if type(specNameLocal) == "string" and specNameLocal ~= "" then
                specName = specNameLocal
            end
        end
    end

    return string.format(
        "PLAYER=%s   CLASS=%s   SPEC=%s   RACE=%s",
        tostring(fullName),
        tostring(className),
        tostring(specName),
        tostring(raceName)
    )
end

-- =========================
-- Casting / Channeling (unit)
-- =========================

local function GetUnitCastInfo(unit)
    -- Returns: stateText, remain, duration, spellId, interruptible
    -- stateText: "casting:<spell>" / "channel:<spell>" / "idle"
    -- interruptible: "yes"/"no"/"-"

    if not unit then
        return "idle", 0, 0, nil, "-"
    end

    local name, _, _, startMS, endMS, _, _, notInterruptible, spellId = SafeCall(UnitCastingInfo, unit)
    if endMS then
        local s = (startMS or 0) / 1000.0
        local e = (endMS or 0) / 1000.0
        local dur = e - s
        local rem = e - Now()
        if rem < 0 then rem = 0 end
        if dur < 0 then dur = 0 end
        local intr = (notInterruptible == true) and "no" or "yes"
        return "casting:" .. tostring(name or "-"), rem, dur, spellId, intr
    end

    local chName, _, _, chStartMS, chEndMS, _, chNotInterruptible, chSpellId = SafeCall(UnitChannelInfo, unit)
    if chEndMS then
        local s = (chStartMS or 0) / 1000.0
        local e = (chEndMS or 0) / 1000.0
        local dur = e - s
        local rem = e - Now()
        if rem < 0 then rem = 0 end
        if dur < 0 then dur = 0 end
        local intr = (chNotInterruptible == true) and "no" or "yes"
        return "channel:" .. tostring(chName or "-"), rem, dur, chSpellId, intr
    end

    return "idle", 0, 0, nil, "-"
end

-- =========================
-- Loss of Control (CC) for player
-- =========================

local function GetLossOfControlText()
    -- Prefer C_LossOfControl (works for player control states)
    if C_LossOfControl and type(C_LossOfControl.GetActiveLossOfControlData) == "function" then
        local data = SafeCall(C_LossOfControl.GetActiveLossOfControlData, 1)
        if type(data) == "table" then
            local locType = data.locType or data.lossOfControlType or data.type or "-"
            local rem = data.timeRemaining or data.remainingTime or 0
            local txt = data.displayText or data.name or nil
            if txt and txt ~= "" then
                return string.format("%s(%s) rem=%s", tostring(locType), tostring(txt), FormatSec(rem))
            end
            if locType ~= "-" then
                return string.format("%s rem=%s", tostring(locType), FormatSec(rem))
            end
        end
    end
    return "-"
end

function M.GetStatusLine()
    local castState, rem, dur = GetUnitCastInfo("player")
    local cc = GetLossOfControlText()

    -- local bar = "0.00/0.00"
    local bar = "0.00"
    if dur and dur > 0 then
        -- bar = FormatSec(rem) .. "/" .. FormatSec(dur)
        bar = FormatSec(rem)
    end

    return string.format("CAST=%s   SCD=%s   CTRL=%s", tostring(castState), tostring(bar), tostring(cc))
end

-- =========================
-- Power helpers
-- =========================

local function GetPowerTypeName(unit)
    local pType = SafeCall(UnitPowerType, unit)
    if pType == 0 then return "mana" end
    if pType == 1 then return "rage" end
    if pType == 2 then return "focus" end
    if pType == 3 then return "energy" end
    if pType == 6 then return "runic" end
    if pType == 8 then return "lunarpower" end
    if pType == 11 then return "maelstrom" end
    if pType == 13 then return "insanity" end
    if pType == 17 then return "fury" end
    if pType == 18 then return "pain" end
    return "power" .. tostring(pType or "-")
end

-- =========================
-- Target resolver (12.0: hard target + soft targets)
-- =========================

local function UnitIsValid(unit)
    if not unit then return false end
    local exists = SafeCall(UnitExists, unit)
    if exists then return true end
    local guid = SafeCall(UnitGUID, unit)
    if guid and guid ~= "" then return true end
    local name = SafeCall(UnitName, unit)
    if name and name ~= "" then return true end
    return false
end

local function ResolveTargetUnit()
    -- Priority order:
    -- 1) Hard target (classic behavior)
    -- 2) Soft targets introduced/used heavily in modern clients
    -- 3) Mouseover fallback for UI hover scenarios

    local candidates = {
        { unit = "target",        tag = "target" },
        { unit = "softenemy",     tag = "softenemy" },
        { unit = "softfriend",    tag = "softfriend" },
        { unit = "softinteract",  tag = "softinteract" },
        { unit = "mouseover",     tag = "mouseover" },
    }

    for _, c in ipairs(candidates) do
        if UnitIsValid(c.unit) then
            return c.unit, c.tag
        end
    end

    return nil, "-"
end

-- =========================
-- Target lines
-- =========================

-- function M.GetTargetLine()
    -- local unit, tag = ResolveTargetUnit()
    -- if not unit then
        -- return "targetUnit=-  target=-  hp=-  power=-"
    -- end

    -- -- target name: Name-Realm for players, Name for NPCs
    -- local name, realm = SafeCall(UnitName, unit)
    -- local fullName = "-"
    -- if name and name ~= "" then
        -- if realm and realm ~= "" then
            -- fullName = tostring(name) .. "-" .. tostring(realm)
        -- else
            -- fullName = tostring(name)
        -- end
    -- end

    -- local hp = SafeCall(UnitHealth, unit) or 0
    -- local hpMax = SafeCall(UnitHealthMax, unit) or 0
    -- local hpPct = FormatPct(hp, hpMax)

    -- local powTypeName = GetPowerTypeName(unit)
    -- local pow = SafeCall(UnitPower, unit) or 0
    -- local powMax = SafeCall(UnitPowerMax, unit) or 0

    -- return string.format(
        -- "targetUnit=%s  target=%s  hp=%d/%d(%s)  %s=%d/%d",
        -- tostring(tag),
        -- tostring(fullName),
        -- tonumber(hp) or 0, tonumber(hpMax) or 0, tostring(hpPct),
        -- tostring(powTypeName),
        -- tonumber(pow) or 0, tonumber(powMax) or 0
    -- )
-- end
function M.GetTargetLine()
    local unit, tag = ResolveTargetUnit()
    if not unit then
        return "TUNIT=-   TARGET=-   PHP=-   BUF=-   DEBUF=-"
    end

    -- Basic name
    local name, realm = SafeCall(UnitName, unit)
    local fullName = "-"
    if name and name ~= "" then
        if realm and realm ~= "" then
            fullName = tostring(name) .. "-" .. tostring(realm)
        else
            fullName = tostring(name)
        end
    end

    -- Health (prefer UnitHealthPercent if available)
    local hpCur = SafeCall(UnitHealth, unit) or 0
    local hpMax = SafeCall(UnitHealthMax, unit) or 0
    local hpPct = SafeCall(UnitHealthPercent, unit) or 0
    -- hpPct = (hpMax > 0) and math.floor((hpCur / hpMax) * 100 + 0.5)
    if hpPct == nil then
        -- Fallback to manual percentage
        -- hpPct = (hpMax > 0) and math.floor((hpCur / hpMax) * 100 + 0.5) or 0
	else
		-- hpPct = math.floor(hpPct * 100 + 0.5) or 0
    end

    -- Position / facing (might return nil/0 when out of range)
    -- local x, y, z = SafeCall(UnitPosition, unit)
    -- local facing = SafeCall(UnitFacing, unit) or 0

    -- local posStr = "-"
    -- if x and y and z then
        -- posStr = string.format("x=%.2f y=%.2f z=%.2f f=%.2f", x, y, z, facing)
    -- end

    -- -- Buff / debuff counts (lightweight; avoids storing full lists)
    -- local buffCount = 0
    -- do
        -- local i = 1
        -- while true do
            -- local auraName = SafeCall(UnitBuff, unit, i)
            -- if not auraName then break end
            -- buffCount = buffCount + 1
            -- i = i + 1
            -- -- Safety cap to avoid pathological loops
            -- if i > 80 then break end
        -- end
    -- end

    -- local debuffCount = 0
    -- do
        -- local i = 1
        -- while true do
            -- local auraName = SafeCall(UnitDebuff, unit, i)
            -- if not auraName then break end
            -- debuffCount = debuffCount + 1
            -- i = i + 1
            -- if i > 80 then break end
        -- end
    -- end

    return string.format(
        "TARGET=%s   HP=%d/%d   PHP=%.2f   BUF=%d   DEBUF=%d",
        tostring(fullName),
        tonumber(hpCur) or 0, tonumber(hpMax) or 0, tonumber(hpPct) or 0,
        tonumber(buffCount) or 0,
        tonumber(debuffCount) or 0
    )
end

function M.GetTargetCastLine()
    local unit = "targettarget"
    local tag = "tt"

    if not UnitExists(unit) then
        return "TTNAME=- TTHP=0/0(0%) TCAST=idle SCD=0.00/0.00 TSPELL=- INTER=- debuffs2=-"
    end

    local name = UnitName(unit) or "-"
    local hp = UnitHealth(unit) or 0
    local hpMax = UnitHealthMax(unit) or 0
    local hpPct = SafeCall(UnitHealthPercent, unit)

    local state, rem, dur, spellId, intr = GetUnitCastInfo(unit)

    local bar = "0.00/0.00"
    if dur and dur > 0 then
        bar = FormatSec(rem) .. "/" .. FormatSec(dur)
    end

    
    local debuffStrList = {}
    local index = 1

    while index <= 5 do
        local name, _, count, debuffType, duration, expires, _, _, _, spellId =
            UnitDebuff(unit, index)

        if not name then break end

        local remain = (expires and expires > 0) and math.max(0, expires - GetTime()) or 0

        table.insert(
            debuffStrList,
            string.format(
                "%s:%d:%.1f",
                spellId or "0",
                count or 0,
                remain
            )
        )

        index = index + 1
    end

    local debuffs = (#debuffStrList > 0) and table.concat(debuffStrList, ",") or "-"

    return string.format(
        "TTNAME=%s TTHP=%d/%d(%.2f) TCAST=%s TSCD=%s TSPELL=%s INTER=%s DEBUF=%s",
        name,
        hp,
        hpMax,
        hpPct,
        tostring(state or "idle"),
        bar,
        spellId and tostring(spellId) or "-",
        tostring(intr or "-"),
        debuffs
    )
end

return M
