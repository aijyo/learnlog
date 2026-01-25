--[[
English:
OneKeyBindSpellViewer main (OCR-minimal UI)
Goal:
- Show ONLY the most important info on line1 for PaddleOCR:
  SPELL=<spellId> GCD=<remain> SCD=<castRemain>
- Keep left icon visible.
- Hide/disable other info lines for now (commented out).
- WoW 12.0 compatible: use C_Spell/C_ActionBar when possible.
]]--

local ADDON_NAME = ...
OneKeyBindSpellViewerDB = OneKeyBindSpellViewerDB or {}

local DEFAULT_ACTION_SLOT = 59
local GCD_SPELL_ID = 61304
local UPDATE_INTERVAL = 0.05

-- =========================
-- Safe helpers
-- =========================

local function SafeCall(fn, ...)
    if type(fn) ~= "function" then return nil end
    local ok, a, b, c, d, e, f, g, h, i, j = pcall(fn, ...)
    if not ok then return nil end
    return a, b, c, d, e, f, g, h, i, j
end

local function Now()
    return SafeCall(GetTime) or 0
end

local function Remaining(startTime, duration)
    if not startTime or not duration or duration <= 0 then return 0 end
    local r = (startTime + duration) - Now()
    if r < 0 then r = 0 end
    return r
end

local function FormatSec(sec)
    if not sec or sec <= 0 then return "0.00" end
    return string.format("%.2f", sec)
end

local function IsMonitoring()
    return OneKeyBindSpellViewerDB.monitoring == true
end

local function SetMonitoring(v)
    OneKeyBindSpellViewerDB.monitoring = (v == true)
end

-- Get/Set monitored action slot
local function GetMonitoredSlot_()
    -- Guard: DB might be corrupted (non-table)
    if type(OneKeyBindSpellViewerDB) ~= "table" then
        OneKeyBindSpellViewerDB = {}
    end

    local v = OneKeyBindSpellViewerDB.slotId

    -- Accept both number and numeric string
    local n = tonumber(v)
    if n and n > 0 then
        return n
    end

    return DEFAULT_ACTION_SLOT
end

local function SetMonitoredSlot_(slotId)
    if type(slotId) ~= "number" or slotId <= 0 then
        OneKeyBindSpellViewerDB.slotId = nil
        return
    end
    OneKeyBindSpellViewerDB.slotId = slotId
end
-- =========================
-- slot → spell
-- =========================

local function GetSpellIdFromSlot(slotId)
    -- Prefer modern API first
    if C_ActionBar and type(C_ActionBar.GetSpell) == "function" then
        local sid = SafeCall(C_ActionBar.GetSpell, slotId)
        if type(sid) == "number" and sid > 0 then return sid end
    end

    -- Fallback: action info
    local aType, aId = SafeCall(GetActionInfo, slotId)
    if aType == "spell" and type(aId) == "number" then
        return aId
    end

    if aType == "macro" and aId then
        local mid = SafeCall(GetMacroSpell, aId)
        if type(mid) == "number" then return mid end
    end

    return nil
end

local function GetSpellIcon(slotId, spellId)
    if spellId then
        if C_Spell and type(C_Spell.GetSpellTexture) == "function" then
            local t = SafeCall(C_Spell.GetSpellTexture, spellId)
            if t then return t end
        end
        local t2 = SafeCall(GetSpellTexture, spellId)
        if t2 then return t2 end
    end
    return SafeCall(GetActionTexture, slotId)
end

-- =========================
-- GCD
-- =========================

local function GetGCDRemain()
    if C_Spell and type(C_Spell.GetSpellCooldown) == "function" then
        local cd = SafeCall(C_Spell.GetSpellCooldown, GCD_SPELL_ID)
        if cd and cd.isEnabled and cd.duration and cd.duration > 0 then
            return Remaining(cd.startTime or 0, cd.duration)
        end
    end
    return 0
end

-- =========================
-- Player casting remain (SCD)
-- =========================

local function GetPlayerCastRemain()
    -- Returns casting/channeling remaining seconds, or 0 if idle.
    local name, _, _, startMS, endMS = SafeCall(UnitCastingInfo, "player")
    if endMS then
        local e = (endMS or 0) / 1000.0
        local rem = e - Now()
        if rem < 0 then rem = 0 end
        return rem
    end

    local chName, _, _, chStartMS, chEndMS = SafeCall(UnitChannelInfo, "player")
    if chEndMS then
        local e = (chEndMS or 0) / 1000.0
        local rem = e - Now()
        if rem < 0 then rem = 0 end
        return rem
    end

    return 0
end

local function GetTargetTypeChar_()
    if not SafeCall(UnitExists, "target") then
        return "0"
    end

    -- Friendly
    if SafeCall(UnitIsFriend, "player", "target") then
        return "1"
    end

    -- Hostile / Attackable
    if SafeCall(UnitCanAttack, "player", "target") or SafeCall(UnitIsEnemy, "player", "target") then
        local isPlayer = SafeCall(UnitIsPlayer, "target")
        if isPlayer then
            return "3" -- hostile player
        end
        return "2" -- hostile npc
    end

    -- Neutral / other
    return "4"
end

local function FormatTargetChar_()
    local c = GetTargetTypeChar_()
    if not c or c == "" then return "0" end
    return c
end

-- local function GetTargetCastRemain_()
    -- -- Default: no cast or not interruptible
    -- if not SafeCall(UnitExists, "target") then
        -- return 0.0
    -- end

    -- -- UnitCastingInfo: name, text, texture, startTimeMS, endTimeMS, ...
    -- local name, _, _, startTimeMS, endTimeMS, _, _, notInterruptible =
        -- SafeCall(UnitCastingInfo, "target")

    -- if not name then
        -- return 0.0
    -- end

    -- -- If not interruptible, ignore
    -- if notInterruptible then
        -- return 0.0
    -- end

    -- local nowMS = GetTime() * 1000
    -- local remain = (endTimeMS - nowMS) / 1000.0

    -- if remain < 0 then
        -- remain = 0.0
    -- end

    -- return remain
-- end
local function GetTargetCastRemain_()
    if not UnitExists("target") then
        return 0.0
    end

    local castName, _, _, startTimeMS, endTimeMS, _, _, notInterruptible = SafeCall(UnitCastingInfo, "target")
    
    if not castName then
        return 0.0
    end

    if notInterruptible then
        return 0.0
    end

    if not startTimeMS or not endTimeMS or type(startTimeMS) ~= "number" or type(endTimeMS) ~= "number" then
        return 0.0
    end

    local currentTimeMS = GetTime() * 1000
    local remainingTimeSec = (endTimeMS - currentTimeMS) / 1000.0

    return math.max(remainingTimeSec, 0.0)
end
-- =========================
-- UI (solid window)
-- =========================

local UI = {}
local acc = 0
local lastSpellId = nil

local function CreateUI()
    if UI.frame then return end

    local f = CreateFrame("Frame", "OneKeyBindSpellViewerFrame", UIParent)
    -- Compact OCR-friendly size
    f:SetSize(256, 40)
    f:SetPoint("CENTER", UIParent, "CENTER", 0, 220)
    f:SetMovable(true)
    f:EnableMouse(true)
    f:RegisterForDrag("LeftButton")
    f:SetScript("OnDragStart", f.StartMoving)
    f:SetScript("OnDragStop", f.StopMovingOrSizing)

    -- Solid background
    local bg = f:CreateTexture(nil, "BACKGROUND")
    bg:SetAllPoints(true)
    bg:SetColorTexture(0.08, 0.08, 0.08, 1.0)
    UI.bg = bg

    -- Border
    local border = f:CreateTexture(nil, "BORDER")
    border:SetAllPoints(true)
    border:SetColorTexture(0.25, 0.25, 0.25, 1.0)
    UI.border = border

    -- Icon (left, vertically centered)
    local icon = f:CreateTexture(nil, "ARTWORK")
    icon:SetSize(32, 32)
    icon:SetPoint("LEFT", f, "LEFT", 8, 0)
    icon:Hide()
    UI.icon = icon

    -- Single OCR line (vertically centered, same row as icon)
    local text1 = f:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    text1:SetPoint("LEFT", icon, "RIGHT", 8, 0)
    text1:SetJustifyH("LEFT")
    text1:SetJustifyV("MIDDLE")
    text1:SetText("0=0.00=0.00")
    text1:SetTextColor(1, 1, 1, 1)
    UI.text1 = text1

    -- Hover hint for slot id (kept outside OCR main line)
    local slotHint = f:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
    slotHint:SetPoint("RIGHT", f, "RIGHT", -8, 0)
    slotHint:SetJustifyH("RIGHT")
    slotHint:SetJustifyV("MIDDLE")
    slotHint:SetText("")
    slotHint:Hide()
    UI.slotHint = slotHint

    UI.frame = f
end

-- Try best-effort to get action slot id from an action button
local function GetActionSlotFromButton_(btn)
    if not btn then return nil end

    -- Many action buttons expose .action
    if type(btn.action) == "number" and btn.action > 0 then
        return btn.action
    end

    -- Secure buttons may store it as an attribute
    local a = SafeCall(btn.GetAttribute, btn, "action")
    if type(a) == "number" and a > 0 then
        return a
    end

    -- Some buttons provide GetPagedID()
    if type(btn.GetPagedID) == "function" then
        local pid = SafeCall(btn.GetPagedID, btn)
        if type(pid) == "number" and pid > 0 then
            return pid
        end
    end

    return nil
end

local function ShowHoverSlot_(slotId)
    if not UI.slotHint then return end
    if type(slotId) ~= "number" or slotId <= 0 then
        UI.slotHint:Hide()
        return
    end
    UI.slotHint:SetText("SLOT=" .. tostring(slotId))
    UI.slotHint:Show()
end

local function HideHoverSlot_()
    if UI.slotHint then
        UI.slotHint:Hide()
    end
end

local function HookActionButtonsOnce_()
    if OneKeyBindSpellViewerDB and OneKeyBindSpellViewerDB._hookedActionButtons then
        return
    end
    OneKeyBindSpellViewerDB._hookedActionButtons = true

    -- Common action bar button name prefixes in Retail
    local prefixes = {
        "ActionButton",
        "MultiBarBottomLeftButton",
        "MultiBarBottomRightButton",
        "MultiBarRightButton",
        "MultiBarLeftButton",
        "MultiBar5Button",
        "MultiBar6Button",
        "MultiBar7Button",
    }

    for _, p in ipairs(prefixes) do
        for i = 1, 12 do
            local name = p .. tostring(i)
            local btn = _G[name]
            if btn and type(btn.HookScript) == "function" then
                btn:HookScript("OnEnter", function(self)
                    local slotId = GetActionSlotFromButton_(self)
                    ShowHoverSlot_(slotId)
                end)
                btn:HookScript("OnLeave", function()
                    HideHoverSlot_()
                end)
            end
        end
    end
end

-- =========================
-- Refresh
-- =========================

local function Refresh()
	local slotId = DEFAULT_ACTION_SLOT
	-- local slotId = GetMonitoredSlot_()

    local spellId = GetSpellIdFromSlot(slotId)
    local gcd = GetGCDRemain()
    local scd = GetPlayerCastRemain()

    -- Icon updates only when spell changes
    if spellId ~= lastSpellId then
        lastSpellId = spellId
        local tex = GetSpellIcon(slotId, spellId)
        if tex then
            UI.icon:SetTexture(tex)
            UI.icon:Show()
        else
            UI.icon:Hide()
        end
    end

    -- OCR-friendly short line: avoid NAME/Chinese text to reduce OCR ambiguity
    -- UI.text1:SetText(string.format(
        -- "SPELL=%s GCD=%s SCD=%s",
        -- spellId and tostring(spellId) or "-",
        -- FormatSec(gcd),
        -- FormatSec(scd)
    -- ))
    -- OCR-friendly format: <spellId>=<gcdRemain>=<castRemain>
    -- Keep everything numeric where possible for easier post-splitting.
    local sid = spellId and tostring(spellId) or "0"
    -- UI.text1:SetText(string.format("%s=%s=%s", sid, FormatSec(gcd), FormatSec(scd)))
	
	-- local tgt = FormatTargetChar_()
	-- UI.text1:SetText(string.format("%s=%s=%s=%s", sid, FormatSec(gcd), FormatSec(scd), tgt))
	local tgtCast = GetTargetCastRemain_()

	local text
	
	-- text = string.format(
		-- "%s=%s=%s",
		-- sid,
		-- FormatSec(gcd),
		-- FormatSec(scd)
	-- )
	-- -- Append target cast remain only when meaningful
	-- text = string.format(
		-- "%s=%s=%s=%.2f",
		-- sid,
		-- FormatSec(gcd),
		-- FormatSec(scd),
		-- tgtCast
	-- )
	if tgtCast > 0.5 then
		-- Append target cast remain only when meaningful
		text = string.format(
			"%s=%s=%s=%.2f",
			sid,
			FormatSec(gcd),
			FormatSec(scd),
			tgtCast
		)
	else
		text = string.format(
			"%s=%s=%s",
			sid,
			FormatSec(gcd),
			FormatSec(scd)
		)
	end

	UI.text1:SetText(text)

    -- Old module lines disabled for now:
    -- if OneKeyBindSpellViewer_PlayerInfo then
    --     ...
    -- end
end

local function StartLoop()
    acc = 0
    UI.frame:SetScript("OnUpdate", function(_, e)
        acc = acc + e
        if acc < UPDATE_INTERVAL then return end
        acc = 0
        Refresh()
    end)
end

local function StopLoop()
    UI.frame:SetScript("OnUpdate", nil)
end

-- =========================
-- Slash
-- =========================

SLASH_ONEKEYBINDSPELLVIEWER1 = "/okbslot"
SlashCmdList["ONEKEYBINDSPELLVIEWER"] = function()
    CreateUI()
    if IsMonitoring() then
        SetMonitoring(false)
        StopLoop()
        UI.text1:SetText("Stopped. /okbslot")
    else
        SetMonitoring(true)
        lastSpellId = nil
        StartLoop()
    end
end

SLASH_ONEKEYBINDSPELLVIEWERSLOT1 = "/okbslot"
SlashCmdList["ONEKEYBINDSPELLVIEWERSLOT"] = function(msg)
    CreateUI()

    msg = (msg or ""):match("^%s*(.-)%s*$")
    if msg == "" then
        -- Reset to default
        SetMonitoredSlot_(nil)
        lastSpellId = nil
        Refresh()
        print("OKB slot reset to default: " .. tostring(DEFAULT_ACTION_SLOT))
        return
    end

    local n = tonumber(msg)
    if not n or n <= 0 then
        print("Usage: /okbslot <number>   e.g. /okbslot 59   (empty to reset)")
        return
    end

    SetMonitoredSlot_(n)
    lastSpellId = nil
    Refresh()
    print("OKB slot set to: " .. tostring(n))
end

-- =========================
-- Init
-- =========================

local ev = CreateFrame("Frame")
ev:RegisterEvent("PLAYER_LOGIN")
ev:SetScript("OnEvent", function()
    CreateUI()
    HookActionButtonsOnce_()
	OneKeyBindSpellViewerDB = OneKeyBindSpellViewerDB or {}
	if type(OneKeyBindSpellViewerDB) ~= "table" then
		OneKeyBindSpellViewerDB = {}
	end
    if OneKeyBindSpellViewerDB.monitoring == nil then
        OneKeyBindSpellViewerDB.monitoring = false
    end
    if IsMonitoring() then
        lastSpellId = nil
        StartLoop()
    end
end)
