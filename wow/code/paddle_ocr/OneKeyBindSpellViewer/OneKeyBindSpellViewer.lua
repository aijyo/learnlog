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

local DEFAULT_ACTION_SLOT = 68
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

-- =========================
-- UI (solid window)
-- =========================

local UI = {}
local acc = 0
local lastSpellId = nil

local function CreateUI()
    if UI.frame then return end

    local f = CreateFrame("Frame", "OneKeyBindSpellViewerFrame", UIParent)
    -- Keep it compact for OCR
    f:SetSize(300, 58)
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

    -- Icon (left)
    local icon = f:CreateTexture(nil, "ARTWORK")
    icon:SetSize(36, 36)
    icon:SetPoint("TOPLEFT", f, "TOPLEFT", 10, -11)
    icon:Hide()
    UI.icon = icon

    -- Line1 (only line for OCR)
    local text1 = f:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    text1:SetPoint("TOPLEFT", f, "TOPLEFT", 56, -18)
    text1:SetJustifyH("LEFT")
    text1:SetText("0=0.00=0.00")
	text1:SetTextColor(1, 1, 1, 1)
    UI.text1 = text1

    -- Disable other lines (kept as comments for future restore)
    -- local text2 = f:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
    -- local text3 = f:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")
    -- local text4 = f:CreateFontString(nil, "OVERLAY", "GameFontHighlightSmall")

    UI.frame = f
end

-- =========================
-- Refresh
-- =========================

local function Refresh()
    local slotId = DEFAULT_ACTION_SLOT
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
    UI.text1:SetText(string.format("%s=%s=%s", sid, FormatSec(gcd), FormatSec(scd)))

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

SLASH_ONEKEYBINDSPELLVIEWER1 = "/okb"
SlashCmdList["ONEKEYBINDSPELLVIEWER"] = function()
    CreateUI()
    if IsMonitoring() then
        SetMonitoring(false)
        StopLoop()
        UI.text1:SetText("Stopped. /okb")
    else
        SetMonitoring(true)
        lastSpellId = nil
        StartLoop()
    end
end

-- =========================
-- Init
-- =========================

local ev = CreateFrame("Frame")
ev:RegisterEvent("PLAYER_LOGIN")
ev:SetScript("OnEvent", function()
    CreateUI()
    OneKeyBindSpellViewerDB = OneKeyBindSpellViewerDB or {}
    if OneKeyBindSpellViewerDB.monitoring == nil then
        OneKeyBindSpellViewerDB.monitoring = false
    end
    if IsMonitoring() then
        lastSpellId = nil
        StartLoop()
    end
end)
