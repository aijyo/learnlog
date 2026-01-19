--[[
English:
OneKeyBindSpellViewer (rich slot monitor)

Commands:
  /okb <barIndex> <slotInBar>
    Example: /okb 3 2
             /okb 103 2   -> MultiActionBar3 slot 2 (typical right bar 1)
  /okb -1
    Stop monitoring and hide UI

UI shows:
- 64x64 action icon (exactly like the monitored slot)
- action slot index, binding keys
- action type (spell/macro/item/none)
- spell name + spellID (if resolvable)
- spell cooldown remaining
- GCD status + remaining (via spellID 61304)
- charges (if any), and charge recovery
- usability & range status
]]

local ADDON_NAME = ...
OneKeyBindSpellViewerDB = OneKeyBindSpellViewerDB or {}

local SLOTS_PER_BAR = 12
local GCD_SPELL_ID = 61304 -- English: universal GCD spell id

-- Blizzard MultiActionBar slot bases:
-- MultiActionBar1 => 61..72
-- MultiActionBar2 => 49..60
-- MultiActionBar3 => 25..36
-- MultiActionBar4 => 37..48
local MULTI_BASE = {
    [101] = 60, -- +n => 61..72
    [102] = 48, -- +n => 49..60
    [103] = 24, -- +n => 25..36
    [104] = 36, -- +n => 37..48
}

local function Trim(s)
    if not s then return "" end
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function Clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function Now()
    return GetTime()
end

local function FormatTime(sec)
    if not sec or sec <= 0 then return "0.0s" end
    if sec >= 60 then
        return string.format("%.1fm", sec / 60.0)
    end
    return string.format("%.1fs", sec)
end

local function Remaining(startTime, duration)
    if not startTime or not duration or duration <= 0 then return 0 end
    local r = (startTime + duration) - Now()
    if r < 0 then r = 0 end
    return r
end

local function GetActionSlot(barIndex, slotInBar)
    slotInBar = Clamp(slotInBar, 1, 12)
    if MULTI_BASE[barIndex] then
        return MULTI_BASE[barIndex] + slotInBar
    end
    barIndex = Clamp(barIndex, 1, 10)
    return (barIndex - 1) * SLOTS_PER_BAR + slotInBar
end

local function FindKeysForActionSlot(actionSlot)
    -- English:
    -- Best-effort reverse mapping from actionSlot to binding command,
    -- then query keys with GetBindingKey(cmd).
    local candidates = {}

    if actionSlot >= 1 and actionSlot <= 12 then
        table.insert(candidates, "ACTIONBUTTON" .. tostring(actionSlot))
    end

    if actionSlot >= 61 and actionSlot <= 72 then
        table.insert(candidates, "MULTIACTIONBAR1BUTTON" .. tostring(actionSlot - 60))
    elseif actionSlot >= 49 and actionSlot <= 60 then
        table.insert(candidates, "MULTIACTIONBAR2BUTTON" .. tostring(actionSlot - 48))
    elseif actionSlot >= 25 and actionSlot <= 36 then
        table.insert(candidates, "MULTIACTIONBAR3BUTTON" .. tostring(actionSlot - 24))
    elseif actionSlot >= 37 and actionSlot <= 48 then
        table.insert(candidates, "MULTIACTIONBAR4BUTTON" .. tostring(actionSlot - 36))
    end

    for _, cmd in ipairs(candidates) do
        local k1, k2 = GetBindingKey(cmd)
        if (k1 and k1 ~= "") or (k2 and k2 ~= "") then
            return k1 or "", k2 or "", cmd
        end
    end
    return "", "", candidates[1] or ""
end

local function GetGCDInfo()
    -- English:
    -- Use spellID 61304 to get global cooldown reliably.
    local s, d, e = GetSpellCooldown(GCD_SPELL_ID)
    if e ~= 1 or not d or d <= 0 then
        return false, 0, 0
    end
    -- Some APIs return very small cooldown when none; treat > 0 as gcd.
    local rem = Remaining(s, d)
    if rem <= 0 then
        return false, 0, 0
    end
    return true, rem, d
end

local function ResolveActionSlot(actionSlot)
    -- Returns a table with lots of fields for UI.
    local info = {
        actionSlot = actionSlot,
        actionType = "none",
        actionId = nil,       -- spellID or macroId or itemId depending on type
        spellID = nil,        -- resolved spellID if possible
        spellName = "-",
        iconTex = nil,        -- exact action texture preferred
        usableStr = "-",
        rangeStr = "-",
        cdRemain = 0,
        cdDur = 0,
        cdStr = "-",
        chargesStr = "-",
        gcdStr = "-",
        gcdRemain = 0,
        key1 = "",
        key2 = "",
        bindCmd = "",
    }

    -- Exact icon from action slot (closest to "like the monitored slot")
    info.iconTex = GetActionTexture(actionSlot)

    local aType, aId = GetActionInfo(actionSlot)
    info.actionType = aType or "none"
    info.actionId = aId

    -- Binding keys (best effort)
    local k1, k2, cmd = FindKeysForActionSlot(actionSlot)
    info.key1, info.key2, info.bindCmd = k1, k2, cmd

    -- GCD
    local gcdActive, gcdRemain, gcdDur = GetGCDInfo()
    info.gcdRemain = gcdRemain
    if gcdActive then
        info.gcdStr = string.format("ON (%s / %s)", FormatTime(gcdRemain), FormatTime(gcdDur))
    else
        info.gcdStr = "OFF"
    end

    -- If not a usable action, stop here
    if not aType or not aId then
        info.spellName = "-"
        info.cdStr = "-"
        return info
    end

    local resolvedSpellID = nil
    local resolvedName = nil
    local resolvedIcon = nil

    if aType == "spell" then
        resolvedSpellID = aId
        resolvedName, _, resolvedIcon = GetSpellInfo(resolvedSpellID)
    elseif aType == "macro" then
        -- Try resolve macro -> spell
        local macroSpellID = GetMacroSpell(aId)
        local macroItemName = GetMacroItem(aId)
        if macroSpellID then
            resolvedSpellID = macroSpellID
            resolvedName, _, resolvedIcon = GetSpellInfo(resolvedSpellID)
        elseif macroItemName then
            info.spellName = "ItemMacro: " .. tostring(macroItemName)
        else
            info.spellName = "Macro"
        end
    elseif aType == "item" then
        -- Item action: spellID might not exist; show item name
        local itemName = GetItemInfo(aId)
        info.spellName = itemName or ("ItemID: " .. tostring(aId))
    end

    if resolvedSpellID then
        info.spellID = resolvedSpellID
        info.spellName = resolvedName or ("Spell " .. tostring(resolvedSpellID))

        -- Prefer exact action texture; fallback to spell icon
        if not info.iconTex then
            info.iconTex = resolvedIcon
        end

        -- Usability
        local usable, noMana = IsUsableSpell(resolvedSpellID)
        if usable then
            info.usableStr = "OK"
        else
            info.usableStr = noMana and "NoMana" or "No"
        end

        -- Range (only meaningful for some spells)
        local inRange = IsSpellInRange(resolvedSpellID, "target")
        if inRange == 1 then
            info.rangeStr = "InRange"
        elseif inRange == 0 then
            info.rangeStr = "OutRange"
        else
            info.rangeStr = "N/A"
        end

        -- Charges
        local cur, max, chStart, chDur = GetSpellCharges(resolvedSpellID)
        if max and max > 0 then
            local chRem = Remaining(chStart, chDur)
            if chRem > 0 then
                info.chargesStr = string.format("%d/%d (regen %s)", cur or 0, max, FormatTime(chRem))
            else
                info.chargesStr = string.format("%d/%d", cur or 0, max)
            end
        else
            info.chargesStr = "-"
        end

        -- Cooldown: prefer spell cooldown, but sometimes action cooldown differs for items/macros
        local s1, d1, e1 = GetSpellCooldown(resolvedSpellID)
        local spellRem = (e1 == 1 and d1 and d1 > 0) and Remaining(s1, d1) or 0

        local s2, d2, e2 = GetActionCooldown(actionSlot)
        local actionRem = (e2 == 1 and d2 and d2 > 0) and Remaining(s2, d2) or 0

        local rem = spellRem
        local dur = d1 or 0
        if actionRem > rem then
            rem = actionRem
            dur = d2 or dur
        end

        info.cdRemain = rem
        info.cdDur = dur
        info.cdStr = FormatTime(rem)
    else
        -- No spell resolvable, still show action cooldown if exists
        local s2, d2, e2 = GetActionCooldown(actionSlot)
        local actionRem = (e2 == 1 and d2 and d2 > 0) and Remaining(s2, d2) or 0
        info.cdStr = FormatTime(actionRem)
    end

    return info
end

-- -----------------------------
-- UI
-- -----------------------------

local frame = CreateFrame("Frame", "OneKeyBindSpellViewerFrame", UIParent, "BackdropTemplate")
frame:SetSize(620, 260)
frame:SetPoint("CENTER")
frame:SetMovable(true)
frame:EnableMouse(true)
frame:RegisterForDrag("LeftButton")
frame:SetScript("OnDragStart", function(self) self:StartMoving() end)
frame:SetScript("OnDragStop", function(self) self:StopMovingOrSizing() end)

frame:SetBackdrop({
    bgFile = "Interface/Tooltips/UI-Tooltip-Background",
    edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
    tile = true, tileSize = 16, edgeSize = 16,
    insets = { left = 4, right = 4, top = 4, bottom = 4 }
})
frame:SetBackdropColor(0, 0, 0, 0.85)
frame:Hide()

local title = frame:CreateFontString(nil, "OVERLAY", "GameFontHighlightLarge")
title:SetPoint("TOPLEFT", 12, -10)
title:SetText("OKB Slot Monitor (Rich)")

local closeBtn = CreateFrame("Button", nil, frame, "UIPanelCloseButton")
closeBtn:SetPoint("TOPRIGHT", -6, -6)

-- 64x64 icon (exact slot icon)
local icon = frame:CreateTexture(nil, "ARTWORK")
icon:SetSize(64, 64)
icon:SetPoint("TOPLEFT", 12, -46)
icon:SetTexture(134400)

local iconBorder = CreateFrame("Frame", nil, frame, "BackdropTemplate")
iconBorder:SetPoint("TOPLEFT", icon, "TOPLEFT", -2, 2)
iconBorder:SetPoint("BOTTOMRIGHT", icon, "BOTTOMRIGHT", 2, -2)
iconBorder:SetBackdrop({
    edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
    edgeSize = 16
})
iconBorder:SetBackdropBorderColor(1, 1, 1, 0.8)

local function MakeLine(yOffset)
    local t = frame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    t:SetPoint("TOPLEFT", 90, yOffset)
    t:SetWidth(520)
    t:SetJustifyH("LEFT")
    t:SetText("-")
    return t
end

local l1 = MakeLine(-48)  -- slot/binding
local l2 = MakeLine(-72)  -- spell name/id
local l3 = MakeLine(-96)  -- action type
local l4 = MakeLine(-120) -- cd
local l5 = MakeLine(-144) -- gcd
local l6 = MakeLine(-168) -- charges
local l7 = MakeLine(-192) -- usable/range

local hint = frame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
hint:SetPoint("BOTTOMLEFT", 12, 10)
hint:SetWidth(590)
hint:SetJustifyH("LEFT")
hint:SetText("Cmd: /okb <bar> <slot>   Stop: /okb -1   (MultiActionBar: 101..104)")

-- -----------------------------
-- Monitoring state + live update
-- -----------------------------

local function IsMonitoring()
    return OneKeyBindSpellViewerDB.monitoring == true
end

local function StopMonitoring()
    OneKeyBindSpellViewerDB.monitoring = false
    OneKeyBindSpellViewerDB.barIndex = nil
    OneKeyBindSpellViewerDB.slotInBar = nil
    frame:Hide()
end

local function StartMonitoring(barIndex, slotInBar)
    OneKeyBindSpellViewerDB.monitoring = true
    OneKeyBindSpellViewerDB.barIndex = barIndex
    OneKeyBindSpellViewerDB.slotInBar = slotInBar
    frame:Show()
end

local function BarLabel(barIndex)
    if MULTI_BASE[barIndex] then
        return string.format("%d (MultiActionBar%d)", barIndex, barIndex - 100)
    end
    return tostring(barIndex)
end

local lastInfo = nil
local function RefreshUI()
    if not IsMonitoring() then return end

    local bar = tonumber(OneKeyBindSpellViewerDB.barIndex or 1) or 1
    local slotInBar = tonumber(OneKeyBindSpellViewerDB.slotInBar or 1) or 1
    slotInBar = Clamp(slotInBar, 1, 12)

    local actionSlot = GetActionSlot(bar, slotInBar)
    local info = ResolveActionSlot(actionSlot)

    lastInfo = info

    -- Icon
    icon:SetTexture(info.iconTex or 134400)

    -- Key formatting
    local keyStr = "-"
    if info.key1 ~= "" and info.key2 ~= "" then
        keyStr = info.key1 .. " , " .. info.key2
    elseif info.key1 ~= "" then
        keyStr = info.key1
    elseif info.key2 ~= "" then
        keyStr = info.key2
    end

    l1:SetText(string.format("Monitor: bar=%s slot=%d  -> actionSlot=%d | Key=%s | Bind=%s",
        BarLabel(bar), slotInBar, actionSlot, keyStr, (info.bindCmd ~= "" and info.bindCmd or "-")))

    if info.spellID then
        l2:SetText(string.format("Spell: %s  |  SpellID: %d", info.spellName or "-", info.spellID))
    else
        l2:SetText(string.format("Spell: %s", info.spellName or "-"))
    end

    l3:SetText(string.format("ActionType: %s  |  ActionID: %s",
        tostring(info.actionType), info.actionId and tostring(info.actionId) or "-"))

    l4:SetText(string.format("Cooldown: %s", info.cdStr or "-"))
    l5:SetText(string.format("GCD: %s", info.gcdStr or "-"))
    l6:SetText(string.format("Charges: %s", info.chargesStr or "-"))
    l7:SetText(string.format("Usable: %s  |  Range: %s", info.usableStr or "-", info.rangeStr or "-"))
end

-- Smooth countdown updates (every ~0.1s)
local accum = 0
frame:SetScript("OnUpdate", function(_, elapsed)
    if not frame:IsShown() or not IsMonitoring() then return end
    accum = accum + elapsed
    if accum < 0.10 then return end
    accum = 0

    -- Recompute only time-dependent parts by full refresh (cheap enough)
    RefreshUI()
end)

-- -----------------------------
-- Events
-- -----------------------------

local evt = CreateFrame("Frame")
evt:RegisterEvent("PLAYER_ENTERING_WORLD")
evt:RegisterEvent("UPDATE_BINDINGS")
evt:RegisterEvent("ACTIONBAR_SLOT_CHANGED")
evt:RegisterEvent("ACTIONBAR_UPDATE_STATE")
evt:RegisterEvent("ACTIONBAR_UPDATE_USABLE")
evt:RegisterEvent("SPELL_UPDATE_COOLDOWN")
evt:RegisterEvent("SPELLS_CHANGED")

evt:SetScript("OnEvent", function(_, event, ...)
    if not frame:IsShown() then return end
    RefreshUI()
end)

-- -----------------------------
-- Slash command
-- -----------------------------

SLASH_ONEKEYBINDSPELLVIEWER1 = "/okb"
SlashCmdList["ONEKEYBINDSPELLVIEWER"] = function(msg)
    msg = Trim(msg)

    if msg == "" then
        if IsMonitoring() then
            if frame:IsShown() then frame:Hide() else frame:Show() RefreshUI() end
        else
            print("|cFF00FF00[OKB]|r Usage:")
            print("  /okb <barIndex> <slotInBar>   e.g. /okb 3 2")
            print("  /okb -1                       stop monitoring")
            print("  MultiActionBars: /okb 101..104 <slot>")
        end
        return
    end

    if msg == "-1" then
        StopMonitoring()
        print("|cFF00FF00[OKB]|r Monitoring stopped.")
        return
    end

    local a, b = msg:match("^(%-?%d+)%s+(%-?%d+)$")
    if not a or not b then
        print("|cFFFF0000[OKB]|r Usage: /okb <barIndex> <slotInBar>  OR  /okb -1")
        return
    end

    local barIndex = tonumber(a)
    local slotInBar = tonumber(b)
    if not barIndex or not slotInBar then
        print("|cFFFF0000[OKB]|r barIndex/slot must be numbers.")
        return
    end

    slotInBar = Clamp(slotInBar, 1, 12)
    if (barIndex < 1 or barIndex > 10) and (not MULTI_BASE[barIndex]) then
        print("|cFFFF0000[OKB]|r barIndex must be 1..10, or 101..104 for MultiActionBars.")
        return
    end

    StartMonitoring(barIndex, slotInBar)
    RefreshUI()
    print(string.format("|cFF00FF00[OKB]|r Monitoring bar=%d slot=%d. Use /okb -1 to stop.", barIndex, slotInBar))
end
