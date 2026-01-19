--[[
English:
OneKeyBindSpellViewer (Rich monitor + UI inputs)

Slash:
  /okb  -> toggle start/stop monitoring

UI:
- User can edit barIndex + slotInBar and click Apply.
- Shows as much info as possible: icon 64x64, spellID, cooldown, GCD, charges,
  usability, range, action states, proc overlay, player states, casting/channeling states.
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
-- We support barIndex:
--   1..10     : linear mapping ( (bar-1)*12 + slot )
--   101..104  : MultiActionBar1..4 fixed ranges above
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

local function Remaining(startTime, duration)
    if not startTime or not duration or duration <= 0 then return 0 end
    local r = (startTime + duration) - Now()
    if r < 0 then r = 0 end
    return r
end

local function FormatTime(sec)
    if not sec or sec <= 0 then return "0.0s" end
    if sec >= 60 then
        return string.format("%.1fm", sec / 60.0)
    end
    return string.format("%.1fs", sec)
end

local function BarLabel(barIndex)
    if MULTI_BASE[barIndex] then
        return string.format("%d (MultiActionBar%d)", barIndex, barIndex - 100)
    end
    return tostring(barIndex)
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

local function GetGCD()
    -- English:
    -- Use spellID 61304 to get global cooldown reliably.
    local s, d, e = GetSpellCooldown(GCD_SPELL_ID)
    if e ~= 1 or not d or d <= 0 then
        return false, 0, 0
    end
    local rem = Remaining(s, d)
    if rem <= 0 then
        return false, 0, 0
    end
    return true, rem, d
end

local function GetCastingState(spellID)
    -- English:
    -- Return casting/channeling info and whether it matches the monitored spellID.
    local castName, _, _, castStartMS, castEndMS, _, _, _, castSpellID = UnitCastingInfo("player")
    if castName then
        local dur = (castEndMS - castStartMS) / 1000.0
        local rem = (castEndMS / 1000.0) - Now()
        if rem < 0 then rem = 0 end
        local match = (spellID and castSpellID and spellID == castSpellID) and true or false
        return "CAST", castName, castSpellID, rem, dur, match
    end

    local chName, _, _, chStartMS, chEndMS, _, _, _, chSpellID = UnitChannelInfo("player")
    if chName then
        local dur = (chEndMS - chStartMS) / 1000.0
        local rem = (chEndMS / 1000.0) - Now()
        if rem < 0 then rem = 0 end
        local match = (spellID and chSpellID and spellID == chSpellID) and true or false
        return "CHAN", chName, chSpellID, rem, dur, match
    end

    return "NONE", nil, nil, 0, 0, false
end

local function BoolStr(v) return v and "YES" or "NO" end

local function SafeIsSpellOverlayed(spellID)
    -- English:
    -- IsSpellOverlayed exists on retail; guard for nil in some clients.
    if not spellID then return false end
    if type(IsSpellOverlayed) == "function" then
        return IsSpellOverlayed(spellID) and true or false
    end
    return false
end

local function ResolveActionSlot(actionSlot)
    -- English:
    -- Returns a rich info table for UI display.
    local info = {
        actionSlot = actionSlot,

        -- binding
        key1 = "", key2 = "", bindCmd = "",

        -- action
        actionType = "none",
        actionId = nil,
        actionTexture = nil,

        -- resolved spell (best effort)
        spellID = nil,
        spellName = "-",
        spellIcon = nil,

        -- cooldown & gcd
        cdRemain = 0, cdDur = 0,
        gcdActive = false, gcdRemain = 0, gcdDur = 0,

        -- charges
        chargesCur = nil, chargesMax = nil,
        chargesRegenRemain = 0, chargesRegenDur = 0,

        -- usability/range
        usableSpell = nil, noMana = nil,
        usableAction = nil,
        actionInRange = nil,
        spellInRange = nil,

        -- action states
        isCurrent = false,
        isAutoRepeat = false,
        isAttackAction = false,

        -- proc overlay
        procOverlay = false,

        -- player states
        inCombat = false,
        isDead = false,
        isGhost = false,
        isMounted = false,
        speed = 0,

        -- casting/channeling
        castMode = "NONE",
        castName = nil,
        castSpellID = nil,
        castRemain = 0,
        castDur = 0,
        castMatch = false,
    }

    -- binding
    info.key1, info.key2, info.bindCmd = FindKeysForActionSlot(actionSlot)

    -- player states
    info.inCombat = UnitAffectingCombat("player") and true or false
    info.isDead = UnitIsDeadOrGhost("player") and true or false
    info.isGhost = UnitIsGhost("player") and true or false
    info.isMounted = IsMounted() and true or false
    info.speed = GetUnitSpeed("player") or 0

    -- action icon (exactly like the monitored slot)
    info.actionTexture = GetActionTexture(actionSlot)

    -- action type/id
    local aType, aId = GetActionInfo(actionSlot)
    info.actionType = aType or "none"
    info.actionId = aId

    -- action state flags
    info.isCurrent = IsCurrentAction(actionSlot) and true or false
    info.isAutoRepeat = IsAutoRepeatAction(actionSlot) and true or false
    info.isAttackAction = IsAttackAction(actionSlot) and true or false

    -- range (action)
    local ar = IsActionInRange(actionSlot)
    info.actionInRange = ar -- 1/0/nil

    -- usability (action)
    local ua = IsUsableAction(actionSlot)
    info.usableAction = ua and true or false

    -- resolve to spell if possible
    local resolvedSpellID = nil
    if aType == "spell" and aId then
        resolvedSpellID = aId
    elseif aType == "macro" and aId then
        local macroSpellID = GetMacroSpell(aId)
        if macroSpellID then
            resolvedSpellID = macroSpellID
        end
    end

    if resolvedSpellID then
        info.spellID = resolvedSpellID
        local name, _, icon = GetSpellInfo(resolvedSpellID)
        info.spellName = name or ("Spell " .. tostring(resolvedSpellID))
        info.spellIcon = icon

        -- spell usability/range
        local usable, noMana = IsUsableSpell(resolvedSpellID)
        info.usableSpell, info.noMana = usable, noMana

        local sr = IsSpellInRange(resolvedSpellID, "target")
        info.spellInRange = sr -- 1/0/nil

        -- proc overlay
        info.procOverlay = SafeIsSpellOverlayed(resolvedSpellID)

        -- charges
        local cur, max, chStart, chDur = GetSpellCharges(resolvedSpellID)
        if max and max > 0 then
            info.chargesCur, info.chargesMax = cur, max
            info.chargesRegenDur = chDur or 0
            info.chargesRegenRemain = Remaining(chStart, chDur)
        end
    else
        -- fallback: keep actionType-based name
        if aType == "item" and aId then
            local itemName = GetItemInfo(aId)
            info.spellName = itemName or ("ItemID: " .. tostring(aId))
        elseif aType == "macro" then
            info.spellName = "Macro"
        elseif aType == "none" then
            info.spellName = "-"
        else
            info.spellName = tostring(aType)
        end
    end

    -- cooldown: combine spell cooldown + action cooldown (take max remaining)
    local bestRem, bestDur = 0, 0
    if info.spellID then
        local s1, d1, e1 = GetSpellCooldown(info.spellID)
        if e1 == 1 and d1 and d1 > 0 then
            local r1 = Remaining(s1, d1)
            if r1 > bestRem then
                bestRem, bestDur = r1, d1
            end
        end
    end

    do
        local s2, d2, e2 = GetActionCooldown(actionSlot)
        if e2 == 1 and d2 and d2 > 0 then
            local r2 = Remaining(s2, d2)
            if r2 > bestRem then
                bestRem, bestDur = r2, d2
            end
        end
    end

    info.cdRemain, info.cdDur = bestRem, bestDur

    -- gcd
    local gcdActive, gcdRemain, gcdDur = GetGCD()
    info.gcdActive = gcdActive
    info.gcdRemain, info.gcdDur = gcdRemain, gcdDur

    -- casting/channeling
    info.castMode, info.castName, info.castSpellID, info.castRemain, info.castDur, info.castMatch =
        GetCastingState(info.spellID)

    return info
end

-- -----------------------------
-- UI
-- -----------------------------

local frame = CreateFrame("Frame", "OneKeyBindSpellViewerFrame", UIParent, "BackdropTemplate")
frame:SetSize(740, 360)
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
title:SetText("OKB Slot Monitor (Rich + Inputs)")

local closeBtn = CreateFrame("Button", nil, frame, "UIPanelCloseButton")
closeBtn:SetPoint("TOPRIGHT", -6, -6)

-- Icon 64x64 (exact action texture)
local icon = frame:CreateTexture(nil, "ARTWORK")
icon:SetSize(64, 64)
icon:SetPoint("TOPLEFT", 12, -42)
icon:SetTexture(134400)

local iconBorder = CreateFrame("Frame", nil, frame, "BackdropTemplate")
iconBorder:SetPoint("TOPLEFT", icon, "TOPLEFT", -2, 2)
iconBorder:SetPoint("BOTTOMRIGHT", icon, "BOTTOMRIGHT", 2, -2)
iconBorder:SetBackdrop({ edgeFile = "Interface/Tooltips/UI-Tooltip-Border", edgeSize = 16 })
iconBorder:SetBackdropBorderColor(1, 1, 1, 0.8)

-- Input area
local inputLabel = frame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
inputLabel:SetPoint("TOPLEFT", 90, -44)
inputLabel:SetText("Monitor Target (edit then Apply):")

local function MakeEditBox(w, h)
    local eb = CreateFrame("EditBox", nil, frame, "InputBoxTemplate")
    eb:SetSize(w, h)
    eb:SetAutoFocus(false)
    eb:SetNumeric(false)
    eb:SetMaxLetters(8)
    eb:SetTextInsets(6, 6, 0, 0)
    eb:SetScript("OnEscapePressed", function(self) self:ClearFocus() end)
    eb:SetScript("OnEnterPressed", function(self) self:ClearFocus() end)
    return eb
end

local barText = frame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
barText:SetPoint("TOPLEFT", inputLabel, "BOTTOMLEFT", 0, -10)
barText:SetText("barIndex (1..10 or 101..104):")

local barEdit = MakeEditBox(80, 22)
barEdit:SetPoint("LEFT", barText, "RIGHT", 8, 0)

local slotText = frame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
slotText:SetPoint("LEFT", barEdit, "RIGHT", 16, 0)
slotText:SetText("slot (1..12):")

local slotEdit = MakeEditBox(50, 22)
slotEdit:SetPoint("LEFT", slotText, "RIGHT", 8, 0)

local applyBtn = CreateFrame("Button", nil, frame, "UIPanelButtonTemplate")
applyBtn:SetSize(70, 22)
applyBtn:SetPoint("LEFT", slotEdit, "RIGHT", 16, 0)
applyBtn:SetText("Apply")

local helpText = frame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
helpText:SetPoint("TOPLEFT", barText, "BOTTOMLEFT", 0, -8)
helpText:SetWidth(620)
helpText:SetJustifyH("LEFT")
helpText:SetText("Tips: 101..104 = MultiActionBar1..4 (e.g. 103=right bar 1). /okb toggles start/stop.")

-- Status bars for CD and GCD
local function MakeStatusBar(y, label)
    local t = frame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
    t:SetPoint("TOPLEFT", 12, y)
    t:SetText(label)

    local sb = CreateFrame("StatusBar", nil, frame, "BackdropTemplate")
    sb:SetSize(700, 14)
    sb:SetPoint("TOPLEFT", t, "BOTTOMLEFT", 0, -4)
    sb:SetMinMaxValues(0, 1)
    sb:SetValue(0)
    sb:SetStatusBarTexture("Interface\\TARGETINGFRAME\\UI-StatusBar")

    sb:SetBackdrop({
        bgFile = "Interface/Tooltips/UI-Tooltip-Background",
        insets = { left = 1, right = 1, top = 1, bottom = 1 }
    })
    sb:SetBackdropColor(0, 0, 0, 0.6)

    local txt = sb:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
    txt:SetPoint("CENTER")
    txt:SetText("-")

    sb.text = txt
    return sb
end

local cdBar = MakeStatusBar(-120, "Cooldown (spell/action max):")
local gcdBar = MakeStatusBar(-160, "GCD:")

local function MakeLine(y)
    local t = frame:CreateFontString(nil, "OVERLAY", "GameFontNormal")
    t:SetPoint("TOPLEFT", 12, y)
    t:SetWidth(720)
    t:SetJustifyH("LEFT")
    t:SetText("-")
    return t
end

local l1 = MakeLine(-205) -- slot/binding/action
local l2 = MakeLine(-228) -- spell/id/type
local l3 = MakeLine(-251) -- usability/range/proc/action state
local l4 = MakeLine(-274) -- player state
local l5 = MakeLine(-297) -- casting/channeling

-- -----------------------------
-- Monitoring state
-- -----------------------------

local function IsMonitoring()
    return OneKeyBindSpellViewerDB.monitoring == true
end

local function SetDefaultsIfNeeded()
    if OneKeyBindSpellViewerDB.barIndex == nil then OneKeyBindSpellViewerDB.barIndex = 1 end
    if OneKeyBindSpellViewerDB.slotInBar == nil then OneKeyBindSpellViewerDB.slotInBar = 1 end
end

local function ApplyInputs()
    -- English:
    -- Validate and apply bar/slot from UI to DB.
    local barIndex = tonumber(barEdit:GetText() or "")
    local slotInBar = tonumber(slotEdit:GetText() or "")
    if not barIndex then return false, "barIndex must be a number" end
    if not slotInBar then return false, "slot must be a number" end

    slotInBar = Clamp(slotInBar, 1, 12)
    if (barIndex < 1 or barIndex > 10) and (not MULTI_BASE[barIndex]) then
        return false, "barIndex must be 1..10 or 101..104"
    end

    OneKeyBindSpellViewerDB.barIndex = barIndex
    OneKeyBindSpellViewerDB.slotInBar = slotInBar
    return true
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

    -- Icon texture: exact action slot texture, fallback to spell icon, then question mark
    local tex = info.actionTexture or info.spellIcon or 134400
    icon:SetTexture(tex)

    -- Cooldown bar
    if info.cdDur and info.cdDur > 0 then
        cdBar:SetMinMaxValues(0, info.cdDur)
        cdBar:SetValue(info.cdDur - info.cdRemain)
        cdBar.text:SetText(string.format("%s / %s  (remain %s)",
            FormatTime(info.cdDur - info.cdRemain), FormatTime(info.cdDur), FormatTime(info.cdRemain)))
    else
        cdBar:SetMinMaxValues(0, 1)
        cdBar:SetValue(0)
        cdBar.text:SetText("No cooldown")
    end

    -- GCD bar
    if info.gcdActive and info.gcdDur and info.gcdDur > 0 then
        gcdBar:SetMinMaxValues(0, info.gcdDur)
        gcdBar:SetValue(info.gcdDur - info.gcdRemain)
        gcdBar.text:SetText(string.format("%s / %s  (remain %s)",
            FormatTime(info.gcdDur - info.gcdRemain), FormatTime(info.gcdDur), FormatTime(info.gcdRemain)))
    else
        gcdBar:SetMinMaxValues(0, 1)
        gcdBar:SetValue(0)
        gcdBar.text:SetText("GCD OFF")
    end

    -- key string
    local keyStr = "-"
    if info.key1 ~= "" and info.key2 ~= "" then
        keyStr = info.key1 .. " , " .. info.key2
    elseif info.key1 ~= "" then
        keyStr = info.key1
    elseif info.key2 ~= "" then
        keyStr = info.key2
    end

    l1:SetText(string.format(
        "Monitor: bar=%s slot=%d -> actionSlot=%d | Bind=%s | Key=%s",
        BarLabel(bar), slotInBar, actionSlot, (info.bindCmd ~= "" and info.bindCmd or "-"), keyStr
    ))

    local sid = info.spellID and tostring(info.spellID) or "-"
    l2:SetText(string.format(
        "Action: type=%s id=%s | Spell: %s | SpellID=%s",
        tostring(info.actionType), info.actionId and tostring(info.actionId) or "-",
        tostring(info.spellName), sid
    ))

    -- usability/range/proc/action state
    local usableSpellStr = "-"
    if info.usableSpell ~= nil then
        if info.usableSpell then usableSpellStr = "OK" else usableSpellStr = (info.noMana and "NoMana" or "No") end
    end

    local usableActionStr = info.usableAction and "OK" or "No"

    local actionRangeStr = "N/A"
    if info.actionInRange == 1 then actionRangeStr = "InRange"
    elseif info.actionInRange == 0 then actionRangeStr = "OutRange"
    end

    local spellRangeStr = "N/A"
    if info.spellInRange == 1 then spellRangeStr = "InRange"
    elseif info.spellInRange == 0 then spellRangeStr = "OutRange"
    end

    local procStr = info.procOverlay and "YES" or "NO"

    local stateStr = string.format("Current=%s AutoRepeat=%s AttackAction=%s",
        BoolStr(info.isCurrent), BoolStr(info.isAutoRepeat), BoolStr(info.isAttackAction))

    local chargesStr = "-"
    if info.chargesMax and info.chargesMax > 0 then
        if info.chargesRegenRemain and info.chargesRegenRemain > 0 then
            chargesStr = string.format("%d/%d (regen %s)", info.chargesCur or 0, info.chargesMax, FormatTime(info.chargesRegenRemain))
        else
            chargesStr = string.format("%d/%d", info.chargesCur or 0, info.chargesMax)
        end
    end

    l3:SetText(string.format(
        "Usable: spell=%s action=%s | Range: action=%s spell=%s | ProcOverlay=%s | Charges=%s | %s",
        usableSpellStr, usableActionStr, actionRangeStr, spellRangeStr, procStr, chargesStr, stateStr
    ))

    -- player state
    l4:SetText(string.format(
        "Player: Combat=%s Dead=%s Ghost=%s Mounted=%s Speed=%.1f",
        BoolStr(info.inCombat), BoolStr(info.isDead), BoolStr(info.isGhost), BoolStr(info.isMounted), info.speed
    ))

    -- casting/channeling state
    if info.castMode == "NONE" then
        l5:SetText("Casting: NONE")
    else
        local matchStr = info.castMatch and "THIS SPELL" or "OTHER"
        l5:SetText(string.format(
            "Casting: %s | %s (ID:%s) | remain %s | %s",
            info.castMode,
            tostring(info.castName),
            info.castSpellID and tostring(info.castSpellID) or "-",
            FormatTime(info.castRemain),
            matchStr
        ))
    end
end

-- -----------------------------
-- UI wiring
-- -----------------------------

applyBtn:SetScript("OnClick", function()
    local ok, err = ApplyInputs()
    if not ok then
        print("|cFFFF0000[OKB]|r Apply failed: " .. tostring(err))
        return
    end
    RefreshUI()
end)

-- -----------------------------
-- Live update
-- -----------------------------

local accum = 0
frame:SetScript("OnUpdate", function(_, elapsed)
    if not frame:IsShown() or not IsMonitoring() then return end
    accum = accum + elapsed
    if accum < 0.10 then return end
    accum = 0
    RefreshUI()
end)

local evt = CreateFrame("Frame")
evt:RegisterEvent("PLAYER_ENTERING_WORLD")
evt:RegisterEvent("UPDATE_BINDINGS")
evt:RegisterEvent("ACTIONBAR_SLOT_CHANGED")
evt:RegisterEvent("ACTIONBAR_UPDATE_STATE")
evt:RegisterEvent("ACTIONBAR_UPDATE_USABLE")
evt:RegisterEvent("SPELL_UPDATE_COOLDOWN")
evt:RegisterEvent("SPELLS_CHANGED")

evt:SetScript("OnEvent", function(_, event, ...)
    if not frame:IsShown() or not IsMonitoring() then return end
    RefreshUI()
end)

-- -----------------------------
-- Slash: /okb (toggle start/stop)
-- -----------------------------

SLASH_ONEKEYBINDSPELLVIEWER1 = "/okb"
SlashCmdList["ONEKEYBINDSPELLVIEWER"] = function(msg)
    msg = Trim(msg)

    -- Only used for start/stop (toggle)
    if IsMonitoring() then
        OneKeyBindSpellViewerDB.monitoring = false
        frame:Hide()
        print("|cFF00FF00[OKB]|r Monitoring stopped.")
        return
    end

    -- Start monitoring
    SetDefaultsIfNeeded()
    OneKeyBindSpellViewerDB.monitoring = true
    frame:Show()

    -- Fill inputs from DB
    barEdit:SetText(tostring(OneKeyBindSpellViewerDB.barIndex or 1))
    slotEdit:SetText(tostring(OneKeyBindSpellViewerDB.slotInBar or 1))

    RefreshUI()
    print("|cFF00FF00[OKB]|r Monitoring started. Change bar/slot in UI and click Apply. Use /okb to stop.")
end
