local ADDON_NAME = ...
KSE_Saved = KSE_Saved or {}

-- =========================================================
-- Compatibility helpers (Retail 11.x/12.x)
-- MySlot uses similar wrappers for GetSpellInfo etc.【filecite】turn0file4
-- =========================================================
local GetSpellNameCompat = (C_Spell and C_Spell.GetSpellName) or _G.GetSpellInfo
local GetSpellLinkCompat = (C_Spell and C_Spell.GetSpellLink) or _G.GetSpellLink

-- =========================================================
-- Minimal JSON encoder (only what we need)
-- =========================================================
local function JsonEscape_(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
         :gsub("\"", "\\\"")
         :gsub("\b", "\\b")
         :gsub("\f", "\\f")
         :gsub("\n", "\\n")
         :gsub("\r", "\\r")
         :gsub("\t", "\\t")
    return s
end

local function JsonEncodeValue_(v)
    local t = type(v)
    if t == "nil" then
        return "null"
    elseif t == "number" then
        return tostring(v)
    elseif t == "boolean" then
        return v and "true" or "false"
    elseif t == "string" then
        return "\"" .. JsonEscape_(v) .. "\""
    elseif t == "table" then
        -- Decide array vs object
        local isArray = true
        local n = 0
        for k, _ in pairs(v) do
            if type(k) ~= "number" then
                isArray = false
                break
            end
            n = math.max(n, k)
        end

        if isArray then
            local out = {}
            for i = 1, n do
                out[#out + 1] = JsonEncodeValue_(v[i])
            end
            return "[" .. table.concat(out, ",") .. "]"
        else
            local out = {}
            for k, vv in pairs(v) do
                out[#out + 1] = "\"" .. JsonEscape_(k) .. "\":" .. JsonEncodeValue_(vv)
            end
            table.sort(out)
            return "{" .. table.concat(out, ",") .. "}"
        end
    end
    return "\"\""
end

-- =========================================================
-- UI (similar idea to MySlot export editbox frame)【filecite】turn0file1
-- =========================================================
local function CreateMainFrame_()
    local f = CreateFrame("Frame", "KSE_MainFrame", UIParent, BackdropTemplateMixin and "BackdropTemplate" or nil)
    f:SetSize(760, 260)
    f:SetPoint("CENTER", 0, 0)
    f:SetFrameStrata("DIALOG")
    f:SetToplevel(true)
    f:EnableMouse(true)
    f:SetMovable(true)
    f:RegisterForDrag("LeftButton")
    f:SetScript("OnDragStart", f.StartMoving)
    f:SetScript("OnDragStop", f.StopMovingOrSizing)
    f:SetBackdrop({
        bgFile = "Interface\\DialogFrame\\UI-DialogBox-Background",
        edgeFile = "Interface\\DialogFrame\\UI-DialogBox-Border",
        tile = true,
        tileSize = 32,
        edgeSize = 32,
        insets = { left = 8, right = 8, top = 10, bottom = 10 },
    })
    f:SetBackdropColor(0, 0, 0, 1)
    f:Hide()

    local title = f:CreateFontString(nil, "ARTWORK", "GameFontNormalLarge")
    title:SetPoint("TOPLEFT", 18, -12)
    title:SetText("SpellID + Keybind JSON  (Ctrl+C)")

    local close = CreateFrame("Button", nil, f, "UIPanelCloseButton")
    close:SetPoint("TOPRIGHT", -6, -6)

    local exportBtn = CreateFrame("Button", nil, f, "GameMenuButtonTemplate")
    exportBtn:SetSize(160, 26)
    exportBtn:SetPoint("TOPLEFT", 18, -40)
    exportBtn:SetText("Export Keybind JSON")

    -- Outer box
    local box = CreateFrame("Frame", nil, f, BackdropTemplateMixin and "BackdropTemplate" or nil)
    box:SetPoint("TOPLEFT", 18, -72)
    box:SetPoint("BOTTOMRIGHT", -18, 18)
    box:SetBackdrop({
        bgFile = "Interface/Tooltips/UI-Tooltip-Background",
        edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
        tile = true,
        tileEdge = true,
        tileSize = 16,
        edgeSize = 16,
        insets = { left = -2, right = -2, top = -2, bottom = -2 },
    })
    box:SetBackdropColor(0, 0, 0, 0.2)

    local scroll = CreateFrame("ScrollFrame", nil, box, "UIPanelScrollFrameTemplate")
    scroll:SetPoint("TOPLEFT", 8, -8)
    scroll:SetPoint("BOTTOMRIGHT", -28, 8)

    local edit = CreateFrame("EditBox", nil, scroll)
    edit:SetMultiLine(true)
    edit:SetAutoFocus(false)
    edit:EnableMouse(true)
    edit:SetFontObject(GameTooltipText)
    edit:SetWidth(680)
    edit:SetMaxLetters(99999999)

    edit:SetScript("OnEscapePressed", function() edit:ClearFocus() end)
    edit:SetScript("OnMouseUp", function()
        edit:HighlightText(0, -1)
    end)

    scroll:SetScrollChild(edit)

    f.ExportButton = exportBtn
    f.EditBox = edit
    return f
end

local UI_ = CreateMainFrame_()

-- =========================================================
-- Keybind mapping
-- We map button global name -> binding command (ACTIONBUTTON / MULTIACTIONBARxBUTTONy)
-- =========================================================
local function GetCommandForButtonName_(btnName)
    local idx = btnName:match("^ActionButton(%d+)$")
    if idx then return "ACTIONBUTTON" .. idx end

    idx = btnName:match("^MultiBarBottomLeftButton(%d+)$")
    if idx then return "MULTIACTIONBAR1BUTTON" .. idx end

    idx = btnName:match("^MultiBarBottomRightButton(%d+)$")
    if idx then return "MULTIACTIONBAR2BUTTON" .. idx end

    idx = btnName:match("^MultiBarRightButton(%d+)$")
    if idx then return "MULTIACTIONBAR3BUTTON" .. idx end

    idx = btnName:match("^MultiBarLeftButton(%d+)$")
    if idx then return "MULTIACTIONBAR4BUTTON" .. idx end

    -- Retail may have MultiBar5/6/7 (TWW/DF changes). Try best-effort.
    idx = btnName:match("^MultiBar5Button(%d+)$")
    if idx then return "MULTIACTIONBAR5BUTTON" .. idx end

    idx = btnName:match("^MultiBar6Button(%d+)$")
    if idx then return "MULTIACTIONBAR6BUTTON" .. idx end

    idx = btnName:match("^MultiBar7Button(%d+)$")
    if idx then return "MULTIACTIONBAR7BUTTON" .. idx end

    return nil
end

local function GetBindingStringForCommand_(command)
    if not command then return nil end
    local k1, k2 = GetBindingKey(command)
    if k1 and k2 then
        return k1 .. "," .. k2
    end
    return k1 or k2
end

-- =========================================================
-- Read spellId from an action slot (button.action)
-- Similar to MySlot:GetActionInfo uses GetActionInfo(slotId)【filecite】turn0file4
-- =========================================================
local function GetSpellIdFromActionSlot_(slotId)
    if not slotId then return nil end
    local actionType, id, subType = GetActionInfo(slotId)
    if actionType == "spell" and id then
        return id
    end

    -- Optional: macro -> try resolve spell
    if actionType == "macro" and id then
        local spellId = GetMacroSpell(id)
        if spellId then return spellId end
    end

    -- Flyout/item/etc are ignored by default
    return nil
end

-- =========================================================
-- Collect buttons (only those existing in current UI)
-- =========================================================
local function EnumerateActionButtons_()
    local list = {}

    local function AddByPrefix(prefix)
        for i = 1, 12 do
            local name = prefix .. i
            local btn = _G[name]
            if btn then
                list[#list + 1] = btn
            end
        end
    end

    AddByPrefix("ActionButton")
    AddByPrefix("MultiBarBottomLeftButton")
    AddByPrefix("MultiBarBottomRightButton")
    AddByPrefix("MultiBarRightButton")
    AddByPrefix("MultiBarLeftButton")
    AddByPrefix("MultiBar5Button")
    AddByPrefix("MultiBar6Button")
    AddByPrefix("MultiBar7Button")

    return list
end

-- =========================================================
-- Export
-- Output JSON: { "<spellId>": {slotId=xx, key="SHIFT-2", name="xxx"} , ... }
-- If duplicated spellId appears on different buttons, we keep an array.
-- =========================================================
local function ExportSpellKeybindJson_()
    local buttons = EnumerateActionButtons_()
    local out = {}

    for _, btn in ipairs(buttons) do
        local btnName = btn:GetName()
        local slotId = btn.action or (btn.GetAttribute and btn:GetAttribute("action")) -- best effort

        local spellId = GetSpellIdFromActionSlot_(slotId)
        if spellId then
            local command = GetCommandForButtonName_(btnName or "")
            local keybind = GetBindingStringForCommand_(command)

            -- Include even if keybind is nil? You can choose. Here: only export those with keybind.
            if keybind and keybind ~= "" then
                local spellName = GetSpellNameCompat and GetSpellNameCompat(spellId) or nil
                local rec = {
                    slotId = slotId,
                    key = keybind,
                    name = spellName or "",
                    button = btnName or "",
                }

                local k = tostring(spellId)
                if not out[k] then
                    out[k] = rec
                else
                    -- If duplicated, convert to array
                    if out[k].slotId ~= nil then
                        out[k] = { out[k], rec }
                    else
                        table.insert(out[k], rec)
                    end
                end
            end
        end
    end

    return JsonEncodeValue_(out)
end

-- =========================================================
-- Hook button
-- =========================================================
UI_.ExportButton:SetScript("OnClick", function()
    local s = ExportSpellKeybindJson_()
    UI_.EditBox:SetText(s)
    UI_.EditBox:HighlightText(0, -1)
    UI_.EditBox:SetFocus()
end)

-- Slash command
SLASH_KSE1 = "/kse"
SlashCmdList["KSE"] = function()
    UI_:SetShown(not UI_:IsShown())
end

-- Quick access: show UI on addon loaded if you want (disabled by default)
local ev = CreateFrame("Frame")
ev:RegisterEvent("ADDON_LOADED")
ev:SetScript("OnEvent", function(_, _, name)
    if name ~= ADDON_NAME then return end
    -- UI_:Show()
end)
