<#
.SYNOPSIS
  Deploy local.settings.json values to Azure Function App Application Settings.

.DESCRIPTION
  - Reads local.settings.json from current folder.
  - Optionally fetches AzureWebJobsStorage connection string from an existing Storage Account.
  - Calls `az functionapp config appsettings set` to set all keys.
  - Restarts the Function App at the end.

USAGE:
  # Run directly (will use the defaults below)
  .\deploy_appsettings_adapted.ps1

  # To fetch storage connection string automatically, edit $StorageAccount below or pass via param (see code comments).

#>

# --- adapted defaults (your environment)
$ResourceGroup = "RG-SEE-D-FUNCTION"
$FunctionApp   = "see-d-crm-orquestradorbot"

# Optional: if you want the script to retrieve AzureWebJobsStorage from a Storage Account,
# set $StorageAccount to the storage account name (must be in the same resource group):
# $StorageAccount = "myuniquestorageacct"
# Leave $StorageAccount = $null if you want to use AzureWebJobsStorage value present in local.settings.json (if any).
$StorageAccount = $null

function AbortIfError($msg) {
    Write-Error $msg
    exit 1
}

# --- pre-checks
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    AbortIfError "Azure CLI 'az' not found. Install Azure CLI and run 'az login' before running this script."
}

# check az login
$account = az account show --query "id" -o tsv 2>$null
if (-not $account) {
    Write-Host "You are not logged in to Azure CLI. Running 'az login' now..."
    az login | Out-Null
    $account = az account show --query "id" -o tsv 2>$null
    if (-not $account) { AbortIfError "Azure login required. Please run 'az login' and try again." }
}

# check file
$localSettingsPath = Join-Path (Get-Location) "local.settings.json"
if (-not (Test-Path $localSettingsPath)) {
    AbortIfError "local.settings.json not found in current folder: $localSettingsPath"
}

Write-Host "Reading local.settings.json from $localSettingsPath ..."
$jsonRaw = Get-Content $localSettingsPath -Raw
try {
    $local = ConvertFrom-Json $jsonRaw
} catch {
    AbortIfError "Failed to parse local.settings.json: $_"
}

if (-not $local.Values) {
    AbortIfError "local.settings.json does not contain a 'Values' object. Structure must be { \"IsEncrypted\": false, \"Values\": { ... } }"
}

# Build settings list
$settingsList = @()
foreach ($p in $local.Values.PSObject.Properties) {
    # convert boolean/null to string representation to avoid issues
    $val = $p.Value
    if ($null -eq $val) {
        $strVal = ""
    } elseif ($val -is [System.Boolean]) {
        $strVal = $val.ToString().ToLower()
    } else {
        $strVal = $val.ToString()
    }

    # If value is an object/array, compress to single-line JSON
    if ($val -is [System.Object] -and ($val.PSObject.Properties.Count -gt 0)) {
        try { $strVal = ($val | ConvertTo-Json -Compress) } catch { $strVal = $val.ToString() }
    }

    $pair = "$($p.Name)=$strVal"
    $settingsList += $pair
}

# If user requested storage account, fetch connection string
if ($StorageAccount) {
    Write-Host "Fetching connection string for storage account '$StorageAccount' in resource group '$ResourceGroup'..."
    try {
        $connStr = az storage account show-connection-string --name $StorageAccount --resource-group $ResourceGroup -o tsv 2>$null
    } catch {
        $connStr = $null
    }
    if (-not $connStr) {
        AbortIfError "Could not retrieve connection string for storage account '$StorageAccount'. Make sure the storage account exists and you have permissions."
    }

    # Place or overwrite AzureWebJobsStorage in settings list
    $settingsList = $settingsList | Where-Object { -not ($_ -like "AzureWebJobsStorage=*") }
    $settingsList += "AzureWebJobsStorage=$connStr"
    Write-Host "AzureWebJobsStorage will be set from storage account: $StorageAccount"
} else {
    # Ensure AzureWebJobsStorage exists in local.settings.json values; if present, it will be used (copied)
    $hasStorage = $local.Values.PSObject.Properties | Where-Object { $_.Name -eq "AzureWebJobsStorage" }
    if ($hasStorage) {
        Write-Host "AzureWebJobsStorage found in local.settings.json and will be applied."
    } else {
        Write-Host "Note: No StorageAccount parameter and AzureWebJobsStorage not found in local.settings.json."
        Write-Host "If your Function App requires AzureWebJobsStorage (most do), set it in Portal or rerun with StorageAccount set."
    }
}

# Preview to user
Write-Host ""
Write-Host "The following settings will be applied to Function App '$FunctionApp' in resource group '$ResourceGroup':"
$settingsList | ForEach-Object { Write-Host "  $_" }

Write-Host ""
$confirm = Read-Host "Apply these settings? Type 'YES' to continue"
if ($confirm -ne "YES") {
    AbortIfError "User aborted."
}

# Apply settings
Write-Host "Applying settings via az..."
try {
    $azArgs = @("functionapp","config","appsettings","set","--resource-group",$ResourceGroup,"--name",$FunctionApp,"--settings")
    $azArgs += $settingsList
    $null = az @azArgs
    if ($LASTEXITCODE -ne 0) {
        AbortIfError "az CLI failed to set appsettings (exit code $LASTEXITCODE)."
    }
} catch {
    AbortIfError "Error invoking az CLI: $_"
}

Write-Host "Settings applied. Restarting Function App..."
az functionapp restart --name $FunctionApp --resource-group $ResourceGroup | Out-Null

Write-Host "Done. Recommended next steps:"
Write-Host " - Open Portal > Function App > Configuration to verify values."
Write-Host " - Open Portal > Function App > Functions > select function and click 'Get function URL' to test with Postman."
Write-Host " - Use Log stream to see runtime logs: Portal > Function App > Log stream."

exit 0
