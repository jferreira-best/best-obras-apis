<#
deploy_appsettings.ps1
- Aplica app settings (comparo seguro do local.settings.json)
- Opcional: zip-deploy do código (Kudu)
- Fallback: upload para Storage + WEBSITE_RUN_FROM_PACKAGE
- Reinicia a Function App
Usage:
  .\deploy_appsettings.ps1            # só settings
  .\deploy_appsettings.ps1 -ZipDeploy # settings + deploy
#>

param(
  [string]$ResourceGroup = "RG-SEE-D-FUNCTION",
  [string]$FunctionApp = "see-d-crm-orquestradorbot",
  [switch]$ZipDeploy = $false,
  [string]$PackageName = "package.zip"
)

function FailIf($msg) { Write-Error $msg; exit 1 }

# pré-requisito
if (-not (Get-Command az -ErrorAction SilentlyContinue)) { FailIf "Azure CLI não encontrado. Instale 'az' e faça 'az login'." }

# checa login/subscription
$accountJson = az account show --only-show-errors 2>&1
if ($LASTEXITCODE -ne 0) { FailIf "Não foi possível obter subscription. Rode 'az login'." }
try { $account = $accountJson | ConvertFrom-Json } catch { FailIf "Erro ao ler conta: $accountJson" }
Write-Host "Subscription: $($account.name) ($($account.id))"

# -------- aplicar app settings com lógica segura (se houver local.settings.json) --------
$localPath = Join-Path (Get-Location) "local.settings.json"
$settingsToSet = @()

if (Test-Path $localPath) {
  Write-Host "local.settings.json encontrado - preparando chaves válidas..."
  try {
    $localJson = Get-Content $localPath -Raw | ConvertFrom-Json
  } catch {
    Write-Warning "Erro ao ler local.settings.json: $_. Pulando aplicação de settings."
    $localJson = $null
  }

  if ($localJson -ne $null -and $localJson.Values) {
    # construir lista local válida (ignorar chaves que comecem com '#' e valores vazios)
    $localPairs = @()
    foreach ($p in $localJson.Values.PSObject.Properties) {
      $k = $p.Name
      $v = $p.Value
      if ($k -like '#*') { continue }
      if ($null -eq $v -or $v -eq '') { continue }
      $localPairs += [PSCustomObject]@{ Name = $k; Value = [string]$v }
    }

    if ($localPairs.Count -gt 0) {
      # pegar app settings atuais do Azure
      $azureOut = az functionapp config appsettings list --name $FunctionApp --resource-group $ResourceGroup 2>&1
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "Falha ao listar app settings do Azure:`n$azureOut"
      } else {
        try { $azureList = $azureOut | ConvertFrom-Json } catch { Write-Warning "Resposta inválida ao listar app settings."; $azureList = @() }
        # comparar e montar array de KEY=VALUE para set
        $toSet = @()
        foreach ($item in $localPairs) {
          $existing = $azureList | Where-Object { $_.name -eq $item.Name }
          if (-not $existing) {
            $toSet += "$($item.Name)=$($item.Value)"
          } elseif ($existing.value -ne $item.Value) {
            $toSet += "$($item.Name)=$($item.Value)"
          }
        }

        # garantir flags úteis
        if ($toSet -notcontains "SCM_DO_BUILD_DURING_DEPLOYMENT=true") { $toSet += "SCM_DO_BUILD_DURING_DEPLOYMENT=true" }
        if ($toSet -notcontains "FUNCTIONS_WORKER_RUNTIME=python") { $toSet += "FUNCTIONS_WORKER_RUNTIME=python" }

        if ($toSet.Count -gt 0) {
          Write-Host "Chaves a aplicar/atualizar:"
          $toSet | ForEach-Object { Write-Host "  $_" }
          $applyOut = az functionapp config appsettings set --name $FunctionApp --resource-group $ResourceGroup --settings $toSet 2>&1
          if ($LASTEXITCODE -ne 0) {
            Write-Warning "Falha ao aplicar app settings. Saída do az:`n$applyOut"
          } else {
            Write-Host "App settings aplicados com sucesso."
          }
        } else {
          Write-Host "Nenhuma alteração detectada nas app settings."
        }
      }
    } else {
      Write-Host "Nenhuma chave válida encontrada no local.settings.json (apenas comentários ou valores vazios)."
    }
  } else {
    Write-Host "local.settings.json não tem 'Values' ou está vazio."
  }
} else {
  Write-Host "local.settings.json não encontrado - pulando aplicação de settings."
}

# -------- função de fallback run-from-package (upload para storage) --------
function Do-RunFromPackageFallback($packagePath) {
  Write-Host "Fallback: upload para Storage e aplicar WEBSITE_RUN_FROM_PACKAGE..."
  $settingsJson = az functionapp config appsettings list --name $FunctionApp --resource-group $ResourceGroup --only-show-errors 2>&1
  if ($LASTEXITCODE -ne 0) { Write-Warning "Não foi possível obter app settings: $settingsJson"; return $false }
  try { $settingsList = $settingsJson | ConvertFrom-Json } catch { Write-Warning "Resposta inválida ao listar app settings."; return $false }

  $awsEntry = $settingsList | Where-Object { $_.name -eq "AzureWebJobsStorage" }
  if (-not $awsEntry) { Write-Warning "AzureWebJobsStorage não encontrado."; return $false }
  $aws = $awsEntry.value
  if ($aws -notmatch "AccountName=([^;]+)") { Write-Warning "Não foi possível extrair AccountName."; return $false }
  $storageAccount = $matches[1]
  $containerName = "function-deploy"

  az storage container create --account-name $storageAccount --name $containerName --auth-mode key --only-show-errors 2>&1
  if ($LASTEXITCODE -ne 0) { Write-Warning "Falha ao criar container"; return $false }

  az storage blob upload --account-name $storageAccount --container-name $containerName --file $packagePath --name package.zip --auth-mode key --only-show-errors 2>&1
  if ($LASTEXITCODE -ne 0) { Write-Warning "Falha no upload do blob"; return $false }

  $expiry = (Get-Date).AddDays(1).ToString("yyyy-MM-ddTHH:mmZ")
  $sas = az storage blob generate-sas --account-name $storageAccount --container-name $containerName --name package.zip --permissions r --expiry $expiry --auth-mode key -o tsv 2>&1
  if ($LASTEXITCODE -ne 0) { Write-Warning "Falha ao gerar SAS: $sas"; return $false }

  $blobUrl = "https://$storageAccount.blob.core.windows.net/$containerName/package.zip`?$sas"
  $applyRunOut = az functionapp config appsettings set --name $FunctionApp --resource-group $ResourceGroup --settings "WEBSITE_RUN_FROM_PACKAGE=$blobUrl" --only-show-errors 2>&1
  if ($LASTEXITCODE -ne 0) { Write-Warning "Falha ao aplicar WEBSITE_RUN_FROM_PACKAGE:`n$applyRunOut"; return $false }

  Write-Host "Fallback aplicado. URL: $blobUrl"
  return $true
}

# -------- zip-deploy (opcional) --------
if ($ZipDeploy) {
  Write-Host "Preparando package.zip para zip-deploy..."
  $cwd = Get-Location
  $packagePath = Join-Path $cwd.Path $PackageName
  if (Test-Path $packagePath) { Remove-Item $packagePath -Force }

  $exclusionRegex = 'local.settings.json|[\\/]venv[\\/]|[\\/]\\.venv[\\/]|[\\/]\\.git[\\/]|__pycache__'
  $files = Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch $exclusionRegex } | ForEach-Object { $_.FullName }
  if (-not $files) { FailIf "Nenhum arquivo encontrado para empacotar." }

  Compress-Archive -Path $files -DestinationPath $packagePath -Force
  if (-not (Test-Path $packagePath)) { FailIf "Falha ao criar $packagePath" }

  Write-Host "Arquivo $packagePath criado. Tentando zip-deploy via Kudu..."
  $deployOut = az functionapp deployment source config-zip --resource-group $ResourceGroup --name $FunctionApp --src $packagePath --only-show-errors 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Zip deploy falhou. Mensagem az:`n$deployOut"
    $ok = Do-RunFromPackageFallback $packagePath
    if (-not $ok) { Write-Warning "Fallback também falhou. Verifique permissões e conectividade." }
  } else {
    Write-Host "Zip deploy concluído com sucesso."
  }
}

# -------- reiniciar (tenta, não falha o script) --------
Write-Host "Tentando reiniciar a Function App..."
$restartOut = az functionapp restart --name $FunctionApp --resource-group $ResourceGroup --only-show-errors 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Restart retornou erro (provavelmente permissões):`n$restartOut"
} else {
  Write-Host "Function App reiniciada."
}

Write-Host "Done."
