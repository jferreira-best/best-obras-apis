# === comparar e aplicar app settings do local.settings.json para a Function App ===
$ResourceGroup = "RG-SEE-D-FUNCTION"
$FunctionApp   = "see-d-crm-orquestradorbot"
$localPath     = ".\local.settings.json"

if (-not (Test-Path $localPath)) { Write-Error "local.settings.json não encontrado em $(Get-Location)"; exit 1 }

# Ler local.settings.json
$localJson = Get-Content $localPath -Raw | ConvertFrom-Json

# Construir lista de pares key=value ignorando 'comentários' e valores vazios
$localPairs = @()
foreach ($p in $localJson.Values.PSObject.Properties) {
  $k = $p.Name
  $v = $p.Value
  if ($k -like '#*') { continue }        # ignora chaves de comentário
  if ($null -eq $v -or $v -eq '') { continue } # ignora valores vazios
  $localPairs += [PSCustomObject]@{ Name = $k; Value = [string]$v }
}

if (-not $localPairs) { Write-Host "Nenhuma chave válida encontrada no local.settings.json"; exit 0 }

# Pegar app settings atuais do Azure
$azureOut = az functionapp config appsettings list --name $FunctionApp --resource-group $ResourceGroup 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Erro ao listar app settings do Azure:`n$azureOut"; exit 1 }
$azureList = $azureOut | ConvertFrom-Json

# Montar lista de keys que precisam ser atualizadas
$toSet = @()
foreach ($item in $localPairs) {
  $existing = $azureList | Where-Object { $_.name -eq $item.Name }
  if (-not $existing) {
    $toSet += "$($item.Name)=$($item.Value)"
  } elseif ($existing.value -ne $item.Value) {
    $toSet += "$($item.Name)=$($item.Value)"
  }
}

if (-not $toSet) {
  Write-Host "Nenhuma alteração detectada. Nada a aplicar."
  exit 0
}

Write-Host "Chaves a aplicar/atualizar:"
$toSet | ForEach-Object { Write-Host "  $_" }

# Aplicar (mostra erro completo se houver)
$applyOut = az functionapp config appsettings set --name $FunctionApp --resource-group $ResourceGroup --settings $toSet 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Error "Falha ao aplicar app settings. Saída do az:`n$applyOut"
  Write-Host "Se quiser, reexecute com --debug para ver mais detalhes:"
  Write-Host "  az functionapp config appsettings set --name $FunctionApp --resource-group $ResourceGroup --settings <KEY=VALUE...> --debug"
  exit 1
}

Write-Host "App settings aplicados com sucesso."
