$ErrorActionPreference = 'Stop'

$expected = @{
    'models/structured_forest.yml.gz' = 'fe3586bca5ec124b73fa769c2b54e195e4526d6ceeddf161fad96bf68a65bf17'
    'models/dexined.pth' = 'bd4c603ef71113b447bffb72a73b9a54ca890e78f4bc8f34cb73f1efb2681e74'
    'models/dexined_model.py' = 'd417d0db386205ede64267ae438f9bcb38f84f3832bced1d41bd2e011c5fab53'
}

foreach ($relativePath in $expected.Keys) {
    $path = Join-Path $PSScriptRoot $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required model file is missing: $relativePath"
    }

    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$relativePath]) {
        throw "SHA-256 mismatch for $relativePath. Expected $($expected[$relativePath]); got $actual."
    }
    Write-Output "OK  $relativePath"
}

$sidecar = Join-Path $PSScriptRoot 'models_ascii/sf.yml.gz'
if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
    $actual = (Get-FileHash -LiteralPath $sidecar -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected['models/structured_forest.yml.gz']) {
        throw "SHA-256 mismatch for models_ascii/sf.yml.gz."
    }
    Write-Output 'OK  models_ascii/sf.yml.gz'
}
