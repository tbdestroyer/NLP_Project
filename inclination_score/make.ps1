# Mirror of the Makefile targets for Windows without GNU make. Usage: .\make.ps1 <target>
param([Parameter(Mandatory = $true)][string]$Target)

$py = ".venv\Scripts\python.exe"
# Package and model caches stay inside the project folder.
$env:UV_CACHE_DIR = ".cache\uv"

function Invoke-Steps([string[][]]$Steps) {
    foreach ($step in $Steps) {
        & $py @step
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$data = @(@("-m", "inclination.load.download"), @("-m", "inclination.load.standardise"))
$train = , @("-m", "inclination.train")
$export = , @("-m", "inclination.export")
$eval = , @("eval/evaluate.py")

switch ($Target) {
    "setup" {
        uv venv --python 3.11 .venv
        uv pip install --python $py -r requirements.txt -e .
    }
    "download" { Invoke-Steps (, @("-m", "inclination.load.download")) }
    "inspect" { Invoke-Steps (, @("-m", "inclination.load.inspect_raw")) }
    "data" { Invoke-Steps $data }
    "train" { Invoke-Steps $train }
    "train-v3" { Invoke-Steps (, @("-m", "inclination.train", "--model", "v3")) }
    "llm" { Invoke-Steps (, @("-m", "inclination.llm_jobs", "all")) }
    "export-v2" { Invoke-Steps (, @("-m", "inclination.export", "--models", "v2", "--append")) }
    "export" { Invoke-Steps $export }
    "eval" { Invoke-Steps $eval }
    "app" { Invoke-Steps (, @("-m", "streamlit", "run", "app/Home.py")) }
    "test" { Invoke-Steps (, @("-m", "pytest", "-q")) }
    "all" { Invoke-Steps ($data + $train + $export + $eval) }
    default {
        Write-Error "Unknown target '$Target'. Targets: setup, download, inspect, data, train, train-v3, llm, export, export-v2, eval, app, test, all"
        exit 1
    }
}
exit $LASTEXITCODE
