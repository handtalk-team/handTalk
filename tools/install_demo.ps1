# ============================================================
# handTalk 시연 노트북 원클릭 설치 + 실행 (Windows)
#
#   powershell -ExecutionPolicy Bypass -File install_demo.ps1
#
# 1) Python 3.11 설치 (없으면)        2) GitHub 에서 코드 받기 (feature/tutor-app)
# 3) 서버 패키지 설치                  4) 공개 웹 라이브러리 받기 (three.js, MediaPipe, 글꼴)
# 5) 모델·메타휴먼 묶음 풀기           6) 서버 실행 + 브라우저 열기
#
# 5번 묶음(AI Hub 학습 모델, 메타휴먼)은 공개 저장소에 올릴 수 없어서 claude.ai 프로젝트 라이브러리에
# handtalk_demo_assets.zip.001 ~ .NNN 으로 나눠 올려 두었다. 다운로드 폴더(또는 -Assets 폴더)에 받아 두면
# 이 스크립트가 합쳐서 푼다. 두 번째 실행부터는 바로 서버만 켠다 (새 묶음을 받아 두면 그것만 다시 푼다).
# ============================================================

param(
    [string]$Dir = "$env:USERPROFILE\handtalk_demo",
    [string]$Assets = "$env:USERPROFILE\Downloads",
    [string]$Branch = 'feature/tutor-app',
    [switch]$Glove,
    [string]$GlovePort = '',
    [switch]$Update          # 코드를 GitHub 최신으로 다시 받기
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force $Dir | Out-Null
# 바로가기가 다운로드 폴더가 아닌 설치 폴더의 스크립트를 가리키게 복사해 둔다
$self = Join-Path $Dir 'install_demo.ps1'
if ($PSCommandPath -and ($PSCommandPath -ne $self)) { Copy-Item $PSCommandPath $self -Force }
$code   = Join-Path $Dir 'handTalk'
$server = Join-Path $code 'server'
$venv   = Join-Path $Dir '.venv'
$py     = Join-Path $venv 'Scripts\python.exe'

function Find-Python {
    foreach ($v in '3.11', '3.12') {
        try { $p = & py "-$v" -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    }
    return $null
}

# 1) Python
if (-not (Test-Path $py)) {
    $base = Find-Python
    if (-not $base) {
        Write-Host "[1/6] Python 3.11 설치 중..." -ForegroundColor Yellow
        winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
        $base = Find-Python
        if (-not $base) { Write-Host "Python 설치 후 PowerShell 을 새로 열고 다시 실행하세요." -ForegroundColor Red; exit 1 }
    }
    Write-Host "[1/6] Python: $base"
    & $base -m venv $venv
}

# 2) 코드
if ($Update -and (Test-Path $code)) {
    # 모델·메타휴먼·라이브러리는 저장소 밖 파일이라 옮겨 두었다가 다시 넣는다
    $keep = @('server\ml\models', 'client\assets\avatar', 'client\assets\lexicon.json', 'client\vendor', 'server\.env')
    $tmpKeep = Join-Path $Dir '_keep'
    foreach ($k in $keep) { $s = Join-Path $code $k; if (Test-Path $s) { $d = Join-Path $tmpKeep $k; New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null; Move-Item $s $d -Force } }
    Remove-Item $code -Recurse -Force
}
if (-not (Test-Path $server)) {
    Write-Host "[2/6] GitHub 에서 코드 받기 ($Branch)"
    $zip = Join-Path $Dir 'handTalk.zip'
    Invoke-WebRequest "https://github.com/handtalk-team/handTalk/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
    $tmp = Join-Path $Dir '_extract'
    Expand-Archive $zip -DestinationPath $tmp -Force
    Move-Item (Get-ChildItem $tmp | Select-Object -First 1).FullName $code
    Remove-Item $tmp, $zip -Recurse -Force
    $tmpKeep = Join-Path $Dir '_keep'
    if (Test-Path $tmpKeep) { Copy-Item (Join-Path $tmpKeep '*') $code -Recurse -Force; Remove-Item $tmpKeep -Recurse -Force }
}

# 3) 서버 패키지
$marker = Join-Path $venv '.installed'
if (-not (Test-Path $marker)) {
    Write-Host "[3/6] 서버 패키지 설치 (몇 분 걸립니다)"
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $server 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Host "패키지 설치 실패 — 위 오류를 확인하세요." -ForegroundColor Red; exit 1 }
    New-Item -ItemType File $marker | Out-Null
}

# 4) 공개 웹 라이브러리 (인터넷 없이도 열리게 미리 받아 둠)
if (-not (Test-Path (Join-Path $code 'client\vendor\mediapipe\hands\hands.js'))) {
    Write-Host "[4/6] 웹 라이브러리 받기 (three.js, MediaPipe Hands, 글꼴)"
    & $py (Join-Path $code 'client\tools\fetch_vendor.py')
}

# 5) 모델·메타휴먼 묶음 (처음이거나, 다운로드 폴더에 새 묶음이 있으면 다시 푼다)
# 라이브러리에서 받으면 이름에 날짜가 붙기도 한다 (handtalk_demo_assets.zip-20261022T...-xxxx.001).
# 조각 번호(확장자)마다 가장 최근에 받은 파일 하나씩만 골라 번호 순으로 합친다.
$parts = Get-ChildItem $Assets -Filter 'handtalk_demo_assets.zip*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -match '^\.\d{3}$' } |
    Group-Object Extension | ForEach-Object { $_.Group | Sort-Object LastWriteTime -Descending | Select-Object -First 1 } |
    Sort-Object { [int]$_.Extension.TrimStart('.') }
$stamp = Join-Path $server 'ml\models\bundle_stamp.txt'
$newStamp = if ($parts) { ($parts | ForEach-Object { "$($_.Name) $($_.Length) $($_.LastWriteTimeUtc.Ticks)" }) -join ';' } else { '' }
$oldStamp = if (Test-Path $stamp) { Get-Content $stamp -Raw } else { '' }
$needAssets = -not (Test-Path (Join-Path $server 'ml\models\sign_motion\meta.json'))
if ($newStamp -and ($newStamp.Trim() -ne $oldStamp.Trim())) { $needAssets = $true }
if ($needAssets) {
    $whole = Join-Path $Assets 'handtalk_demo_assets.zip'
    if (-not $parts -and -not (Test-Path $whole)) {
        Write-Host "[5/6] 모델 묶음이 없습니다. claude.ai 프로젝트 라이브러리에서 handtalk_demo_assets.zip.001 ~ 을 모두" -ForegroundColor Red
        Write-Host "      '$Assets' 에 받은 뒤 다시 실행하세요." -ForegroundColor Red
        exit 1
    }
    if ($parts) {
        Write-Host "[5/6] 묶음 합치기 ($($parts.Count)개)"
        $out = [System.IO.File]::Create((Join-Path $Dir 'assets.zip'))
        foreach ($p in $parts) { $b = [System.IO.File]::ReadAllBytes($p.FullName); $out.Write($b, 0, $b.Length) }
        $out.Close()
        $whole = Join-Path $Dir 'assets.zip'
    }
    Write-Host "[5/6] 묶음 풀기"
    Expand-Archive $whole -DestinationPath $code -Force
    if ($whole -eq (Join-Path $Dir 'assets.zip')) { Remove-Item $whole -Force }
    if ($newStamp) { [System.IO.File]::WriteAllText($stamp, $newStamp) }
}
$envFile = Join-Path $server '.env'
# PowerShell 5.1 의 Set-Content -Encoding utf8 은 BOM 을 붙여 첫 키를 깨뜨리므로 BOM 없이 쓴다
if (-not (Test-Path $envFile)) { [System.IO.File]::WriteAllText($envFile, "LLM_PROVIDER=claude_code`nCLAUDE_CODE_MODEL=sonnet`n", (New-Object System.Text.UTF8Encoding $false)) }

# 바탕화면 바로가기 (다음부터 더블클릭으로 실행)
$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'handTalk 시작.lnk'
if (-not (Test-Path $lnk)) {
    $sh = New-Object -ComObject WScript.Shell
    $s = $sh.CreateShortcut($lnk)
    $s.TargetPath = 'powershell.exe'
    $s.Arguments = "-ExecutionPolicy Bypass -File `"$self`" -Dir `"$Dir`""
    $s.Save()
}

# 글러브
if ($Glove -and -not $GlovePort) {
    $ports = [System.IO.Ports.SerialPort]::GetPortNames() | Sort-Object
    if ($ports.Count -ge 1) { $GlovePort = $ports[-1] }
}
$env:GLOVE_PORT_RIGHT = $GlovePort

# 6) 실행
Write-Host "[6/6] 서버 시작 — 브라우저가 http://localhost:8000/app/ 로 열립니다 (종료: 이 창 닫기)" -ForegroundColor Green
Start-Process 'http://localhost:8000/app/'
Set-Location $server
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
