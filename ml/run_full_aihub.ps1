# ============================================================
# handTalk — AI Hub 전체 단어 추출 + 학습 + 평가 (Windows, CPU)
#
# 사용법 (레포 루트에서):
#   powershell -ExecutionPolicy Bypass -File ml\run_full_aihub.ps1
#
# - 이미 추출된 클립은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행
# - 학습은 <OUT>\models\last.pt 가 있으면 거기서 이어서 학습
# - 실행 중에는 PC가 절전 모드로 들어가지 않음 (스크립트 종료 시 해제)
# ============================================================

param(
    [string]$Src     = 'E:\wordmorpheme\수어 영상',
    [string]$Out     = 'E:\handtalk_data',
    [int]   $Workers = 6
)

# 'Stop' 이면 PS 5.1 에서 python stderr 한 줄만 나와도 중단되므로 exit code 로만 판단
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

# 절전 방지 (이 프로세스가 살아 있는 동안만)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$py     = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$train  = Join-Path $Out 'npy_train'
$test   = Join-Path $Out 'npy_test'
$models = Join-Path $Out 'models'
$logDir = Join-Path $Out 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("run_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))

function Run([string]$title, [string[]]$pyArgs) {
    "`n##### $title  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))" | Tee-Object -FilePath $log -Append
    & $py -W ignore @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "$title 실패 (exit $LASTEXITCODE) — 로그: $log" }
}

$tr = Join-Path $Src '1.Training'
$va = Join-Path $Src '2.Validation'

# ── 1) 학습용: real_word 01~16 ────────────────────────────────
foreach ($i in 1..16) {
    $b = '{0:D2}' -f $i
    Run "extract real_word $b" @('-m', 'ml.preprocessing.extract_aihub',
        '--keypoint', "$tr\[라벨]${b}_real_word_keypoint",
        '--morpheme', "$tr\[라벨]01_real_word_morpheme\morpheme",
        '--output', $train, '--batches', $b, '--all_words', '--workers', $Workers)
}

# ── 2) 학습용: syn_word 01, 02 ────────────────────────────────
Run 'extract syn_word 01 02' @('-m', 'ml.preprocessing.extract_aihub',
    '--keypoint', "$tr\[라벨]01_syn_word_keypoint\WORD\keypoint",
    '--morpheme', "$tr\[라벨]01_syn_word_keypoint\WORD\morpheme",
    '--output', $train, '--batches', '01', '02', '--all_words', '--workers', $Workers)

# ── 3) 테스트용: 2.Validation real_word 17, 18 + syn_word 03 ──
Run 'extract validation real_word 17 18' @('-m', 'ml.preprocessing.extract_aihub',
    '--keypoint', "$va\[라벨]09_real_word_keypoint\keypoint",
    '--morpheme', "$va\[라벨]01_real_word_morpheme\morpheme",
    '--output', $test, '--batches', '17', '18', '--all_words', '--workers', $Workers)
Run 'extract validation syn_word 03' @('-m', 'ml.preprocessing.extract_aihub',
    '--keypoint', "$va\[라벨]02_syn_word_keypoint\WORD\keypoint",
    '--morpheme', "$va\[라벨]02_syn_word_keypoint\WORD\morpheme",
    '--output', $test, '--batches', '03', '--all_words', '--workers', $Workers)

# ── 4) 학습 (끊겼으면 last.pt 에서 이어서) ─────────────────────
$trainArgs = @('-m', 'ml.training.train', '--data_dir', $train, '--model_dir', $models,
               '--packed', '--no_test_split', '--num_workers', '2', '--log_every', '500', '--export')
$last = Join-Path $models 'last.pt'
if (Test-Path $last) { $trainArgs += @('--resume', $last) }
Run 'train' $trainArgs

# ── 5) 학습에 안 쓴 수어자(17, 18)로 평가 ─────────────────────
Run 'evaluate' @('-m', 'ml.training.evaluate', '--data_dir', $train, '--test_dir', $test,
                 '--model_dir', $models, '--packed')

"`n완료 — 모델: $models\sign_recognizer.onnx, 로그: $log" | Tee-Object -FilePath $log -Append
