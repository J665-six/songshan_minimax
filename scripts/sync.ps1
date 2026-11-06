# 把本地 infinilm/ 打包上传到集群 ~/work/infinilm
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync.ps1

$ErrorActionPreference = 'Stop'

$local  = 'C:\Users\ASUS\Desktop\infinilm'
$tar    = Join-Path $local '_sync.tar.gz'
$remote = '/public/home/xdzs2026_b0413/work/infinilm.tar.gz'
$key    = 'C:\Users\ASUS\Desktop\minimax\tools\zzeshell_key'
$upload = 'C:\Users\ASUS\Desktop\minimax\tools\scp_up.py'
$ssh    = 'C:\Users\ASUS\Desktop\minimax\tools\scnet.py'

$env:SCNET_USER = 'xdzs2026_b0413'
$env:SCNET_HOST = 'zzeshell.scnet.cn'
$env:SCNET_PORT = '65032'
$env:SCNET_KEY  = $key

if (Test-Path $tar) { Remove-Item $tar -Force }

$parent = Split-Path $local
$leaf   = Split-Path $local -Leaf
$tarArgs = @('-czf', $tar, '-C', $parent,
             '--exclude=*/results/*.json',
             '--exclude=*/__pycache__/*',
             $leaf)
& tar.exe @tarArgs | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'tar failed' }
Write-Host ("archive {0:N1} KB" -f ((Get-Item $tar).Length / 1KB))

& python $ssh "mkdir -p ~/work && echo REMOTE_DIR_OK"
if ($LASTEXITCODE -ne 0) { throw 'mkdir failed' }

& python $upload $tar $remote
if ($LASTEXITCODE -ne 0) { throw 'upload failed' }

$remoteScript = @'
mkdir -p ~/work
tar -xzf ~/work/infinilm.tar.gz -C ~/work
cd ~/work/infinilm
python3 -m py_compile bench/bench_op.py bench/cases.py && echo SYNTAX_OK
find . -maxdepth 2 -type f | sort
'@
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
& python $ssh ("echo $b64 | base64 -d | bash")
