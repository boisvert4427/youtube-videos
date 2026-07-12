Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

try {
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

$python = (Get-Command python).Source
$workdir = 'C:\Users\leona\Documents\python\youtube-videos\youtube-videos'
$log = 'C:\Users\leona\Documents\python\youtube-videos\youtube-videos\tmp\wta_api_only_2000_scrape.log'
$err = 'C:\Users\leona\Documents\python\youtube-videos\youtube-videos\tmp\wta_api_only_2000_scrape.err.log'

$batch = 0
while ($true) {
  $batch++
  Write-Host "[wta-loop] starting batch $batch"
  $p = Start-Process -FilePath $python -ArgumentList @(
    '-u',
    '-m',
    'scraper.tennis.build_wta_rankings_api_timeseries',
    '--api-only',
    '--start-date',
    '2000-11-27',
    '--end-date',
    '2026-06-29',
    '--sleep-seconds',
    '10',
    '--max-new-snapshots',
    '200'
  ) -WorkingDirectory $workdir -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru

  Wait-Process -Id $p.Id
  Start-Sleep -Seconds 1

  $tail = Get-Content $log -Tail 8 -ErrorAction SilentlyContinue
  foreach ($line in $tail) {
    Write-Host $line
  }

  if (($tail -join "`n") -match 'batch limit reached') {
    Write-Host "[wta-loop] batch limit reached, continuing"
    continue
  }

  Write-Host "[wta-loop] no batch limit reached, stopping"
  break
}
