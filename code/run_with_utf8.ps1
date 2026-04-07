# 设置控制台编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

Write-Host "语音系统启动器 (UTF-8模式)" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan

# 获取传递的参数
$argsString = $args -join " "

if ($argsString -eq "") {
    Write-Host "语音对话系统启动器" -ForegroundColor Green
    Write-Host "用法: .\run_with_utf8.ps1 main.py [参数...]" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "参数选项:" -ForegroundColor Cyan
    Write-Host "  --file <文件>    处理指定的wav文件" -ForegroundColor White
    Write-Host "  --list           列出当前目录的所有wav文件" -ForegroundColor White
    Write-Host "  --preload        预加载模式（推荐）- 模型加载一次，连续对话" -ForegroundColor Green
    Write-Host "" -ForegroundColor Yellow
    Write-Host "示例:" -ForegroundColor Gray
    Write-Host "  .\run_with_utf8.ps1 main.py --file test1.wav" -ForegroundColor Gray
    Write-Host "  .\run_with_utf8.ps1 main.py --list" -ForegroundColor Gray
    Write-Host "  .\run_with_utf8.ps1 main.py --preload" -ForegroundColor Green
    Write-Host "" -ForegroundColor Yellow
    Write-Host "💡 提示: 使用 --preload 模式可以避免重复加载模型，提升响应速度!" -ForegroundColor Magenta
    exit 1
}

# 构建命令
$batFile = "run_with_utf8.bat"
$fullCommand = "cmd /c `"$batFile $argsString`""

Write-Host "执行命令: $fullCommand" -ForegroundColor Yellow
Write-Host ""

# 运行命令
Invoke-Expression $fullCommand
