# 环境

教程在 Python 3.11 和 Conda 环境 `geoml` 中开发。项目不要求创建新的虚拟环境。

地理依赖（特别是 GDAL、Rasterio、GeoPandas 和 PROJ）建议通过 Conda 安装并放在同一个环境中。其他直接依赖的版本见仓库根目录的 `requirements.txt`。在已有的地理分析环境中运行前，请先确认 Rasterio、GDAL 与 PROJ 可以正常导入。

生成 Notebook 内核配置：

```powershell
conda run -n geoml python tools\prepare_kernel.py
$env:JUPYTER_PATH = "$PWD\.jupyter"
```

项目局部 `.jupyter/` 配置由脚本在本机生成，包含本地解释器路径，因此不会随仓库分发。
