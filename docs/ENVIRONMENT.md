# 环境配置

教程基于 Python 3.11 开发。地理空间库（GDAL、Rasterio、GeoPandas、PROJ）建议通过 `conda-forge` 统一安装，避免单独使用 `pip` 安装 GDAL 引发二进制依赖冲突。

## 1. 创建 Conda 环境

首次使用可通过仓库根目录的 [environment.yml](../environment.yml) 创建 `geoml` 环境：

```bash
conda env create -f environment.yml
```

如果你本机已有包含上述依赖的 `geoml` 环境，可直接复用，无需重建。已验证的精确版本清单见 [requirements.txt](../requirements.txt)。

## 2. 注册项目局部 Jupyter 内核

为了不污染全局 Jupyter 配置，可使用脚本在项目内生成 `.jupyter/kernels/geoml/kernel.json`（兼容 Windows / macOS / Linux）：

```powershell
# Windows PowerShell
conda run -n geoml python tools/prepare_kernel.py
$env:JUPYTER_PATH = "$PWD\.jupyter"
```

```bash
# macOS / Linux
conda run -n geoml python tools/prepare_kernel.py
export JUPYTER_PATH="$PWD/.jupyter"
```

`.jupyter/` 目录包含本机解释器路径，已被 `.gitignore` 忽略，不会随仓库分发。
