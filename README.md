# Geo-ML｜地理人的机器学习课

![Geo-ML 教程封面](assets/social-preview.png)

> 用黑龙江甘南县的多源栅格，走一遍“认数据 → 对齐 → 建模 → 空间验证”的完整流程。

这是一本写给**地理、GIS、遥感方向**读者的中文实战教程。你不需要先学完机器学习：只要有 Python 基础，能看懂经纬度、投影、栅格和像元，就可以跟着 8 个 Notebook 一步步做。

教程不追求“把模型刷到最高分”。它更关心地理人最容易踩的坑：**不同分辨率怎么对齐、缺测怎么处理、随机划分为什么过于乐观、测试区为什么不能反复用、SHAP 为什么不是因果。**

```mermaid
flowchart LR
    A["六张原始栅格<br/>LST · DEM · NDVI · 气温 · 降水 · 土壤水分"] --> B["01 认识数据"]
    B --> C["02 对齐到 LST 网格"]
    C --> D["03 构建训练表"]
    D --> E["04 空间验证"]
    E --> F["05 回归基线"]
    F --> G["06 决策树与随机森林"]
    G --> H["07 XGBoost 与 SHAP"]
    H --> I["08 KNN / SVR / MLP"]
```

## 这本书在做什么

案例区是**黑龙江省齐齐哈尔市甘南县**（约 122.90–124.47°E、47.59–48.53°N），不是甘肃省甘南藏族自治州。目标变量是 2021 年的地表温度（LST），预测变量是高程、NDVI、气温、2020 年降水和 2020 年土壤水分。

需要先说明三件事：

1. **单位未核实**：源数据没有注明 LST、气温、降水、土壤水分的物理单位。教程图件统一写“原始像元值”，不写 °C、mm 或 %。
2. **时间不完全同期**：LST 和气温只知 2021 年，降水和土壤水分是 2020 年，NDVI 没有时间血缘。教程把它们当作环境背景变量，不解释成同日机制。
3. **这是方法演示**：目标不是生产甘南县 LST 产品，而是演示“在未参与训练的空间块上估计 LST”的完整流程与评估方式。

## 你会学到什么

- 从 GeoTIFF 元数据检查、栅格对齐到训练表构建；
- 空间分块验证，以及它和随机划分的区别；
- 从线性回归、树模型到 XGBoost、SHAP、KNN、SVR 和 MLP；
- 用地图和误差指标检查预测表现，并讨论模型解释的边界。

![Geo-ML 竖版封面](assets/cover-portrait.png)

## 章节地图

| 章 | 一句话问题 | 主要内容 |
|---|---|---|
| 01 | 这些数据是什么？ | 元数据、坐标系、NoData 与空间预览 |
| 02 | 不同网格怎么对齐？ | 重投影、重采样与有效区检查 |
| 03 | 怎样从栅格构建样本？ | 特征表、缺测检查与结果写回 |
| 04 | 随机划分为什么可能乐观？ | 随机验证与空间分块验证 |
| 05 | 简单回归能做到什么？ | 均值基线、线性回归与岭回归 |
| 06 | 树模型怎样拟合空间数据？ | 决策树、过拟合与随机森林 |
| 07 | 提升树如何预测与解释？ | XGBoost、空间调参与 SHAP |
| 08 | KNN、SVR、MLP 怎么比较？ | 非线性回归、参数与误差评估 |

每章围绕一个主问题展开，结合操作步骤、结果解读、地理视角和练习题。

## 案例与数据

教程示例使用黑龙江省齐齐哈尔市甘南县的多源栅格，演示在留出空间区域上估计地表温度。源数据的单位、时间和再分发许可并未全部核实，因此公开仓库不包含原始数据、派生数据或模型结果。请按 [数据说明](docs/DATA.md) 自行准备数据，并先确认你有权使用这些数据。

Notebook 已清除运行输出。准备好同结构数据后，按章节顺序运行即可生成自己的表格和图件。

## 快速开始

### 1. 环境

本项目复用已有的 Conda 环境 `geoml`，不新建虚拟环境：

```powershell
conda run -n geoml python --version
```

依赖版本见 [requirements.txt](requirements.txt)。地理库建议通过 Conda 环境安装，避免单独用 pip 安装 GDAL。

### 2. 数据

把六张栅格和甘南县边界放到 `data/raw/`：

```text
data/raw/
├── GNLST.tif
├── GNDEM.tif
├── GNNDVI.tif
├── GNTEM.tif
├── precip_sum_2020.tif
├── soil_moist_mean_2020.tif
└── GanNan.shp / .shx / .dbf / .prj / .cpg
```

字段、单位、时间与校验值见 [docs/DATA.md](docs/DATA.md) 和 [data/raw/README.md](data/raw/README.md)。

### 3. 运行

```powershell
cd Geo-ML

# 生成项目局部 kernelspec（只需一次）
conda run -n geoml python tools\prepare_kernel.py

# 让 Jupyter 找到项目内 kernel
$env:JUPYTER_PATH = "$PWD\.jupyter"

# 跑单章
conda run --no-capture-output -n geoml python tools\run_notebook.py 01_认识数据.ipynb

# 跑全部 8 章
conda run --no-capture-output -n geoml python tools\run_notebook.py `
  01_认识数据.ipynb 02_栅格对齐.ipynb 03_构建训练表.ipynb 04_空间验证.ipynb `
  05_回归基线.ipynb 06_决策树与随机森林.ipynb 07_XGBoost与SHAP.ipynb 08_KNN_SVR_MLP.ipynb
```

每章第一个代码单元都会打印 `sys.executable` 和 `CONDA_PREFIX`，用来确认没有误用 base 环境。

## 项目结构

```text
Geo-ML/
├── notebooks/              # 8 个章节 Notebook（输出已清空）
├── src/
│   ├── geoml_data.py       # 栅格读取、对齐、训练表、空间划分
│   ├── geoml_models.py     # 模型工厂、指标、CV
│   └── geoml_style.py      # 中文制图样式、三联图、模型对比图
├── data/raw/               # 原始数据（不随公开仓库分发）
├── docs/
│   ├── DATA.md             # 数据准备与许可说明
│   └── ENVIRONMENT.md      # 环境配置说明
├── tools/
│   ├── run_notebook.py     # 无界面执行
│   ├── prepare_kernel.py   # 生成本地 kernelspec
│   └── check_notebooks.py  # 交付前自检
└── requirements.txt
```

## 复现与自检

准备好数据后可运行 Notebook。公开版不附带预先生成的结果图和表格。

```powershell
conda run -n geoml python tools\check_notebooks.py
```

教程代码与讲解已在本地数据环境验证。由于公开版不包含数据，运行完整教程前请先按数据说明放置自己的数据。

## 数据与许可

**数据再分发许可尚未核实。** 本仓库不分发原始栅格、边界矢量、训练表、预测结果或带运行输出的 Notebook。因此：

- 教程代码与文字采用 MIT 许可（见 [LICENSE](LICENSE)）；
- 数据与模型结果不属于 MIT 许可覆盖范围；
- 使用者需自行确认数据来源和使用权限。

更多信息见 [数据说明](docs/DATA.md)。

## 局限

- 源数据的观测时段、单位和来源信息不完整，不能仅凭数值范围推断。
- 不同变量可能来自不同年份和空间分辨率，不能把重采样误认为增加了细节。
- 测试区与训练区可能存在分布差异，指标需要结合空间划分和误差地图解读。
- 教程的 SVR 网格、树深范围、MLP 结构都很小，目的是教学，不是刷榜。

## 适合谁

- 地理、GIS、遥感专业的本科高年级/研究生；
- 会一点 Python，但没系统做过机器学习的人；
- 想给论文补上“空间验证”“基线模型”“可解释性”的地理工作者。

如果你已经熟悉随机森林和 SHAP，可以直接从第 04、07 章看验证设计和归因边界。

---
