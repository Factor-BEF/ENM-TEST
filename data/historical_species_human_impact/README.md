# Historical species and human-impact data for ENM

本目录基于仓库中的 `Awesome-GEE` 资源入口，为 ENM 测试整理 1 种植物、1 种动物和 1 套人类活动历史数据。

## 数据总览

| 类型 | 目标 | 数据集 | 时间范围 | 空间/记录分辨率 | 仓库中的获取方式 |
|---|---|---|---|---|---|
| 植物 | 黑云杉 (*Picea mariana*) | Canada Long-Term Tree Species | 1984–2022，年度 | 30 m raster | GEE asset + JS 批量导出脚本；类别值 18 |
| 动物 | 赤狐 (*Vulpes vulpes*) | GBIF occurrence download | 1980–2017 | point occurrences | 固定 DOI/download key + Python 下载脚本 |
| 人类活动 | Global Human Modification v3 | GHM 1990–2020 overall | 1990、1995、2000、2005、2010、2015、2020 | 300 m raster | GEE asset + JS 批量导出脚本 |

## 1. 植物：黑云杉 Picea mariana

- 数据集：Canada Long-Term Tree Species (1984–2022)
- GEE ImageCollection：`projects/sat-io/open-datasets/CA_FOREST/SPECIES-1984-2022`
- 时间：1984–2022，年度
- 空间分辨率：30 m
- 黑云杉类别值：`18`
- 许可：Open Government Licence – Canada
- 来源说明：https://gee-community-catalog.org/projects/ca_species_ts/
- 加拿大官方开放数据：https://open.canada.ca/data/en/dataset/17396c45-8100-4ce6-a5d6-147c3d566d2e

仓库文件：
- `plant_black_spruce/species_legend.csv`：完整 0–37 类别图例并标记黑云杉。
- `plant_black_spruce/gee_export_black_spruce.js`：按年导出 1984–2022 黑云杉 presence/absence GeoTIFF。

> 官方逐年 GeoTIFF 体量约为 GB 级，因此不将 39 年原始栅格直接塞入 Git 仓库；脚本从固定 GEE asset 导出，与保存原始大文件相比更适合复现和版本控制。

## 2. 动物：赤狐 Vulpes vulpes

- 数据源：GBIF occurrence archive
- GBIF taxon key：`5219243`
- 固定历史下载：`0008885-200221144449610`
- DOI：`10.15468/dl.ktotdn`
- 时间：1980–2017
- 记录数：256,439
- 该下载已限定 `hasCoordinate=true` 和 `hasGeospatialIssue=false`，适合作为 ENM 历史出现点的起始数据。
- DOI 页面：https://doi.org/10.15468/dl.ktotdn

仓库文件：
- `animal_red_fox/gbif_source_manifest.csv`：固定下载键、DOI、筛选条件和时间范围。
- `animal_red_fox/download_gbif_red_fox.py`：下载已归档 GBIF ZIP，不需要重新发起异步 GBIF 下载任务。

> GBIF 完整 occurrence archive 较大，因此仓库记录不可变 DOI/download key 和自动下载脚本，而不是重复提交大 ZIP。这样能保证原始数据可追踪到同一归档版本。

## 3. 人类活动：Global Human Modification v3

- 数据集：Global Human Modification v3 historical overall modification
- GEE ImageCollection：`projects/sat-io/open-datasets/GHM/HM_1990_2020_OVERALL_300M`
- 时间：1990、1995、2000、2005、2010、2015、2020
- 空间分辨率：300 m
- 指标范围：0–1；数值越高表示人类改造程度越强。
- 历史数据 DOI：`10.5281/zenodo.14449495`
- 方法论文：Theobald et al. (2025), Scientific Data, DOI `10.1038/s41597-025-04892-2`
- 许可：CC BY 4.0
- 来源说明：https://gee-community-catalog.org/projects/global_human_modification/

仓库文件：
- `human_activity/gee_export_ghm_v3.js`：批量导出 7 个历史年份，并额外生成 1990→2020 的变化量栅格。

## ENM 使用建议

1. 将黑云杉和 GHM 栅格统一投影、范围与分辨率；若用于 MaxEnt/BIOMOD 等模型，可根据研究尺度将 GHM 重采样到环境变量网格。
2. 赤狐 GBIF 记录下载后先做时间分层、重复点清理、空间稀疏化和采样偏差处理，再用于历史 ENM。
3. 不建议把 2022 GHM 静态产品直接与 1990–2020 历史序列计算变化，因为该版本与历史变化序列的可比性需要单独确认。

## 可复现性

`data_catalog.csv` 汇总所有固定资产 ID、DOI、许可和时间信息。所有大体量原始数据均通过固定 GEE asset 或 GBIF DOI 获取，仓库仅保存小型元数据与可执行获取脚本，避免 Git 仓库膨胀并保持来源可追踪。
