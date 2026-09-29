# Data and model sources

Source files are linked here, not copied into this repository. Licence terms belong to each provider; check them before any reuse.

| Use in DepthWizard | Source | Terms note |
|---|---|---|
| Training and urban evaluation: ISPRS Vaihingen benchmark (9 cm IRRG orthophoto and a dense image-matching DSM, `dsm_09cm_matching`) | https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx | Academic/research-use benchmark. Not redistributed here. The `dsm_09cm_matching` product is an image-matching DSM, not LiDAR ground truth. `PHASE2_RED_FLAGS/scripts/fetch_area26_data.py` fetches a copy from an unofficial third-party Hugging Face mirror; that mirror is not endorsed, so obtain the data from the ISPRS source and follow its terms. |
| Base backbone: Depth Anything V2 Small (HF format) | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf | Apache-2.0 (Small model) |
| Depth Anything V2 paper | https://arxiv.org/abs/2406.09414 | |
| Satellite optical imagery (Mojave and forest checks): Sentinel-2 L2A COGs | https://registry.opendata.aws/sentinel-2-l2a-cogs/ | Contains modified Copernicus Sentinel data 2021 |
| Reference elevation and hilly-terrain tests: USGS 3D Elevation Program (3DEP) 1 m DEM, lidar-derived bare-earth | https://prd-tnm.s3.amazonaws.com/index.html (StagedProducts/Elevation) | USGS-authored or produced data are in the U.S. public domain (USGS copyrights-and-credits policy); credit "U.S. Geological Survey" |
| Aerial imagery for hilly and demo scenes: USDA NAIP (Washington 2023 program used for the screenshots) | https://naip-usdaonline.hub.arcgis.com/ and https://planetarycomputer.microsoft.com/dataset/naip | USDA FSA policy: most FSA information is public domain, attribution "U. S. Department of Agriculture, Farm Service Agency" (https://www.fsa.usda.gov/help/policies-and-links/); the Planetary Computer NAIP record links to that policy as "Public Domain" |
| Copernicus DEM GLO-30 (research lane, not part of the shipped runner) | https://registry.opendata.aws/copernicus-dem/ and https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM | See the attribution block in `README.md` |
| ETH Global Canopy Height 2020 (proxy reference for the forest check) | https://doi.org/10.1038/s41559-023-02206-6 | CC BY 4.0. A model-derived product, not survey truth. |

The frozen hilly-terrain and Mojave evidence folders under `10_DOCUMENTATION/evidence/` document the scenes, seeds and commands used for those checks.
