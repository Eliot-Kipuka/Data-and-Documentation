Spatial grids aligned to the current manuscript and response

EPSG:32649, WGS 1984 / UTM zone 49N; centre coordinates are projected metres. Import SHP with SHX, DBF, PRJ and CPG.

grid_600m_model contains the 3,570 selected model polygons, not every Guangzhou grid. grid_1200m_model contains 1,648 nominal aggregation squares; display squares are not clipped and area_m2 is the sum of contributing source areas. CV block layers contain 526 occupied 3 km blocks and 245 occupied 5 km blocks. All four categories share geometry and folds. 3/5 km blocks are validation groups for the 600 m model, not independently estimated larger-grid outcomes.

Y1 HER, Y2 HSR, Y3 LER, Y4 LSR. Count and fold attributes match the sibling reproducibility package. These layers do not contain X17 predictors; model input correction is documented in that package and does not change their geometry or shop counts. 600m_to_1200m_membership.csv records contributing source cells. Count aggregation used the full supplied source fishnet.

Run python export_spatial_grids.py --output regenerated with pyshp==2.3.1 and the sibling reproducibility package present. Geometry is read from these included shapefiles; no original local absolute path is required. Attributes and coarse centres are updated from the shared input data. Existing files are not overwritten by this command.
