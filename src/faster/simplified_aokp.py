#This is a re-organised version of Maria's (Wunsch) verison.
# It takes into account only the files that change during the optimisation
# The files that do not need to change (e.g., all the GIS stuff), are in a database that is accessed by the optimiser
# Database: in folder experimental_results, which are produced by another set of codes in src/faster/experiments,
# many of them are using the original AOKP functions that run GDAL (they need to run one by one separated and take time)

import os
from subprocess import run

import numpy as np
import pandas as pd
import pygeoprocessing.routing
import rtree
from osgeo import gdal, gdalconst, ogr, osr

import src.user_inputs as usin
from src.faster.custom_typing import Connection, Coordinate


def reset_files():
    for folder in [usin.path_temp, usin.path_results, usin.path_outputs]:
        for root_path, _, files in os.walk(folder):
            for filename in files:
                filepath = os.path.join(root_path, filename)
                os.remove(filepath)


def create_pour_points(coordinates: list[Coordinate]) -> None:
    """
    This function defines the pour points based on the chosen geographic coordinates for the WWTP construction,
    saving them into a shapefile.

    :param: coordinates: Matrix ( n x 2) of geographic coordinates in UTM
    :return: None
    """

    xs = [x for x, y in coordinates]
    ys = [y for x, y in coordinates]

    # Outlet.shp: creation of the pour points shapefile
    # Get the spatial reference from Proj4
    spatialReference = osr.SpatialReference()
    #    spatialReference.ImportFromProj4('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #WGS83/30N ##Juan Carlos DEM
    spatialReference.ImportFromProj4(
        '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs')  # WGS83/36N ## Abu Dies, West Bank

    # Define the directory (the path where you want to save your shapefile)

    # Delete pour point files
    for entry in os.listdir(usin.path_pour_points):
        absolute_path = os.path.join(usin.path_pour_points, entry)
        if os.path.isfile(absolute_path):
            os.remove(absolute_path)

    # Select the driver for the shp-file creation.
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Here the data are stored
    shapeData = driver.CreateDataSource(usin.path_pour_points)
    # This will create a corresponding layer for the data with given spatial information.
    layer = shapeData.CreateLayer('pour_points', spatialReference, ogr.wkbPoint)
    layer_defn = layer.GetLayerDefn()  # gets parameters of the current shapefile
    point = ogr.Geometry(ogr.wkbPoint)

    # Populate the attribute table
    for fid, (x, y) in enumerate(zip(xs, ys)):
        # What created before will be written into the layer/shape file
        feature = ogr.Feature(layer_defn)
        # Create a new point at given coordinates
        point.AddPoint(x, y)
        feature.SetGeometry(point)
        feature.SetFID(fid)
        layer.CreateFeature(feature)

    # Close the shapefile
    shapeData.Destroy()


def order_watershed():
    """
    This function sorts the features in the watershed.shp in a decreasing area order. In the rasterization,
    this step allows to not lose the small watersheds that could be overlappped by the biggest ones, as
    in the rasterization process the last value at the corresponding pixel is taken into account.

    :param: None
    :return: ordered_watershed
    """

    # Open the data source file
    ogr.UseExceptions()  # if something goes wrong, we want to know about it
    driver = ogr.GetDriverByName('ESRI Shapefile')
    watershed_data = driver.Open(usin.path_watershed_shp, 0)  # 0--read-only

    layer = watershed_data.GetLayer()
    layerName = layer.GetName()  # returns string 'watersheds'
    ordered_watershed = watershed_data.ExecuteSQL(
        'select * from "%s" order by Area DESC' % layerName)

    # print("Watersheds are ordered in a decreasing order")

    return watershed_data, ordered_watershed


def attribute_areas():
    """
    This function calculates the area of each watershed and adds the calculated value to the attribute table of the
    watershed shapefile.

    :param: None
    :return: None
    """

    dataset = ogr.Open(usin.path_watershed_folder)
    driver = dataset.GetDriver()
    dataSource = driver.Open(usin.path_watershed_shp, 1)
    # define floating point field named Area
    areaFldDef = ogr.FieldDefn('Area', ogr.OFTInteger)
    # define floating point field named CatchID
    idFldDef = ogr.FieldDefn('CatchID', ogr.OFTInteger)
    # get layer and add the 2 fields:
    layer = dataSource.GetLayer()
    layer.CreateField(areaFldDef)
    layer.CreateField(idFldDef)
    # Populate
    for i, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        feature.SetField("Area", area)
        # Use i + 1 to distinguish first layer from 0 in raster
        feature.SetField("CatchID", i + 1)
        layer.SetFeature(feature)


def order_and_rasterize_watershed():
    """
    This function rasterizes the watershed. In the rasterization process the last value at the corresponding
    pixel is taken into account (that is why the order_watershed is done before).

    :param: None
    :return:
    """

    watershed_data, ordered_watershed = order_watershed()

    # driver = ogr.GetDriverByName('ESRI Shapefile')
    # ds = driver.Open(path_watershed_shp, 0)  # 0--read-only
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(usin.path_filled, gdal.GA_ReadOnly)
    # Fetch number of rows and columns
    ncol, nrow = raster_ds.RasterXSize, raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(
        usin.path_watershed_dem, ncol, nrow, 1, gdal.GDT_UInt32)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    # Rasterize the shapefile layer to our new dataset

    status = gdal.RasterizeLayer(dataset=out_raster_ds,  # output to our new dataset
                                 # output to our new dataset's first band
                                 bands=[1],
                                 layer=ordered_watershed,  # rasterize this layer
                                 # pfnTransformer=None,
                                 # pTransformArg=None,
                                 burn_values=[0],
                                 options=[
                                     'ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                     'ATTRIBUTE=CatchID',  # put raster values according to the 'CatchID' field values
                                 ],
                                 callback=0,
                                 callback_data=None
                                 )

    watershed_data.ReleaseResultSet(ordered_watershed)

    if status != 0:
        # release layer
        raise RuntimeError("Watershed rasterization failed!")


def polygonize_watershed():
    """
    This function converts the rasterfile (watershed_dem.tif) in vector layer, essential in the next steps
    for the extraction of information of each subcatchment.

    :param: None
    :return: None
    """
    # POLIGONIZE (from raster to shapefile)

    # Save the spatial reference of the shape file watershed in order to use for the
    # new shapefile that will result from Polygonized process
    # ( ,1) means it will be also written on
    dataSource = ogr.Open(usin.path_watershed_shp, 1)
    layer = dataSource.GetLayer()
    srs = layer.GetSpatialRef()

    # call the raster
    sourceRaster = gdal.Open(usin.path_watershed_dem)
    band = sourceRaster.GetRasterBand(1)
    # bandArray = band.ReadAsArray()

    # create a new shapefile
    outShapefile = "polygonized"
    dataset = ogr.Open(usin.path_watershed_folder)
    driver = dataset.GetDriver()

    if os.path.exists(outShapefile + ".shp"):
        driver.DeleteDataSource(outShapefile + ".shp")

    # We are going to store the raster in the main folder (where the raster is contained)
    # Here the code, if you instead want to store your shapefile in a specific folder
    outDatasource = driver.CreateDataSource(usin.path_subcatchments)

    outLayer = outDatasource.CreateLayer("polygonized", srs)
    #  I need to create a field to store the raster band
    newField = ogr.FieldDefn('ID', ogr.OFTInteger)
    outLayer.CreateField(newField)

    # The following script will give you as result the all polygons in the raster also the one to be created automatically from the value 0
    # status=gdal.Polygonize( band, None, outLayer, 0, [], callback=None )

    # Here the script if you instead want to obtain only the polygons in the shapefile. In this case, a mask is applied
    # in order to obtain only the polygons that are inside the biggest watershed
    status = gdal.Polygonize(band, band, outLayer, -1, [], callback=None)
    outDatasource.Destroy()

    if status != 0:
        raise RuntimeError("Watershed polygonization failed!")


def count_features(path_to_shapefile: str) -> int:
    """
    This function counts how many features there are in particular layer.

    :param: Vector file with features to be count .shp
    :return: number_features
    """
    # How many points did you give as input?
    dataSource = ogr.Open(path_to_shapefile)
    lyr = dataSource.GetLayer()
    # print('The layer is named: {n}\n'.format(n=lyr.GetName()))
    # We need to account before the total features
    feature_count = lyr.GetFeatureCount()
    # We need to account the number of fields
    # First we need to capture the layer definition
    defn = lyr.GetLayerDefn()
    # How many fields
    field_count = defn.GetFieldCount()
    number_features = int(feature_count / field_count)
    return number_features


def get_feature_grid(path_data: str, tmp_filename: str, path_temp: str) -> rtree.index.Index:
    ds = ogr.Open(path_data, gdalconst.GA_ReadOnly)
    layer = ds.GetLayer()

    file_path = os.path.join(path_temp, tmp_filename)
    if os.path.exists(file_path + ".dat"):
        index = rtree.index.Index(file_path, interleaved=False)
    else:
        index = rtree.index.Index(file_path, interleaved=False)
        for fid1 in range(0, layer.GetFeatureCount()):
            feature1 = layer.GetFeature(fid1)
            geometry1 = feature1.GetGeometryRef()
            xmin, xmax, ymin, ymax = geometry1.GetEnvelope()
            index.insert(fid1, (xmin, xmax, ymin, ymax))

    # index.close()
    return index


def reorder_outlets(path_pour_points_shp: str, path_ordered_outlets: str, path_polygonized: str) -> dict[int, int]:
    # Get first layer for intersection
    ds_data_outlet = ogr.Open(path_pour_points_shp,
                              gdalconst.GA_ReadOnly)  # old path_temp_outlet
    layer_data_outlet = ds_data_outlet.GetLayer()
    # layer_geom_outlet = layer_data_outlet.GetGeometryRef()

    # Get second layer for intersection
    ds_data_polygons = ogr.Open(path_polygonized, gdalconst.GA_ReadOnly)
    layer_data_polygons = ds_data_polygons.GetLayer()

    # create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path_ordered_outlets):
        driver.DeleteDataSource(path_ordered_outlets)
    outlets_ordered_ds = driver.CreateDataSource(path_ordered_outlets)
    outlets_ordered_lyr = outlets_ordered_ds.CreateLayer(
        path_ordered_outlets, geom_type=ogr.wkbPoint)
    # featureDefn = outlets_ordered_lyr.GetLayerDefn()

    # Order the outlets based on the polygons
    outlets = [outlet for outlet in layer_data_outlet]

    sc2outlet = {}
    for i_sc, polygon in enumerate(layer_data_polygons):
        polygon_geometry = polygon.GetGeometryRef()
        for i_outlet, outlet in enumerate(outlets):
            outlet_geometry = outlet.GetGeometryRef()
            if outlet_geometry.Intersects(polygon_geometry):
                outlets_ordered_lyr.CreateFeature(outlet)
                sc2outlet[i_sc] = i_outlet

    return sc2outlet


def join_attribute_network(path_to_datafile: str, network_grid_data, path_polygonized_file: str, field_name: str) -> np.ndarray:
    ds_data = ogr.Open(path_to_datafile, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()

    ds_polygonized = ogr.Open(path_polygonized_file, gdalconst.GA_ReadOnly)
    layer_polygonized = ds_polygonized.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    n_subcatchments = layer_polygonized.GetFeatureCount()
    data_per_subcatchment = np.zeros(n_subcatchments)

    for idx_subcatchment in range(n_subcatchments):
        polygon = layer_polygonized.GetFeature(idx_subcatchment)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for channel_index in list(network_grid_data.intersection((xmin, xmax, ymin, ymax))):
                channel = layer_data.GetFeature(channel_index)
                channel_geometry = channel.GetGeometryRef()
                if channel_geometry.Intersects(polygon_geometry):
                    field_data = channel.GetField(field_name)
                    if field_data is not None:
                        data_per_subcatchment[idx_subcatchment] += float(field_data)

    return data_per_subcatchment


def join_attribute_data_in_boundary(path_to_datafile: str, feature_grid_data,
                                    polygonized_file: str, field_name: str,
                                    location_index_field_name: str, num_types_data: int) -> np.ndarray:

    ds_data = ogr.Open(path_to_datafile, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()

    ds_polygonized = ogr.Open(polygonized_file, gdalconst.GA_ReadOnly)
    layer_polygonized = ds_polygonized.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    num_subcatchments = layer_polygonized.GetFeatureCount()
    # Creates a matrix (list in a list) with: num_types_data (towns) as rows and subcatchments as columns
    data_per_town_per_subcatch = np.zeros((num_types_data, num_subcatchments))

    idx_features_seen = set()

    for idx_subcatchment in range(layer_polygonized.GetFeatureCount()):
        polygon = layer_polygonized.GetFeature(idx_subcatchment)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry is not None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for idx_feature in list(feature_grid_data.intersection((xmin, xmax, ymin, ymax))):
                data_feature = layer_data.GetFeature(idx_feature)
                data_geometry = data_feature.GetGeometryRef()
                if data_geometry.Intersects(polygon_geometry):
                    field_data = data_feature.GetField(field_name)
                    # Gets the field with the index of each town
                    idx_location = data_feature.GetField(
                        location_index_field_name)
                    if field_data is not None and idx_location is not None:
                        if idx_feature not in idx_features_seen:
                            data_per_town_per_subcatch[idx_location][idx_subcatchment] += field_data
                            idx_features_seen.add(idx_feature)

    return data_per_town_per_subcatch


def join_attribute_data_at_outlet(path_ordered_outlets, feature_grid_data, path_land_cost_data, field_name, no_data_value):

    ds_data = ogr.Open(path_land_cost_data, gdalconst.GA_ReadOnly)
    ds_outlets = ogr.Open(path_ordered_outlets, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()
    layer_outlets = ds_outlets.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    n_outlets = layer_outlets.GetFeatureCount()
    data_per_outlets = []

    for idx_subcatchment in range(n_outlets):
        polygon = layer_outlets.GetFeature(idx_subcatchment)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry is not None:
            for feature_index in feature_grid_data.intersection(polygon_geometry.GetEnvelope()):
                field_data = layer_data.GetFeature(feature_index)
                field_data_geometry = field_data.GetGeometryRef()
                if field_data_geometry.Intersects(polygon_geometry):
                    field_data = field_data.GetField(field_name)
                    if field_data is not None:
                        data_per_outlets.append(float(field_data))
                    else:
                        data_per_outlets.append(no_data_value)

    return data_per_outlets


def create_temporary_circular_buffer(idx_sc: int, radius: float, path_ds_in: str, path_buffer_out: str) -> str:
    n_points_polygon = 100

    ds_in = ogr.Open(path_ds_in)
    layer_in = ds_in.GetLayer()

    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Re-create output datasource that contains buffer
    if os.path.exists(path_buffer_out):
        driver.DeleteDataSource(path_buffer_out)
    ds_buffer_out = driver.CreateDataSource(path_buffer_out)

    layer_out = ds_buffer_out.CreateLayer(
        "buffer_area", layer_in.GetSpatialRef())
    feature_out_defn = layer_out.GetLayerDefn()

    feature_sc = layer_in.GetFeature(idx_sc)
    geom_feature_sc = feature_sc.GetGeometryRef()
    # Create polygon that approximates a circle
    geom_buffer = geom_feature_sc.Buffer(radius, n_points_polygon)

    feature_out = ogr.Feature(feature_out_defn)
    feature_out.SetGeometry(geom_buffer)
    layer_out.CreateFeature(feature_out)

    return path_buffer_out


def remove_small_subcatchments(threshold: float = 1000):
    ds_polygonized = ogr.Open(usin.path_subcatchments, gdalconst.GA_Update)
    layer_polygonized = ds_polygonized.GetLayer()
    # Counts the number of subcatchments in the polygons layer
    n_subcatchments = layer_polygonized.GetFeatureCount()
    # print(f"Found {n_subcatchments} subcatchments.")

    for i_subcatchment in range(layer_polygonized.GetFeatureCount()):
        polygon = layer_polygonized.GetFeature(i_subcatchment)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            # xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            boundary = polygon_geometry.GetBoundary()
            area = boundary.Area()
            # print(f"Area of sc {i_subcatchment} is {area}.")
            if area <= threshold:
                # print(f"SC {i_subcatchment} is too small, deleting.")
                layer_polygonized.DeleteFeature(polygon.GetFID())


def read_outlet_locations(csv_path: str) -> tuple[list[Coordinate], list[Connection]]:
    df = pd.read_csv(csv_path)
    coordinates = [(float(x), float(y)) for x, y in zip(df["X"], df["Y"])]
    connections = [(int(id), int(conn_to), 1.0)
                   for id, conn_to in zip(df["id"], df["connects_to"]) if not np.isnan(conn_to)]
    return coordinates, connections


def pre_AOKP(coordinates: list[Coordinate]) -> dict[int, int]:
    # reset result, output and temporary files
    reset_files()

    create_pour_points(coordinates)

    ########################### CONVERTION TO Py3 ################################
    # TODO: convert path_watershed from shp to gpkg
    # path_watershed_converted = os.path.join(root, "temp",'output_watershed.gpkg')
    # run(['ogr2ogr', '-f', 'GPusin.KG', path_watershed_converted, path_watershed])
    flow_dir_path = os.path.join(usin.root, "temp", 'flow_dir_d8.tif')
    pygeoprocessing.routing.flow_dir_d8((usin.path_filled, 1), flow_dir_path)
    gdal.UseExceptions()
    pygeoprocessing.routing.delineate_watersheds_d8(
        (flow_dir_path, 1), usin.path_pour_points_shp, usin.path_watershed)

    # convert watersheds.gpkg into watershed.shp
    run(['ogr2ogr', '-f', "ESRI Shapefile", usin.path_watershed_folder, usin.path_watershed])

    ########################### CONVERTION TO Py3 ################################

    attribute_areas()

    order_and_rasterize_watershed()

    polygonize_watershed()

    num_WWTP = count_features(usin.path_pour_points_shp)  # path_temp_outlet

    # Delete subcatchments that are the size of "one-pixel", they can occur when
    # The subcatchments overlap on 1 pixel or smth like that.
    # The size of one pixel seems to be 822 m2.
    one_pixel_area = 1000
    remove_small_subcatchments(threshold=one_pixel_area)

    num_subcatchments = count_features(usin.path_subcatchments)

    assert num_subcatchments == num_WWTP, "Number of WWTP must match number of subcatchments."

    # Order outlets, so they can have the same index as the polygons (subcatchments)
    # old path_temp_outlets before path_pourpoints...
    sc2outlet = reorder_outlets(
        usin.path_pour_points_shp, usin.path_ordered_outlets, usin.path_subcatchments)

    return sc2outlet
