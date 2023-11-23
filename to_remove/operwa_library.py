import logging
import random
from subprocess import run

import pandas as pd
import pygeoprocessing.routing
import rtree
from numpy import *
from osgeo import gdal, gdalconst, ogr, osr
from platypus import Binary
from shapely import speedups

from .user_inputs import *

gdal.UseExceptions()

logging.basicConfig()

speedups.disable()  # based on https://stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

################################################
## Description of functions in AOKP algorithm ##
################################################


def create_pour_points(coordinates):
    """
        This function defines the pour points based on the chosen geographic coordinates for the WWTP construction,
        saving them into a shapefile.

        :param: coordinates: Matrix ( n x 2) of geographic coordinates in UTM
        :return: None
    """

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    # Outlet.shp: creation of the pour points shapefile
    # Get the spatial reference from Proj4
    spatialReference = osr.SpatialReference()
    #    spatialReference.ImportFromProj4('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #WGS83/30N ##Juan Carlos DEM
    spatialReference.ImportFromProj4(
        '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs')  # WGS83/36N ## Abu Dies, West Bank

    # Define the directory (the path where you want to save your shapefile)

    # Delete pour point files
    for entry in os.listdir(path_pour_points):
        absolute_path = os.path.join(path_pour_points, entry)
        if os.path.isfile(absolute_path):
            os.remove(absolute_path)

    driver = ogr.GetDriverByName('ESRI Shapefile')  # select the driver for the shp-file creation.
    shapeData = driver.CreateDataSource(path_pour_points)  # here the data are stored
    layer = shapeData.CreateLayer('pour_points', spatialReference,
                                  ogr.wkbPoint)  # this will create a corresponding layer for the data with given spatial information.
    layer_defn = layer.GetLayerDefn()  # gets parameters of the current shapefile
    point = ogr.Geometry(ogr.wkbPoint)

    # Populate the attribute table
    for i in range(len(X)):
        point.AddPoint(X[i], Y[i])  # create a new point at given coordinates
        featureIndex = i  # this will be the index points in the dataset

        # What created before will be written into the layer/shape file:
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(point)
        feature.SetFID(featureIndex)
        layer.CreateFeature(feature)

    shapeData.Destroy()  # close the shapefile


def attribute_areas():
    """
        This function calculates the area of each watershed and adds the calculated value to the attribute table of the
        watershed shapefile.

        :param: None
        :return: None
    """

    dataset = ogr.Open(path_watershed_folder)
    driver = dataset.GetDriver()
    dataSource = driver.Open(path_watershed_shp, 1)
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


def order_watershed():
    """
        This function sorts the features in the watershed.shp in a decreasing area order. In the rasterization,
        this step allows to not loose the small watersheds that could be overlappped by the biggest ones, as
        in the raterization process the last value at the corresponding pixel is taken into account.

        :param: None
        :return: ordered_watershed
    """

    # Open the data source file
    ogr.UseExceptions()  # if something goes wrong, we want to know about it
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(path_watershed_shp, 0)  # 0--read-only

    layer = ds.GetLayer()
    layerName = layer.GetName()  # returns string 'three_points'
    ordered_watershed = ds.ExecuteSQL('select * from "%s" order by Area DESC' % layerName)

    return ds, ordered_watershed


def rasterize_watershed(ordered_watershed):
    """
            This function rasterizes the watershed. In the rasterization process the last value at the corresponding
            pixel is taken into account (that is why the order_watershed is done before).

            :param: None
            :return:
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(path_watershed_shp, 0)  # 0--read-only
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(path_filled, gdal.GA_ReadOnly)
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(path_watershed_dem, ncol, nrow, 1, gdal.GDT_UInt32)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    # Rasterize the shapefile layer to our new dataset

    status = gdal.RasterizeLayer(dataset=out_raster_ds,  # output to our new dataset
                                 bands=[1],  # output to our new dataset's first band
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

    if status != 0:
        raise RuntimeError("Rasterization failed")


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
    dataSource = ogr.Open(path_watershed_shp, 1)  # ( ,1) means it will be also written on
    layer = dataSource.GetLayer()
    srs = layer.GetSpatialRef()

    # call the raster
    sourceRaster = gdal.Open(path_watershed_dem)
    band = sourceRaster.GetRasterBand(1)
    # bandArray = band.ReadAsArray()

    # create a new shapefile
    outShapefile = "polygonized"
    dataset = ogr.Open(path_watershed_folder)
    driver = dataset.GetDriver()

    if os.path.exists(outShapefile + ".shp"):
        driver.DeleteDataSource(outShapefile + ".shp")

    # We are going to store the raster in the main folder (where the raster is contained)
    # Here the code, if you instead want to store your shapefile in a specific folder
    outDatasource = driver.CreateDataSource(path_subcatchments)

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
        raise RuntimeError("Polygonize failed")


def count_features(path_to_shapefile):
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
    # print('Outlet.shp has {n} WWTPs'.format(n=WWTP))


def calculate_area(path_to_shapefile, number_features):
    """
              This function counts what is the area of features in a particular layer.

              :param: Vector file with features to be count .shp
              :return: Area (list)
    """

#                            'AREA OF EACH SUB-CATCHMENT [m2]'
    dataSource = ogr.Open(path_to_shapefile)
    layer = dataSource.GetLayer()
    Area = zeros(number_features)
    i = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        Area[i] = geom.GetArea()
        i = i + 1
    #        print ("Area =",Area)
    return Area


def join_attribute_network(path_to_datafile, network_grid_data, polygonized_file, field_name):

    ds_data = ogr.Open(path_to_datafile, gdalconst.GA_ReadOnly)
    ds_polygonized = ogr.Open(polygonized_file, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()
    layer_polygonized = ds_polygonized.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    num_subcatchments = layer_polygonized.GetFeatureCount()
    # Creates a matrix (list in a list) with: num_locations (towns) as rows and subcatchments as columns
    data_per_subcatchment = np.zeros(num_subcatchments)

    for subcatchment_index in range(layer_polygonized.GetFeatureCount()):
        polygon = layer_polygonized.GetFeature(subcatchment_index)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for channel_index in list(network_grid_data.intersection((xmin, xmax, ymin, ymax))):
                channel = layer_data.GetFeature(channel_index)
                channel_geometry = channel.GetGeometryRef()
                if channel_geometry.Intersects(polygon_geometry):
                    length = channel.GetFieldAsString(field_name)
                    if length != '':
                        data_per_subcatchment[subcatchment_index] += float(length)

    print(('The sum of', field_name, 'in all subcatchments is', sum(data_per_subcatchment)))
    return data_per_subcatchment


def join_attribute_data_in_boundary(path_to_datafile, feature_grid_data, polygonized_file, field_name, location_index_field_name, num_types_data):

    ds_data = ogr.Open(path_to_datafile, gdalconst.GA_ReadOnly)
    ds_polygonized = ogr.Open(polygonized_file, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()
    layer_polygonized = ds_polygonized.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    num_subcatchments = layer_polygonized.GetFeatureCount()
    # Creates a matrix (list in a list) with: num_types_data (towns) as rows and subcatchments as columns
    data_per_town_per_subcatch = np.zeros((num_types_data, num_subcatchments))

    index_counted_features = []

    for subcatchment_index in range(layer_polygonized.GetFeatureCount()):
        polygon = layer_polygonized.GetFeature(subcatchment_index)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for data_feature_index in list(feature_grid_data.intersection((xmin, xmax, ymin, ymax))):
                data_feature = layer_data.GetFeature(data_feature_index)
                data_geometry = data_feature.GetGeometryRef()
                if data_geometry.Intersects(polygon_geometry):
                    field_data = data_feature.GetFieldAsString(field_name)
                    # Gets the field with the index of each town
                    field_location = data_feature.GetFieldAsString(location_index_field_name)
                    if field_location != '':
                        # Transforms the index of each town in integer
                        loc_index = int(field_location)

                        if field_data != '':

                            if data_feature_index not in index_counted_features:

                                data_per_town_per_subcatch[loc_index][subcatchment_index] += float(
                                    field_data)

                                index_counted_features.append(data_feature_index)

    return data_per_town_per_subcatch


def join_attribute_data_in_buffer(path_to_datafile, feature_grid_data, polygonized_file, field_name, location_index_field_name, num_types_data):

    ds_data = ogr.Open(path_to_datafile, gdalconst.GA_ReadOnly)
    ds_polygonized = ogr.Open(polygonized_file, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()
    layer_polygonized = ds_polygonized.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    num_subcatchments = layer_polygonized.GetFeatureCount()
    data_per_town_per_subcatch = np.zeros((num_types_data, num_subcatchments))

    for subcatchment_index in range(layer_polygonized.GetFeatureCount()):
        polygon = layer_polygonized.GetFeature(subcatchment_index)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for data_feature_index in list(feature_grid_data.intersection((xmin, xmax, ymin, ymax))):
                data_feature = layer_data.GetFeature(data_feature_index)
                data_geometry = data_feature.GetGeometryRef()
                if data_geometry.Intersects(polygon_geometry):
                    field_data = data_feature.GetFieldAsString(field_name)
                    # Gets the field with the index of each town
                    field_location = data_feature.GetFieldAsString(location_index_field_name)
                    if field_location != '':
                        # Transforms the index of each town in integer
                        loc_index = int(field_location)

                        if field_data != '':
                            data_per_town_per_subcatch = float(field_data)

    return data_per_town_per_subcatch


def present_value(DR, n, c_op_year):
    """
                This function calculates the present value cost for determined number of operation years.

                :param: DR = Discount rate ()
                        n = expected operated life (years)
                        cost_year = cost expended per year for operation (ILS/year)

                :return: cost_total (ILS)
    """
    c_op_total = c_op_year * (1 - (1 + DR) ** (-n)) / DR  # [ILS/year]
    return c_op_total


def calculate_scale_factor(Inv_1, Inv_2, Q1, Q2):
    """
                This function calculates the scale factor for the cost to capacity method.

                :param: Inv_1 = Factor scale for the Investment Cost - Price 1 [-]
                        Inv_2 = Factor scale for the O&M Cost - Price 2
                        Q1 = Design Capacity of WWTP1 [m3/day]
                        Q2 = Design Capacity of WWTP2 [m3/day]
                :return: sc_factor
    """

    sc_factor = (log(Inv_2 / Inv_1)) / (log(Q2 / Q1))

    if sc_factor > 1:
        # print (sc_factor,'sc_factor is bigger than 1, this means that\
        # diseconomies of scale exist and the incremental cost becomes more\
        # expensive for every added unit of capacity. This does not match\
        # with the literature, where the factor scale for WWTP is smaller\
        # than 1. For this reason the value is changed to 0.70')
        return 0.7
    else:
        # print ('sc_factor is ', sc_factor)
        return sc_factor


def calc_pipeline_ww_inv_cost(pipe_cost, hRan, length):
    """
                    This function calculates the costs of the construction of pipelines.

                    :param: pipe_cost = Cost pipe [ILS/m] (list with 8 elements)
                            hRan= highest range value of the length of the pipes per specific diameter [m] (list with 7 elements)
                            length = length of the network in each catchment [m]
                    :return: pipe_inv_cost
    """
    # hRan = np.array(hRan)
    # pipe_cost = np.array(pipe_cost)
    pipe_inv_cost = zeros(*length.shape)

    for i in range(len(length)):
        if length[i] < hRan[0]:
            pipe_inv_cost[i] = length[i] * pipe_cost[0]
        elif length[i] < hRan[1]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (length[i] - hRan[0]) * pipe_cost[1]
        elif length[i] < hRan[2]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + \
                (hRan[1] - hRan[0]) * pipe_cost[1] + (length[i] - hRan[1]) * pipe_cost[2]
        elif length[i] < hRan[3]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
                length[i] - hRan[2]) * pipe_cost[3]
        elif length[i] < hRan[4]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
                hRan[3] - hRan[2]) * pipe_cost[3] + (length[i] - hRan[3]) * pipe_cost[4]
        elif length[i] < hRan[5]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
                hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (length[i] - hRan[4]) * pipe_cost[5]
        elif length[i] < hRan[6]:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
                hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (hRan[5] - hRan[4]) * pipe_cost[5] + (
                length[i] - hRan[5]) * pipe_cost[6]
        else:
            pipe_inv_cost[i] = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
                hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (hRan[5] - hRan[4]) * pipe_cost[5] + (
                hRan[6] - hRan[5]) * pipe_cost[6] + (length[i] - hRan[6]) * pipe_cost[7]

    #            print(pipe_inv_cost)
    return pipe_inv_cost


def calculate_flow_at_wwtp(pop_per_town_subcatchment, water_consumption, L, p, num_locations, num_WWTP):
    """
                    This function calculates the flow at each treatment plant, considering the designed flow and
                    population in each subcatchment.

                    :param: pop_per_catchment = population in each subcatchment connected to WWTP(inhab)
                            flow_design = flow in m3/d
                    :return: flow_wwtp - Capacity of the WWTP [m3/day]
    """

    flow_wwtp_per_town = np.zeros((num_locations, num_WWTP))

    # Volume of water consumption from each source according to user's input, per year

    for location_index in range(num_locations):
        for subcatchment_index in range(num_WWTP):
            flow_per_town = pop_per_town_subcatchment[location_index,
                                                      subcatchment_index] * (1 - L) * water_consumption[location_index] * p
            flow_wwtp_per_town[location_index][subcatchment_index] += float(flow_per_town)

    print('Flow of wwtp per town per subcatchment = ', flow_wwtp_per_town)

    flow_wwtp = np.sum(flow_wwtp_per_town, axis=0)

    return flow_wwtp


def coverage_check(pop_per_subcatchment, real_pop):
    """
                        This function calculates the coverage of the wastewater network in the region.

                        :param: connected_pop = population in the region that is connected to a WWTP(inhab)
                                real_pop = total population given by user
                        :return: coverage = percentage of connected and total population
    """

    connected_pop = (float(sum(pop_per_subcatchment)))
    real_pop = (float(real_pop))
    coverage = connected_pop / real_pop
    # print 'Connected pop =', connected_pop
    # print 'Real population =', real_pop
    return coverage


def calc_benefit_connections(pop_per_catchment, inhabit_per_household, fee_connection):
    """
                        This function calculates the benefits obtained through the payment of connections fees by
                        the population. This payment is done once in the investment time (normally in the first
                        year), considers the average number of floors per building in the region and the area of
                        buildings per subcatchment.

                        :param:
                        :return:
    """

    benefit_connection = zeros(*pop_per_catchment.shape)
    # Connection fee to the WWTP
    for i in range(len(pop_per_catchment)):
        benefit_connection[i] = (pop_per_catchment[i] / inhabit_per_household) * fee_connection

    return benefit_connection


def calc_benefit_ww_yearly_fees(pop_per_town_per_subcatchment, fee_sanitary, water_consumption, num_WWTP, num_locations):
    """
                        This function calculates the benefits obtained through the payment of yearly fees by the
                        population to the water authorities. Present value calculation is made to the investment period
                        time.

                        :param:
                        :return:
    """

    benefit_per_town_per_subcatch = np.zeros((num_locations, num_WWTP))

    for location_index in range(num_locations):
        for subcatchment_index in range(num_WWTP):
            benefit_per_town_per_sub = pop_per_town_per_subcatchment[location_index, subcatchment_index] * \
                water_consumption[location_index] * fee_sanitary[location_index] * 365
            benefit_per_town_per_subcatch[location_index][subcatchment_index] += float(
                benefit_per_town_per_sub)

    benefit_fees_per_subcatchment = np.sum(benefit_per_town_per_subcatch, axis=0)

    return benefit_fees_per_subcatchment


def cost_to_capacity(flow, known_flow, sc_factor, known_cost):
    """
                                This function calculates the investment costs, based on the cost to capacity method.
                                :param:
                                :return:
    """

    investment_costs = zeros(*flow.shape)

    for i in range(len(investment_costs)):
        investment_costs[i] = ((flow[i] / known_flow) ** (sc_factor)) * known_cost

    return investment_costs


def reorder_outlets(path_pour_points_shp, path_ordered_outlets, path_polygonized):  # old path_temp_outlet
    """
                    This function

                    :param:
                    :return:
    """

    # Get first layer for intersection
    ds_data_outlet = ogr.Open(path_pour_points_shp, gdalconst.GA_ReadOnly)  # old path_temp_outlet
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

    # # Order the outlets based on the polygons
    outlets = [outlet for outlet in layer_data_outlet]

    for polygon in layer_data_polygons:
        polygon_geometry = polygon.GetGeometryRef()
        for outlet in outlets:
            outlet_geometry = outlet.GetGeometryRef()
            if outlet_geometry.Intersects(polygon_geometry):
                outlets_ordered_lyr.CreateFeature(outlet)

    return None


def join_attribute_data_at_outlet(ordered_outlets, feature_grid_data, path_land_cost_data, field_name, no_data_value):

    ds_data = ogr.Open(path_land_cost_data, gdalconst.GA_ReadOnly)
    ds_outlets = ogr.Open(ordered_outlets, gdalconst.GA_ReadOnly)
    layer_data = ds_data.GetLayer()
    layer_outlets = ds_outlets.GetLayer()

    # Counts the number of subcatchments in the polygons layer
    num_outlets = layer_outlets.GetFeatureCount()
    data_per_outlets = []

    for subcatchment_index in range(layer_outlets.GetFeatureCount()):
        polygon = layer_outlets.GetFeature(subcatchment_index)
        polygon_geometry = polygon.GetGeometryRef()
        if polygon_geometry != None:
            xmin, xmax, ymin, ymax = polygon_geometry.GetEnvelope()
            for feature_index in list(feature_grid_data.intersection((xmin, xmax, ymin, ymax))):
                field_data = layer_data.GetFeature(feature_index)
                field_data_geometry = field_data.GetGeometryRef()
                if field_data_geometry.Intersects(polygon_geometry):
                    field_data = field_data.GetFieldAsString(field_name)
                    if field_data != '':
                        data_per_outlets.append(float(field_data))
                    else:
                        data_per_outlets.append(no_data_value)

    # print ('The sum of', field_name, 'in all subcatchments is', sum(data_per_outlets))
    return data_per_outlets


def get_land_cost_by_wwtp(land_cost_in_wwtp_location, flow_wwtp, type_treatment):

    land_cost_by_wwtp = []
    for i in range(len(land_cost_in_wwtp_location)):
        if type_treatment[i] == 'cas with agr. reuse':
            land_cost_by_wwtp.append(area_usage_cas * land_cost_in_wwtp_location[i] * flow_wwtp[i])
            print("Land area requirement for CAS plant = ", area_usage_cas * flow_wwtp[i], "m2")

        elif type_treatment[i] in ['mbr with urb. reuse', 'mbr with no reuse']:
            land_cost_by_wwtp.append(area_usage_mbr * land_cost_in_wwtp_location[i] * flow_wwtp[i])

            print("Land area requirement for MBR plant = ", area_usage_mbr * flow_wwtp[i], "m2")
        else:
            land_cost_by_wwtp.append(0)

    return land_cost_by_wwtp


def create_temporary_buffer(inputfn, radius, wwtp_index):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()
    srs = inputlyr.GetSpatialRef()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    outputBufferfn = path_outputBufferfn
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    outLayer = outputBufferds.CreateLayer("buffer_area", srs)
    featureDefn = outLayer.GetLayerDefn()

    wwtp = inputlyr.GetFeature(wwtp_index)
    ingeom = wwtp.GetGeometryRef()
    geomBuffer = ingeom.Buffer(radius)

    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geomBuffer)
    outLayer.CreateFeature(outFeature)

    return outputBufferfn


def calculate_applied_irr_flow(radius_wwtp, inputfn, feature_grid_data, wwtp_index):
    """
                    This function calculates the optimal flow to be used in an area.

                    :param:
                    :return:
    """

    temporary_buffer_file = create_temporary_buffer(inputfn, radius_wwtp, wwtp_index)
    # number_buffers = count_features(outputBufferfn)
    areas_buffer_size = np.pi * radius_wwtp**2

    area_per_reuse = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                     feature_grid_data=feature_grid_data,
                                                     polygonized_file=temporary_buffer_file,
                                                     field_name='area_pixel',
                                                     location_index_field_name='Type_R_ind',
                                                     num_types_data=3)
    area_per_reuse = area_per_reuse[:, 0]

    demand_irr_flow_all_types = np.zeros(len(area_per_reuse))
    if area_per_reuse[0] != 0:
        optimal_flow = irr_oper_agr * area_per_reuse[0]
        demand_irr_flow_all_types[0] += float(optimal_flow)
    if area_per_reuse[1] != 0:
        optimal_flow = irr_oper_urb * area_per_reuse[1]
        demand_irr_flow_all_types[1] += float(optimal_flow)

    type_treatment_flow_based = choose_type_treatment_flow_based(
        demand_irr_flow_all_types, areas_buffer_size)

    if type_treatment_flow_based == 'cas with agr. reuse':
        applied_irrigation_flow = demand_irr_flow_all_types[0]
    elif type_treatment_flow_based == 'mbr with urb. reuse':
        applied_irrigation_flow = demand_irr_flow_all_types[1]
    else:
        applied_irrigation_flow = 0

    return applied_irrigation_flow, type_treatment_flow_based, area_per_reuse


def choose_type_treatment_flow_based(demand_irr_flow_all_types, areas_buffer_size):

    type_treatment_flow_based = []
    if areas_buffer_size == 0:
        return 'no_wwtp'
    else:
        agr_flow = demand_irr_flow_all_types[0]
        urb_flow = demand_irr_flow_all_types[1]
        if (agr_flow == 0) and (urb_flow == 0):
            return 'mbr with no reuse'
        else:
            if agr_flow >= urb_flow:
                return 'cas with agr. reuse'
            else:
                return 'mbr with urb. reuse'


def find_optimal_radii(flow_at_each_wwtp, inputfn, outputBufferfn, feature_grid_data, tol=1e-3):
    """
                    This function calculates the radius of influence of each treatment plant, in terms of areas
                    where the reuse of reclaimed wastewater could take place.

                    :param: flow_at_each_wwtp
                    :return: radius_influence_wwtp
    """
    radii_wwtp = []
    applied_irrigation_flows = []
    types_treatment_flow_based = []
    areas_per_reuse = []

    for wwtp_index in range(len(flow_at_each_wwtp)):
        if flow_at_each_wwtp[wwtp_index] == 0:
            radius_wwtp = 0
            applied_irrigation_flow = 0
            type_treatment_flow_based = 'no_wwtp'
            area_per_reuse = np.zeros(nr_reuse_options)
        else:

            max_radius = 0.1
            applied_irrigation_flow = -1
            # Find sufficiently large max_radius
            while applied_irrigation_flow < flow_at_each_wwtp[wwtp_index]:
                max_radius *= 10
                applied_irrigation_flow, \
                    type_treatment_flow_based, \
                    area_per_reuse = calculate_applied_irr_flow(radius_wwtp=max_radius,
                                                                inputfn=inputfn,
                                                                feature_grid_data=feature_grid_data,
                                                                wwtp_index=wwtp_index)

            # Bisection method
            min_radius = 0
            tol = 0.01
            radius_wwtp = (min_radius + max_radius) / 2.0

            # Check that we calculate flow at least once
            assert (max_radius - min_radius) / 2.0 > tol

            while (max_radius - min_radius) / 2.0 > tol:
                applied_irrigation_flow, \
                    type_treatment_flow_based, \
                    area_per_reuse = calculate_applied_irr_flow(radius_wwtp=radius_wwtp,
                                                                inputfn=inputfn,
                                                                feature_grid_data=feature_grid_data,
                                                                wwtp_index=wwtp_index)

                if applied_irrigation_flow - flow_at_each_wwtp[wwtp_index] == 0:
                    break
                elif applied_irrigation_flow - flow_at_each_wwtp[wwtp_index] > 0:
                    max_radius = radius_wwtp
                else:
                    min_radius = radius_wwtp
                radius_wwtp = (min_radius + max_radius) / 2.0

        radii_wwtp.append(radius_wwtp)
        applied_irrigation_flows.append(applied_irrigation_flow)
        types_treatment_flow_based.append(type_treatment_flow_based)
        areas_per_reuse.append(area_per_reuse)

    return radii_wwtp, applied_irrigation_flows, types_treatment_flow_based, areas_per_reuse


def createBuffer(inputfn, outputBufferfn, bufferDist):

    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()
    srs = inputlyr.GetSpatialRef()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    outLayer = outputBufferds.CreateLayer("buffer_area", srs)
    # bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()

    feature_index = 0
    for feature in inputlyr:

        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist[feature_index])

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        outLayer.CreateFeature(outFeature)
        outFeature = None
        feature_index += 1


def calculate_area_buffer(path_to_shapefile, number_features):
    """
              This function counts what is the area of features in a particular layer.

              :param: Vector file with features to be count .shp
              :return: Area (list)
    """

#                            'AREA OF EACH SUB-CATCHMENT [m2]'
    dataSource = ogr.Open(path_to_shapefile)
    layer = dataSource.GetLayer()
    areas_buffer = zeros(number_features)
    i = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom != None:
            areas_buffer[i] = geom.GetArea()
        else:
            pass
        i = i + 1
    #        print ("Area =",Area)
    return areas_buffer


def wwtp_costs(flow_wwtp, known_flow_1, known_flow_2, known_inv_1, known_inv_2, known_oem_1, known_oem_2, DR, n):

    # Present value for O&M - costs  1
    pv_oem_1 = present_value(DR=DR, n=n, c_op_year=known_oem_1)

    # Present value for O&M - costs 2
    pv_oem_2 = present_value(DR=DR, n=n, c_op_year=known_oem_2)

    # Scale factor for O&M
    sc_oem = calculate_scale_factor(Inv_1=pv_oem_1, Inv_2=pv_oem_2,
                                    Q1=known_flow_1, Q2=known_flow_2)

    # Cost to capacity for O&M
    oem_costs = ((flow_wwtp / known_flow_1) ** (sc_oem)) * known_oem_1

    print("O&M costs (PV) = ", oem_costs)

    # Scale factor for investment
    sc_inv = calculate_scale_factor(
        Inv_1=known_inv_1, Inv_2=known_inv_2, Q1=known_flow_1, Q2=known_flow_2)

    investment_costs = ((flow_wwtp / known_flow_1) ** (sc_inv)) * known_inv_1

    print("Investments in WWTP costs  = ", investment_costs)

    wwtp_total_cost = investment_costs + oem_costs
    return wwtp_total_cost


def calc_pipeline_reuse_inv_cost(pipe_cost, hRan, length):
    """
                    This function calculates the costs of the construction of pipelines.

                    :param: pipe_cost = Cost pipe [ILS/m] (list with 8 elements)
                            hRan= highest range value of the length of the pipes per specific diameter [m] (list with 7 elements)
                            length = length of the network in each catchment [m]
                    :return: pipe_inv_cost
    """

    if length < hRan[0]:
        pipe_inv_cost = length * pipe_cost[0]
    elif length < hRan[1]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (length - hRan[0]) * pipe_cost[1]
    elif length < hRan[2]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + \
            (hRan[1] - hRan[0]) * pipe_cost[1] + (length - hRan[1]) * pipe_cost[2]
    elif length < hRan[3]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
            length - hRan[2]) * pipe_cost[3]
    elif length < hRan[4]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
            hRan[3] - hRan[2]) * pipe_cost[3] + (length - hRan[3]) * pipe_cost[4]
    elif length < hRan[5]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
            hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (length - hRan[4]) * pipe_cost[5]
    elif length < hRan[6]:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
            hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (hRan[5] - hRan[4]) * pipe_cost[5] + (
            length - hRan[5]) * pipe_cost[6]
    else:
        pipe_inv_cost = hRan[0] * pipe_cost[0] + (hRan[1] - hRan[0]) * pipe_cost[1] + (hRan[2] - hRan[1]) * pipe_cost[2] + (
            hRan[3] - hRan[2]) * pipe_cost[3] + (hRan[4] - hRan[3]) * pipe_cost[4] + (hRan[5] - hRan[4]) * pipe_cost[5] + (
            hRan[6] - hRan[5]) * pipe_cost[6] + (length - hRan[6]) * pipe_cost[7]

    #            print(pipe_inv_cost)
    return pipe_inv_cost


def calculate_flows_reuse(area_per_reuse_per_buffer, flow_wwtp, type_treatment):

    type_treatment = np.array(type_treatment)
    flow_agriculture_reuse = []
    flow_urban_reuse = []
    flow_no_reuse = []
    for i in range(len(type_treatment)):
        if type_treatment[i] == 'cas with agr. reuse':
            flow_agriculture_reuse.append(flow_wwtp[i])
            flow_urban_reuse.append(0)
            flow_no_reuse.append(0)
        elif type_treatment[i] == 'mbr with urb. reuse':
            flow_agriculture_reuse.append(0)
            flow_urban_reuse.append(flow_wwtp[i])
            flow_no_reuse.append(0)
        else:
            flow_agriculture_reuse.append(0)
            flow_urban_reuse.append(0)
            flow_no_reuse.append(flow_wwtp[i])

    applied_flow_agriculture = []
    applied_flow_urban = []

    for i in range(len(flow_agriculture_reuse)):
        if area_per_reuse_per_buffer[i][0] == 0:
            applied_flow_agriculture.append(0)
        else:
            applied_flow_agriculture.append(
                flow_agriculture_reuse[i] / area_per_reuse_per_buffer[i][0])

    for i in range(len(flow_urban_reuse)):
        if area_per_reuse_per_buffer[i][1] == 0:
            applied_flow_urban.append(0)
        else:
            applied_flow_urban.append(flow_urban_reuse[i] / area_per_reuse_per_buffer[i][1])

    return flow_agriculture_reuse, flow_urban_reuse, flow_no_reuse, applied_flow_agriculture, applied_flow_urban


def benefits_freshwater_with_reuse(sum_pop, number_buffers, num_locations, flow_urban_reuse):
    """
             #   This function
#
             #       :param:
             #       :return:
    #"""

    benefits_freshwater_management = np.zeros((num_locations, number_buffers))

    for location_index in range(num_locations):
        for buffer_index in range(number_buffers):
            population = sum_pop[location_index][buffer_index]
            if flow_urban_reuse[buffer_index] == 0:
                volume_consumption = population * water_consumption[location_index] * 365
            else:
                volume_consumption = population * \
                    (water_consumption[location_index] - urb_irrig_use_coef *
                     water_consumption[location_index]) * 365
            volume_consumption = population * water_consumption[location_index] * 365
            total_benefits = volume_consumption * \
                tariff[location_index] * collection_efficiency[location_index]

            benefits_freshwater_management[location_index][buffer_index] += float(total_benefits)

    benefit_freshwater_per_subcatch = np.sum(benefits_freshwater_management, axis=0)
    return benefit_freshwater_per_subcatch


def benefits_reuse_selling(number_buffers, num_locations, flow_agriculture_reuse, flow_urban_reuse, coef_peak):

    benefits_agriculture = np.zeros(number_buffers)
    benefits_urban = np.zeros(number_buffers)

    for location_index in range(num_locations):
        for buffer_index in range(number_buffers):
            flow_agr = flow_agriculture_reuse[buffer_index]
            flow_urb = flow_urban_reuse[buffer_index]
            benefits_agr = flow_agr * 365 * \
                tariff_reclaimed[location_index] * collection_efficiency[location_index]
            benefits_urb = flow_urb * 365 * \
                tariff_reclaimed[location_index] * collection_efficiency[location_index]

            benefits_agriculture[buffer_index] += float(benefits_agr)
            benefits_urban[buffer_index] += float(benefits_urb)

    return benefits_agriculture, benefits_urban


def calculate_energy_savings(feature_grid_data, applied_irrigation_flow):
    """
                 #   This function
    #
                 #       :param:
                 #       :return:
        #"""

    # Analyse where each wwtp is placed
    town_in_wwtp_location = join_attribute_data_at_outlet(ordered_outlets=path_ordered_outlets,
                                                          feature_grid_data=feature_grid_data,
                                                          path_land_cost_data=path_population,
                                                          field_name='Loc_index',
                                                          no_data_value=no_data_value_loc_index)
    print('The treatment plant is located at the position: ', town_in_wwtp_location)

    energy_savings = []

    for i in range(len(applied_irrigation_flow)):
        if town_in_wwtp_location[i] == 0:
            energy_savings_for_town = applied_irrigation_flow[i] * c_source_substitute[0] * 365
            energy_savings.append(energy_savings_for_town)
        elif town_in_wwtp_location[i] == 1:
            energy_savings_for_town = applied_irrigation_flow[i] * c_source_substitute[1] * 365
            energy_savings.append(energy_savings_for_town)
        elif town_in_wwtp_location[i] == 2:
            energy_savings_for_town = applied_irrigation_flow[i] * c_source_substitute[2] * 365
            energy_savings.append(energy_savings_for_town)
        elif town_in_wwtp_location[i] == 3:
            energy_savings_for_town = applied_irrigation_flow[i] * c_source_substitute[3] * 365
            energy_savings.append(energy_savings_for_town)

    return energy_savings


def calculate_environmental_savings(applied_irrigation_flow, flow_no_reuse):
    """
                 #   This function
    #
                 #       :param:
                 #       :return:
        #"""

    environmental_savings = []
    for i in range(len(applied_irrigation_flow)):
        env_saving = (applied_irrigation_flow[i] -
                      flow_no_reuse[i]) * ww_treatment_after_discharge * 365
        environmental_savings.append(env_saving)

    return environmental_savings


def costs_freshwater_without_reuse(number_buffers, sum_pop, num_locations):
    """
             #   This function
#
             #       :param:
             #       :return:
    #"""

    total_costs_water_management = np.zeros((num_locations, number_buffers))

    # Volume of water consumption from each source according to user's input, per year

    for location_index in range(num_locations):
        for buffer_index in range(number_buffers):
            population = sum_pop[location_index][buffer_index]
            volume_consumption = population * water_consumption[location_index] * 365
            volume_consumption_s1 = volume_consumption * portion_s1[location_index] / 100
            volume_consumption_s2 = volume_consumption * portion_s2[location_index] / 100
            volume_consumption_s3 = volume_consumption * portion_s3[location_index] / 100
            # print('volume_water_consumption =', volume_consumption)

            # Conversion of consumption volume to distribution volumes
            volume_dist_s1 = volume_consumption_s1 / (1 - losses_distribution[location_index])
            volume_dist_s2 = volume_consumption_s2 / (1 - losses_distribution[location_index])
            volume_dist_s3 = volume_consumption_s3 / (1 - losses_distribution[location_index])
            total_volume_dist = volume_dist_s1 + volume_dist_s2 + volume_dist_s3
            # print('total_volume_dist  =', total_volume_dist )

            # Conversion of distribution volume to transmission volumes
            volume_tran_s1 = volume_dist_s1 / (1 - losses_transmission[location_index])
            volume_tran_s2 = volume_dist_s2 / (1 - losses_transmission[location_index])
            volume_tran_s3 = volume_dist_s3 / (1 - losses_transmission[location_index])
            total_volume_tran = volume_tran_s1 + volume_tran_s2 + volume_tran_s3
            # print('total_volume_tran  =', total_volume_tran)

            # Costs for water production related to each source
            costs_prod_s1 = volume_tran_s1 * c_prod_s1[location_index]
            costs_prod_s2 = volume_tran_s2 * c_prod_s2[location_index]
            costs_prod_s3 = volume_tran_s3 * c_prod_s3[location_index]
            total_costs_prod = costs_prod_s1 + costs_prod_s2 + costs_prod_s3
            # print('total_costs_prod  =', total_costs_prod)

            # Costs for water transmission related to each source
            costs_tran_s1 = volume_tran_s1 * c_tran_s1[location_index]
            costs_tran_s2 = volume_tran_s2 * c_tran_s2[location_index]
            costs_tran_s3 = volume_tran_s3 * c_tran_s2[location_index]

            # Costs for water transmission administration related to the whole transmited volume
            costs_tran_adm = total_volume_tran * c_tran_adm[location_index]
            total_costs_tran = costs_tran_s1 + costs_tran_s2 + costs_tran_s3 + costs_tran_adm
            # print('total_costs_tran  =', total_costs_tran)

            # Costs for water distribution related to to the whole distributed volume
            costs_dist_staff = total_volume_dist * c_dist_staff[location_index]
            costs_dist_energy = total_volume_dist * c_dist_energy[location_index]
            costs_dist_adm = total_volume_dist * c_dist_adm[location_index]
            total_costs_dist = costs_dist_staff + costs_dist_energy + costs_dist_adm

            total_costs_water = total_costs_prod + total_costs_tran + total_costs_dist

            total_costs_water_management[location_index][buffer_index] += float(total_costs_water)

    total_costs_water_per_subcatch_without_reuse = np.sum(total_costs_water_management, axis=0)
    return total_costs_water_per_subcatch_without_reuse


def costs_freshwater_with_reuse(sum_pop, number_buffers, num_locations, flow_urban_reuse, urb_irrig_use_coef):
    """
             #   This function
#
             #       :param:
             #       :return:
    #"""

    total_costs_water_management = np.zeros((num_locations, number_buffers))

    # Volume of water consumption from each source according to user's input, per year

    for location_index in range(num_locations):
        for buffer_index in range(number_buffers):
            population = sum_pop[location_index][buffer_index]
            if flow_urban_reuse[buffer_index] == 0:
                volume_consumption = population * water_consumption[location_index] * 365
            else:
                volume_consumption = population * \
                    (water_consumption[location_index] - urb_irrig_use_coef *
                     water_consumption[location_index]) * 365
            volume_consumption_s1 = volume_consumption * portion_s1[location_index] / 100
            volume_consumption_s2 = volume_consumption * portion_s2[location_index] / 100
            volume_consumption_s3 = volume_consumption * portion_s3[location_index] / 100
            # print('volume_water_consumption =', volume_consumption)

            # Conversion of consumption volume to distribution volumes
            volume_dist_s1 = volume_consumption_s1 / (1 - losses_distribution[location_index])
            volume_dist_s2 = volume_consumption_s2 / (1 - losses_distribution[location_index])
            volume_dist_s3 = volume_consumption_s3 / (1 - losses_distribution[location_index])
            total_volume_dist = volume_dist_s1 + volume_dist_s2 + volume_dist_s3
            # print('total_volume_dist  =', total_volume_dist )

            # Conversion of distribution volume to transmission volumes
            volume_tran_s1 = volume_dist_s1 / (1 - losses_transmission[location_index])
            volume_tran_s2 = volume_dist_s2 / (1 - losses_transmission[location_index])
            volume_tran_s3 = volume_dist_s3 / (1 - losses_transmission[location_index])
            total_volume_tran = volume_tran_s1 + volume_tran_s2 + volume_tran_s3
            # print('total_volume_tran  =', total_volume_tran)

            # Costs for water production related to each source
            costs_prod_s1 = volume_tran_s1 * c_prod_s1[location_index]
            costs_prod_s2 = volume_tran_s2 * c_prod_s2[location_index]
            costs_prod_s3 = volume_tran_s3 * c_prod_s3[location_index]
            total_costs_prod = costs_prod_s1 + costs_prod_s2 + costs_prod_s3
            # print('total_costs_prod  =', total_costs_prod)

            # Costs for water transmission related to each source
            costs_tran_s1 = volume_tran_s1 * c_tran_s1[location_index]
            costs_tran_s2 = volume_tran_s2 * c_tran_s2[location_index]
            costs_tran_s3 = volume_tran_s3 * c_tran_s2[location_index]

            # Costs for water transmission administration related to the whole transmited volume
            costs_tran_adm = total_volume_tran * c_tran_adm[location_index]
            total_costs_tran = costs_tran_s1 + costs_tran_s2 + costs_tran_s3 + costs_tran_adm
            # print('total_costs_tran  =', total_costs_tran)

            # Costs for water distribution related to to the whole distributed volume
            costs_dist_staff = total_volume_dist * c_dist_staff[location_index]
            costs_dist_energy = total_volume_dist * c_dist_energy[location_index]
            costs_dist_adm = total_volume_dist * c_dist_adm[location_index]
            total_costs_dist = costs_dist_staff + costs_dist_energy + costs_dist_adm

            total_costs_water = total_costs_prod + total_costs_tran + total_costs_dist

            total_costs_water_management[location_index][buffer_index] += float(total_costs_water)

    total_costs_water_per_subcatch = np.sum(total_costs_water_management, axis=0)
    return total_costs_water_per_subcatch


def calc_pumping_reuse_costs(radius_buffer, flow_wwtp, type_treatment, max_slope_area, diameter, f, h_suction,
                             h_outlet,
                             local_losses_coef, h_reservoir, density, pump_operation_time, price_average_kwh,
                             fixed_cost, tax_percent):

    costs_pumping_reuse_year = []
    for i in range(len(type_treatment)):
        if type_treatment[i] in ['no_wwtp', 'mbr with no reuse', 'mbr with no reuse data available']:
            costs_pumping_reuse_year.append(0)
        elif type_treatment[i] in ['cas with agr. reuse', 'mbr with urb. reuse']:
            distance = radius_buffer[i]
            delta_z = (max_slope_area / 100) * distance
            flow = flow_wwtp[i] / 24
            velocity = (flow / (pi * (diameter ** 2) / 4)) / 3600
            linear_headlosses = f * (distance / diameter) * ((velocity ** 2) / (2 * 9.81))
            local_losses = local_losses_coef * distance
            head_to_be_saved = delta_z + h_suction + h_outlet + h_reservoir + linear_headlosses + local_losses
            hydraulic_power = (flow * density * 9.81 * head_to_be_saved) / (3.6 * (10 ** 6))  # KW
            required_energy = pump_operation_time * hydraulic_power  # kwh
            cost_of_energy_month = required_energy * price_average_kwh * 30
            costs_with_fixed = cost_of_energy_month + fixed_cost
            taxes = costs_with_fixed * tax_percent
            total_costs_month = costs_with_fixed + taxes
            total_costs_year = 12 * total_costs_month
            costs_pumping_reuse_year.append(total_costs_year)

    return costs_pumping_reuse_year


def get_feature_grid(path_data, tmp_filename):
    ds = ogr.Open(path_data, gdalconst.GA_ReadOnly)
    layer = ds.GetLayer()

    file_path = os.path.join('temp', tmp_filename)
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


def onsite_treat_costs(total_pop, cluster_pop):

    # assumptions
    people_by_unit = 6.2 * 4  # person/septictank 4 people/building
    empty_frequency_year = 2  # times/year/septictank
    empty_costs = 1500  # ILS/time
    construction_costs = 8500  # ILS/unit

    onsite_pop = total_pop - cluster_pop
    num_units = onsite_pop/people_by_unit
    operation_costs_year = empty_frequency_year * empty_costs * num_units
    investment_costs = construction_costs * num_units

    return investment_costs, operation_costs_year, onsite_pop


def Join_Calcul(num_WWTP, coordinates):

    # WASTEWATER SYSTEM RELATED

    # Get the features index in the grid file, to be used in further calculations
    feature_grid_data = get_feature_grid(path_population, 'feature_grid')
    print('got grid_data')
    network_grid_data = get_feature_grid(path_channel, 'channel_grid')
    print('got channel_data')
    # Calculation of length of pipeline network
    network_length = join_attribute_network(
        path_channel, network_grid_data, path_subcatchments, 'LENGTH')

    print('Network length for each subcatchment =', network_length)

    # Calculation of pipeline for sewage network costs
    ww_pipe_inv_costs = calc_pipeline_ww_inv_cost(pipe_cost=Pw, hRan=hRan, length=network_length)
    print('Total cost of wastewater pipeline = ', ww_pipe_inv_costs, ' ILS')

    # Calculation of population in each subcatchment
    pop_per_town_per_subcatchment = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                                    feature_grid_data=feature_grid_data,
                                                                    polygonized_file=path_subcatchments,
                                                                    field_name='sumInhabit',
                                                                    location_index_field_name='Loc_index',
                                                                    num_types_data=4)
    print('Population for each town per subcatchment =', pop_per_town_per_subcatchment)
    pop_per_subcatchment = np.sum(pop_per_town_per_subcatchment, axis=0)
    print('Population per subcatchment =', pop_per_subcatchment)

    pop_per_town = np.sum(pop_per_town_per_subcatchment, axis=1)

    print('Population per town =', pop_per_town)

    total_pop = np.sum(pop_per_town_per_subcatchment)

    print('Total Population supplied =', total_pop)

    # Calculation of flow at each treatment plant
    flow_wwtp = calculate_flow_at_wwtp(pop_per_town_subcatchment=pop_per_town_per_subcatchment,
                                       water_consumption=water_consumption,
                                       L=losses_infiltration,
                                       p=coef_peak,
                                       num_locations=4,
                                       num_WWTP=num_WWTP)
    print('Flows at each treatment plant = ', flow_wwtp, 'm3/day')

    # Calculate flow generated (without the design peak consideration)

    ww_flow_generated = []
    for i in range(len(flow_wwtp)):
        flow_generation = flow_wwtp[i] / coef_peak
        ww_flow_generated.append(flow_generation)

    print('The generated wastewater flow (without peak) = ', ww_flow_generated)

    # Calculation of benefit of connection fees
    benefit_connection = calc_benefit_connections(pop_per_catchment=pop_per_subcatchment,
                                                  inhabit_per_household=inhabit_per_household,
                                                  fee_connection=fee_connection)

    print('Benefit_connection = ', benefit_connection)

    # Calculation of benefit of sanitary fees
    benefit_ww_fees = calc_benefit_ww_yearly_fees(pop_per_town_per_subcatchment=pop_per_town_per_subcatchment,
                                                  fee_sanitary=fee_sanitary,
                                                  water_consumption=water_consumption,
                                                  num_WWTP=num_WWTP,
                                                  num_locations=num_locations)

    print('Benefit_ww_fees = ', benefit_ww_fees)

    # RECLAIMED WATER SYSTEM RELATED

    # Order outlets so they can have the same index as the poygons (subcatchments)
    # old path_temp_outlets before path_pourpoints...
    reorder_outlets(path_pour_points_shp, path_ordered_outlets, path_subcatchments)
    # print 'Outlets are reordered'

    # Calculate radius of influence of each wwtp
    radii_wwtp, \
        applied_irrigation_flows, \
        types_treatment_flow_based, \
        areas_per_reuse_per_buffer = find_optimal_radii(flow_at_each_wwtp=ww_flow_generated,
                                                        # vol_area_ratio=vol_area_ratio, (deleted, it is not being used for anything)
                                                        inputfn=path_ordered_outlets,
                                                        outputBufferfn=None,
                                                        feature_grid_data=feature_grid_data)

    # Draw buffers around wwtp based on calculated radius
    createBuffer(inputfn=path_ordered_outlets,
                 outputBufferfn=path_buffer_areas, bufferDist=radii_wwtp)

    # Count the number of buffers
    number_buffers = count_features(path_buffer_areas)

    # Calculate the area of each buffer
    areas_buffer_total = calculate_area_buffer(path_to_shapefile=path_buffer_areas,
                                               number_features=number_buffers)
    # print 'The area of each buffer is', areas_buffer_total

    print("The area per type of reuse per buffer = ", areas_per_reuse_per_buffer)

    print("The type of treatment at each wwtp = ", types_treatment_flow_based)

    print("The radius for reuse at each treatment plant = ", radii_wwtp, 'meters')

    print("The flow applied for irrigation around each wwtp = ", applied_irrigation_flows, 'm3/d')

    # Calculate pipeline network inside of buffer area (reuse network)
    reuse_network_length_theoric = join_attribute_network(
        path_channel, network_grid_data, path_buffer_areas, 'Length')

    print('Network length for each reuse area would be =', reuse_network_length_theoric)

    # Calculate pipeline for reuse network costs
    reuse_network_length_theoric = np.array(reuse_network_length_theoric)
    reuse_network_costs = []
    for i in range(len(types_treatment_flow_based)):
        if types_treatment_flow_based[i] in ['no_wwtp', 'mbr with no reuse']:
            reuse_network_costs.append(0)
        elif types_treatment_flow_based[i] in ['cas with agr. reuse', 'mbr with urb. reuse']:
            reuse_network_costs.append(
                calc_pipeline_reuse_inv_cost(pipe_cost=Pw, hRan=hRan, length=reuse_network_length_theoric[i]))
    print('Costs of reuse network pipeline = ', reuse_network_costs, 'a total  = ', sum(
        reuse_network_costs), 'ILS')

    # WASTEWATER TREATMENT/RECLAMATION SYSTEM RELATED

    # Costs of treatment plants
    flow_wwtp = np.array(flow_wwtp)
    types_treatment_flow_based = np.array(types_treatment_flow_based)

    treatment_costs = []
    for i in range(len(types_treatment_flow_based)):
        if types_treatment_flow_based[i] == 'no_wwtp':
            treatment_costs.append(0)
        elif types_treatment_flow_based[i] == 'cas with agr. reuse':
            treatment_costs.append(
                wwtp_costs(flow_wwtp[i], known_flow_cas_1, known_flow_cas_2, known_inv_cost_cas_1, known_inv_cost_cas_2,
                           known_oem_cost_cas_1, known_oem_cost_cas_2, DR, n))
        elif types_treatment_flow_based[i] in ['mbr with urb. reuse', 'mbr with no reuse']:
            treatment_costs.append(
                wwtp_costs(flow_wwtp[i], known_flow_mbr_1, known_flow_mbr_2, known_inv_cost_mbr_1,
                           known_inv_cost_mbr_2, known_oem_cost_mbr_1, known_oem_cost_mbr_2, DR, n))
    print('Treatment costs with wwtps varying between MBR and CAS (investment and OeM) = ', treatment_costs, 'a total = ', sum(
        treatment_costs), 'ILS')

    # Costs of land purchase

    land_cost_in_wwtp_location = join_attribute_data_at_outlet(ordered_outlets=path_ordered_outlets,
                                                               feature_grid_data=feature_grid_data,
                                                               path_land_cost_data=path_population,
                                                               field_name='Land_price',
                                                               no_data_value=no_data_value_land_price)
    print('Land cost in each wwtp area = ', land_cost_in_wwtp_location)

    land_cost_by_wwtp = get_land_cost_by_wwtp(
        land_cost_in_wwtp_location, flow_wwtp, types_treatment_flow_based)

    print('Land purchase cost for each wwtp =', land_cost_by_wwtp,
          ' ILS, a total = ', sum(land_cost_by_wwtp))

    # Reservoir costs
    # Scale factor calculation for known reservoir
    sc_reservoir = calculate_scale_factor(Inv_1=known_inv_cost_reservoir_1, Inv_2=known_inv_cost_reservoir_2,
                                          Q1=known_reservoir_volume_1, Q2=known_reservoir_volume_2)
    # print 'The scale factor for the reservoirs is', sc_reservoir
    reuse_reservoir_costs = []
    for i in range(len(types_treatment_flow_based)):
        if types_treatment_flow_based[i] in ['no_wwtp', 'mbr with no reuse', 'mbr with no reuse data available']:
            reuse_reservoir_costs.append(0)
        elif types_treatment_flow_based[i] in ['cas with agr. reuse', 'mbr with urb. reuse']:
            reuse_reservoir_costs.append(
                ((flow_wwtp[i] * 1.25 / known_reservoir_volume_1) ** (sc_reservoir)) * known_inv_cost_reservoir_1)
    print('Costs of investment for the reservoirs = ', reuse_reservoir_costs)

    # Calculate flow for agriculture, reuse and no reuse

    flow_agriculture_reuse, flow_urban_reuse, flow_no_reuse, applied_flow_agriculture, applied_flow_urban = calculate_flows_reuse(
        area_per_reuse_per_buffer=areas_per_reuse_per_buffer,
        flow_wwtp=ww_flow_generated,
        type_treatment=types_treatment_flow_based)
    print('Flow used for agriculture in each buffer = ', flow_agriculture_reuse)
    print('Flow used for urban reuse in each buffer = ', flow_urban_reuse)
    print('Flow not used for reuse in each buffer = ', flow_no_reuse)
    print('Ratio m3/m2  used for agriculture in each buffer = ', applied_flow_agriculture)
    print('Ratio m3/m2 used for urban reuse in each buffer = ', applied_flow_urban)

    costs_reuse_pumping = calc_pumping_reuse_costs(radius_buffer=radii_wwtp, flow_wwtp=flow_wwtp,
                                                   type_treatment=types_treatment_flow_based,
                                                   max_slope_area=max_slope_area, diameter=diameter, f=f,
                                                   h_suction=h_suction, h_outlet=h_outlet,
                                                   local_losses_coef=local_losses_coef,
                                                   h_reservoir=h_reservoir, density=density,
                                                   pump_operation_time=pump_operation_time,
                                                   price_average_kwh=price_average_kwh, fixed_cost=fixed_cost,
                                                   tax_percent=tax_percent)

    print('Costs with reuse pumping, per year, = ', costs_reuse_pumping)

    # Costs with Onsite sanitation

    onsite_inv_costs, onsite_oem_year, onsite_pop = onsite_treat_costs(total_population, total_pop)

    pv_oem_onsite = present_value(DR, n, onsite_oem_year)

    print('Costs with construction of septic tanks (ILS)= ', onsite_inv_costs)
    print('Costs with operation of septic tanks (ILS/year) = ', onsite_oem_year)
    print('Population supplied with onsite treatment (inhabitants) = ', onsite_pop)

    # Benefits onsite sanitation

    benefit_connection_onsite = (onsite_pop / int(6.2)) * 1500
    print(benefit_connection_onsite)
    benefit_ww_fees_onsite_year = onsite_pop * 0.07 * 1.5 * 365
    print(benefit_ww_fees_onsite_year)

    pv_fees_onsite = present_value(DR, n, benefit_ww_fees_onsite_year)
    print(pv_fees_onsite)

    # FRESHWATER MANAGEMENT RELATED

    # Calculation of population in each buffer
    pop_per_town_per_buffer = join_attribute_data_in_boundary(path_to_datafile=path_population,
                                                              feature_grid_data=feature_grid_data,
                                                              polygonized_file=path_buffer_areas,
                                                              field_name='sumInhabit',
                                                              location_index_field_name='Loc_index',
                                                              num_types_data=4)

    print('Population in each town in each buffer = ', pop_per_town_per_buffer)

    # Costs with freshwater management in the buffer area (with reuse included as source)

    total_costs_freshwater_without_reuse = costs_freshwater_without_reuse(number_buffers=number_buffers,
                                                                          sum_pop=pop_per_town_per_buffer,
                                                                          num_locations=4)

    print('Costs in the buffers, in a year, with water management, WITHOUT reuse = ',
          total_costs_freshwater_without_reuse)

    total_costs_water_per_buffer = costs_freshwater_with_reuse(sum_pop=pop_per_town_per_buffer, number_buffers=number_buffers,
                                                               num_locations=4, flow_urban_reuse=flow_urban_reuse,
                                                               urb_irrig_use_coef=urb_irrig_use_coef)
    print('Costs in the buffers, in a year, with water management,considering reuse = ',
          total_costs_water_per_buffer)

    # Benefit with freshwater selling with_reuse
    benefit_freshwater_per_subcatch = benefits_freshwater_with_reuse(sum_pop=pop_per_town_per_buffer,
                                                                     number_buffers=number_buffers, num_locations=4,
                                                                     flow_urban_reuse=flow_urban_reuse)
    print('Benefit received, in a year, from the benefit of freshwater selling = ',
          benefit_freshwater_per_subcatch)

    # Benefit with reclaimed water
    benefits_agriculture, benefits_urban = benefits_reuse_selling(number_buffers=number_buffers,
                                                                  num_locations=4,
                                                                  flow_agriculture_reuse=flow_agriculture_reuse,
                                                                  flow_urban_reuse=flow_urban_reuse,
                                                                  coef_peak=coef_peak)

    print('Benefit received, in a year, for reuse selling for agriculture = ', benefits_agriculture)
    print('Benefit received, in a year, for reuse selling for urban use = ', benefits_urban)

    # Benefit as energy savings

    benefit_energy_savings = calculate_energy_savings(feature_grid_data=feature_grid_data,
                                                      applied_irrigation_flow=(applied_irrigation_flows))
    print('The equivalent in energy savings, by year = ', benefit_energy_savings)

    # Benefit as environmental savings

    environmental_savings = calculate_environmental_savings(applied_irrigation_flow=applied_irrigation_flows,
                                                            flow_no_reuse=flow_no_reuse)

    print('The equivalent in wastewater tariff saved is =', environmental_savings)

    # Total benefits

    total_benefit_ww = sum(benefit_connection) + present_value(DR=DR,
                                                               n=n, c_op_year=sum(benefit_ww_fees))

    print('Total benefit wastewater = ', total_benefit_ww)

    benefits_reclaimed_water_management = present_value(DR=DR, n=n, c_op_year=sum(benefits_agriculture)) + \
        present_value(DR=DR, n=n, c_op_year=sum(benefits_urban))

    print('Total benefit reclaimed wastewater = ', benefits_reclaimed_water_management)

    benefits_freshwater_management = present_value(
        DR=DR, n=n, c_op_year=sum(benefit_freshwater_per_subcatch))

    print('Total benefit freshwater management  = ', benefits_freshwater_management)

    benefits_energy_savings = present_value(DR=DR, n=n, c_op_year=sum(benefit_energy_savings))

    print('Total benefits of energy savings =  ', benefits_energy_savings)

    benefits_env_savings = present_value(DR=DR, n=n, c_op_year=sum(environmental_savings))

    print('Total benefits of environmental savings =  ', benefits_env_savings)

    total_benefits_onsite = benefit_connection_onsite + pv_fees_onsite

    total_benefits = total_benefit_ww + benefits_reclaimed_water_management + benefits_freshwater_management + benefits_energy_savings + benefits_env_savings \
        + total_benefits_onsite

    print("The total benefits are = ", total_benefits)

    # Total costs

    total_costs_ww = sum(ww_pipe_inv_costs) + sum(treatment_costs) + sum(land_cost_by_wwtp)

    print("The total costs with wastewater is = ", total_costs_ww)

    total_costs_reclaimed_ww = sum(reuse_reservoir_costs) + sum(reuse_network_costs) + \
        present_value(DR=DR, n=n, c_op_year=sum(costs_reuse_pumping))

    print("The total costs with reclaimed ww management is = ", total_costs_reclaimed_ww)

    total_costs_freshwater_mngt = present_value(
        DR=DR, n=n, c_op_year=sum(total_costs_water_per_buffer))

    print("The total costs with costs freshwater in the buffer is = ", total_costs_freshwater_mngt)

    total_costs_onsite = onsite_inv_costs + pv_oem_onsite

    total_costs = total_costs_ww + total_costs_reclaimed_ww + total_costs_freshwater_mngt \
        + total_costs_onsite

    print("The total costs are = ", total_costs)

    costs_minus_benefits = total_costs - total_benefits

    print("The costs - benefits = ", costs_minus_benefits)

    # Calculation of coverage with cluster level sanitation
    coverage_region = coverage_check(
        pop_per_subcatchment=pop_per_subcatchment, real_pop=total_population)

    print('The coverage of the wastewater network will be =', int(
        coverage_region * 100), '% of the total population in the region')

    coverage_catchment = coverage_check(
        pop_per_subcatchment=pop_per_subcatchment, real_pop=population_catchment)

    coverage_onsite = onsite_pop / total_population

    print('The coverage of onsite sanitation will be of =', int(
        coverage_onsite * 100), '% of the total population in the region')

    print('The coverage of the wastewater network will be =', int(
        coverage_catchment * 100), '% of the catchment')

    # Saving results to excel csv file

    partial_results = {'Geographic coordinates (UTM)': list(coordinates),
                       'Network length of ww pipeline (m)': list(network_length),
                       'Cost of wastewater pipeline (ILS)': list(ww_pipe_inv_costs),
                       'Population supplied by one WWTP (inhab)': list(pop_per_subcatchment),
                       # 'Population per town (inhab)': pop_per_town,
                       'Flows at each subcatchment (m3/d)': list(flow_wwtp),
                       'Generated ww flow (without peak) (m3/d)': list(ww_flow_generated),
                       'Benefit from connections (ILS)': list(benefit_connection),
                       'Benefit from ww fees (ILS / year)': list(benefit_ww_fees),
                       # 'Area per type of reuse per subcatchment (m2 - Agriculture, Urban, No reuse)': areas_per_reuse_per_buffer,
                       'Type of treatment of each wwtp': list(types_treatment_flow_based),
                       'Radius for reuse at each wwtp (m)': list(radii_wwtp),
                       'Flow applied in irrigation (m3/d)': list(applied_irrigation_flows),
                       'Length of reuse pipeline distribution (m)': list(reuse_network_length_theoric),
                       'Costs of reuse network pipeline (ILS)': list(reuse_network_costs),
                       'Treatment costs (ILS)': list(treatment_costs),
                       'Land costs in each wwtp area (ILS / m2)': list(land_cost_in_wwtp_location),
                       'Land purchase costs at each WWTP (ILS) ': list(land_cost_by_wwtp),
                       'Costs of reservoir investment (ILS)': list(reuse_reservoir_costs),
                       'Flow used for agriculture (m3 / d)': list(flow_agriculture_reuse),
                       'Flow used for urban reuse (m3/d)': list(flow_urban_reuse),
                       'Flow not used for reuse (m3/d)': list(flow_no_reuse),
                       'Ratio m3 / m2  used for agriculture in each buffer': list(applied_flow_agriculture),
                       'Ratio m3/m2 used for urban reuse in each buffer': list(applied_flow_urban),
                       'Costs with reuse pumping (ILS / year)': list(costs_reuse_pumping),
                       'Costs in the buffers with water management - WITHOUT reuse (ILS/year)': list(total_costs_freshwater_without_reuse),
                       'Costs in the buffers with water management - considering reuse (ILS/year)': list(total_costs_water_per_buffer),
                       'Benefit received from the benefit of freshwater selling (ILS/year)': list(benefit_freshwater_per_subcatch),
                       'Benefit received (in a year) for reuse selling for agriculture (ILS / year)': list(benefits_agriculture),
                       'Benefit received (in a year) for reuse selling for urban use  (ILS/year)': list(benefits_urban),
                       'Equivalent in energy savings (ILS/year)': list(benefit_energy_savings),
                       'Equivalent in wastewater tariff saved (ILS/year)': list(environmental_savings)
                       }

    # print partial_results
    df = pd.DataFrame(data=partial_results)
    # print df
    export_csv = df.to_csv(path_partial_results, index=None, header=True)

    total_results = {'Total population supplied (inhab)': total_pop,
                     'Benefit wastewater (Total with PV summed)': total_benefit_ww,
                     'Total benefit reclaimed wastewater (total PV)': benefits_reclaimed_water_management,
                     'Total benefit freshwater management (total PV)': benefits_freshwater_management,
                     'Total benefits of energy savings (total PV)': benefits_energy_savings,
                     'Total benefits of environmental savings (total PV)': benefits_env_savings,
                     'Total benefits  (ILS)': total_benefits,
                     'Total costs with wastewater (ILS PV)': total_costs_ww,
                     'Total costs with reclaimed ww management (ILS)': total_costs_reclaimed_ww,
                     'Total costs with costs freshwater in the buffer (ILS)': total_costs_freshwater_mngt,
                     'Total costs (ILS)': total_costs,
                     'Costs - benefits (ILS)': total_costs - total_benefits,
                     'Coverage (region)': coverage_region,
                     'Coverage (town)': coverage_catchment,
                     'Benefits/costs': (total_benefits / total_costs), 'Onsite construction costs ': onsite_inv_costs,
                     'Onsite OeM costs (year) ': onsite_oem_year,
                     'Onsite OeM costs (total) ': pv_oem_onsite,
                     'Benefits onsite treatment (connection fee) ': benefit_connection_onsite,
                     'Benefits onsite treatment (sanitation fee) ': pv_fees_onsite,
                     'Population supplied with onsite treatment (inhabitants)': onsite_pop
                     }
    df_total = pd.DataFrame(data=total_results, index=[0])
    # print total_results
    export_csv = df_total.to_csv(path_total_results, index=None, header=True)

    return total_benefits, total_costs,  coverage_catchment


def reset_files():
    for folder in [path_temp, path_results, path_outputs]:
        for root_path, _, files in os.walk(folder):
            for filename in files:
                filepath = os.path.join(root_path, filename)
                os.remove(filepath)


def AOKP(coordinates):
    # reset result, output and remporary files
    reset_files()

    create_pour_points(coordinates)

    # COMMENT FOR NOW, FIXIN OF pourpoints-outletpoints
    # shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    # if os.path.exists(path_temp_outlet):
    #     shpdriver.DeleteDataSource(path_temp_outlet)

    # rt.delineate_watershed(dem_uri=path_filled,
    #                        outlet_shapefile_uri=path_pour_points,
    #                        snap_distance=snap_distance,
    #                        flow_threshold=flow_threshold,
    #                        watershed_out_uri=path_watershed,
    #                        snapped_outlet_points_uri=path_temp_outlet,
    #                        stream_out_uri=path_stream)
    # print ("Watersheds delineation ran")

    ########################### CONVERTION TO Py3 ################################
    # convert path_watershed from shp to gpkg
    # path_watershed_converted = os.path.join(root, "temp",'output_watershed.gpkg')
    # run(['ogr2ogr', '-f', 'GPKG', path_watershed_converted, path_watershed])
    flow_dir_path = os.path.join(root, "temp", 'flow_dir_d8.tif')
    pygeoprocessing.routing.flow_dir_d8((path_filled, 1), flow_dir_path)
    print('flow direction successfull')
    print('path_pour_points is ', path_pour_points_shp)
    print('path_watershed is ', path_watershed)
    print('flow_dir_path is ', flow_dir_path)
    gdal.UseExceptions()
    pygeoprocessing.routing.delineate_watersheds_d8(
        (flow_dir_path, 1), path_pour_points_shp, path_watershed)

    print("Watersheds delineation ran")

    # convert watersheds.gpkg into watershed.shp
    run(['ogr2ogr', '-f', "ESRI Shapefile", path_watershed_folder, path_watershed])
    print("watershed converted from gpkg to shp")

    ########################### CONVERTION TO Py3 ################################

    attribute_areas()
    print("Watersheds areas were attributed to shapefile")

    ds, ordered_watershed = order_watershed()
    print("Watersheds are ordered in a decreasing order")

    rasterize_watershed(ordered_watershed)
    print("Watersheds are rasterized")

    # release layer
    ds.ReleaseResultSet(ordered_watershed)

    polygonize_watershed()
    print("Watersheds are polygonized and converted in vector layer")

    num_WWTP = count_features(path_pour_points_shp)  # path_temp_outlet
    print(('Outlet.shp has {n} WWTPs'.format(n=num_WWTP)))

    num_subcatchments = count_features(path_subcatchments)
    print(('Polygonized.shp has {n} catchments'.format(n=num_subcatchments)))

    # %% CHECK

    # In this cell the number of sub-catchmnets present in polygon.shp is
    # compared with the number of pour points (points in outlet.shp) insert
    # as input. This will tell us if all the sub-catchments were delineated along
    # the previous steps.

    #                                   'COMPARISON'
    if num_WWTP == num_subcatchments:
        print("All the sub-catchments were delineated")
        total_benefits, total_costs,  coverage_catchment = Join_Calcul(num_WWTP, coordinates)

        ben_cost_ratio = total_benefits/total_costs

        print('BENEFITS/COSTS = ', ben_cost_ratio, ' COVERAGE = ', coverage_catchment * 100, '%')

    else:
        print(("Error in the watershed delineation"
              'Abundancing(-) or missing(+) of {n} sub-catchments'.format(n=num_WWTP - num_subcatchments), sys.exc_info()[1]))
        total_benefits = 0
        total_costs = 0
        coverage_catchment = 0

        ben_cost_ratio = 0

    return (ben_cost_ratio, coverage_catchment)


### FUNCTIONS USED IN THE OPTIMIZATION FILES ###

def random_combination(matrix, size):
    seen = set()
    n = len(matrix)
    while True:
        new_sample = tuple(sorted(random.sample(range(n), size)))
        if new_sample not in seen:
            seen.add(new_sample)
            yield tuple(matrix[i] for i in new_sample)


class LimitedBinary(Binary):
    def __init__(self, nbits, min_num_true=1, max_num_true=None):
        super(LimitedBinary, self).__init__(nbits)

        if max_num_true is None:
            max_num_true = nbits

        self.min_num_true = min_num_true
        self.max_num_true = max_num_true
        # print('self.min_num_true is ', self.min_num_true)

    def rand(self):
        num_true = np.random.randint(self.min_num_true, self.max_num_true)
        # print('num_true is ', num_true)
        true_indices = np.random.choice(list(range(self.nbits)), size=num_true, replace=False)

        return [(index in true_indices) for index in range(self.nbits)]
