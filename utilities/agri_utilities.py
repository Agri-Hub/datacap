import datacube
import math
import calendar
import ipywidgets
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import colors as mcolours
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime,date, timedelta
from pyproj import Proj, transform
from datacube.utils import geometry
from datacube.utils.geometry import CRS,Geometry
from datacube.model import Range
import shapely
from shapely.geometry import shape
import fiona
import rasterio.features
from fiona.crs import from_epsg
import os,sys
from hdstats import nangeomedian_pcm
from tqdm import tqdm
import pandas as pd
import gdal
import fnmatch
import csv
import json
import stat
import types
from glob import glob
import shutil
import time
import zipfile
import requests
import logging
import hashlib
import re
import socket
from dateutil.relativedelta import relativedelta

from utilities.utils import *


def show_options():
    classification_type = ipywidgets.Select(
        options=['Pixel-Based', 'Object-Based'],
        description='Type',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )
    sentinel1 = ipywidgets.Checkbox(description='Use Sentinel-1',)
    sentinel2 = ipywidgets.Checkbox(description='Use Sentinel-2',)
    text1 = ipywidgets.Text(description='Shapefile Path',)
    text2 = ipywidgets.Text(description='Name of "ID" column',)
    text3 = ipywidgets.Text(description='Name of "Crop Type" column',)
    box = ipywidgets.VBox([classification_type,sentinel1,sentinel2,text1,text2,text3])
    return box


def get_dates(timeStart,timeEnd):

    query = {
        'time': (timeStart,timeEnd),
        'product': 's2_preprocessed_v2',
    }
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bands = ['B02']#,'B03','B04','B05','B06','B07','B08','B8A','B11','B12','ndvi','ndwi','psri']
    data = dc.load(**query,dask_chunks={})

    dates = []
    for day in list(data.time.values):
        a = str(day-np.timedelta64(1,'D'))[:10]
        b = str(day+np.timedelta64(1,'D'))[:10]
        dates.append((a,b))
    return dates

def generate_feature_space_preload(dates,filename,outfile,colID,colType,timeStart,timeEnd,samples = None,classficationType='object',sentinel1=True,sentinel2=True):
    '''
    Generates a pixel-based or object-based feature space in the format of csv
    :param filename: the full path of the shapefile
    :param outfile: the full path of the feature space csv to be generated
    :param colID: the column name that holds the id for each parcel
    :param colType: the column name that holds the crop code for each parcel
    :param timeStart: starting date of acquitions in the YYYY-MM-DD format
    :param timeEnd: ending date of acquitions in the YYYY-MM-DD format
    :param classficationType: the choice whether the classification will be pixel-based or object-based. Valid values 'o' or 'p'
    :param sentinel1: boolean value for generating or not features based on sentinel-1 products
    :param sentinel2: boolean value for generating or not features based on sentinel-2 products
    '''
    from osgeo import osr,ogr

    if filename is None:
        sys.exit('No filename has been given')
    if colID is None:
        sys.exit('No column for ID has been given')
    if colType is None:
        sys.exit('No column for crop type has been given')

    if not os.path.exists(filename):
        sys.exit('File does not exist')

    for day in dates:
        query = {
                'time': (day[0], day[1]),
                'product': 's2_preprocessed_v2',
        }
        dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
        bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','ndvi','ndwi','psri']

        print("Loading data for range {}...please wait approximately 3-7 minutes".format(day[0]+'to'+day[1]))
        data = dc.load(**query)
        print("Data has been loaded")

        data['ndvi'] = calculate_index(data,'ndvi')
        data['ndwi'] = calculate_index(data,'ndmi')
        data['psri'] = calculate_index(data,'psri')

        outfile = outfile + str(data.time.values[0])[:10] + '.csv'



        if sentinel2:

            ras = gdal.Open('/data2/netherlands/s2/2017/31UFT/03/24/March242017_B05.tif')
            gt = ras.GetGeoTransform()
            inv_gt = gdal.InvGeoTransform(gt)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataSource = driver.Open(filename, 0)
            ds = dataSource.GetLayer()

            parcels = {}
            headings = [colID]
            iterations = 0
            parcel_data = {}
            for f in tqdm(ds):

                id = int(f['id'])

                geom = f.GetGeometryRef()
                geom = geom.ExportToWkt()
                vect_tmp_drv = ogr.GetDriverByName('MEMORY')
                vect_tmp_src = vect_tmp_drv.CreateDataSource('')
                vect_tmp_lyr = vect_tmp_src.CreateLayer('', srs, ogr.wkbPolygon)
                vect_tmp_lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

                feat = ogr.Feature(vect_tmp_lyr.GetLayerDefn())
                feat.SetField("id", id)
                feat_geom = ogr.CreateGeometryFromWkt(geom)
                feat.SetGeometry(feat_geom)
                vect_tmp_lyr.CreateFeature(feat)

                xmin, xmax, ymin, ymax = feat_geom.GetEnvelope()

                off_ulx, off_uly = map(int, gdal.ApplyGeoTransform(inv_gt, xmin, ymax))
                off_lrx, off_lry = map(int, gdal.ApplyGeoTransform(inv_gt, xmax, ymin))
                rows, columns = (off_lry - off_uly) + 1, (off_lrx - off_ulx) + 1

                ras_tmp = gdal.GetDriverByName('MEM').Create('', columns, rows, 1, gdal.GDT_Byte)
                ras_tmp.SetProjection(ras.GetProjection())
                ras_gt = list(gt)
                ras_gt[0], ras_gt[3] = gdal.ApplyGeoTransform(gt, off_ulx, off_uly)
                ras_tmp.SetGeoTransform(ras_gt)

                gdal.RasterizeLayer(ras_tmp, [1], vect_tmp_lyr, burn_values=[1])
                mask = ras_tmp.GetRasterBand(1).ReadAsArray()

                aa = off_uly
                bb = off_lry + 1
                cc = off_ulx
                dd = off_lrx + 1

                iterations += 1

                if samples is not None and samples == iterations:
                    break


                parcels[f[colID]] = {}
                values = [f[colID]]

                if data is None or geom is None:
                    continue


                for band in bands:
                    if band == "SCL":
                        continue
                    for i in range(data[band].shape[0]):
                        if iterations == 1:
                            headings.append(band+'_'+str(data.time[i].values).split('.')[0][:10])
                        try:
                            cloud_indices = np.where(np.logical_and(data['SCL'][i].values==3,data['SCL'][i].values>7))
                            data[band][i].values[aa:bb,cc:dd][cloud_indices] = np.nan
                            if (np.all(data[band][i].values[aa:bb,cc:dd]==np.nan)):
                                values.append(np.nan)
                            else:
                                values.append(round(np.nanmean(data[band][i].values[aa:bb,cc:dd]),3))
                        except Exception as e:
                            print(e)
                            values.append(np.nan)

                if iterations == 1:
                    headings.append('CropType')

                values.append(f[colType])
                parcels[f[colID]] = values

            df = pd.DataFrame.from_dict(parcels, orient='index')
            df.to_csv(outfile,header=headings)



def generate_feature_space_coherence(filename,outfile,colID,colType,timeStart,timeEnd,samples=None,classficationType='object'):
    '''
    Generates a pixel-based or object-based feature space in the format of csv
    :param filename: the full path of the shapefile
    :param outfile: the full path of the feature space csv to be generated
    :param colID: the column name that holds the id for each parcel
    :param colType: the column name that holds the crop code for each parcel
    :param timeStart: starting date of acquitions in the YYYY-MM-DD format
    :param timeEnd: ending date of acquitions in the YYYY-MM-DD format
    :param classficationType: the choice whether the classification will be pixel-based or object-based. Valid values 'o' or 'p'
    '''
    from osgeo import osr,ogr

    if filename is None:
        sys.exit('No filename has been given')
    if colID is None:
        sys.exit('No column for ID has been given')
    if colType is None:
        sys.exit('No column for crop type has been given')

    if not os.path.exists(filename):
        sys.exit('File does not exist')


    query = {
            'time': (timeStart, timeEnd),
            'product': 'sentinel1_coherence',
    }
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bands = ['vv','vh','vv_vh']

    print("Loading data for range {}...please wait approximately 3-7 minutes".format(timeStart+' to '+timeEnd))
    data = dc.load(**query)
    print("Data has been loaded")

    if(len(data)==0):
        return

    data['vv_vh'] = data.vv / data.vh

    ws = outfile
    outfile = outfile + 'coherence_' + str(timeStart) + '_to_' + str(timeEnd) + '.csv'

    if True:
        for rasterfile in os.listdir(os.path.join(ws,'s1',timeStart[:4],'coherence')):
            if '.tif' in rasterfile:
                basemap = rasterfile
                break
        if basemap is None:
            return

        ras = gdal.Open(os.path.join(ws,'s1',timeStart[:4],'coherence',basemap))
        gt = ras.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(filename, 0)
        ds = dataSource.GetLayer()

        parcels = {}
        headings = [colID]
        iterations = 0
        parcel_data = {}
        for f in tqdm(ds):

            id = int(f['id'])

            geom = f.GetGeometryRef()
            geom = geom.ExportToWkt()
            vect_tmp_drv = ogr.GetDriverByName('MEMORY')
            vect_tmp_src = vect_tmp_drv.CreateDataSource('')
            vect_tmp_lyr = vect_tmp_src.CreateLayer('', srs, ogr.wkbPolygon)
            vect_tmp_lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

            feat = ogr.Feature(vect_tmp_lyr.GetLayerDefn())
            feat.SetField("id", id)
            feat_geom = ogr.CreateGeometryFromWkt(geom)
            feat.SetGeometry(feat_geom)
            vect_tmp_lyr.CreateFeature(feat)

            xmin, xmax, ymin, ymax = feat_geom.GetEnvelope()

            off_ulx, off_uly = map(int, gdal.ApplyGeoTransform(inv_gt, xmin, ymax))
            off_lrx, off_lry = map(int, gdal.ApplyGeoTransform(inv_gt, xmax, ymin))
            rows, columns = (off_lry - off_uly) + 1, (off_lrx - off_ulx) + 1

            ras_tmp = gdal.GetDriverByName('MEM').Create('', columns, rows, 1, gdal.GDT_Byte)
            ras_tmp.SetProjection(ras.GetProjection())
            ras_gt = list(gt)
            ras_gt[0], ras_gt[3] = gdal.ApplyGeoTransform(gt, off_ulx, off_uly)
            ras_tmp.SetGeoTransform(ras_gt)

            gdal.RasterizeLayer(ras_tmp, [1], vect_tmp_lyr, burn_values=[1])
            mask = ras_tmp.GetRasterBand(1).ReadAsArray()

            aa = off_uly
            bb = off_lry + 1
            cc = off_ulx
            dd = off_lrx + 1

            iterations += 1

            if samples is not None and samples == iterations:
                break


            parcels[f[colID]] = {}
            values = [f[colID]]

            if data is None or geom is None:
                continue


            for band in bands:
                for i in range(data[band].shape[0]):
                    if iterations == 1:
                        headings.append(band+'_mean_'+str(data.time[i].values).split('.')[0][:10])
                        headings.append(band+'_std_'+str(data.time[i].values).split('.')[0][:10])

                    values.append(round(np.nanmean(data[band][i].values[aa:bb,cc:dd]),3))
                    values.append(round(np.nanstd(data[band][i].values[aa:bb,cc:dd]),3))


            if len(values)==1:
                continue
            if iterations == 1:
                headings.append('CropType')

            values.append(f[colType])
            parcels[f[colID]] = values

        df = pd.DataFrame.from_dict(parcels, orient='index')
        df.to_csv(outfile,header=headings)





def generate_feature_space(filename,outfile,colID,colType,timeStart,timeEnd,samples = None,classficationType='object',sentinel1=True,sentinel2=True):
    '''
    Generates a pixel-based or object-based feature space in the format of csv.
    It is strongly advised to use this function when the number of parcels is <= 1000.
    :param filename: the full path of the shapefile
    :param outfile: the full path of the feature space csv to be generated
    :param colID: the column name that holds the id for each parcel
    :param colType: the column name that holds the crop code for each parcel
    :param timeStart: starting date of acquitions in the YYYY-MM-DD format
    :param timeEnd: ending date of acquitions in the YYYY-MM-DD format
    :param classficationType: the choice whether the classification will be pixel-based or object-based. Valid values 'o' or 'p'
    :param sentinel1: boolean value for generating or not features based on sentinel-1 products
    :param sentinel2: boolean value for generating or not features based on sentinel-2 products
    '''

    if filename is None:
        sys.exit('No filename has been given')
    if colID is None:
        sys.exit('No column for ID has been given')
    if colType is None:
        sys.exit('No column for crop type has been given')

    if not os.path.exists(filename):
        sys.exit('File does not exist')


    if sentinel2:


        # bands = ['B02','B03_10m','B04_10m','B05_20m','B06_20m','B07_20m','B08_10m','B8A_20m','B09_60m','B11_20m','B12_20m','SCL_20m']
        bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']

        # open shapefile
        ds = fiona.open(filename)
        crs = geometry.CRS(ds.crs_wkt)



        parcels = {}
        headings = [colID]
        iterations = 0
        for f in tqdm(ds):


            feature_geom = f['geometry']
            geom = Geometry(geom=feature_geom,crs=crs)
            bounds = shape(feature_geom).bounds
            crop_type = f['properties'][colType]
            print
            if 'MULTIPOLYGON' in geom.wkt:
                continue

            iterations += 1

            if samples is not None and samples == iterations:
                break

            query = {
                    'geopolygon': geom,
                    'time': (timeStart, timeEnd),
                    'product': 's2_preprocessed_v2'
            }

            parcels[f[colID]] = {}
            values = [f[colID]]

            # data = dc.load(output_crs="EPSG:3857",resolution=(-10,10),measurements=bands,**query)

            data = dc.load(measurements=bands,**query)

            if data is None or geom is None:
                continue

            data['ndvi'] = calculate_index(data,'ndvi')
            data['ndwi'] = calculate_index(data,'ndmi')
            data['psri'] = calculate_index(data,'psri')



            mask = geometry_mask([geom], data.geobox, invert=True)
            data = data.where(mask)
            data = data.where(data.SCL != 3)
            data = data.where(data.SCL <= 7)
            for band in bands:
                raw_data = data[band].values
                for i in range(raw_data.shape[0]):
                    if iterations == 1:
                        headings.append(band+'_'+str(data.time[i].values).split('.')[0][:10])
                    try:
                        if (np.all(raw_data[i]==np.nan)):
                            values.append(np.nan)
                        else:
                            values.append(round(np.nanmean(raw_data[i]),3))
                    except:
                        values.append(-9999)
            if iterations == 1:
                headings.append('CropType')
            values.append(crop_type)
            parcels[f[colID]] = values

        df = pd.DataFrame.from_dict(parcels, orient='index')
        df.to_csv(outfile,header=headings)


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)

def display_rgb(img,alpha=1., figsize=(10, 10)):
    # rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis=-1)
    rgb = rgb/rgb.max() * alpha
    plt.figure(figsize=figsize)
    plt.imshow(rgb)


def read_shapefile(shape_file='/data2/cyprus36SWDparcels3857.shp', threshold=0, selected_ids=None,product="s2a_sen2cor_granule"):
    ds = fiona.open(shape_file)
    crs = geometry.CRS(ds.crs_wkt)
    cnt = 0
    geometries = []
    parcels_data = []
    for f in ds:
        feature_geom = f['geometry']
        geom = Geometry(feature_geom, crs)
        bounds = shape(feature_geom).bounds
        if 'MULTIPOLYGON' in geom.wkt:
            continue

        if threshold == 0:
            geometries.append(geom)
        elif selected_ids == None:
            geometries.append(geom)
            cnt += 1
        else:
            if f['id'] in selected_ids:
                geometries.append(geom)
        if threshold > 0 and cnt == threshold:
            break
    return geometries


def check_index(index):
    if index.lower() not in ['ndvi', 'ndwi', 'ndmi', 'psri', 'savi']:
        print("Error in name of index. Calculation for '{}' is not supported.".format(index))
        return False
    return True


def calculate_index(data, index):
    if index.lower() == 'ndvi':
        return (data.B08 - data.B04) / (data.B08 + data.B04)
    elif index.lower() == 'ndwi':
        return (data.B08 - data.B03) / (data.B08 + data.B03)
    elif index.lower() == 'ndmi':
        return (data.B08 - data.B11) / (data.B08 + data.B11)
    elif index.lower() == 'psri':
        return (data.B04 - data.B02) / data.B06
    else:
        return None


def plot_index_timeseries(geom, index='ndvi', start_time='2019-01-01', end_time='2019-12-31', cloud_free_percentage=60):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    query = {
        'geopolygon': geom,
        'time': (start_time, end_time),
        'product': "s2a_sen2cor_granule"
    }
    if not check_index(index):
        return None

    print("Please wait for data loading from the cube...")
    data = dc.load(output_crs="EPSG:3857", measurements=['B04_10m', 'B08_10m', 'SCL_20m'], resolution=(-10, 10),**query)
    print("Loading has been finished")
    mask = geometry_mask([geom], data.geobox, invert=True)
    data = data.where(mask)

    data[index] = calculate_index(data, index)

    if cloud_free_percentage == 0:
        data.ndvi.plot(col='time', vmin=-1, vmax=1, cmap='YlGn', col_wrap=6)
        del data
        return

    all_pixels = np.count_nonzero(~np.isnan(data['ndvi'][0].values))
    data[index][np.where(np.logical_or(data['SCL']==3,data['SCL']>7))] = np.nan

#     fig = plt.figure(figsize=(120, 70))

    cols = 2
    rows = len(data.time.values) // cols
    if len(data.time.values) % cols > 0:
        rows += 1
    cloudy_images = []
    real_i = 0

    total = 0
    for i in range(len(data.time.values)):
        img = data['ndvi'][i].values
        free_pixels =  np.count_nonzero(~np.isnan(data['ndvi'][i].values))
        cloud_perc = 100.0 - free_pixels / all_pixels * 100.0
        if cloud_perc > cloud_free_percentage:
            cloudy_images.append(str(data.time.values[i])[:10] + '(' + str(round(cloud_perc,3)) + ')')
            continue
        total += 1

    rows = total // cols
    if total % cols > 0:
        rows += 1

    cloudy_images = []
    real_i = 0

    fig, axarr = plt.subplots(rows,cols,figsize=(20, 7))

    row = 0
    col = 0
    for i in range(len(data.time.values)):
        img = data['ndvi'][i].values
        free_pixels =  np.count_nonzero(~np.isnan(data['ndvi'][i].values))
        cloud_perc = 100.0 - free_pixels / all_pixels * 100.0
        if cloud_perc > cloud_free_percentage:
            cloudy_images.append(str(data.time.values[i])[:10] + '(' + str(round(cloud_perc,3)) + ')')
            continue
#         fig.add_subplot(rows, cols,real_i + 1)
        axarr[row,col].imshow(img, vmin=-0.5, vmax=1, cmap='YlGn')
#         plt.imshow(img, vmin=-0.5, vmax=1, cmap='YlGn')
#         plt.title('NDVI for ' + str(data.time.values[i])[:10])
        axarr[row,col].set_title('NDVI for ' + str(data.time.values[i])[:10])
        col += 1
        if col == cols:
           row += 1
           col = 0
        real_i += 1
#     plt.show()
    print("Excluded images due to high cloud percentage:")
    for item in cloudy_images:
        print(item)
    del data,dc


def create_rgb(geom,start_date='2019-01-01',end_date='2019-02-28'):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    query = {
        'geopolygon': geom,
        'time': (start_date,end_date),
        'product': "s2a_sen2cor_granule"
    }
    data = dc.load(output_crs="EPSG:3857", measurements=['B02_10m', 'B03_10m', 'B04_10m'], resolution=(-10, 10), **query)
    mask = geometry_mask([geom], data.geobox, invert=True)
    data = data.where(mask)
    timestamps = data.time.values
    rgb = np.stack((data['B02_10m'].values, data['B03_10m'].values, data['B04_10m'].values), axis=2)
    del data,dc
    return rgb,timestamps


def plot_rgb(rgb,timestamps,width=50,height=50,cols=6,rows=2):
    fig = plt.figure(figsize=(50, 50))
    cols = 6
    rows = 2
    for i in range(len(timestamps)):
        img = rgb[:, :, :, i]
        array_min, array_max = np.nanmin(img), np.nanmax(img)
        img = (img - array_min) / (array_max - array_min)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(str(timestamps[i])[:10], fontsize=35)


def plot_rgb_geomedian(rgb,width=50,height=50,cols=6,rows=2):
    fig = plt.figure(figsize=(50, 50))
    cols = 6
    rows = 2
    for i in range(1):
        img = rgb
        array_min, array_max = np.nanmin(img), np.nanmax(img)
        img = (img - array_min) / (array_max - array_min)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title("Geomedian RGB", fontsize=35)



def cloud_trend(geom,start_time='2019-01-01', end_time='2019-12-31'):

    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    query = {
        'geopolygon': geom,
        'time': (start_time, end_time),
        'product': "s2a_sen2cor_granule"
    }
    print("Please wait for data loading from the cube...")
    data = dc.load(output_crs="EPSG:3857", measurements=['SCL_20m'], resolution=(-10, 10),**query)
    print("Data loading has been finished...")

    mask = geometry_mask([geom], data.geobox, invert=True)
    data = data.where(mask)
    all_pixels = np.count_nonzero(~np.isnan(data['SCL_20m'][0].values))

    cloud_data = {}
    for i in range(len(data.time.values)):
        img = data['SCL_20m'][i].values
        img[np.logical_or(data['SCL_20m'][i].values < 4, data['SCL_20m'][i].values > 6)] = -9999
        cloudy_pixels = np.count_nonzero(img == -9999)
        cloud_perc = cloudy_pixels / all_pixels * 100.0
        cloud_perc = cloudy_pixels / all_pixels * 100.0
        cloud_data[str(data.time.values[i])[:10]] = cloud_perc

    del data, dc
    return cloud_data



def preprocess_s2(ws_tmp,clip=False,bbox=[]):

    months = {'01': 'January_all', '02': 'February_all', '03': 'March_all', '04': 'April_all', '05': 'May_all',
              '06': 'June_all', '07': 'July_all', '08': 'August_all', '09': 'September_all', '10': 'October_all',
              '11': 'November_all', '12': 'December_all'}


    bands = ['B02_10m.tif', 'B03_10m.tif', 'B04_10m.tif', 'B05_10m.tif', 'B06_10m.tif', 'B07_10m.tif', 'B08_10m.tif',
             'B8A_10m.tif', 'B11_10m.tif', 'B12_10m.tif', 'SCL_10m.tif','TCI_10m.tif']


    # if clip = True please get the bounds (3857 projection)
    xmin,ymin,xmax,ymax = bbox[0],bbox[1],bbox[2],bbox[3]

    logging.info('Starting the preprocess of Sentinel-2 products...')
    for mon in os.listdir(ws_tmp):
        mon_path = os.path.join(ws_tmp, mon)
        # transforming (converting and resampling) each band (10m and 20m) of SAFE product to geotiff
        for safe in glob(os.path.join(mon_path, '*.SAFE')):
            p = Sentinel2_pre_process(mon_path, safe)
            p.converting_rasters()
            p.delete_jp2_files()
        logging.info('Transforming (converting and resampling of basic rasters) process has been successfully finished')

        for b in bands:
            for safe in glob(os.path.join(mon_path, '*.SAFE')):
                rasterdate = safe[-14:-12] + safe[-20:-16]
                fulldate = safe.split('_MSIL2A_')[1][:8]
                corr_year = safe.split('_MSIL2A_')[1][:4]
                corr_day = safe.split('_MSIL2A_')[1][6:8]
                correct_date = corr_day+corr_year
                for root, dirs, files in os.walk(mon_path):
                    if fulldate in root:
                        for tif in fnmatch.filter(files, '*' + b):
                            gdal.Warp(os.path.join(mon_path, months[mon].split('_')[0] + correct_date + '_all' + '_' +b.split('_')[0] + '.tif'), os.path.join(root, tif),
                                      dstSRS='EPSG:3857',xRes=10,yRes=10)
        logging.info('Reproject with GDAL and SRC files of %s files has been successfully generated',
                    months[mon].split('_')[0])

        # delete L2A files
        for safe in glob(os.path.join(mon_path, '*.SAFE')):
            shutil.rmtree(safe)
        logging.info('L2A files removing process has been successfully finished')

        if clip==1:
            # clip mosaic images
            for tif in glob(os.path.join(mon_path, '*.tif')):
                if '_TCI' in tif:
                    x = tif.split("_all_")[0]
                    xx = x + "_RGB.tif"
                    cmd = "gdalwarp -te %f %f %f %f %s %s" % (xmin, ymin, xmax, ymax, tif, xx)
                    os.system(cmd)
                    os.remove(tif)
                elif not "_RGB" in tif:
                    out_tif = os.path.split(tif)[1].split('_all')[0] + os.path.split(tif)[1].split('_all')[1]
                    # print out_tif
                    Sentinel2_pre_process.clip_raster(os.path.dirname(tif), tif, out_tif, xmin=xmin, ymin=ymin,
                                                      xmax=xmax,
                                                      ymax=ymax)
                    os.remove(tif)
            logging.info('Clipping process has been successfully finished')

        else:
            #without clip process
            for tif in glob(os.path.join(mon_path, '*.tif')):
                if '_TCI' in tif:
                    x = tif.split("_all_TCI")[0]
                    xx = x + "_RGB.tif"
                    os.rename(tif, xx)
                elif 'all' in tif:
                    x = tif.split('_all')
                    xx = x[0] + x[1]
                    os.rename(tif, xx)

        dates = []
        for tif in glob(os.path.join(mon_path, '*.tif')):
            rasterdate = tif.split('.')[0].split('/')[-1].split('_')[0][-6:-4]
            print(rasterdate)
            if rasterdate not in dates:
                dates.append(rasterdate)

        for date in dates:
            os.mkdir(os.path.join(mon_path, date))
            for tif in glob(os.path.join(mon_path, '*.tif')):
                rasterdate = tif.split('.')[0].split('/')[-1].split('_')[0][-6:-4]
                if rasterdate == date:
                    cmd = 'echo %s | sudo -S mv %s %s' % ('userdev', tif, os.path.join(mon_path, date))
                    os.system(cmd)
        for date in dates:
            mon_path2 = os.path.join(mon_path, date)
            s2 = Sentinel2_process(mon_path2)
            s2.maskcloudvegatation()
            # s2.create_ndvi()
            # s2.create_psri()
            # s2.create_ndwi()
            # s2.create_savi()
            # s2.create_bori()
            # s2.create_bari()

            # delete 8bit files
            for tif_8bit in glob(os.path.join(mon_path2, '*8bit.tif')):
                    os.remove(tif_8bit)
            logging.info('Removing process of intermediate files (8bit rasters) has been successfully finished')


def download_s2(tiles,start_date,end_date,outdir,username,password,cloud_cover='90'):
    cloud_cover = str(cloud_cover)
    for tile_name in tiles:
        query_url = 'https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?maxRecords=10&rocessingLevel=LEVEL2A&productIdentifier=%25'+tile_name+'%25&startDate='+start_date+'T00%3A00%3A00Z&completionDate='+end_date+'T23%3A59%3A59Z&sortParam=startDate&sortOrder=descending&status=0%7C34%7C37&dataset=ESA-DATASET&cloudCover=%5B0%2C'+cloud_cover+'%5D&'

        initial_outdir = outdir

        def _get_next_page(links):
            for link in links:
                if link['rel'] == 'next':
                    return link['href']
            return False

        query_response = {}
        while query_url:
            response = requests.get(query_url)
            response.raise_for_status()
            data = response.json()
            for feature in data['features']:
                query_response[feature['id']] = feature
            query_url = _get_next_page(data['properties']['links'])

        import shutil
        from pathlib import Path
        import concurrent.futures
        from multiprocessing.pool import ThreadPool

        import requests
        # from tqdm import tqdm

        DOWNLOAD_URL = 'https://zipper.creodias.eu/download'
        TOKEN_URL = 'https://auth.creodias.eu/auth/realms/DIAS/protocol/openid-connect/token'

        def _get_token(username, password):
            token_data = {
                'client_id': 'CLOUDFERRO_PUBLIC',
                'username': username,
                'password': password,
                'grant_type': 'password'
            }
            response = requests.post(TOKEN_URL, data=token_data).json()
            try:
                return response['access_token']
            except KeyError:
                raise RuntimeError('Unable to get token. Response was {response}')



        ids = [result['id'] for result in query_response.values()]

        i = 0
        for id in ids:
            i+=1
            time.sleep(0.1)
            outdir = initial_outdir
            identifier = query_response[id]['properties']['productIdentifier'].split('/')[-1]
            if 'MSIL1C' in identifier:
                continue
            sensing_date = identifier.split('.')[0].split('_')[2]
            sensing_day = sensing_date[-2:]
            sensing_year = sensing_date[:4]
            sensing_month = sensing_date[4:6]
            size = query_response[id]['properties']['services']['download']['size']/1024.0/1024.0
            token = _get_token(username, password)
            url = '{}/{}?token={}'.format(DOWNLOAD_URL,id,token)
            outdir = outdir + '/'+sensing_year
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + tile_name
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir +'/'+sensing_month
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            print(outdir)
            outfile = Path(outdir) / '{}.zip'.format(identifier)
            if os.path.exists(outfile):
                continue
            outfile_temp = str(outfile) + '.incomplete'
            try:
                downloaded_bytes = 0
                print(url)
                req =  requests.get(url, stream=True, timeout=100)
                chunk_size = 2 ** 20  # download in 1 MB chunks
                with open(outfile_temp, 'wb') as fout:
                    for chunk in req.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            fout.write(chunk)
                            downloaded_bytes += len(chunk)
                shutil.move(outfile_temp, str(outfile))
            finally:
                try:
                    Path(outfile_temp).unlink()
                except OSError:
                    pass
