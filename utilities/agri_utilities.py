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
# from hdstats import nangeomedian_pcm
from tqdm import tqdm
import pandas as pd
import gdal
#from utils import *
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
import operator
import time
sys.path.append('/home/noa/.snap/snap-python')
import snappy
from snappy import GPF
from snappy import ProductIO
from snappy import HashMap
from snappy import jpy
import subprocess
from snappy import WKTReader
from snappy import File
from snappy import ProgressMonitor
from time import *
import datetime as dt
from osgeo import osr,ogr


def change_colnames(df):
    cols = ["id"] + [x.split("_")[1] + "_" + str(datetime.strptime(x.split("_")[0], '%B%d%Y').timetuple().tm_yday) for x in df.columns[1:]]
    return cols

def doy_to_date(doy, year):
    return datetime.strptime(str(doy) + year, '%j%Y')

def get_band(df, band_name, with_id=False):
    bands = [x for x in df.columns if band_name in x]
    if with_id:
        bands = ["id"] + bands
    return df.loc[:, bands]

def simple_daily_interpolation(df, name, start_doy, end_doy, year='2021', interp_method='linear', ):
    n_cols = df.shape[1]

    df.index = df.index.astype(int)
    df.columns = [pd.to_datetime(x.split("_")[-1]) for x in df]

    date_of_doy = doy_to_date(end_doy, year)
    if df.columns[-1] < date_of_doy:
        df[date_of_doy] = np.nan
    dfT = df.T
    dfT = dfT.resample('1d').asfreq()

    df_daily = dfT.T.interpolate(interp_method, axis=1).ffill(axis=1).bfill(axis=1)
#     df_daily.columns = df_daily.columns.map(lambda t: "{}_{}".format(name, t.timetuple().tm_yday))
    df_daily = df_daily[df.columns]
    df_daily.columns = df_daily.columns.map(lambda t: "{}_mean_{}".format(name, t.date()))
    return df_daily


def daily_fs(fs, year, start_doy, end_doy, bandnames, s1 = False, s1_names = [], has_id=False, keep_init = True):
    band_list = []
#     print("Interpolation...")
    for b in tqdm(bandnames):
        band_df = get_band(fs, b)
        band_df = simple_daily_interpolation(band_df, b, start_doy, end_doy, year)
        cols = [x for x in band_df.columns if start_doy <= pd.to_datetime(x.split("_")[-1]).dayofyear <= end_doy]
        band_df = band_df[cols]
        band_list.append(band_df)
    
    if s1:
        band_list.append(fs.filter(regex = 'vv|vh'))
    fs_daily = pd.concat(band_list, axis=1, join='inner')
    if has_id:
        fs_daily.insert(0, 'id', fs["id"])
    return fs_daily


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

def generate_feature_space_preload(filename,outfile_dir,colID,
                                   timeStart,timeEnd,samples = None,
                                   classficationType='object',sentinel1=True,sentinel2=True):
    '''
    Generates a pixel-based or object-based feature space in the format of csv
    :param filename: the full path of the shapefile
    :param outfile: the full path of the feature space csv to be generated
    :param colID: the column name that holds the id for each parcel
    :param timeStart: starting date of acquitions in the YYYY-MM-DD format
    :param timeEnd: ending date of acquitions in the YYYY-MM-DD format
    :param classficationType: the choice whether the classification will be pixel-based or object-based. Valid values 'o' or 'p'
    :param sentinel1: boolean value for generating or not features based on sentinel-1 products
    :param sentinel2: boolean value for generating or not features based on sentinel-2 products
    '''

    bands_indices = [0,1,2,3,4,5,6,7,8,9,11,12,13]
    if filename is None:
        sys.exit('No filename has been given')
    if colID is None:
        sys.exit('No column for ID has been given')
    # if colType is None:
    #     sys.exit('No column for crop type has been given')

    if not os.path.exists(filename):
        sys.exit('File does not exist')

    dates = get_dates(timeStart, timeEnd)
    for day in dates:
        stime = time()
        query = {
                'time': (day[0], day[1]),
                'product': 's2_preprocessed_v2',
        }
        dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
        bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','ndvi','ndwi','psri']

        #print("Loading data for range {}...Patience is a great asset!".format(day[0]+' to '+day[1]))
        data = dc.load(**query)

        data['ndvi'] = calculate_index(data,'ndvi')
        data['ndwi'] = calculate_index(data,'ndmi')
        data['psri'] = calculate_index(data,'psri')

        for index in bands:
            data[index] = data[index].where(((data['SCL']>=3) & (data['SCL']<=7)), np.nan)

        outfile = outfile_dir + 'fs' + str(data.time.values[0])[:10] + '.csv'
        data = data.to_array()
        data.loc['SCL'] = (data.loc['SCL']>=3) & (data.loc['SCL']<7)
        #print("Data has been loaded")
        cloud_free_ratio = data[10].values.sum() / (data.shape[2]*data.shape[3])
        if cloud_free_ratio < 0.05:
            print("Cloud coverage for {} is {}%, thus this date is skipped.".format(str(pd.to_datetime(data.time[0].values).date()), 100-cloud_free_ratio*100))
            continue
        if sentinel2:

            ras = gdal.Open('/data2/netherlands/s2/2017/31UFT_clipped/03/27/March272017_B02.tif')
            gt = ras.GetGeoTransform()

            inv_gt = gdal.InvGeoTransform(gt)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataSource = driver.Open(filename, 0)
            ds = dataSource.GetLayer()

            parcels = {}
            iterations = 0

            for f in tqdm(ds):

                geom = f.GetGeometryRef()
                geom = geom.ExportToWkt()

                vect_tmp_drv = ogr.GetDriverByName('MEMORY')
                vect_tmp_src = vect_tmp_drv.CreateDataSource('')
                vect_tmp_lyr = vect_tmp_src.CreateLayer('', srs, ogr.wkbPolygon)
                vect_tmp_lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

                feat = ogr.Feature(vect_tmp_lyr.GetLayerDefn())
                feat.SetField("id", f['id'])
                feat_geom = ogr.CreateGeometryFromWkt(geom)
                feat.SetGeometry(feat_geom)
                vect_tmp_lyr.CreateFeature(feat)

                xmin, xmax, ymin, ymax = feat_geom.GetEnvelope()

                off_ulx, off_uly = map(int, gdal.ApplyGeoTransform(inv_gt, xmin, ymax))
                off_lrx, off_lry = map(int, gdal.ApplyGeoTransform(inv_gt, xmax, ymin))

                # Specify offset and rows and columns to read

                rows, columns = (off_lry - off_uly) + 1, (off_lrx - off_ulx) + 1

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


                ras_tmp = gdal.GetDriverByName('MEM').Create('', columns, rows, 1, gdal.GDT_Byte)
                ras_tmp.SetProjection(ras.GetProjection())
                ras_gt = list(gt)
                ras_gt[0], ras_gt[3] = gdal.ApplyGeoTransform(gt, off_ulx, off_uly)
                ras_tmp.SetGeoTransform(ras_gt)

                gdal.RasterizeLayer(ras_tmp, [1], vect_tmp_lyr, burn_values=[1])
                mask = ras_tmp.GetRasterBand(1).ReadAsArray()
                parcel_data = data[:,0,aa:bb,cc:dd].where(mask)
                parcel_size = mask.sum()
                cloudfree_parcel_size = parcel_data[10].values.sum()
                cloud_coverage = cloudfree_parcel_size / (parcel_size + 1e-7)

                if cloud_coverage < 30:
                    values.extend([np.nan for _ in range(len(bands))])
                else:
                    values.extend(parcel_data[bands_indices].mean(axis = (1,2)).round(3).values)

                if len(values)==1:
                    continue

                parcels[f[colID]] = values
            headings = [colID] + [(band+'_mean_'+str(pd.to_datetime(data.time[0].values).date())) for band in bands]
            df = pd.DataFrame.from_dict(parcels, orient='index')
            df.to_csv(outfile,header=headings,index=False)
            print("Time elapsed for creating feature space for {} is {}s".format(str(pd.to_datetime(data.time[0].values).date()), stime-time()))

def generate_feature_space_preload_indices(dates,filename,outfile,colID,colType,timeStart,timeEnd,samples = None,classficationType='object',sentinel1=True,sentinel2=True):
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
        bands = ['ndvi','ndwi','psri']

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
                if iterations == 1000:
                    break
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
            df.to_csv(outfile,header=headings,index = False)


def generate_feature_space_backscatter(filename,outfile,colID,colType,timeStart,timeEnd,samples=None,classficationType='object'):
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
            'product': 'sentinel1',
    }
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bands = ['vv','vh']

    print("Loading data for range {}...please wait approximately 3-7 minutes".format(timeStart+' to '+timeEnd))
    data = dc.load(**query)
    print("Data has been loaded")

    if(len(data)==0):
        print("Empty data")
        return

    ws = outfile
    outfile = outfile + 'fs_sar_' + str(timeStart) + '_to_' + str(timeEnd) + '.csv'

    if True:
        for rasterfile in os.listdir(os.path.join(ws,'s1',timeStart[:4],'backscatter')):
            if '.tif' in rasterfile:
                basemap = rasterfile
                break

        if basemap is None:
            return
        print("Using ", basemap, " as basemap image")
        ras = gdal.Open(os.path.join(ws,'s1',timeStart[:4],'backscatter',basemap))
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
        df.to_csv(outfile,header=headings,index = False)



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
                        headings.append('coherence_'+band+'_mean_'+str(data.time[i].values).split('.')[0][:10])
                        headings.append('coherence_'+band+'_std_'+str(data.time[i].values).split('.')[0][:10])

                    values.append(round(np.nanmean(data[band][i].values[aa:bb,cc:dd]),3))
                    values.append(round(np.nanstd(data[band][i].values[aa:bb,cc:dd]),3))


            if len(values)==1:
                continue
            if iterations == 1:
                headings.append('CropType')

            values.append(f[colType])
            parcels[f[colID]] = values

        df = pd.DataFrame.from_dict(parcels, orient='index')
        df.to_csv(outfile,header=headings,index = False)


def generate_mask(outfile_dir, filename, colID = 'id', timeStart = '2017-03-01', timeEnd = '2017-10-30', method = 'object', sentinel2=True):
    bands_indices = [0,1,2,3,4,5,6,7,8,9,11,12,13]
    if filename is None:
        sys.exit('No filename has been given')
    if colID is None:
        sys.exit('No column for ID has been given')
    # if colType is None:
    #     sys.exit('No column for crop type has been given')

    if not os.path.exists(filename):
        sys.exit('File does not exist')

    dates = get_dates(timeStart, timeEnd)
    parcels = {}
    iterations = 0
    headings = []
    for day in dates:
        stime = time()
        query = {
                'time': (day[0], day[1]),
                'product': 's2_preprocessed_v2',
        }
        dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")

        #print("Loading data for range {}...Patience is a great asset!".format(day[0]+' to '+day[1]))
        data = dc.load(measurements=['SCL'], **query)


        outfile = outfile_dir + 'mask' + str(data.time.values[0])[:10] + '.csv'
        data = data.to_array()
        data.loc['SCL'] = (data.loc['SCL']>=3) & (data.loc['SCL']<7)
        cloud_free_ratio = data[0].values.sum() / (data.shape[2]*data.shape[3])
        if cloud_free_ratio < 0.05:
            print("Cloud coverage for {} is {}%, thus this date is skipped.".format(str(pd.to_datetime(data.time[0].values).date()), 100-cloud_free_ratio*100))
            continue
        headings.append(str(pd.to_datetime(data.time[0].values.date())))
        if sentinel2:

            ras = gdal.Open('/data2/netherlands/s2/2017/31UFT_clipped/03/27/March272017_B02.tif')
            gt = ras.GetGeoTransform()

            inv_gt = gdal.InvGeoTransform(gt)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            dataSource = driver.Open(filename, 0)
            ds = dataSource.GetLayer()

            for f in tqdm(ds):

                geom = f.GetGeometryRef()
                geom = geom.ExportToWkt()

                vect_tmp_drv = ogr.GetDriverByName('MEMORY')
                vect_tmp_src = vect_tmp_drv.CreateDataSource('')
                vect_tmp_lyr = vect_tmp_src.CreateLayer('', srs, ogr.wkbPolygon)
                vect_tmp_lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

                feat = ogr.Feature(vect_tmp_lyr.GetLayerDefn())
                feat.SetField("id", f['id'])
                feat_geom = ogr.CreateGeometryFromWkt(geom)
                feat.SetGeometry(feat_geom)
                vect_tmp_lyr.CreateFeature(feat)

                xmin, xmax, ymin, ymax = feat_geom.GetEnvelope()

                off_ulx, off_uly = map(int, gdal.ApplyGeoTransform(inv_gt, xmin, ymax))
                off_lrx, off_lry = map(int, gdal.ApplyGeoTransform(inv_gt, xmax, ymin))

                # Specify offset and rows and columns to read

                rows, columns = (off_lry - off_uly) + 1, (off_lrx - off_ulx) + 1

                aa = off_uly
                bb = off_lry + 1
                cc = off_ulx
                dd = off_lrx + 1


                if iterations == 0:
                    parcels[f[colID]] = []

                if data is None or geom is None:
                    continue


                ras_tmp = gdal.GetDriverByName('MEM').Create('', columns, rows, 1, gdal.GDT_Byte)
                ras_tmp.SetProjection(ras.GetProjection())
                ras_gt = list(gt)
                ras_gt[0], ras_gt[3] = gdal.ApplyGeoTransform(gt, off_ulx, off_uly)
                ras_tmp.SetGeoTransform(ras_gt)

                gdal.RasterizeLayer(ras_tmp, [1], vect_tmp_lyr, burn_values=[1])
                mask = ras_tmp.GetRasterBand(1).ReadAsArray()
                parcel_data = data[:,0,aa:bb,cc:dd].where(mask)
                parcel_size = mask.sum()
                cloudfree_parcel_size = (parcel_data[0].values == 1).sum()
                cloud_free_ratio = cloudfree_parcel_size / (parcel_size + 1e-7)

                parcels[f[colID]].append(cloud_free_ratio)
        iterations += 1

    df = pd.DataFrame.from_dict(parcels, orient='index')
    df.to_csv(outfile_dir,header=headings,index=False)
    return df


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    return rasterio.features.geometry_mask([geom.to_crs(geobox.crs) for geom in geoms],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)

def display_rgb(img,alpha=1., figsize=(10, 10)):
    # rgb = np.stack([img[b_r], img[b_g], img[b_b]], axis=-1)
    rgb = img/img.max() * alpha
    plt.figure(figsize=figsize)
    plt.imshow(rgb)


def read_shapefile(shape_file, exhaustive = False, selected_ids = None, threshold=1):
    ds = fiona.open(shape_file)
    crs = geometry.CRS(ds.crs_wkt)
    cnt = 0
    geometries = []
    ids = []
    for f in ds:
        feature_id = f['properties']['id']
        feature_geom = f['geometry']
        geom = Geometry(feature_geom, crs)
#       bounds = shape(feature_geom).bounds
#       if 'MULTIPOLYGON' in geom.wkt:
#             continue
        if selected_ids is None:
            geometries.append(geom)
            ids.append(feature_id)
            cnt += 1
            if exhaustive:
                continue
            elif cnt == threshold:
                return geometries, ids
        elif feature_id in selected_ids:
            geometries.append(geom)
            ids.append(feature_id)
            selected_ids.remove(feature_id)
            if len(selected_ids) == 0:
                return geometries, ids

def read_shapefile_simple(shape_file, exhaustive = False, selected_ids = None, threshold=1):
    ds = fiona.open(shape_file)
    crs = geometry.CRS(ds.crs_wkt)
    cnt = 0
    geometries = []
    ids = []
    for f in ds:
        feature_id = f['properties']['id']
        geom = f['geometry']
#       bounds = shape(feature_geom).bounds
#       if 'MULTIPOLYGON' in geom.wkt:
#             continue
        if selected_ids is None:
            geometries.append(geom)
            ids.append(feature_id)
            cnt += 1
            if exhaustive:
                continue
            elif cnt == threshold:
                return geometries, ids
        elif feature_id in selected_ids:
            geometries.append(geom)
            ids.append(feature_id)
            selected_ids.remove(feature_id)
            if len(selected_ids) == 0:
                return geometries, ids

def check_index(index):
    if index.lower() not in ['ndvi', 'ndwi', 'ndmi', 'psri', 'savi']:
        print("Error in name of index. Calculation for '{}' is not supported.".format(index))
        return False
    return True


def calculate_index(data, index):
    if index.lower() == 'ndvi':
        return (data.B08.astype('float16')-data.B04.astype('float16'))/(data.B08.astype('float16')+data.B04.astype('float16'))
    elif index.lower() == 'ndwi':
        return (data.B08.astype('float16')-data.B03.astype('float16'))/(data.B08.astype('float16')+data.B03.astype('float16'))
    elif index.lower() == 'ndmi':
        return (data.B08.astype('float16')-data.B11.astype('float16'))/(data.B08.astype('float16')+data.B11.astype('float16'))
    elif index.lower() == 'psri':
        return (data.B04.astype('float16')-data.B02.astype('float16'))/data.B06.astype('float16')
    else:
        return None


def round5 (x):
    return (round((x-5)/10))*10+5

def plot_coherence_timeseries(geom, index, start_time='2017-01-01', end_time='2017-12-31', cols=4, 
                              masked=False, buffer=300, show = True):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bounds = [(round5(x), round5(y)) for x, y in geom.exterior.coords]
    
    if not masked:
        geom = geom.buffer(buffer)
    
    query = {
        'geopolygon': geom,
        'time': (start_time, end_time),
        'product': "sentinel1_coherence"
    }
    
    data = dc.load(measurements=[index], **query)
    mask = geometry_mask([geom], data.geobox, invert=True)

    bounds_xy = [(np.where(data.x == x)[0][0], np.where(data.y == y)[0][0]) for x, y in bounds]
    x = [point[0] for point in bounds_xy]
    y = [point[1] for point in bounds_xy]
    all_pixels = len(data.x) * len(data.y)  # np.count_nonzero(mask)

    rows = len(data.time.values)//cols
    if len(data.time.values) % cols > 0:
        rows += 1

    if masked:
        data = data.where(mask)

    fig = plt.figure(figsize=(20, 30))
    for i in range(len(data.time.values)):
        img = data[index][i].values
        fig.add_subplot(rows, cols, i+1)
        fig.subplots_adjust(hspace=0.2)
        im = plt.imshow(img, vmin=0, vmax=1, cmap='binary')
        plt.title(index + ' for ' + str(data.time.values[i])[:10], size=10)
        plt.plot(x, y, color='r')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if show:
        plt.show()
    else:
        return fig


def plot_dindex_timeseries(geom, index='ndvi', start_time='2019-01-01',
                          cols=5, end_time='2019-12-31',
                          cloud_free_percentage=60, masked=False,
                          buffer=300, show=True):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bounds = [(round5(x), round5(y)) for x, y in geom.exterior.coords]
    if not masked and buffer is None:
        buffer = 300
        geom2 = geom.buffer(buffer)
    elif not buffer is None and not masked:
        geom2 = geom.buffer(buffer)
    else:
        geom2 = geom

    query = {
        'geopolygon': geom2,
        'time': (start_time, end_time),
        'product': "s2_preprocessed_v2"
    }

    if not check_index(index):
        return None
    if index.lower() == 'ndvi':
        data = dc.load(measurements=['B04', 'B08', 'SCL'], **query)
        colormap = 'RdYlGn'
    elif index.lower() == 'ndwi':
        data = dc.load(measurements=['B03', 'B08', 'SCL'], **query)
        colormap = 'YlGnBu'
    elif index.lower() == 'ndmi':
        data = dc.load(measurements=['B08', 'B11', 'SCL'], **query)
        colormap = 'YlGnBu'
    elif index.lower() == 'psri':
        data = dc.load(measurements=['B02', 'B04', 'B06', 'SCL'], **query)
        colormap = 'YlOrRd'

    mask = geometry_mask([geom], data.geobox, invert=True)
    all_pixels = np.count_nonzero(mask)  # len(data.x) * len(data.y)
    nan_array = data['SCL'].where((data['SCL'] >= 3) & (data['SCL'] < 7) & mask)

    timestamps = data.time.values
    if masked:
        data = data.where(mask)

    data[index] = calculate_index(data, index)
    cloud_percs = []
    for i in range(len(timestamps)):
        free_pixels = nan_array[i].count().values
        cloud_perc = (free_pixels / all_pixels) * 100
        cloud_percs.append(cloud_perc)

    bounds_xy = [(np.where(data.x == x)[0][0], np.where(data.y == y)[0][0]) for x, y in bounds]
    x = [point[0] for point in bounds_xy]
    y = [point[1] for point in bounds_xy]

    free_index = [i for i in range(len(cloud_percs)) if cloud_percs[i] > cloud_free_percentage]
    timestamps = timestamps[free_index]
    ndvis = data[index][free_index]
    cloud_percs = np.array(cloud_percs)[free_index]

    rows = len(timestamps) // cols + 1

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.tight_layout()

    for i in range(1,len(timestamps)):
        img = ndvis[i].values.astype('float64')
        img2 = ndvis[i-1].values.astype('float64')
        img = abs(img - img2)
        fig.add_subplot(rows, cols, i + 1)
        im = plt.imshow(img, vmin=0, vmax=2, cmap=colormap)
        plt.title('d{} for {} and {}'.format(index.upper(), pd.to_datetime(timestamps[i]).date(), pd.to_datetime(timestamps[i-1]).date()), size=10)
        plt.plot(x, y, color='r')
        plt.axis('off')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if show:
        plt.show()
    else:
        return fig


def plot_index_timeseries(geom, index='ndvi', start_time='2019-01-01',
                          cols=5, end_time='2019-12-31',
                          cloud_free_percentage=60, masked=False,
                          buffer=300, show = True):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bounds = [(round5(x), round5(y)) for x, y in geom.exterior.coords]
    if not masked and buffer is None:
        buffer = 300
        geom2 = geom.buffer(buffer)
    elif not buffer is None and not masked:
        geom2 = geom.buffer(buffer)
    else:
        geom2 = geom

    query = {
        'geopolygon': geom2,
        'time': (start_time, end_time),
        'product': "s2_preprocessed_v2"
    }

    if not check_index(index):
        return None
    if index.lower() == 'ndvi':
        data = dc.load(measurements=['B04', 'B08', 'SCL'], **query)
        colormap = 'RdYlGn'
    elif index.lower() == 'ndwi':
        data = dc.load(measurements=['B03', 'B08', 'SCL'], **query)
        colormap = 'YlGnBu'
    elif index.lower() == 'ndmi':
        data = dc.load(measurements=['B08', 'B11', 'SCL'], **query)
        colormap = 'YlGnBu'
    elif index.lower() == 'psri':
        data = dc.load(measurements=['B02', 'B04', 'B06', 'SCL'], **query)
        colormap = 'YlOrRd'
    
    mask = geometry_mask([geom], data.geobox, invert=True)
    all_pixels = np.count_nonzero(mask) #len(data.x) * len(data.y)  
    nan_array = data['SCL'].where((data['SCL'] >= 3) & (data['SCL'] < 7) & mask)
    
    timestamps = data.time.values
    if masked:
        data = data.where(mask)

    data[index] = calculate_index(data, index)
    cloud_percs = []
    for i in range(len(timestamps)):
        free_pixels = nan_array[i].count().values
        cloud_perc = (free_pixels / all_pixels) * 100
        cloud_percs.append(cloud_perc)

    bounds_xy = [(np.where(data.x == x)[0][0], np.where(data.y == y)[0][0]) for x, y in bounds]
    x = [point[0] for point in bounds_xy]
    y = [point[1] for point in bounds_xy]

    free_index = [i for i in range(len(cloud_percs)) if cloud_percs[i] > cloud_free_percentage]
    timestamps = timestamps[free_index]
    ndvis = data[index][free_index]
    cloud_percs = np.array(cloud_percs)[free_index]

    rows = len(timestamps) // cols + 1

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.tight_layout()

    for i in range(len(timestamps)):
        img = ndvis[i].values.astype('float64')
        fig.add_subplot(rows, cols, i + 1)
        im = plt.imshow(img, vmin=-0.5, vmax=1, cmap=colormap)
        plt.title('{} for {}'.format(index.upper(), pd.to_datetime(timestamps[i]).date()), size=10)

        plt.plot(x, y, color='r')
        plt.axis('off')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if show:
        plt.show()
    else:
        return fig

def create_rgb(geom,start_time,end_time, buffer = None):
    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    bounds = [(round5(x), round5(y)) for x, y in geom.exterior.coords]

    
    if not buffer is None:
        geom2 = geom.buffer(buffer)
    else:
        geom2 = geom

    query = {
        'geopolygon': geom2,
        'time': (start_time, end_time),
        'product': "s2_preprocessed_v2"
    }
    data = dc.load(measurements=['B02', 'B03', 'B04', 'SCL'],**query)

    mask = geometry_mask([geom], data.geobox, invert=True)
    all_pixels = np.count_nonzero(mask)
    
    nan_array = data['SCL'].where((data['SCL']>=3) & (data['SCL']<7) & mask)
    timestamps = data.time.values

    cloud_percs = []
    for i in range(len(timestamps)):

        free_pixels = nan_array[i].count().values
        cloud_perc = (free_pixels / all_pixels) * 100
        cloud_percs.append(cloud_perc)



    bounds_xy = [(np.where(data.x == x)[0][0], np.where(data.y == y)[0][0]) for x, y in bounds]
    x = [point[0] for point in bounds_xy]
    y = [point[1] for point in bounds_xy]


    rgb = np.stack((data['B02'].values, data['B03'].values, data['B04'].values), axis=2)
    rgb = rgb.swapaxes(0, 3).swapaxes(0, 1).astype('float32')
    return rgb,timestamps, x, y, cloud_percs


def plot_rgb(rgb,timestamps, x, y, cloud_percs, cloud_free_percentage = 100, cols=6):

    free_index = [i for i in range(len(cloud_percs)) if  cloud_percs[i] > cloud_free_percentage]
    rgb = rgb[:,:,:,free_index]
    timestamps = timestamps[free_index]

    rows = len(timestamps) // cols + 1

    fig = plt.figure(figsize=(cols*4,rows*4))
    fig.tight_layout()
    for i in range(len(timestamps)):
        img = rgb[:, :, :, i]
        array_min, array_max = np.nanmin(img), np.nanmax(img)
        img = (img - array_min) / (array_max - array_min + 1e-7)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title('RGB for {}'.format(pd.to_datetime(timestamps[i]).date()), size=10)

        plt.plot(x, y, color='r')
        plt.axis('off')
    plt.show()


def plot_rgb_geomedian(rgb,width=50,height=50,cols=6,rows=2):
    fig = plt.figure(figsize=(50, 50))
    for i in range(1):
        img = rgb
        array_min, array_max = np.nanmin(img), np.nanmax(img)
        img = (img - array_min) / (array_max - array_min)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title("Geomedian RGB", fontsize=35)

def plot_rgb_multiple_parcels(geoms, ids, start_time = '2017-03-01', cols=4, cloud_free_percentage = 100,
                                end_time = '2017-11-01', buffer = None):
    for geom, i in zip(geoms, ids):
        rgb,timestamps, x, y, cloud_percs = create_rgb(geom, start_time = start_time,
                                                       end_time = end_time, buffer = buffer)
        print('--------------------------------------------------------------------')
        print('-------------------------- parcel {0:>5} -----------------------------'.format(i))
        print('-------------------------------------------------------------------')
        plot_rgb(rgb,timestamps,x, y, cloud_percs, cloud_free_percentage, cols=cols)
        
def plot_index_multiple_parcels(geoms, ids,index = 'ndvi', start_time = '2017-03-01', cols = 4,
                                end_time = '2017-11-01', cloud_free_percentage=70, masked = False, 
                                buffer = None):
    for geom, i in zip(geoms, ids):
        print('--------------------------------------------------------------------')
        print('-------------------------- parcel {0:>5} -----------------------------'.format(i))
        print('-------------------------------------------------------------------')
        plot_index_timeseries(geom, index = index, start_time = start_time, end_time = end_time,cols=cols,
                      cloud_free_percentage=cloud_free_percentage, masked = masked, buffer = buffer)
        

def cloud_trend(geom,start_time='2019-01-01', end_time='2019-12-31'):

    dc = datacube.Datacube(app="test", config="/home/noa/datacube.conf")
    query = {
        'geopolygon': geom,
        'time': (start_time, end_time),
        'product': "s2_preprocessed_v2"
    }
    print("Please wait for data loading from the cube...")
    data = dc.load(measurements=['SCL'],**query)
    print("Data loading has been finished...")

    mask = geometry_mask([geom], data.geobox, invert=True)
    data = data.where(mask)
    all_pixels = np.count_nonzero(~np.isnan(data['SCL'][0].values))

    cloud_data = {}
    for i in range(len(data.time.values)):
        img = data['SCL'][i].values
        img[np.logical_or(data['SCL'][i].values < 4, data['SCL'][i].values > 6)] = -9999
        cloudy_pixels = np.count_nonzero(img == -9999)
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

def download_s2(tiles,start_date,end_date,outdir,username,password,cloud_cover='90', bounding_box = None):
    cloud_cover = str(cloud_cover)
    for tile_name in tiles:
        if len(tiles) > 0:
            query_url = 'https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?maxRecords=10&rocessingLevel=LEVEL2A&productIdentifier=%25'+tile_name+'%25&startDate='+start_date+'T00%3A00%3A00Z&completionDate='+end_date+'T23%3A59%3A59Z&sortParam=startDate&sortOrder=descending&status=0%7C34%7C37&dataset=ESA-DATASET&cloudCover=%5B0%2C'+cloud_cover+'%5D&'
        elif not bounding_box in None:
            query_url = 'https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?maxRecords=10&rocessingLevel=LEVEL2A&geometry='+bounding_box+'%25&startDate='+start_date+'T00%3A00%3A00Z&completionDate='+end_date+'T23%3A59%3A59Z&sortParam=startDate&sortOrder=descending&status=0%7C34%7C37&dataset=ESA-DATASET&cloudCover=%5B0%2C'+cloud_cover+'%5D&'
        else:
            print("Error! Please provide either the name of the tile(s) or a bounding box")
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

        import requests

        DOWNLOAD_URL = 'https://zipper.creodias.eu/download'
        # TOKEN_URL = 'https://auth.creodias.eu/auth/realms/DIAS/protocol/openid-connect/token'

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
            sensing_year = sensing_date[:4]
            sensing_month = sensing_date[4:6]
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
            # print(outdir)
            outfile = Path(outdir) / '{}.zip'.format(identifier)
            if os.path.exists(outfile):
                continue
            outfile_temp = str(outfile) + '.incomplete'
            try:
                downloaded_bytes = 0
                # print(url)
                req = requests.get(url, stream=True, timeout=100)
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
        for month in os.listdir(os.path.join(outdir,sensing_year,tile_name)):
            os.chdir(os.path.join(outdir,sensing_year,tile_name,month))
            os.system('unzip \*.zip')
            os.system('rm *.zip')


def download_s1(wkt,start_date,end_date,outdir,username,password,product_type):
    import requests
    query_url = 'https://finder.creodias.eu/resto/api/collections/Sentinel1/search.json?maxRecords=10&startDate='+start_date+'00%3A00%3A00Z&completionDate='+end_date+'T23%3A59%3A59Z&productType='+product_type+'&geometry='+wkt+'&sortParam=startDate&sortOrder=ascending&status=all&dataset=ESA-DATASET'

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

    ids = [result['id'] for result in query_response.values()]

    i = 0
    for id in ids:
        i+=1
        time.sleep(0.1)
        outdir = initial_outdir
        identifier = query_response[id]['properties']['productIdentifier'].split('/')[-1]
        if '_SLC_' in identifier:
            sensing_date = identifier.split('.')[0].split('__')[1].split('_')[1]
            sensing_date = sensing_date.split('T')[0]
            sensing_day = int(sensing_date[-2:])
        else:
            sensing_date = identifier.split('.')[0].split('_')[4]
            sensing_date = sensing_date.split('T')[0]
            sensing_day = int(sensing_date[-2:])

        sensing_year = sensing_date[:4]
        sensing_month = sensing_date[4:6]

        logging.info(id)
        size = query_response[id]['properties']['services']['download']['size'] / 1024 / 1024
        if size < 600:
            continue
        token = _get_token(username, password)
        url = '{}/{}?token={}'.format(DOWNLOAD_URL, id, token)
        # pbar =  tqdm(total = len(ids), unit='files')
        outdir = os.path.join(outdir, sensing_month)

        if not os.path.exists(outdir):
            print(outdir)
            os.mkdir(outdir)
        try:
            zipped_file = os.path.join(outdir, identifier.split('.SAF')[0])
            zipped_file += '.zip'
            print(zipped_file)
            unzipped_file = zipped_file.split('.zip')[0] + '.SAFE'
            if os.path.exists(zipped_file) or os.path.exists(unzipped_file):
                print("File exists")
                continue
            # exists = 0
            # for tif in os.listdir(outdir):
            #     if identifier[:25] in tif:
            #         exists = 1
            # if exists == 1:
            #     print("Exists!")
            #     continue

            while True:
                session = requests.Session()
                session.stream = True
                resp = session.get(url)
                print("Response Status:", resp.status_code)
                if resp.status_code == 200:
                    logging.info('Session status code: %d of %s', resp.status_code)
                    break
                else:
                    logging.warning('Something happened: \tStatus Code: %d, \tReason: %s, \tFilename: %s ',
                                    resp.status_code,
                                    resp.reason)
                    time.sleep(300)
            # download and save
            print("Download Process starts...")

            outfile = zipped_file
            outfile_temp = str(zipped_file) + '.incomplete'
            try:
                downloaded_bytes = 0
                print(url)
                req = requests.get(url, stream=True, timeout=100)
                # progress = tqdm(unit='B', unit_scale=True, disable=False)
                chunk_size = 2 ** 20  # download in 1 MB chunks
                with open(outfile_temp, 'wb') as fout:
                    for chunk in req.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            fout.write(chunk)
                            # progress.update(len(chunk))
                            downloaded_bytes += len(chunk)
                shutil.move(outfile_temp, str(outfile))
            finally:
                Path(outfile_temp).unlink()
        except:
            continue


def index(ws,year,tile,product):
    if product == 's2':
        full_path = os.path.join(ws, 's2', year,tile)
        os.system('python /home/noa/Desktop/paper/datacap/product_yamls/s2_preprocessed/create_yamls.py %s' %(full_path))
    elif product == 's1_insar':
        full_path = os.path.join(ws, 's1',year,'coherence')
        os.system('python /home/noa/Desktop/paper/datacap/product_yamls/s1_coherence/s1prepare_coh.py %s' % (full_path))
    elif product == 's1_sar':
        full_path = os.path.join(ws, 's1', year, 'backscatter')
        os.system('python /home/noa/Desktop/paper/datacap/product_yamls/s1_backscatter/s1prepare.py %s' % (full_path))
