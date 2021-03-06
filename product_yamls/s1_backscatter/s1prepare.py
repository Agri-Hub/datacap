# coding=utf-8
"""
Ingest data from the command-line.
"""
from __future__ import absolute_import, division

from __future__ import print_function
import logging
import uuid
from xml.etree import ElementTree
import re
from pathlib import Path
import yaml
from dateutil import parser
from datetime import timedelta
import rasterio.warp
import click
from osgeo import osr
import os,sys

_STATIONS = {'023': 'TKSC', '022': 'SGS', '010': 'GNC', '011': 'HOA',
             '012': 'HEOC', '013': 'IKR', '014': 'KIS', '015': 'LGS',
             '016': 'MGR', '017': 'MOR', '032': 'LGN', '019': 'MTI', '030': 'KHC',
             '031': 'MLK', '018': 'MPS', '003': 'BJC', '002': 'ASN', '001': 'AGS',
             '007': 'DKI', '006': 'CUB', '005': 'CHM', '004': 'BKT', '009': 'GLC',
             '008': 'EDC', '029': 'JSA', '028': 'COA', '021': 'PFS', '020': 'PAC'}

# Parses a string and returns from -> to : YYYY-MM-DD_YYYY-MM-DD
def time_parse(timestr):
    # S1A_IW_GRDH_1SDV_20190106T043507_20190106T043532_025352_02CE64_04CB_VH
    aos = timestr.split('_')[4][:4]+'-'+timestr.split('_')[4][4:6]+'-'+timestr.split('_')[4][6:8]
    los = timestr.split('_')[5][:4]+'-'+timestr.split('_')[5][4:6]+'-'+timestr.split('_')[5][6:8]
    return aos, los


def band_name(path):
    name = path.stem
    if 'VH' in str(path):
        layername = 'vh'
    if 'VV' in str(path):
        layername = 'vv'
    return layername


def get_projection(path):
    with rasterio.open(str(path)) as img:
        left, bottom, right, top = img.bounds
        return {
            'spatial_reference': str(str(getattr(img, 'crs_wkt', None) or img.crs.wkt)),
            'geo_ref_points': {
                'ul': {'x': left, 'y': top},
                'ur': {'x': right, 'y': top},
                'll': {'x': left, 'y': bottom},
                'lr': {'x': right, 'y': bottom},
            }
        }


def get_coords(geo_ref_points, spatial_ref):
    spatial_ref = osr.SpatialReference(spatial_ref)
    t = osr.CoordinateTransformation(spatial_ref, spatial_ref.CloneGeogCS())

    def transform(p):
        lon, lat, z = t.TransformPoint(p['x'], p['y'])
        return {'lon': lon, 'lat': lat}

    return {key: transform(p) for key, p in geo_ref_points.items()}


def populate_coord(doc):
    proj = doc['grid_spatial']['projection']
    doc['extent']['coord'] = get_coords(proj['geo_ref_points'], proj['spatial_reference'])


def crazy_parse(timestr):
    try:
        return parser.parse(timestr)
    except ValueError:
        if not timestr[-2:] == "60":
            raise
        return parser.parse(timestr[:-2] + '00') + timedelta(minutes=1)


def prep_dataset(fields, path):
    # ws = '/data/lithuania/SENTINEL_1A'
    docs = []
    for tif in os.listdir(path):
        if not tif.endswith('.tif'):
            continue
        if 'VV_' in tif:
            continue

        aos, los = time_parse(tif)

        fields['creation_dt'] = aos
        fields['satellite'] = 'SENTINEL_1A'
        images = {band_name(im_path): {
            'path': str(im_path.relative_to(path))
        } for im_path in path.glob(tif.split('VH_3857')[0]+'*.tif')}
        print(tif)
        print(images)

        doc = {
            'id': str(uuid.uuid4()),
            # Strips the path of the suffix and generating a name for the generated yaml
            'name': list(images.values())[0]['path'][:-4].replace('VV', '').replace('VH', ''),
            'processing_level': "terrain",
            'product_type': "gamma0",
            'creation_dt': aos,
            'platform': {'code': 'SENTINEL_1'},
            'instrument': {'name': 'SAR_C'},
            'extent': {
                'from_dt': str(aos),
                'to_dt': str(los),
            #      'center_dt': str(aos)
            },
            'format': {'name': 'GeoTiff'},
            'grid_spatial': {
                'projection': get_projection(path / next(iter(images.values()))['path'])
            },
            'image': {
                'bands': images
            },
        }

        fields['id'] = doc['id']
        populate_coord(doc)
        docs.append(doc)
    return docs


def dataset_folder(fields):
    fmt_str = "{vehicle}_{instrument}_{type}_{level}_{type}{product}-{groundstation}_{path}_{row}_{date}"
    return fmt_str.format(**fields)


# INPUT path is parsed for elements - below hardcoded for testing
def prepare_datasets(s1a_path):
    fields = re.match(
        (r"(?P<platform>SENTINEL_1A)"), s1a_path.stem).groupdict()
    print("Fields:",fields)
    # timedelta(days=int(fields["julianday"]))
    # , 'creation_dt': ((crazy_parse(fields["productyear"]+'0101T00:00:00'))+timedelta(days=int(fields["julianday"])))})
    fields.update({'level': 'gamma0', 'type': 'intensity'})
    total_s1a = prep_dataset(fields, s1a_path)
    results = []
    for s1a in total_s1a:
        results.append((s1a, s1a_path))
    print("Results:",results)
    return results

@click.command(help="Prepare SENTINEL_1A SAR dataset for ingestion into the Data Cube.")
@click.argument('datasets',
                type=click.Path(exists=True, readable=True, writable=True),
                nargs=-1)
def main(datasets):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    print("Testing datasets:",datasets)
    for dataset in datasets:
        path = Path(dataset)
        logging.info("Processing %s", path)
        documents = prepare_datasets(path)
        for document in documents:
            dataset, folder = document
            yaml_path = str(folder.joinpath(dataset['name'] + '.yaml'))
            logging.info("Writing %s", yaml_path)
            with open(yaml_path, 'w') as stream:
                yaml.dump(dataset, stream)


if __name__ == "__main__":
    main()
    import os
    import sys

    os.chdir('/home/noa')

    for yaml in os.listdir(dataset):
        if not f.endswith('.yaml'):
            continue
        print("Adding...",f)
        ws_file = os.path.join(dataset,yaml)
        os.system('datacube dataset add %s' %ws_file)
