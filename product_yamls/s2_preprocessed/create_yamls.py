import os
import sys

os.chdir('/home/noa')

if (len(sys.argv)) < 2:
    print("No argument given")

ws = sys.argv[1]

months = ['01','02','03','04','05','06','07','08','09','10','11','12']

for month in months:
    ws_month = os.path.join(ws,month)
    if not os.path.exists(ws_month):
        continue
    for day in os.listdir(ws_month):
        ws_day = os.path.join(ws_month,day)
        os.system('python /home/noa/Desktop/product_yamls/s2_preprocessed/s2prepare.py %s' %ws_day)


for month in months:
    ws_month = os.path.join(ws,month)
    if not os.path.exists(ws_month):
        continue
    for day in os.listdir(ws_month):
        ws_day = os.path.join(ws_month,day)
        for f in os.listdir(ws_day):
            if not f.endswith('.yaml'):
                continue
            print("Adding...",f)
            ws_file = os.path.join(ws_day,f)
            os.system('datacube dataset add %s' %ws_file)
