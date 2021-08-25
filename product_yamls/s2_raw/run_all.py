import os,sys
import sen2cor_prepare


def loop_directory(directory):
	x=0
	for filename in os.listdir(directory):
		if filename.endswith(".SAFE"):
			file_directory = os.path.join(directory, filename)
			print("Creating yaml for: " + filename)
			os.system('python /home/noa/Downloads/prepare_scripts/sen2cor_prepare.py'+ ' ' + directory + "/" + filename + " " + '--output=/home/noa/Desktop/index-yamls')
			x=x+1
	print('Total yamls created: ' + str(x))

def index_yamls(directory):
	y=0
	for filename in os.listdir("/home/noa/Desktop/index-yamls"):
		if filename.endswith(".yaml"):
			file_directory = os.path.join(directory, filename)
			print("Indexing: " + filename)
			os.system('datacube dataset add' + ' ' + directory + '/' + filename)
			y=y+1
	print('Total datasets indexed: ' + str(y))

if __name__=='__main__':
	ws = "/data2/greece/s2/2021/34TFK"
	for month in os.listdir(ws):
		loop_directory(os.path.join(ws,month))
	index_yamls("/home/noa/Desktop/index-yamls")


