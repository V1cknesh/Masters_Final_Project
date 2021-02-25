import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import glob
import os
import gc
from zlib import crc32
import multiprocessing as mp


def ipaddressconverter(ipaddress):
	return '.'.join(ipaddress.split('.')[0:-1]) + ".0-255"

def packet_size(row):
	return F[row['GROUP'] - 1][1]


thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]



rootdir = '/home/student/MachineLearningTest/AWS_zip/instagram/original_proc'
training = []
testing = []
count = 0
for subdir, dirs, files in os.walk(rootdir):
	print(subdir)
	test = subdir.split("/")[-1]
	if test == "training":
		for subdirs, dirs, files in os.walk(subdir):
			page_number = subdir.split("/")[-2]
			for file in files:
				filename = subdir + '/' + file
				df = pd.read_csv(filename, names=['Time', 'SRC', 'DST', 'PACKET_SIZE'], index_col=None)
				df['PAGE'] = page_number
				df['PACKET_SIZE'].apply(pd.to_numeric, errors='coerce')
				df['SRC'] = df['SRC'].map(ipaddressconverter)
				df.drop(['DST'], axis=1, inplace=True)
				training.append(df)
			final_training = pd.concat(training,axis=0,ignore_index=True)
			q1 = final_training['PACKET_SIZE'].quantile(0.25)
			q3 = final_training['PACKET_SIZE'].quantile(0.75)
			mask = final_training['PACKET_SIZE'].between(2 * q1 - 1.5 * q3, 2.5 * q3 - 1.5 * q1, inclusive=True)
			final_filtered_training=final_training.loc[mask]
			for threshold in thresholds:
				#Feature Engineering based of CDN bursts
				F = []
				k = 1
				B = []
				initial_time = final_filtered_training['Time'].iloc[0]
				group_packet_size = 0
				for index, row in final_filtered_training.iterrows():
					if (row[0] - initial_time < threshold):
						group_packet_size += row[2]
						B.append([row[0], row[1], row[2], k, group_packet_size, row[3]])
					else:
						F.append([k, group_packet_size])
						group_packet_size = 0
						k += 1
				train_data = pd.DataFrame(B, columns=['TIME', 'SRC', 'PACKET_SIZE', 'GROUP', 'CUMULATIVE_PACKET_SIZE', 'PAGE_NUMBER']) 
				train_data['TOTAL_PACKET_SIZE']  = train_data.apply(packet_size, axis=1)
				train_data.drop(['GROUP'], axis=1, inplace=True)
				train_data.to_csv("training_data_"+str(page_number)+"-"+str(threshold).replace(".", "_")+".csv", sep=",", encoding='utf-8')
			training=[]
			gc.collect()
	elif test == "testing":
		for subdirs, dirs, files in os.walk(subdir):
			page_number = subdir.split("/")[-2]
			for file in files:
				filename = subdir + '/' + file
				df = pd.read_csv(filename, names=['Time', 'SRC', 'DST', 'PACKET_SIZE'], index_col=None)
				df['PAGE'] = page_number
				df['PACKET_SIZE'].apply(pd.to_numeric, errors='coerce')
				df['SRC'] = df['SRC'].map(ipaddressconverter)
				df.drop(['DST'], axis=1, inplace=True)
				testing.append(df)
			final_testing = pd.concat(testing,axis=0,ignore_index=True)
			q1 = final_testing['PACKET_SIZE'].quantile(0.25)
			q3 = final_testing['PACKET_SIZE'].quantile(0.75)
			mask = final_testing['PACKET_SIZE'].between(2 * q1 - 1.5 * q3, 2.5 * q3 - 1.5 * q1, inclusive=True)
			final_filtered_testing=final_testing.loc[mask]
			for threshold in thresholds:
				#Feature Engineering based of CDN bursts
				F = []
				k = 1
				B = []
				initial_time = final_filtered_testing['Time'].iloc[0]
				group_packet_size = 0
				for index, row in final_filtered_testing.iterrows():
					if (row[0] - initial_time < threshold):
						group_packet_size += row[2]
						B.append([row[0], row[1], row[2], k, group_packet_size, row[3]])
					else:
						F.append([k, group_packet_size])
						group_packet_size = 0
						k += 1
				test_data = pd.DataFrame(B, columns=['TIME', 'SRC', 'PACKET_SIZE', 'GROUP', 'CUMULATIVE_PACKET_SIZE', 'PAGE_NUMBER']) 
				test_data['TOTAL_PACKET_SIZE']  = test_data.apply(packet_size, axis=1)
				test_data.drop(['GROUP'], axis=1, inplace=True)
				test_data.to_csv("testing_data_"+str(page_number)+"-"+str(threshold).replace(".", "_")+".csv", sep=",", encoding='utf-8')
			testing=[]
			gc.collect()