# Master of Computing Final Project

Research on Intradomain Website Fingerprinting

Supervisors: Professor Dong Jin Song, Wang Kailong

Goal/Objectives: The main goal of this project is to analyze the both the deep fingerprinting and triplet fingerprining model on the dataset describe in the first paper. It is to serve as an extension to the methods described in the paper. It is a extension of the existing research done in the below papers. Some parts of the implementation might look similar due to the nature of the project (Keras as mostly the same methods for model building and the algorithms describes in the papers were followed but applied to the main research paper). This is purely a research project and not for commercial use.

Main research paper: It's Not Just the Site, It's the Contents: Intra-domain Fingerprinting Social Media Websites through CDN Bursts

Deep Fingerpring Model Paper: Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning

Triplet Fingerprinting Paper: Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning

Feature Engineering

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

