# move to code repo
import os
import matplotlib.pyplot as plt



sensors = os.listdir('pac_data/')
for sensor in sensors:
	labels = os.listdir(os.path.join('pac_data', sensor))
	for label in labels:
		print sensor, label
		files = os.listdir(os.path.join('pac_data', sensor, label))
		times = [int(file[9:11]) for file in files]
		fig = plt.figure()
		plt.hist(times, bins=24, range=(0, 24))	
		plt.xlim(0, 24)
		plt.xlabel('Time (hr. of day, 0 means 12am, 1 means 1am, etc.)')
		plt.title('Sensor %s, true label = %s' % (sensor, label))
		fig.savefig('%s-%s.png' % (sensor, label))


