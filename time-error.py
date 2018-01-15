import numpy as np
import matplotlib.pyplot as plt

lines = [line.strip().split(',')[1:] for line in open('test-time.csv')]
hours = np.asarray(range(24), dtype=int)
tn = np.zeros(24, dtype=int)
fp = np.zeros(24, dtype=int)
fn = np.zeros(24, dtype=int)
tp = np.zeros(24, dtype=int)
for line in lines:
	if line[1] == 'TN':
		tn[int(line[0])] += 1
	elif line[1] == 'FP':
		fp[int(line[0])] += 1
	elif line[1] == 'FN':
		fn[int(line[0])] += 1
	elif line[1] == 'TP':
		tp[int(line[0])] += 1

tn = tn.astype(np.float64)
fp = fp.astype(np.float64)
fn = fn.astype(np.float64)
tp = tp.astype(np.float64)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)


fig = plt.figure()
plt.bar(hours, fpr, align='edge')
plt.xlim(0, 24)
plt.xlabel('Time (hr. of day, 0 means 12am, 1 means 1am, etc.)')
plt.ylabel('False positive rate')
plt.title('False positive rates by time of day')
fig.savefig('fpr.png')

fig2 = plt.figure()
plt.bar(hours, fnr, align='edge')
plt.xlim(0, 24)
plt.xlabel('Time (hr. of day, 0 means 12am, 1 means 1am, etc.)')
plt.ylabel('False negative rate')
plt.title('False negative rates by time of day')
fig2.savefig('fnr.png')


