from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_style('darkgrid',{"axes.facecolor": ".92"}) # (1)
sns.set_context('notebook')

n = ['128', '256', '512', '1024','2048']

#normal_matrix_mul
flops = [681, 674, 562, 384, 306]
_min = [611, 663, 553, 368, 296]
_max = [702, 681, 566, 398, 310]

#with loop unrolling
#flops = [787, 813, 802, 396, 309]
#_min = [760, 799, 785, 367, 304]
#_max = [792, 823, 816, 412, 312]

#with loop swap
#flops = [743, 653, 590, 385, 367]
#_min = [707, 622, 565, 376, 364]
#_max = [750, 747, 597, 407, 378]

#with tiling
#flops = [709, 702, 710, 736, 717]
#_min = [663, 693, 664, 721, 555]
#_max = [711, 706, 713, 746, 733]

df = pd.DataFrame({'n':n,'FLOPS':flops})
print("Accuracy")

#display(df) # in jupyter

fig, ax = plt.subplots(figsize = (8,6))

x = n
y = flops

plt.xlabel("n", size=14)
plt.ylim(-0.3, 900)
width = 0.1

for i, j in zip(x,y):
    ax.bar(i,j, edgecolor = "black",
        error_kw=dict(lw=1, capsize=1, capthick=1))
    ax.set(ylabel = 'MFLOPS')
#insert error bars
yerr = [np.subtract(flops, _min), np.subtract(_max, flops)]
ax.errorbar(x, y, fmt='_',yerr=yerr, color='black', ecolor='black', elinewidth=3, capsize=7)

plt.title('Matrix Multiplikation ohne Optimierungen')
#plt.title('Matrix Multiplikation mit Loop Unrolling')
#plt.title('Matrix Multiplikation mit Loop Swap')
#plt.title('Matrix Multiplikation mit Tiling (Faktor 4)')

from matplotlib import ticker
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
plt.savefig("Try.png", dpi=500, bbox_inches='tight')
plt.show()