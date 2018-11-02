#!/usr/bin/env python3
###
### Example of reading IRTECON GRD format file
###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

filename = 'hd19-05#01_FFO-01a.grd' # GRD to read
Ig = 7.850622e-05 # Ig value from SIS IV
Igfraction = 0.35 # in percent of Ig

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap0=plt.cm.gist_rainbow_r
cmap = truncate_colormap(cmap0, 0.2, 1.0)

# Using contourf to provide my colorbar info, then clearing the figure
tmpdata = [[0,0],[0,1]]
fig=plt.figure()
colinfo = plt.imshow(tmpdata, cmap=cmap, vmin=0, vmax=1)
plt.clf()
plt.close(fig)

plt.rc('font', family='serif')
plt.rc('font', size='16')
fig, ax = plt.subplots(figsize=(8.5,6))
#ax.set_title('FFO IVC')

def tokenizer(fname):
    nc = 0
    with open(fname) as f:
        chunk = []
        for line in f:
            if '#END Curve description' in line:
                continue
            elif '#END Curve' in line:
                print(line)            
                yield chunk
                chunk = []
                nc += 1
            chunk.append(line)

cvs = [np.loadtxt(cv,comments=('#',' ','--','IRTECON')) for cv in tokenizer(filename)]

lw=0.05
s=1.5

for cv in cvs:
    V=cv[:,0]*1000
    I0=cv[:,1]*1000
    Isis=cv[:,2]

    Iratio = Isis/Ig    
    SISscale = np.minimum(Iratio/Igfraction,np.ones(Iratio.size)) ### cut what is above Igfraction
    norm = colors.Normalize(vmin=0,vmax=1)

    cols = cmap(norm(SISscale))

    ax.plot(V,I0, 'k', lw=lw)
    ax.scatter(V,I0,s=s, color=cols)

ax.set_xlabel(r'$V\rm\;(mV)$')
ax.set_ylabel(r'$I\rm\;(mA)$')
ax.set_xlim([0,1.9])
ax.set_ylim([0.,42])

inax = plt.axes([.7, .25, .2, .02])
cb=fig.colorbar(colinfo, cax=inax, orientation='horizontal', ticks=[0,1])
inax.tick_params(labelsize=12) 
cb.set_label(label=r'$\Delta I_{det}/I_g$',fontsize=13,labelpad=-8)
cb.set_ticklabels(['0','%.f'% (Igfraction*100) + '%'])
       
plt.tight_layout()
plt.show()
