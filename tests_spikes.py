# spike class tests

import silver
import litus.spikes
# cd e.g. to /home/jacob/Projects/Silversight/notebooks/tmp/2015_12_01_rr_sf_noise_model_noisy_BVUPU/retina/4/
retina = silver.glue.RetinaSimulation('simulation.txt')
num_layers = len(retina.layers)
ls = retina.layers
unit_positions = retina.retinafile.get_units()
positions = [unit_positions[unit_positions[:,0]==n,1:] for n in range(num_layers)]
spike_times = ls[0]
reload(litus.spikes)
sc = ls[0]#
#litus.spikes.SpikeContainer(np.concatenate([spike_times[:,1:2],spike_times[:,0:1],positions[0][np.array(spike_times[:,0],dtype=int)]],1),labels=[('t','ms'),'N',('x','mm'),('y','mm')])




figure(figsize=(10,10))
H,xed,yed = sc.spatial_firing_rate(bins=160)
X, Y = np.meshgrid(xed, yed)
pcolormesh(X, Y, H, cmap='gray')
gca().set_aspect('equal')

bins = (sc.linspace_bins('x',bins=100),sc.linspace_bins('y',bins=100))
for n in sc.generate('y',bins=3,remove_dimensions=False):
    for nn in n.generate('x',bins=3,remove_dimensions=False):
        figure()
        H,xed,yed = nn.spatial_firing_rate(bins=bins)
        X, Y = np.meshgrid(xed, yed)
        pcolormesh(X, Y, H, cmap='gray')
        gca().set_aspect('equal')

figure(figsize=(6,6))
bins = (sc.linspace_bins('x',resolution=0.1),sc.linspace_bins('y',resolution=0.1))
for i,n in enumerate(sc.generate('y',bins=3,remove_dimensions=False)):
    print i,
    for ii,nn in enumerate(n.generate('x',bins=3,remove_dimensions=False)):
        subplot(5,5,i+5*ii+1)
        title(str([i,ii]))
        H,xed,yed = nn.spatial_firing_rate(bins=bins)
        X, Y = np.meshgrid(xed, yed)
        pcolormesh(X, Y, H, cmap='gray')
        gca().set_aspect('equal')

# Adding the distance from the center:
sc.spike_times.add_label_dimension('d',sc.spike_times['x']**2 + sc.spike_times['y']**2)
# And plotting rings with about equal distance:
for n in sc.generate('d',bins=5,remove_dimensions=False):
    figure()
    n.plot_spatial_firing_rate(bins=100)


import litus.spikes
reload(litus.spikes)
m = litus.spikes.LabeledMatrix(np.random.rand(100,4),['a','b','c','d'])
st = np.array( [np.random.rand(100),np.round(np.random.rand(100)*4),np.round(np.random.rand(100)*4),np.round(np.random.rand(100)*4)]).transpose()
sc =litus.spikes.SpikeContainer(st,labels=[('a','s'),('b','m'),('c','m'),('d','m')])
for a in sc.generate('b',remove_dimensions=False):
    figure()
    a.plot_spatial_firing_rate('c','d')
    figure()
    plot(a.spike_times.matrix)

## Testing labeling spikes and dealing with data from Goettingen
import utils
gd = utils.GoettingenData(*utils.goettingen_files[2])
time_signals = gd.time_signals
ts = gd.time_signals.copy()
ts[0,:] = ts[0,:]/1000.0
test = litus.spikes.SpikeContainer(gd.file_list[0],data_units='s')
s = test.label_by_time(gd.time_signals,gd.label_names,copy=True,backup_original_spike_times_to='old_spikes')
num_stim = s.len('stimulus')+1
for i,c,k in litus.colorate(s(spike_times__lt=500).generate('stimulus'),len=num_stim):
    subplot(num_stim,1,i)
    k.plot_raster(cell_dimension='rep',color=c)

figure()
plot(s[0],s['old_spikes'],'.')
xlabel('remapped time')
ylabel('real time')
