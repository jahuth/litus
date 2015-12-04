# spike class tests

import silver
import litus.spikes
retina = silver.glue.RetinaSimulation('/home/jacob/Projects/Silversight/notebooks/tmp/2015_12_01_rr_sf_noise_model_noisy_BVUPU/retina/4/simulation.txt')
num_layers = len(retina.layers)
ls = retina.layers
unit_positions = retina.retinafile.get_units()
positions = [unit_positions[unit_positions[:,0]==n,1:] for n in range(num_layers)]
spike_times = ls[0].spikefile.spikes
reload(litus.spikes)
sc = litus.spikes.SpikeContainer(np.concatenate([spike_times[:,1:2],spike_times[:,0:1],positions[0][np.array(spike_times[:,0],dtype=int)]],1),labels=[('t','ms'),'N',('x','mm'),('y','mm')])




figure(figsize=(10,10))
H,xed,yed = sc.spatial_firing_rate_by_label(bins=160)
X, Y = np.meshgrid(xed, yed)
pcolormesh(X, Y, H, cmap='gray')
gca().set_aspect('equal')

for n in sc.generate('y',bins=3,preserve_labels=True):
    for nn in n.generate('x',bins=3,preserve_labels=True):
        figure()
        H,xed,yed = nn.spatial_firing_rate_by_label()
        X, Y = np.meshgrid(xed, yed)
        pcolormesh(X, Y, H, cmap='gray')
        gca().set_aspect('equal')

# Adding the distance from the center:
sc.spike_times.add_label_dimension('d',sc.spike_times['x']**2 + sc.spike_times['y']**2)
# And plotting rings with about equal distance:
for n in sc.generate('d',bins=5,preserve_labels=True):
    figure()
    H,xed,yed = n.spatial_firing_rate_by_label()
    X, Y = np.meshgrid(xed, yed)
    pcolormesh(X, Y, H, cmap='gray')
    gca().set_aspect('equal')


import litus.spikes
reload(litus.spikes)
m = litus.spikes.LabeledMatrix(np.random.rand(100,4),['a','b','c','d'])
st = np.array( [np.random.rand(100),np.round(np.random.rand(100)*4),np.round(np.random.rand(100)*4),np.round(np.random.rand(100)*4)]).transpose()
sc =litus.spikes.SpikeContainer(st,labels=[('a','s'),('b','m'),('c','m'),('d','m')])
for a in sc.generate('b',preserve_labels=True):
    figure()
    imshow(a.spatial_firing_rate_by_label('c','d')[0])
    figure()
    plot(a.spike_times.matrix)

import utils
gd = utils.GoettingenData(*utils.goettingen_files[2])
time_signals = gd.time_signals
ts = gd.time_signals.copy()
ts[0,:] = ts[0,:]/1000.0

