import xml.etree.ElementTree as ET
import scipy
import math
import numpy as np
try:
    import brian2
except:
    brian2 = False
import numpy
import pandas
import json
from scipy.stats import binned_statistic

from litus import cartesian, cartesian_to_index, icartesian, icartesian_to_index

# Spike Container Class

_default_time_unit = 'ms'

conversion_factors_to_seconds = {
        'picoseconds': 0.000000000001, 'ps': 0.000000000001, 
        'nanoseconds': 0.000000001, 'ns': 0.000000001, 
        'microseconds': 0.000001, 'mus': 0.000001, 
        'milliseconds': 0.001, 'ms': 0.001, 
        'seconds': 1.0, 's': 1.0,
        'minutes': 60.0, 'min': 60.0, 
        'hours': 60.0*60.0, 'h': 60.0*60.0 
    }

conversion_factors_to_metres = {
        'kilometre':0.001,'km':0.001,
        'metre':1.0,'m':1.0,
        'decimetre':10.0,'dm':10.0,
        'centimetre':100.0,'cm':100.0,
        'millimetre':1000.0,'mm':1000.0,
        'micrometre':1e-6,'mum':1e-6,
        'nanometre':1e-9,'nm':1e-9
    }


def create_timing_signals(timings,ranges=[3,8]):
    """
        Creates a linear list/np.array of timestamps and a list of integers, specifying the sequence 
        and number of levels.

        The levels are first rotated on the first dimension, then the second and so on. If the 
        last dimension is -1, the number of levels is computed from the timestamps.

        returns a 2 dimensional array with len(ranges) + 1 columns
            * the timestamp is in the first column
            * the following columns contain the specific level for this time stamp

        Example:

            utils.create_timing_signals(range(100),[2,5,-1])

        will return a matrix with the first column being the nubmers from 0 to 99, 
        the second alternates between 0 and 1 (eg. whether the number is odd), the third column 
        contains numbers 0 to 4, each bracketing a block of 2*5 = 10 numbers. The last column is
        computed according to the length of the timeseries (in this case 100/(2*5) is 10, ie the 
        column counts blocks of 10 numbers).
    """
    if ranges[-1] == -1:
        ranges[-1] = np.floor(len(timings)/float(np.prod(ranges[:-1]))) + 1
    return np.array([([t]+[cc for cc in c]) for t,c in zip(timings,icartesian([range(int(r)) for r in ranges]))]).transpose()


def set_default_time_unit(units):
    global _default_time_unit
    if units in conversion_factors_to_seconds.keys():
        _default_time_unit = units
    else:
        raise Exception('\'units\' must be a valid time unit. Eg. '+(','.join(conversion_factors_to_seconds.keys())))

def convert_time(times, from_units, to_units):
    if from_units == to_units or to_units is None:
        return times
    if from_units in conversion_factors_to_seconds.keys():
        conversion = conversion_factors_to_seconds[from_units]/conversion_factors_to_seconds[to_units]
        if conversion == 1.0:
            return times
        return times * conversion
    if from_units in conversion_factors_to_metres.keys():
        conversion = conversion_factors_to_metres[from_units]/conversion_factors_to_metres[to_units]
        if conversion == 1.0:
            return times
        return times * conversion
    return times

def cell_filter_filename_C(x,cell_designation):
    return 'filename' in x and x['filename'].endswith('C'+str(cell_designation)+'.txt')

class SpikeContainerCollection:
    def __init__(self,filename_or_data=None,units=None,data_units=None,min_t=0.0,max_t=None,meta=None,labels=None,label_names=None,copy_from=None,**kwargs):
        self.meta = None
        self.units = None
        self.min_t = None
        self.max_t = None
        self.index_dimensions = None
        units = self._default_units(units)
        if copy_from is not None:
            for key in ['units','meta','min_t','max_t','data_format']:
                self.__dict__[key] = copy_from.__dict__[key]
        self.spike_containers = []
        if meta is not None:
            self.meta = meta
        if units is not None:
            self.units = units
        if min_t is not None:
            self.min_t = min_t
        if max_t is not None:
            self.max_t = max_t
        self.__dict__.update(kwargs)
        if type(filename_or_data) == str:
            try:
                self.load_from_csv(filename_or_data,units,min_t=min_t,max_t=max_t,data_units=data_units)
            except:
                raise
        if type(filename_or_data) == list and len(filename_or_data) > 1:
            self.data_format = 'spike_containers'
            self.spike_containers = []
            for c,f in enumerate(filename_or_data):
                if f.__class__ == SpikeContainer:
                    sc = f
                else:
                    try:
                        sc = SpikeContainer(f,units=units,min_t=min_t,max_t=max_t,data_units=data_units,meta={'c':c,'filename':f,'parent':self})
                    except:
                        raise
                if self.min_t is None or sc.min_t < self.min_t:
                    self.min_t = sc.min_t
                if self.max_t is None or sc.max_t > self.max_t:
                    self.max_t = sc.max_t
                self.spike_containers.append(sc)
        if type(filename_or_data) == np.ndarray:
            try:
                self.set_spike_times(filename_or_data,units,min_t=min_t,max_t=max_t,data_units=data_units,labels=labels)
            except:
                raise
    def find(self,cell_designation,cell_filter=lambda x,c: 'c' in x and x['c'] == c):
        """
            finds spike containers in a multi spike containers collection
        """
        res = [i for i,sc in enumerate(self.spike_containers) if cell_filter(sc.meta,cell_designation)]
        if len(res) > 0:
            return res[0]
    def _default_units(self,units=None):
        global _default_time_unit
        if units is None:
            units = _default_time_unit
        if units is None:
            units = self.units
        if units is None:
            units = 'ms'
        return units
    def convert(self,units=None,min_t=None,max_t=None):
        # TODO: adapt to container collection
        units = self._default_units(units)
        if min_t is None:
            min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
        if max_t is None:
            max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
        return SpikeContainerCollection([sc.convert(units=units,min_t=min_t,max_t=max_t) for sc in self.spike_containers],units=units,min_t=min_t,max_t=max_t)
    def label_by_time(self,time_signals,label_names=[],copy=False, **kwargs):
        """
            creates a labeled spike data structure

                time_label_array is list of lists (or matrix), containing a timestamp in the 
                first column (or first element of each element) and indizes that are to be 
                applied to the data in the remaining columns / elements.

            This function will not add or remove spikes, but only shift spikes according to the 
            adjecent time signals.

            If you want to get spikes relative to a time signal with fixed limits, use `label_peri_signals`,
            which will leave out and duplicate spikes, but can manage overlapping time signals.


        """
        new_containers = []
        for sc in self.spike_containers:
            new_sc = sc.label_by_time(time_signals=time_signals,copy=True, label_names=label_names, **kwargs)
            new_containers.append(new_sc)
        return SpikeContainerCollection(new_containers,
                        units=self.units,
                        min_t=self.min_t,
                        max_t=self.max_t, 
                        copy_from=self, 
                        label_names = label_names, 
                        index_dimensions = np.max(time_signals[1:,:],1)+1)
    def label_peri_signals(self,time_signals,label_names=[],copy=False, pre_signal = 100.0, post_signal = 1000.0, **kwargs):
        """
            creates a labeled spike data structure

            time_label_array is list of lists (or matrix), containing a timestamp in the 
            first column (or first element of each element) and indizes that are to be 
            applied to the data in the remaining columns / elements.


            This function will leave out and duplicate spikes to manage overlapping time signals.

            If you want to get spikes relative to a time signal with felxible limits, use `label_by_time`,
            which will not add or remove spikes, but only shift spikes according to the 
            adjecent time signals.

        """
        new_containers = []
        for sc in self.spike_containers:
            new_sc = sc.label_peri_signals(time_signals=time_signals,copy=True, pre_signal = pre_signal, post_signal = post_signal, label_names=label_names, **kwargs)
            new_containers.append(new_sc)
        return SpikeContainerCollection(new_containers,
                        units=self.units,
                        min_t=self.min_t,
                        max_t=self.max_t, 
                        copy_from=self, 
                        label_names = label_names, 
                        index_dimensions = np.max(time_signals[1:,:],1)+1)
    def set_empty(self):
        self.data_format = 'empty'
        self.spike_containers = None
        self.min_t = None
        self.max_t = None
    def get_spike_times(self,units=None,min_t=None,max_t=None):
        units = self._default_units(units)
        return np.array([sc.get_spike_times(units=units,min_t=min_t,max_t=max_t) for sc in self.spike_containers])
    def bins(self,units=None,min_t=None,max_t=None,resolution=1.0):
        units = self._default_units(units)
        if min_t is None:
            min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
        if max_t is None:
            max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
        return np.arange(min_t,max_t,resolution)
    def add_label(self,name,label_data):
        for sc in self.spike_containers:
            sc.add_label(self,name=name,label_data=label_data)
    def add_index(self,name='index',order=None):
        for sc in self.spike_containers:
                sc.add_index(name=name,order=order)
    def generate(self,*args,**kwargs):
        for spike_container in self.spike_containers:
            yield spike_container
    def len(self,*args,**kwargs):
        return len(self.spike_containers)
    def __len__(self,*args,**kwargs):
        return len(self.spike_containers)
    def __getitem__(self, key):
        return self.spike_containers[key]

class LabelDimension(object):
    def __init__(self,name=None,units='1',min=None,max=None,**kwargs):
        if type(name) == LabelDimension:
            self.name = name.name
            self.units = name.units
            self.min = name.min
            self.max = name.max
        else:
            self.name = name
            self.units = units
            self.min = min
            self.max = max
    def __str__(self):
        return "LabelDimension('"+self.name+"',"+str(self.units)+","+str(self.min)+","+str(self.max)+")"
    def __repr__(self):
        return "LabelDimension('"+self.name+"',"+str(self.units)+","+str(self.min)+","+str(self.max)+")"
    def len(self,resolution=1.0,units=None,conversion_function=convert_time):
        if units is not None:
            resolution = conversion_function(resolution,from_units=self.units,to_units=units)
        if self.min is None:
            return self.max / resolution
        if self.max is None:
            return 0
        return np.ceil((self.max - self.min) / resolution)
    def linspace(self,bins=None,units=None,conversion_function=convert_time,resolution=None):
        """ bins overwrites resolution """
        min = conversion_function(self.min,from_units=self.units,to_units=units)
        max = conversion_function(self.max,from_units=self.units,to_units=units)
        if resolution is None:
            resolution = 1.0
        if bins is None:
            bins = self.len(resolution=resolution,units=units,conversion_function=conversion_function) + 1
        return np.linspace(min,max,bins)
    def linspace_bins(self,bins=None,units=None,conversion_function=convert_time,resolution=None):
        """Generates bin edges for a linspace tiling: there is one edge more than bins and each bin is in the middle between two edges"""
        bins = self.linspace(bins=bins,units=units,conversion_function=conversion_function,resolution=resolution)
        resolution = bins[1] - bins[0]
        bins = np.concatenate([bins,bins[-1:]+resolution]) - 0.5*resolution
        return bins
    def linspace_by_resolution(self,resolution=1.0,units=None,conversion_function=convert_time):
        return self.linspace(bins=None,resolution=resolution,units=units,conversion_function=conversion_function)
    def convert(self,units=None,conversion_function=convert_time):
        min = conversion_function(self.min,from_units=self.units,to_units=units)
        max = conversion_function(self.max,from_units=self.units,to_units=units)
        return LabelDimension(self.name, units, min, max)
    def constraint_range_dict(self,*args,**kwargs):
        """ bins overwrites resolution """
        space = self.linspace(*args,**kwargs)
        resolution = space[1] - space[0]
        return [{self.name+'__gte': s,self.name+'__lt': s+resolution} for s in space]

class LabeledMatrix(object):
    """
        Magical class that contains a 2d data structure and labels associated with each column.

        Tries to be mostly transparent to the 2d numpy structure (len, shape, etc.).

        With square bracket indexing, label names can be found. Function calls can contain magic label functions:

            m(a__lt=0.5,b__evals=lambda x: np.abs(x-0.5)<0.2,remove_dimensions=True)
            # all a smaller than 0.5, all b that evaluate the function as True
            # dimensions that were used for filtering can be retained (default) or removed.

    """
    def __init__(self,matrix=None,labels=None,**kwargs):
        if matrix is not None:
            self.matrix = matrix.copy()
        else:
            self.matrix = None
        self.labels = []
        if labels is not None:
            for l in labels:
                if type(l) == str:
                    self.labels.append(LabelDimension(l))
                elif type(l) == tuple or type(l) == list:
                    self.labels.append(LabelDimension(*l))
                else:
                    self.labels.append(LabelDimension(l.name,l.units,l.min,l.max))
        self.expand_maxima()
    def __str__(self):
        return "LabeledMatrix with label dimensions: "+", ".join([str(l) for l in self.labels])
    def __repr__(self):
        return "LabeledMatrix with label dimensions: "+", ".join([repr(l) for l in self.labels])
    def expand_maxima(self):
        if self.matrix is None or self.matrix.shape[0] == 0:
            return
        i = 0
        while len(self.labels) < self.matrix.shape[1]:
            print len(self.labels),'<', self.matrix.shape[1]
            self.labels.append('I'+str(i))
            i += 1
        for i in range(self.matrix.shape[1]):
            if self.labels[i].min is None or self.labels[i].min > np.min(self.matrix[:,i]):
                self.labels[i].min = np.min(self.matrix[:,i])
            if self.labels[i].max is None or self.labels[i].max < np.max(self.matrix[:,i]):
                self.labels[i].max = np.max(self.matrix[:,i])
    def find_labels(self,key,find_in_name=True,find_in_units=False):
        if type(key) is str:
            found_keys = []
            for label_no,label in enumerate(self.labels):
                    if find_in_name and key in label.name:
                        found_keys.append(label_no)
                    if find_in_units and key == label.units:
                        found_keys.append(label_no)
            return found_keys
        if hasattr(key, '__call__'):
            found_keys = []
            for label_no,label in enumerate(self.labels):
                    if key(label):
                        found_keys.append(label_no)
            return found_keys
        if type(key) is int:
            return [key] if key < self.matrix.shape[1] else []
        return [key]
    def get_label(self,key):
        if type(key) is str:
            for label_no,label in enumerate(self.labels):
                    if key == label.name:
                        return label
            raise Exception("Key not found! "+str(key))
        return self.labels[key]
    def get_label_no(self,key):
        if type(key) is str:
            for label_no,label in enumerate(self.labels):
                    if key == label.name:
                        return label_no
            raise Exception("Key not found! "+str(key))
        return key
    def __getitem__(self, key):
        if type(key) is tuple:
            if type(key[1]) is not slice:
                key = list(key)
                #if not type(key[1]) is list or type(key[1]) is tuple:
                #    key[1] = [key[1]]
                new_key = []
                for k in key[1]:
                    new_key.extend(self.find_labels(k))
                key[1] = new_key
                key = tuple(key)
        else:
            key = tuple([slice(None),self.get_label_no(key)])
        return self.matrix.__getitem__(key)
    def __len__(self):
        return self.matrix.__len__()
    @property
    def shape(self):
        return self.matrix.shape
    def get_converted(self,label=0,units=None,conversion_function=convert_time):
        label_no = self.get_label_no(label)
        min = conversion_function(self.labels[label_no].min,from_units=self.labels[label_no].units,to_units=units)
        max = conversion_function(self.labels[label_no].max,from_units=self.labels[label_no].units,to_units=units)
        new_label = LabelDimension(self.labels[label_no].name,units=units,min=min,max=max)
        return new_label,conversion_function(self.matrix[:,label_no],from_units=self.labels[label_no].units,to_units=units)
    def convert(self,label,units=None,conversion_function=convert_time):
        """ converts a dimension in place """
        label_no = self.get_label_no(label)
        new_label, new_column = self.get_converted(label_no,units,conversion_function)
        labels = [LabelDimension(l) for l in self.labels]
        labels[label_no] = new_label
        matrix = self.matrix.copy()
        matrix[:,label_no] = new_column
        return LabeledMatrix(matrix,labels)
    def __call__(self,remove_dimensions=False,**kwargs):
        constraints = []
        remaining_label_dimensions = range(len(self.labels))
        new_labels = []
        for label_no,label in enumerate(self.labels):
            new_label = LabelDimension(label)
            for k in kwargs:
                if k == label.name:
                    constraints.append(self.matrix[:,label_no] == kwargs[k])
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
                if k == label.name+'__lt':
                    constraints.append(self.matrix[:,label_no] < kwargs[k])
                    new_label.max = np.min([new_label.max,kwargs[k]])
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
                if k == label.name+'__lte':
                    constraints.append(self.matrix[:,label_no] <= kwargs[k])
                    new_label.max = np.min([new_label.max,kwargs[k]])
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
                if k == label.name+'__gt':
                    constraints.append(self.matrix[:,label_no] > kwargs[k])
                    new_label.min = np.max([new_label.min,kwargs[k]])
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
                if k == label.name+'__gte':
                    constraints.append(self.matrix[:,label_no] >= kwargs[k])
                    new_label.min = np.max([new_label.min,kwargs[k]])
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
                if k == label.name+'__evals':
                    constraints.append(kwargs[k](self.matrix[:,label_no]))
                    if label_no in remaining_label_dimensions:
                        remaining_label_dimensions.remove(label_no)
            new_labels.append(new_label)

        st = self.matrix[np.all(constraints,0),:]
        if remove_dimensions:
            return LabeledMatrix(st[:,[r for r in remaining_label_dimensions]],[l for li,l in enumerate(new_labels) if li in remaining_label_dimensions])
        return LabeledMatrix(st,new_labels)
    def add_label_dimension(self,name,label_data):
        if len(label_data.shape) == 1:
            self.labels.append(LabelDimension(name))
            self.matrix = np.concatenate([self.matrix,label_data[:,np.newaxis]],1)
        if len(label_data.shape) == 2:
            if type(name) is list or type(name) is tuple:
                self.labels.extend([LabelDimension(n) for n in name])
                self.matrix = np.concatenate([self.matrix,label_data],1) 
        self.expand_maxima()               
    def add_index(self,name='index',order=None):
        if order is not None:
            order = [self.get_label_no(o) for o in order]
            new_index = icartesian_to_index(self.matrix[:,order])
        else:
            new_index = cartesian_to_index(self.matrix[:,1:])
        self.add_label(name, new_index)
    def __nonzero__(self):
        return self.matrix.shape[0] > 0
    def generate(self,*args,**kwargs):
        """
        Creates an iterator over one of the index dimensions of the object.

        Example::

            for l in s.generate('layer',remove_dimensions=False,resolution=2):
                # taking two layers at a time
                print l

            remove_dimensions: False
                Whether the labels that are used for generation should be removed from the generated objects.
            resolution:     None
                If set to an integer, the dimension used for generation will be split into parts of size `resolution`.
            bins:           None
                If set to an integer, the dimension used for generation will be split into `bins` many, equal parts.
            reversed:       True
                Whether argument list should be reversed, such that the first argument is rotated first.
        """
        constraints = []
        remove_dimensions = kwargs.get('remove_dimensions',False)
        resolution = kwargs.get('resolution',None) 
        bins = kwargs.get('bins',None) 
        if len(args) == 0:
            generator_indizes = range(1,len(self.labels))
        else:
            generator_indizes = [kk for k in args for kk in self.find_labels(k)]
        if kwargs.get("reversed",True):
            generator_indizes = list(reversed(generator_indizes))
        generator_ranges = cartesian([self.labels[i].constraint_range_dict(resolution=resolution,bins=bins) for i in generator_indizes])
        remaining_label_dimensions = [i for i,l in enumerate(self.labels) if i not in generator_indizes]
        for t in generator_ranges:
            constraints = {}
            for ri, i in enumerate(generator_indizes):
                constraints.update(t[ri])
            yield self(remove_dimensions=remove_dimensions, **constraints)


class SpikeContainer:
    """
        Contains labeled Spikes

        The following terms are used:

            `units`: 
                a string representing a time unit (in most cases 'ms' or 's')
            `spike_times`: 
                float values that give the time of the spike event (in the specified `units`)
            `t_min`,`t_max`: 
                The borders of the spike train (in `units`)
            `labels`: 
                a 2d matrix that has a line for each spike and a column for each label that is used
            `label_names`: 
                a list of strings that label the columns of the labels

            `data_format`:
                'empty', 'spike_times' or 'spike_containers'



        Enables the following interactions:

        `sc(some_dimension=2)` 
            -> returns a filtered object where only spikes that are labeled 2 in the column `'some_dimension'`.
        `sc.generate('some_dimension')`
            -> returns a Generator that goes through all values of 'some_dimension' and returns a SpikeContainer
        `sc[:,:]`
            -> to get all spikes and their labels in a 2d matrix
        `sc[:,['some_dimension','some_other_dimension']]`
            -> constrains the columns to be `spike_times, 'some_dimension', 'some_other_dimension'`
        `sc[1:100:2,:]`
            -> samples only the spikes designated with the slice
        `sc[100.0:1100.0,:]`
            -> (not yet implemented) constrains spikes in time window between 100 and 1100 time units
        `sc[[100.0,1100.0],:]`
            -> (not yet implemented) constrains spikes in time window between 100 and 1100 time units


    """
    def __init__(self,filename_or_data=None,units=None,data_units=None,min_t=0.0,max_t=None,meta=None,labels=None,label_names=None,copy_from=None,**kwargs):
        self.meta = None
        self.units = None
        self.min_t = None
        self.max_t = None
        self.index_dimensions = None
        units = self._default_units(units)
        if copy_from is not None:
            for key in ['units','meta','min_t','max_t','data_format']:
                self.__dict__[key] = copy_from.__dict__[key]
        self.data_format = 'empty'
        self.spike_times = LabeledMatrix(None,None)
        if meta is not None:
            self.meta = meta
        if units is not None:
            self.units = units
        if min_t is not None:
            self.min_t = min_t
        if max_t is not None:
            self.max_t = max_t
        self.__dict__.update(kwargs)
        if type(filename_or_data) == str:
            try:
                self.load_from_csv(filename_or_data,units,min_t=min_t,max_t=max_t,data_units=data_units)
            except:
                raise
        if type(filename_or_data) == np.ndarray or type(filename_or_data) == LabeledMatrix:
            try:
                self.set_spike_times(filename_or_data,units,min_t=min_t,max_t=max_t,data_units=data_units,labels=labels)
            except:
                raise
    def __str__(self):
        return "Spike Container with dimensions: "+", ".join([str(l) for l in self.spike_times.labels])
    def __repr__(self):
        return "Spike Container with dimensions: "+", ".join([repr(l) for l in self.spike_times.labels])
    def __nonzero__(self):
        return bool(self.spike_times)
    def store_meta(self,meta):
        "Inplace method that adds meta information to the meta dictionary"
        if self.meta is None:
            self.meta = {}
        self.meta.update(meta)
        return self
    def find(self,cell_designation,cell_filter=lambda x,c: 'c' in x and x['c'] == c):
        """
            finds spike containers in multi spike containers collection offspring
        """
        if 'parent' in self.meta:
            return (self.meta['parent'],self.meta['parent'].find(cell_designation,cell_filter=cell_filter))
    def _default_units(self,units=None):
        global _default_time_unit
        if units is None:
            units = _default_time_unit
        if units is None:
            units = self.units
        if units is None:
            units = 'ms'
        return units
    def convert(self,label_no=0,units=None,conversion_function=convert_time):
        return SpikeContainer(self.spike_times.convert(label_no,units=units,conversion_function=conversion_function),copy_from=self)
    def label_by_time(self,time_signals,label_names=[],copy=True, **kwargs):
        """
            creates a labeled spike data structure

                time_label_array is list of lists (or matrix), containing a timestamp in the 
                first column (or first element of each element) and indizes that are to be 
                applied to the data in the remaining columns / elements.

            This function will not add or remove spikes, but only shift spikes according to the 
            adjecent time signals.

            If you want to get spikes relative to a time signal with fixed limits, use `label_peri_signals`,
            which will leave out and duplicate spikes, but can manage overlapping time signals.


        """
        if self.data_format == 'empty':
            return SpikeContainer(None,units=self.units,copy_from=self)
        spike_times = self.get_spike_times('ms').copy() # this is read only
        re_zeroed_spike_times = spike_times.copy() # this will be modified
        indizes = np.zeros((len(spike_times),time_signals.shape[0] -1 ))
        for t in range(len(time_signals[0])):
            if t + 1 < len(time_signals[0]):
                # we are past the last time signal
                spike_range = (spike_times > time_signals[0][t]) * (spike_times <= time_signals[0][t+1])
                indizes[spike_range,:] = [time_signals[_i][t] for _i in range(1,time_signals.shape[0])]
                re_zeroed_spike_times[spike_range] = (
                        spike_times[spike_range] - time_signals[0][t]
                    )
            else:
                # we move all spikes in the future back by this time signal
                # (this will overwrite the spike times multiple times)
                indizes[spike_times > time_signals[0][t],:] = [time_signals[_i][t] for _i in range(1,time_signals.shape[0])]
                re_zeroed_spike_times[spike_times > time_signals[0][t]] = (
                        spike_times[spike_times > time_signals[0][t]] - time_signals[0][t]
                    )
        new_spike_times = LabeledMatrix(self.spike_times.matrix,self.spike_times.labels)
        new_spike_times.add_label_dimension(label_names,indizes)
        new_spike_times.labels[0].units = units
        new_spike_times.matrix[:,0] = re_zeroed_spike_times
        new_spike_times.labels[0].min = np.min(re_zeroed_spike_times)
        new_spike_times.labels[0].max = np.max(re_zeroed_spike_times)
        if copy:
            s = SpikeContainer(new_spike_times, copy_from=self)
            return s
        else:
            self.set_spike_times(new_spike_times)
            return self
    def label_peri_signals(self,time_signals,label_names=[],units=None,data_units=None,copy=True, pre_signal = 100.0, post_signal = 1000.0, **kwargs):
        """
            creates a labeled spike data structure

            time_label_array is list of lists (or matrix), containing a timestamp in the 
            first column (or first element of each element) and indizes that are to be 
            applied to the data in the remaining columns / elements.


            This function will leave out and duplicate spikes to manage overlapping time signals.

            If you want to get spikes relative to a time signal with felxible limits, use `label_by_time`,
            which will not add or remove spikes, but only shift spikes according to the 
            adjecent time signals.

        """
        if self.data_format == 'empty':
            return SpikeContainer(None,units=self.units,copy_from=self)
        time_signals[0] = convert_time(time_signals[0],from_units=data_units,to_units=units)
        spike_times = self.spike_times.convert(0,units).matrix.copy() # this is read only
        new_matrix = []
        for t in range(len(time_signals[0])):
            condition = (spike_times[:,0] >= time_signals[0][t] - pre_signal)*(spike_times[:,0] < time_signals[0][t] + post_signal)
            new_spikes = (spike_times[condition,0] - time_signals[0][t])[:,np.newaxis]
            old_labels = spike_times[condition,1:]
            new_labels = [[time_signals[_i][t] for _i in range(1,time_signals.shape[0])]] * np.sum(condition)
            if np.sum(condition) > 0:
                new_matrix.append(np.concatenate([new_spikes,old_labels,new_labels],axis=1))
        if len(new_matrix) == 0:
            return SpikeContainer(None,copy_from=self)
        new_matrix = np.concatenate(new_matrix,0)
        if copy:
            new_spike_times = LabeledMatrix(new_matrix,self.spike_times.labels + label_names)
            new_spike_times.labels[0].units = units
            new_spike_times.labels[0].min = pre_signal
            new_spike_times.labels[0].max = post_signal
            s = SpikeContainer(new_spike_times, copy_from=self)
            return s
        else:
            new_spike_times = LabeledMatrix(new_matrix,self.spike_times.labels + label_names)
            new_spike_times.labels[0].units = units
            new_spike_times.labels[0].min = pre_signal
            new_spike_times.labels[0].max = post_signal
            self.set_spike_times(new_spike_times)
            return self
    def set_empty(self):
        self.data_format = 'empty'
        self.spike_times = LabeledMatrix(None,None)
        self.spike_containers = None
        self.min_t = None
        self.max_t = None
    def set_spike_times(self,data,units=None,min_t=0.0,max_t=None,data_units=None,labels=None,correct_missing_time_dimensions=False):
        units = self._default_units(units)
        if type(data) is LabeledMatrix:
            self.spike_times = data.convert(0,units)
        else:
            if labels is not None:
                self.labels = list(labels)
                if correct_missing_time_dimensions:
                    if type(self.labels[0]) == str and self.labels[0] != 'spike_times':
                        self.labels = [LabelDimension('spike_times',units=data_units,min=min_t,max=max_t)] + self.labels
                    elif type(self.labels[0]) != str and self.labels[0].name != 'spike_times':
                        self.labels = [LabelDimension('spike_times',units=data_units,min=min_t,max=max_t)] + self.labels
                if type(self.labels[0]) is str:
                    self.labels[0] = LabelDimension(self.labels[0])
                if type(self.labels[0]) is tuple or type(self.labels[0]) is list:
                    self.labels[0] = LabelDimension(*self.labels[0])
                if data_units is not None:
                    self.labels[0].units = data_units
                if min_t is not None:
                    self.labels[0].min = min_t
                if max_t is not None:
                    self.labels[0].max = max_t
            else:
                self.labels = [LabelDimension('spike_times',units=data_units, min=min_t, max=max_t)]
            self.spike_times = LabeledMatrix(data,self.labels).convert(0,units)
        self.data_format = 'spike_times'
    def get_spike_times(self,units=None,min_t=None,max_t=None):
        return self.spike_times.get_converted(units=units)[1]
    def load_from_csv(self,filename,units=None,delimiter=' ',min_t=0.0,max_t=None,data_units=None):
        units = self._default_units(units)
        import csv
        with open(filename, 'rb') as csvfile:
            c = csv.reader(csvfile, delimiter=delimiter)
            floats = [[float(r) for r in l] for l in c]
            if len(floats) <= 0:
                self.set_empty()
                return
            spike_times = np.squeeze(floats)
            if len(spike_times.shape) == 0:
                spike_times = np.array([spike_times])
            self.set_spike_times(spike_times, units=units,min_t=min_t,max_t=max_t,data_units=data_units)
    def spatial_firing_rate(self,label_x='x',label_y='y',bins=10,geometry=None,weight_function=None,normalize_time=True,normalize_n=False,start_units_with_0=True):
        """

                imshow(s.spatial_firing_rate(weight_function=lambda x: (x[:,1]>0.5)*(x[:,1]<0.6)))

        """
        if bool(self):
            if weight_function is None:
                bins_x = self.spike_times.get_label(label_y).linspace_bins(bins=bins)
                bins_y = self.spike_times.get_label(label_x).linspace_bins(bins=bins)
                H,xed,yed = np.histogram2d(self.spike_times[label_x],self.spike_times[label_y],bins=(bins_x,bins_y))
            else:
                if start_units_with_0:
                    ## TODO : reintroduce a neuron number thing
                    spike_numbers = np.transpose([self.spikes[:,0]-self.n_range[0],self.spikes[:,1]])
                    weights = weight_function(spike_numbers)
                else:
                    weights = weight_function(self.spike_times[label_x],self.spike_times[label_y])
                bins_x = self.spike_times.get_label(label_y).linspace_bins(bins=bins)
                bins_y = self.spike_times.get_label(label_x).linspace_bins(bins=bins)
                H,xed,yed = np.histogram2d(self.spike_times[label_x],self.spike_times[label_y],bins=(bins_x,bins_y),weights=weights)
            if normalize_time:
                H = H/(self.spike_times.labels[0].convert('s').len())
            if normalize_n:
                H = H/cells_per_bin
            return H,xed,yed
        else:
            return ([[0]],[0],[0])
    def plot_spatial_firing_rate(self,label_x='x',label_y='y',bins=10,geometry=None,weight_function=None,normalize_time=True,normalize_n=False,start_units_with_0=True,**kwargs):
        """


        """
        if bool(self):
            import matplotlib.pylab as plt
            H,xed,yed = self.spatial_firing_rate(label_x=label_x,label_y=label_y,bins=bins,geometry=geometry,weight_function=weight_function,normalize_time=normalize_time,normalize_n=False,start_units_with_0=start_units_with_0)
            X, Y = np.meshgrid(xed, yed)
            kwargs['cmap'] = kwargs.get('cmap','gray')
            plt.pcolormesh(X, Y, H,**kwargs)
            plt.gca().set_aspect('equal')
    def temporal_firing_rate(self,time_dimension=0,resolution=1.0,units=None,min_t=None,max_t=None,weight_function=None,normalize_time=True,normalize_n=True,start_units_with_0=True,cell_dimension='N'):
        """
            Outputs a time histogram of spikes.

            `bins`: number of bins (default is 1ms bins from 0 to t_max)
            `weight_function`: if set, computes a weighted histogram, dependent on the (index, time) tuples of each spike

                    weight_function = lambda x: weight_map.flatten()[array(x[:,0],dtype=int)]

            `normalize_time`
            `normalize_n`:  normalize by the length of time (such that normal output is Hz) and/or number of units (such that output is Hz/unit, determined with unique values in cell_dimension)
                            Generally does not make sense when using a weight_function other than 'count'.

            `start_units_with_0`: starts indizes from 0 instead from the actual index
        """
        units = self._default_units(units)
        if self.data_format == 'spike_times':
            converted_dimension,st = self.spike_times.get_converted(0,units)
            if min_t is None:
                min_t = converted_dimension.min
            if max_t is None:
                max_t = converted_dimension.max
            st = st[(st>=min_t)*(st<max_t)]
            bins = converted_dimension.linspace_by_resolution(resolution)
            H,edg = np.histogram(st,bins=bins)
            if normalize_time:
                H = H/(convert_time(resolution,from_units=units,to_units='s')) # make it Hertz
            if normalize_n:
                H = H/(len(np.unique(self.spike_times[cell_dimension])))
            return H,edg
    def plot_temporal_firing_rate(self,time_dimension=0,resolution=1.0,units=None,min_t=None,max_t=None,weight_function=None,normalize_time=True,normalize_n=True,start_units_with_0=True,cell_dimension='N',**kwargs):
        """


        """
        if bool(self):
            import matplotlib.pylab as plt
            H,ed = self.temporal_firing_rate(time_dimension=time_dimension,resolution=resolution,units=units,min_t=min_t,max_t=max_t,weight_function=weight_function,normalize_time=normalize_time,normalize_n=normalize_n,start_units_with_0=start_units_with_0,cell_dimension=cell_dimension)
            plt.plot(ed[1:],H,**kwargs)
    def smoothed_temporal_firing_rate(self, gaussian_width=10.0, **kwargs):
        if self.data_format == 'spike_times':
            from scipy.ndimage import gaussian_filter1d
            H, xed = self.temporal_firing_rate(**kwargs)
            firing_rates = gaussian_filter1d(H,gaussian_width)
            return firing_rates, xed
    def plot_smoothed_temporal_firing_rate(self, gaussian_width=10.0,time_dimension=0,resolution=1.0,units=None,min_t=None,max_t=None,weight_function=None,normalize_time=True,normalize_n=True,start_units_with_0=True,cell_dimension='N',**kwargs):
        """


        """
        if bool(self):
            import matplotlib.pylab as plt
            from scipy.ndimage import gaussian_filter1d
            H,ed = self.temporal_firing_rate(time_dimension=time_dimension,resolution=resolution,units=units,min_t=min_t,max_t=max_t,weight_function=weight_function,normalize_time=normalize_time,normalize_n=normalize_n,start_units_with_0=start_units_with_0,cell_dimension=cell_dimension)
            firing_rates = gaussian_filter1d(H,gaussian_width)
            plt.plot(ed[1:],firing_rates,**kwargs)
    def __call__(self,**kwargs):
        return SpikeContainer(self.spike_times(**kwargs), copy_from=self)
    def add_label(self,name,label_data):
        if self.data_format == 'spike_times':
            if name in self.spike_times.labels:
                raise Exception('Already labeled with "'+str(name)+': '+str(self.spike_times.labels)+'"!')
                return
            if len(label_data.shape) == 1:
                self.spike_times.labels.append(str(name))
                self.index_dimensions = np.concatenate([self.index_dimensions,np.max(label_data[:,np.newaxis],0)+1],1)                
                self.spike_times = np.concatenate([self.spike_times,label_data[:,np.newaxis]],1)
            if len(label_data.shape) == 2:
                if type(name) is list or type(name) is tuple:
                    self.spike_times.labels.extend(name)
                    self.index_dimensions = np.concatenate([self.index_dimensions,np.max(label_data,0)+1],1)                
                    self.spike_times = np.concatenate([self.spike_times,label_data],1)                
    def add_index(self,name='index',order=None):
        if name in self.spike_times.labels:
            return
        if order is not None:
            order = [o for o in order]
            new_index = icartesian_to_index(self.spike_times[:,order])
        else:
            new_index = cartesian_to_index(self.spike_times[:,len(self.time_dimensions):])
        self.add_label(name, new_index)
    def generate(self,*args,**kwargs):
        """
        Creates an iterator over one of the index dimensions of the object.

        Example for a discrete variable::

            for l in s.generate('layer',remove_dimensions=False,resolution=2):
                # taking two layers at a time
                print l

        Example for two continuous variables::

            for l in s.generate('x','y',remove_dimensions=False,bins=3,reversed=True):
                # will generate x0,y0 | x1,y0 | x2,y0 | x0,y1 | x1,y1 | ...
                # (default is reversed=True)
                print l
            for l in s.generate('x','y',remove_dimensions=False,bins=3,reversed=False):
                # will generate x0,y0 | x0,y1 | x0,y2 | x1,y0 | x1,y1 | ...
                print l

        remove_dimensions: False
            Whether the labels that are used for generation should be removed in the generated objects.
        resolution:     None
            If set to an integer, the dimension used for generation will be split into parts of size `resolution`.
        bins:           None
            If set to an integer, the dimension used for generation will be split into `bins` many, equal parts.
        reversed:       True
            Whether argument list should be reversed, such that the first argument is rotated first.
        """
        for st in self.spike_times.generate(*args,**kwargs):
            yield SpikeContainer(st, copy_from=self)
    def len(self,*args,**kwargs):
        constraints = []
        if len(args) == 0 or args == None:
            generator_indizes = range(1,len(self.spike_times.labels))
        else:
            generator_indizes = [kk for k in args for kk in self._find_keys(k)]
        generator_ranges = cartesian([np.arange(self.index_dimensions[i-len(self.time_dimensions)]) for i in generator_indizes])
        return len(generator_ranges)
    def _find_keys(self,key):
        if type(key) is str:
            found_keys = []
            for label_no,label in enumerate(self.spike_times.labels):
                    if key == label.name:
                        return [label_no]
                    if key in label.name:
                        found_keys.append(label_no)
            if found_keys is not []:
                return found_keys
        return [key]
    def _find_key(self,key):
        if type(key) is str:
            found_keys = []
            for label_no,label in enumerate(self.spike_times.labels):
                    if key == label.name:
                        return label_no
                    if key in label.name:
                        found_keys.append(label_no)
            if len(found_keys) == 1:
                return found_keys
            if len(found_keys) > 1:
                import warnings
                warnings.warn("Key has multiple matches, but none is exact! Returning first match. (This might not be consistent)")
                return found_keys[0]
        return key
    def __getitem__(self, key):
        if type(key) is tuple:
            if type(key[1]) is not slice:
                key = list(key)
                if not type(key[1]) is list or type(key[1]) is tuple:
                    key[1] = [key[1]]
                new_key = []
                for k in key[1]:
                    new_key.extend(self._find_keys(k))
                key[1] = new_key
                key = tuple(key)
        else:
            key = self._find_keys(key)[0]
        st = self.spike_times
        return st[key]
        if type(key) == slice:
            print key.start, key.stop, key.step
        else:
            print key
    # Below are some functions that might be deprecated at some point
    def _deprecated__get_spike_array(self,resolution=1.0,units=None,min_t=None,max_t=None):
        units = self._default_units(units)
        if self.data_format == 'spike_times':
            times = convert_time(self.spike_times,from_units=self.units,to_units=units)
            if min_t is None:
                min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
            if max_t is None:
                max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
            spike_array = np.zeros(int(np.ceil((max_t-min_t) / resolution)))
            for t in times:
                # assuming one dimensional data for now?
                if min_t is not None and t < min_t:
                    continue
                if  max_t is not None and t >= max_t:
                    continue
                spike_array[int((t-min_t) / resolution)] += 1
            return spike_array
        if self.data_format == 'empty':
            return np.zeros(int(np.ceil((max_t-min_t) / resolution)))
        if self.data_format == 'spike_containers':
            if min_t is None:
                min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
            if max_t is None:
                max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
            return np.array([sc.get_spike_array(resolution=resolution,units=units,min_t=min_t,max_t=max_t) for sc in self.spike_containers])
    def _deprecated__get_spike_array_index(self,resolution=1.0,units=None,min_t=None,max_t=None):
        units = self._default_units(units)
        if min_t is None:
            min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
        if max_t is None:
            max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
        return np.arange(min_t,max_t,resolution)
    def _deprecated__get_filtered_spike_times(self,label_filter,units=None,min_t=None,max_t=None):
        units = self._default_units(units)
        if self.data_format == 'spike_times':
            filt = np.all(self.labels[:,:len(label_filter)] == label_filter,1)
            #return self.spike_times[]
            return np.concatenate([np.array([self.spike_times[filt]]).transpose(), self.labels[filt,:]], 1)
        if self.data_format == 'spike_containers':
            return np.array([sc.get_filtered_spike_times(label_filter=label_filter,units=units,min_t=min_t,max_t=max_t) for sc in self.spike_containers])
    def _deprecated__get_dimensions_index(self,filter=''):
        if self.data_format == 'spike_containers':
            return self.spike_containers[0]._deprecated__get_dimensions_index(filter)
        if self.data_format == 'empty':
            return None
        if self.spike_times.labels is not None:
            for i,l in enumerate(self.spike_times.labels):
                if filter in l:
                    return i
    def _deprecated__get_dimensions_indizes(self,filter=''):
        if self.data_format == 'spike_containers':
            return self.spike_containers[0]._deprecated__get_dimensions_indizes(filter)
        if self.data_format == 'empty':
            return []
        if self.spike_times.labels is not None:
            return [i for i,l in enumerate(self.spike_times.labels) if filter in l]
    def _deprecated__get_dimensions_with(self,filter=''):
        if self.data_format == 'spike_times':
            return icartesian([range(int(self.index_dimensions[t])) for t in range(len(self.index_dimensions)) if 
                                                                                    filter == '' 
                                                                                    or len(self.spike_times.labels) <= t 
                                                                                    or filter in self.spike_times.labels[t]])
        if self.data_format == 'spike_containers':
            return self.spike_containers[0]._deprecated__get_dimensions_with(filter)
        if self.data_format == 'empty':
            return np.array([[]])
    def _deprecated__get_dimensions_without(self,filter=''):
        if self.data_format == 'spike_times':
            return icartesian([range(int(self.index_dimensions[t])) for t in range(len(self.index_dimensions)) if 
                                                                                    filter == '' 
                                                                                    or len(self.spike_times.labels) <= t 
                                                                                    or not filter in self.spike_times.labels[t]])
        if self.data_format == 'spike_containers':
            return self.spike_containers[0]._deprecated__get_dimensions_without(filter)
        if self.data_format == 'empty':
            return np.array([[]])
    def _deprecated__plot(self,units=None,y=0,marker='|',min_t=None,max_t=None,**kwargs):
        """
        Plots the pointprocess as points at line `y`.

        `marker` determines the color and shape of the marker. Default is a vertical line '|'
        """
        units = self._default_units(units)
        if self.data_format == 'spike_times':
            spike_times = self.get_spike_times(units)
            if spike_times is not None:
                return pl.plot(spike_times,[y]*len(spike_times),marker,**kwargs)
        if self.data_format == 'spike_containers':
            for y_plus,sc in enumerate(self.spike_containers):
                sc.plot(units=units,y=y+y_plus,marker=marker,**kwargs)
            return y+y_plus
    def _deprecated__plot_arr(self,resolution=1.0,units=None,min_t=None,max_t=None,**kwargs):
        units = self._default_units(units)
        return pl.plot(
                    sc.get_spike_array_index(resolution=resolution,units=units,min_t=min_t,max_t=max_t),
                    sc.get_spike_array(resolution=resolution,units=units,min_t=min_t,max_t=max_t),**kwargs)
    def create_SpikeGeneratorGroup(self,time_label=0,index_label=1,reorder_indices=False,index_offset=True):
        """
            Creates a brian 2 create_SpikeGeneratorGroup object that contains the spikes in this container.

                time_label:     Name or number of the label that contains the spike times (default: 0 / first column)
                index_label:    Name or number of the label that contains the cell indices (default: 1 / the second column)
                reorder_indices:   If the cell indices do not matter, the SpikeGeneratorGroup can be created with only as many unique neurons as necessary (default: False / The indices are preserved)
                index_offset:      If set to a number, this will be subtracted from every index (default: True)
                                   If set to True, the `.min` of the label dimension will be subtracted.
                                   If set to False, nothing will be subtracted.
        """
        import brian2
        spike_times = self.spike_times.convert(time_label,'s')[time_label]*brian2.second
        indices = [0] * len(spike_times)
        if len(self.spike_times.find_labels(index_label)):
            indices = self.spike_times[index_label]
        if index_offset is not False:
            if index_offset is True:
                indices = indices - self.spike_times.get_label(index_label).min
            else:
                indices = indices - index_offset
            N = np.max(indices)
        else:
            N = self.spike_times.get_label(index_label).max
        if reorder_indices:
            indices_levels = np.sort(np.unique(indices)).tolist()
            indices = np.array([indices_levels.index(i) for i in indices])
            N = len(indices_levels)
        return brian2.SpikeGeneratorGroup(N+1,indices = indices,
                                            times = spike_times)


