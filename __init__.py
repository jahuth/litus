import numpy as np
from subprocess import call
import json
from copy import copy
import re

_session_description = ""
_last_inputs = []
_tagged_inputs = {}

def set_session_description(s=""):
    global _session_description
    _session_description = s

#############################################
## IPython History snippet saving

def _lines_as_comments(s):
    return "\n".join(["# "+l for l in s.split("\n")])

def snip(tag="",start=-2,write_date=True):
    import IPython
    i = IPython.get_ipython()
    last_history = i.history_manager.get_range(start=start,stop=start+1,output=True)
    with open("ipython_history.py",'a') as output_file:
        for l in last_history:
            global _session_description
            output_file.write('\n\n\n'+('#'*80)+'\n')
            if _session_description != "":
                output_file.write('#\n'+_lines_as_comments(_session_description)+'\n#\n')
            if tag != "":
                output_file.write(_lines_as_comments(tag)+'\n')
            if write_date:
                import datetime
                output_file.write('# '+datetime.datetime.now().isoformat()+'\n')
            output_file.write('\n\n# In ['+str(l[1])+']:\n'+l[2][0])
            _last_inputs.append(l[2][0])
            _tagged_inputs[tag] = _tagged_inputs.get(tag,[])
            _tagged_inputs[tag].append(l[2][0])
            output_file.write('\n\n# Out ['+str(l[1])+']:\n'+_lines_as_comments(repr(l[2][1])))

def snip_this(tag="",write_date=True):
    snip(tag=tag,start=-1,write_date=write_date)

def unsnip(tag=None,start=-1):
    import IPython
    i = IPython.get_ipython()
    if tag in _tagged_inputs.keys():
        if len(_tagged_inputs[tag]) > 0:
            i.set_next_input(_tagged_inputs[tag][start])
    else:
        if len(_last_inputs) > 0:
            i.set_next_input(_last_inputs[start])

def animate(a,r=25,every_nth_frame=5,cmap='gray',tmp_dir='tmp',frame_prefix='frame_',animation_name='animation.mp4',func=None):
    import os,io,glob
    import base64
    import matplotlib.pylab as plt
    from IPython.display import HTML
    try:
        import tqdm
        trange = tqdm.trange
    except:
        tqdm = None
        trange = xrange
    try:
        os.mkdir(tmp_dir)
    except:
        pass
    for f in glob.glob(tmp_dir+'/'+frame_prefix+'*.png'):
        os.remove(f)
    try:
        os.remove(tmp_dir+'/'+animation_name)
    except:
        pass
    if func == None:
        max_shape = np.max([aa.shape for aa in a if type(aa)==np.ndarray],0)

    for ti,t in enumerate(trange(0,len(a),every_nth_frame)):
        if func == None:
            plt.figure()
            plt.title(ti)
            if type(a[t]) == list:
                a[t] = np.zeros(max_shape)
            plt.imshow(a[t],cmap=cmap,vmin=np.min(a),vmax=np.max(a))
            plt.axis('off')
        else:
            func(a[t])
        plt.savefig(tmp_dir+'/'+frame_prefix+'%04d.png'%ti)
        plt.close()
    os.system("avconv -i "+tmp_dir+'/'+frame_prefix+"%04d.png -r "+str(r)+" "+tmp_dir+'/'+animation_name)
    video = io.open(tmp_dir+'/'+animation_name, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" autoplay loop controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))


#############################################
## Sort things with numbers in them

def alert(msg,body="",icon="/home/jacob/Projects/Silversight/icons/kernel.svg"):
    """
        alerts the user of something happening via `notify-send`. If it is not installed, the alert will be printed to the console.
    """
    if type(body) == str:
        body = body[:200]
    if call(["which","notify-send"]) == 0:
        if icon is not None:
            call(["notify-send",msg,"-i",icon,body])
        else:
            call(["notify-send",msg,body])
    else:
        print "ALERT: ", msg

#############################################
## Math things

def exponential(length=100,steepness=2.0):
    kernel = np.exp(-np.arange(0.0,steepness,float(steepness)/float(length)))
    return kernel

def sig(x,sigma_sharpness=1.0,sigma_threshold=0.0):
    return (1.0/(1.0+np.exp(sigma_sharpness*(sigma_threshold-np.array(x,dtype=np.float128)))))

def exponential_filter(ar,length=100,steepness=2.0):
    kernel = np.exp(-np.arange(0.0,steepness,float(steepness)/float(length)))
    return np.convolve(ar,kernel)

def gauss(a=-1.0,b=1.0,resolution=100,m = 0.0,s = 0.5):
    return np.exp(-0.5*(np.linspace(float(a),float(b),resolution)-float(m))**2/float(s))

class FlatKernel(object):
    """
        This class can hold a two dimensional object and return one dimensional indizes given a stride length for each row.
        Together with the flattened object this can be used for more efficient convolution.
    """
    def __init__(self,k):
        self.k = k
        ks0 = self.k.shape[0]/2.0
        ks1 = self.k.shape[1]/2.0
        self.i = np.meshgrid(np.arange(-np.floor(ks0),np.ceil(ks0),1.0),
                              np.arange(-np.floor(ks1),np.ceil(ks1),1.0))
    def get_kernel(self):
        return self.k.flatten()
    def get_indizes(self, stride_length):
        return (self.i[0] + self.i[1] * stride_length).flatten()
    def get(self, stride_length):
        k,i = self.get_kernel(), self.get_indizes(stride_length)
        print k.shape, i.shape
        return k[k!=0.0],i[k!=0.0]

from cmath import rect, phase
from math import radians, degrees

def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))

def mean_orientation(deg):
    v = [rect(1, radians(d)) for d in deg]
    v = np.sign(np.imag(v))*np.real(v) + np.sign(np.imag(v))*np.imag(v)*1j
    return degrees(phase(np.sum(v)/len(deg)))

def orientation_difference(deg1,deg2):
    d = min((deg1 - deg2)%180,(deg2 - deg1)%180)
    return d

#############################################
## Sort things with numbers in them

def _make_an_int_if_possible(text):
    return int(text) if text.isdigit() else text

def _natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ _make_an_int_if_possible(c) for c in re.split('(\d+)', text) ]

def natural_sorted(l):
    """ sorts a sortable in human order (0 < 20 < 100) """
    ll = copy(l)
    ll.sort(key=_natural_keys)
    return ll


#############################################
## json that converts numpy arrays to lists


class _NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def json_dump(x, fp):
    return json.dump(x, fp, cls=_NumpyAwareJSONEncoder)
def json_dumps(x):
    return json.dumps(x, cls=_NumpyAwareJSONEncoder)

#############################################
## cartesian functions: generate combinations of factors

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def icartesian(arrays, out=None):
    """
        Like cartesian, but uses inverted place values, ie. the last value changes slowest, the first fastest.

        >>> icartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [2, 4, 6],
               [3, 4, 6],
               [1, 5, 6],
               [2, 5, 6],
               [3, 5, 6],
               [1, 4, 7],
               [2, 4, 7],
               [3, 4, 7],
               [1, 5, 7],
               [2, 5, 7],
               [3, 5, 7]])
    """
    return cartesian(np.array(arrays)[::-1])[:,::-1]

def cartesian_to_index(ranges, maxima=None):
    """
        Inverts tuples from a cartesian product to a numeric index ie. the index this 
        tuple would have in a cartesian product.

        Each column gets multiplied with a place value according to the preceding columns maxmimum
        and all columns are summed up.

        This function in the same direction as utils.cartesian, ie. the first column has the largest value.
    """
    if maxima is None:
        return reduce(lambda y,x: (x*y[1] + y[0],(np.max(x)+1)*y[1]), ranges[:,::-1].transpose(), (np.array([0]*ranges.shape[0]),1))[0]
    else:
        maxima_prod = np.concatenate([np.cumprod(np.array(maxima)[::-1])[1::-1],[1]])
        return np.sum(ranges * maxima_prod, 1)

def icartesian_to_index(ranges, maxima=None):
    """
        Inverts tuples from a cartesian product to a numeric index ie. the index this 
        tuple would have in a cartesian product.

        Each column gets multiplied with a place value according to the preceding columns maxmimum.

        This function in the opposite direction of utils.cartesian, ie. the last column has the highest value.
    """
    if maxima is None:
        return reduce(lambda y,x: (x*y[1] + y[0],(np.max(x)+1)*y[1]), ranges.transpose(), (np.array([0]*ranges.shape[0]),1))[0]
    else:
        maxima_prod = np.concatenate([[1],np.cumprod(np.array(maxima))[:-1]])
        return np.sum(ranges * maxima_prod, 1)
        maxima_prod = np.concatenate([[1],np.cumprod(np.array(maxima))[:-1]])
        return np.sum(ranges[::-1] * maxima_prod, 1)

def cartesian_dicts(dicts, sort=False):
    if len(dicts.keys()) == 0:
        return []
    if sort is True:
        sorted_keys = sorted(dicts.keys())
    else:
        sorted_keys = (dicts.keys())
    if len(dicts.keys()) == 1:
        this_iteration_key = sorted_keys[0]
        return [{ this_iteration_key: x } for x in dicts[this_iteration_key]]
    n = []
    this_iteration_key = sorted_keys[0]
    next_iteration = cartesian_dicts({ k: dicts[k] for k in sorted_keys[1:]})
    for x in dicts[this_iteration_key]:
        for o in next_iteration:
            new_item = {}
            new_item.update(o)
            new_item.update({this_iteration_key: x})
            n.append(new_item)
    return n

def fillzip(*l):
    """like zip (for things that have a length), but repeats the last element of all shorter lists such that the result is as long as the longest."""
    maximum = max(len(el) for el in l)
    return zip(*[el + [el[-1]]*(maximum-len(el)) for el in l])

#############################################
## recursive generator flattener

def recgen(gen):
    """iterates through generators recursively and flattening them"""
    if not hasattr(gen,'__iter__'):
        yield gen
    else:
        for i in gen:
            for ii in recgen(i):
                yield ii

def recgen_enumerate(gen,n=tuple()):
    """iterates through generators recursively and flattening them"""
    if not hasattr(gen,'__iter__'):
        yield (n,gen)
    else:
        for i_,i in enumerate(gen):
            for element in recgen_enumerate(i,n+(i_,)):
                yield element

#############################################
## plot things

def color_space(colormap, a, start=0.1, stop=0.9):
    if type(colormap) is str:
        from matplotlib import cm
        if colormap in cm.__dict__:
            colormap = cm.__dict__[colormap]
        else:
            colormap = cm.gnuplot
    if type(a) is int or type(a) is float:
        return colormap(np.linspace(start,stop,int(a)))
    return colormap(np.linspace(start,stop,len(a)))

def colorate(sequence, colormap="", start=0):
    """ like enumerate, but with colors """
    n = start
    colors = color_space(colormap, sequence, start=0.1, stop=0.9)
    for elem in sequence:
        yield n, colors[n-start], elem
        n += 1


#################################################
## Parameter / Data Collectors
##
##  They still need a catchy name!
##


def unnpfy(x):
    if type(x) is dict:
        return {k:unnpfy(x[k]) for k in x.keys()}
    if type(x) is np.ndarray:
        return x.tolist()
    if type(x) in [np.int,np.int16,np.int32,np.int64,np.int8]:
        return int(x)
    if type(x) in [np.float,np.float16,np.float32,np.float64,np.float128]:
        return float(x)
    return x

class PDFileHandler(object):
    def __init__(self,parent=None):
        self.parent = parent
    def __call__(self,key):
        if key not in self.parent.files.keys():
            self.parent.files[key] = self.parent.parent.create_path(self.parent,key)
            self.parent.parent.save()
        import os
        if not os.path.exists("/".join(self.parent.files[key].split("/")[:-1])):
            os.makedirs("/".join(self.parent.files[key].split("/")[:-1]))
        return self.parent.files[key]
    def __getitem__(self,key):
        filename = self(key)
        if filename.endswith('.npz') or filename.endswith('.npy'):
            o = np.load(filename)
            if type(o) == np.lib.npyio.NpzFile:
                if o.keys() == ['arr_0']:
                    return o['arr_0']
            return o
        elif filename.endswith('.json'):
            import json
            with open(filename,'r') as f:
                return json.load(value,f)
        else:
            with open(filename,'r') as f:
                return f.read()
    def __setitem__(self,key,value):
        filename = self(key)
        if filename.endswith('.npz'):
            if type(value) == dict:
                np.savez(filename,**value)
            else:
                np.savez(filename,np.array(value))
        elif filename.endswith('.npy'):
            np.save(filename,np.array(value))
        elif filename.endswith('.json'):
            import json
            with open(filename,'w') as f:
                json.dump(value,f)
        else:
            with open(filename,'w') as f:
                f.write(str(value))

class PDContainer(object):
    def __init__(self,name=None,params=None,files=None,data=None,parent=None):
        self.name = name
        self.parameters = params
        self.files = files
        self.data = data
        self.parent = parent
        if self.parameters is None:
            self.parameters = {}
        if self.files is None:
            self.files = {}
        if self.data is None:
            self.data = {}
        if self.name is None:
            self.name = "_".join([str(k) + ":"+ str(self.parameters[k]) for k in self.parameters.keys()])
        self.file = PDFileHandler(self)
    def param(self,key,default=None):
        """for accessing parameters"""
        if key in self.parameters:
            return self.parameters[key]
        return self.parent.param(key,default)
    def __getitem__(self,key):
        return self.data[key]
    def __setitem__(self,key,value):
        self.data[key] = value
        self.parent.save()
    def dumpd(self):
        return { 'name': str(self.name), 'parameters': self.parameters, 'files': self.files, 'data': self.data }
    def loadd(self,d):
        print d
        self.__dict__.update(d)
        return self
    def open(self,key,*args):
        return open(self.file(key),*args)
    """def file(self,key,*args):
        if key not in self.files.keys():
            self.files[key] = self.parent.create_path(self,key)
            self.parent.save()
        import os
        if not os.path.exists("/".join(self.files[key].split("/")[:-1])):
            os.makedirs("/".join(self.files[key].split("/")[:-1]))
        return self.files[key]"""


class PDContainerList(object):
    def __init__(self,path="",filename=None, name_mode = 'int', parameters = None):
        self.path = path
        self.name_mode = name_mode
        import os
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.filename = filename
        self.containers = []
        self.parameters = parameters
        if self.parameters is None:
            self.parameters = {}
    def generate(self,**kwargs):
        """run once to create all children containers for each combination of the keywords"""
        import collections
        all_params = cartesian_dicts({k:kwargs[k] for k in kwargs.keys() if isinstance(kwargs[k], collections.Iterable)})
        for pi,p in enumerate(all_params):
            if self.name_mode == 'int':
                n = str(pi)
            else:
                n = None
            self.containers.append(PDContainer(name=n,params=p,parent=self))
        self.parameters.update({ k: kwargs[k] for k in kwargs.keys() if not isinstance(kwargs[k], collections.Iterable) })
        self.save()
    def find(self,**kwargs):
        results = []
        for c in self.containers:
            match = True
            for k in kwargs.keys():
                if c.param(k) != kwargs[k]:
                    match = False
                    break
            if match:
                results.append(c)
        return results
    def __iter__(self):
        return self.containers.__iter__()
    def save(self):
        import yaml
        cs = yaml.dump([unnpfy(c.dumpd()) for c in self.containers])
        if self.filename is None:
            return cs
        with open(self.path+"/"+self.filename,'w') as f:
            f.write(cs)
    def load(self):
        import yaml
        self.containers = []
        with open(self.path+"/"+self.filename,'r') as f:
            for c in yaml.load(f):
                pd = PDContainer(parent=self)
                self.containers.append(pd.loadd(c))
        return self
    def create_path(self,container,key):
        return self.path+"/."+ container.name + "_" + key
    def param(self,key,default=None):
        """for accessing global parameters"""
        if key in self.parameters:
            return self.parameters[key]
        return default