import numpy as np
from subprocess import call
import json
from copy import copy
import re
from . import lindex

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
    """ 
        This function records a previously execute notebook cell into a file (default: ipython_history.py)

        a tag can be added to sort the cell

        `start` defines which cell in the history to record. Default is -2, ie. the one executed previously to the current one.

    """
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
    """ When this function is invoced in a notebook cell, the cell is snipped. """
    snip(tag=tag,start=-1,write_date=write_date)

def unsnip(tag=None,start=-1):
    """ This function retrieves a tagged or untagged snippet. """
    import IPython
    i = IPython.get_ipython()
    if tag in _tagged_inputs.keys():
        if len(_tagged_inputs[tag]) > 0:
            i.set_next_input(_tagged_inputs[tag][start])
    else:
        if len(_last_inputs) > 0:
            i.set_next_input(_last_inputs[start])

def animate(a,r=25,every_nth_frame=1,cmap='gray',tmp_dir='tmp',frame_prefix='frame_',animation_name='animation.mp4',func=None):
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

class Figure:
    """
    Figure Context Manager

    Can be used with the **with** statement::

        import litus
        import numpy as np
        import matplotlib.pylab as plt
        x = np.arange(0,10,0.1)
        with litus.figure("some_test.png") as f:
            plt.plot(x,np.cos(x))    # plots to a first plot
            with litus.figure("some_other_test.png"):
                plt.plot(-1*np.array(x)) # plots to a second plot
            plt.plot(x,np.sin(x))    # plots to the first plot again
            f.set_tight_layout(True) # using the figure object


    Or if they are to be used in an interactive console::


        import litus
        import numpy as np
        import matplotlib.pylab as plt
        x = np.arange(0,10,0.1)
        with litus.figure("some_test.png",display=True):
            plt.plot(x,np.cos(x))    # plots to a first plot
            with litus.figure("some_other_test.png",close=False):
                plt.plot(-1*np.array(x)) # plots to a second plot
            plt.plot(x,np.sin(x))    # plots to the first plot again

    Both figures will be displayed, but the second one will remain available after the code is executed. (But keep in mind that in the iPython pylab console, after every input, all figures will be closed)

    """
    def __init__(self,path,display=False,close=True):
        self.path = path
        self.display = display
        self._close = close
        self.fig_stack = []
        self.axis_stack = []
        self.fig = None
        self.axis = None
    def __enter__(self):
        import matplotlib.pyplot as plt
        self.fig_stack.append(plt.gcf())
        self.axis_stack.append(plt.gca())
        if self.fig is None:
            self.fig = plt.figure()
            self.axis = self.fig.gca()
        else:
            plt.figure(self.fig.number)
        return self.fig
    def __exit__(self, type, value, tb):
        import matplotlib.pyplot as plt
        if self.path is not None and self.path != "":
            self.fig.savefig(self.path)
        if self.display:
            try:
                # trying to use ipython display
                IPython.core.display.display(self.fig)
            except:
                # otherwise presume that we run with some other gui backend. If we don't, nothing will happen.
                self.fig.show(warn=False)
        if self._close:
            plt.close(self.fig)
            self.fig = None
            self.fig_stack.pop()
            self.axis_stack.pop()
        else:
            fig = self.fig_stack.pop()
            ax = self.axis_stack.pop()
            plt.figure(fig.number)
            plt.sca(ax)
    def show(self,close=True):
        if self.path is not None and self.path != "":
            self.fig.savefig(self.path)
        try:
            # trying to use ipython display
            IPython.core.display.display(self.fig)
        except:
            # otherwise presume that we run with some other gui backend. If we don't, nothing will happen.
            self.fig.show(warn=False)
        if close == True:
            self.close()
    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)

def figure(path,display=False,close=True):
    """
    Can be used with the **with** statement::

        import litus
        import numpy as np
        import matplotlib.pylab as plt
        x = np.arange(0,10,0.1)
        with litus.figure("some_test.png") as f:
            plt.plot(x,np.cos(x))    # plots to a first plot
            with litus.figure("some_other_test.png"):
                plt.plot(-1*np.array(x)) # plots to a second plot
            plt.plot(x,np.sin(x))    # plots to the first plot again
            f.set_tight_layout(True) # using the figure object


    Or if they are to be used in an interactive console::


        import litus
        import numpy as np
        import matplotlib.pylab as plt
        x = np.arange(0,10,0.1)
        with litus.figure("some_test.png",display=True):
            plt.plot(x,np.cos(x))    # plots to a first plot
            with litus.figure("some_other_test.png",close=False):
                plt.plot(-1*np.array(x)) # plots to a second plot
            plt.plot(x,np.sin(x))    # plots to the first plot again

    Both of these figures will be displayed, but the second one will remain open and can be activated again.


    """
    return Figure(path,display=display,close=close)


#############################################
## Alert someone

def alert(msg,body="",icon=None):
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
        print(("ALERT: ", msg))

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
        print(( k.shape, i.shape))
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

def recgen(gen, fix_type_errors=True):
    """
        Iterates through generators recursively and flattens them.

        If `fix_type_errors` is True, `TypeError`s that are generated by
        iterating are ignored and the generator that raised the Exception
        is yielded instead. This is the case eg. for theano tensor variables.

        If you want `TypeError`s to be re-raised, set `fix_type_errors` to False.


        Aliases for this function: `generate_recursively` and `flatten_generator`.

        `recgen_enumerate` enumerates the result with tuples corresponding to each index used.

    """
    if not hasattr(gen,'__iter__'):
        yield gen
    else:
        try:
            for i in gen:
                for ii in recgen(i):
                    yield ii
        except TypeError:
            # oops, it seems it was not an iterable even if it had an __iter__ method...
            # this happens eg. with theano tensor variables as they try to trick you to sum them.
            if not fix_type_errors:
                raise # maybe you want this Exception?
            yield gen

generate_recursively = recgen
flatten_generator = recgen

def recgen_enumerate(gen,n=tuple(), fix_type_errors=True):
    """
        Iterates through generators recursively and flattens them. (see `recgen`)

        This function adds a tuple with enumerators on each generator visited.
    """
    if not hasattr(gen,'__iter__'):
        yield (n,gen)
    else:
        try:
            for i_,i in enumerate(gen):
                for element in recgen_enumerate(i,n+(i_,)):
                    yield element
        except TypeError:
            if not fix_type_errors:
                raise
            yield (n,gen)

#############################################
## list of dictionaries / dictioanries of lists

def list_of_dicts_to_dict_of_lists(list_of_dictionaries):
    """
        Takes a list of dictionaries and creates a dictionary with the combined values for 
        each key in each dicitonary.
        Missing values are set to `None` for each dicitonary that does not contain a key 
        that is present in at least one other dicitonary.

            >>> litus.list_of_dicts_to_dict_of_lists([{'a':1,'b':2,'c':3},{'a':3,'b':4,'c':5},{'a':1,'b':2,'c':3}])

            {'a': [1, 3, 1], 'b': [2, 4, 2], 'c': [3, 5, 3]}

        Shorthand: `litus.ld2dl(..)`
    """
    result = {}
    all_keys = set([k for d in  list_of_dictionaries for k in d.keys()])
    for d in list_of_dictionaries:
        for k in all_keys:
            result.setdefault(k,[]).append(d.get(k,None))
    return result

ld2dl = list_of_dicts_to_dict_of_lists

def dict_of_lists_to_list_of_dicts(dictionary_of_lists):
    """
        Takes a dictionary of lists and creates a list of dictionaries.
        If the lists are of unequal length, the remaining entries are set to `None`.

        Shorthand: `litus.dl2ld(..)`:

            >>> litus.dl2ld({'a': [1, 3, 1], 'b': [2, 4, 2], 'c': [3, 5, 3]})

            [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 4, 'c': 5}, {'a': 1, 'b': 2, 'c': 3}]

    """
    return [{key: dictionary_of_lists[key][index] if len(dictionary_of_lists[key]) > index else None for key in dictionary_of_lists.keys()}
         for index in range(max(map(len,dictionary_of_lists.values())))]

dl2ld = dict_of_lists_to_list_of_dicts


#############################################
## list of dictionaries / dictioanries of lists

def list_of_dicts_to_dict_of_lists(list_of_dictionaries):
    """
        Takes a list of dictionaries and creates a dictionary with the combined values for 
        each key in each dicitonary.
        Missing values are set to `None` for each dicitonary that does not contain a key 
        that is present in at least one other dicitonary.

            >>> litus.list_of_dicts_to_dict_of_lists([{'a':1,'b':2,'c':3},{'a':3,'b':4,'c':5},{'a':1,'b':2,'c':3}])

            {'a': [1, 3, 1], 'b': [2, 4, 2], 'c': [3, 5, 3]}

        Shorthand: `litus.ld2dl(..)`
    """
    result = {}
    all_keys = set([k for d in  list_of_dictionaries for k in d.keys()])
    for d in list_of_dictionaries:
        for k in all_keys:
            result.setdefault(k,[]).append(d.get(k,None))
    return result

ld2dl = list_of_dicts_to_dict_of_lists

def dict_of_lists_to_list_of_dicts(dictionary_of_lists):
    """
        Takes a dictionary of lists and creates a list of dictionaries.
        If the lists are of unequal length, the remaining entries are set to `None`.

        Shorthand: `litus.dl2ld(..)`:

            >>> litus.dl2ld({'a': [1, 3, 1], 'b': [2, 4, 2], 'c': [3, 5, 3]})

            [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 4, 'c': 5}, {'a': 1, 'b': 2, 'c': 3}]

    """
    return [{key: dictionary_of_lists[key][index] if len(dictionary_of_lists[key]) > index else None for key in dictionary_of_lists.keys()}
         for index in range(max(map(len,dictionary_of_lists.values())))]

dl2ld = dict_of_lists_to_list_of_dicts


#############################################
## plot things

def color_space(colormap, a, start=0.1, stop=0.9, length=None):
    if type(colormap) is str:
        from matplotlib import cm
        if colormap in cm.__dict__:
            colormap = cm.__dict__[colormap]
        else:
            colormap = cm.gnuplot
    if type(a) is int or type(a) is float:
        return colormap(np.linspace(start,stop,int(a)))
    return colormap(np.linspace(start,stop,length if length is not None else len(a)))

def colorate(sequence, colormap="", start=0, length=None):
    """ like enumerate, but with colors """
    n = start
    colors = color_space(colormap, sequence, start=0.1, stop=0.9, length=length)
    for elem in sequence:
        yield n, colors[n-start], elem
        n += 1

def plot_corridor(x,y,axis=1,alpha=0.2,**kwargs):
    x = np.array(x,dtype=float)
    from matplotlib.pylab import fill_between, plot
    fill_between(x, y.mean(axis)-y.std(axis), y.mean(axis)+y.std(axis),alpha=alpha,**kwargs)
    plot(x, y.mean(axis)+y.std(axis),'-',alpha=alpha,**kwargs)
    plot(x, y.mean(axis)-y.std(axis),'-',alpha=alpha,**kwargs)
    plot(x,y.mean(axis),**kwargs)

#################################################
## Parameter / Data Collectors
##
##  They still need a catchy name!
##


def unnumpyfy(x):
    if type(x) is dict:
        return {k:unnumpyfy(x[k]) for k in x.keys()}
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
        secondary_keys = []
        if type(key) is tuple:
            secondary_keys = key[1:]
            key = key[0]
        filename = self(key)
        if filename.endswith('.npz') or filename.endswith('.npy'):
            o = np.load(filename)
            r = o
            if type(o) == np.lib.npyio.NpzFile:
                if len(secondary_keys) > 0:
                    if len(secondary_keys) > 1:
                        r = {k: o[k] for k in secondary_keys if k in o.keys()}
                    else:
                        r = o[secondary_keys[0]]
                else:
                    if o.keys() == ['arr_0']:
                        r = o['arr_0']
                    else:
                        r = {k: o[k] for k in o.keys()}
                o.close()
            return r
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
                n = str(len(self.containers))
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
        cs = yaml.dump([unnumpyfy(c.dumpd()) for c in self.containers] + [{'root_parameters':unnumpyfy(self.parameters)}])
        if self.filename is None:
            return cs
        with open(self.path+"/"+self.filename,'w') as f:
            f.write(cs)
    def load(self):
        import yaml
        self.containers = []
        with open(self.path+"/"+self.filename,'r') as f:
            for c in yaml.load(f):
                if 'name' in c.keys():
                    pd = PDContainer(parent=self)
                    self.containers.append(pd.loadd(c))
                else:
                    if 'root_parameters' in c.keys():
                        self.parameters.update(c['root_parameters'])
                    # otherwise maybe a Warning? Maybe not?
        return self
    def create_path(self,container,key):
        return self.path+"/."+ container.name + "_" + key
    def param(self,key,default=None):
        """for accessing global parameters"""
        if key in self.parameters:
            return self.parameters[key]
        return default


class Lists:
    """

    This class provides nested Lists that can be turned into n-dimensional numpy arrays.
    The class is not and does not behave a like a numpy array.

    When called as a context manager (with `with ... as ...:`), a new list 
    is added and reference to the new list is returned. Arbitrary things can be added to
    the list and once the context is closed the list itself is appended to the 
    list one layer higher in the hierarchy.

    To structure data, context manager calls can be used inside each other to nest the lists.
    In this case, only the innermost list should be used, such that the resulting 
    nested list has equal dimensions in all elements.

    To access the latest list (after a context block was closed), `array()` 
    can be used.  `array()` takes an index argument `n` which list should be taken, so `array(-2)` 
    gives the list from the previous iteration.
    The first layer (-0) of the list is hidden: even outside of all context managers the 
    default will give you the list of the last closed context.
    If instead all lists should be accessed while the context block 
    is active (or the object is used outside of a context block), you can either
    access the `.list` attribute itself or call `.array(None)`.

    The functions `.transpose()` and `.mean()` also accept `n` as an argument and 
    behave identical to `.array(n).transpose()` and `.array(n).mean(dims)`.
    Note that `transpose` inverts the order of all the dimensions.

    Example (recommended usage)::

        import litus
        reload(litus)
        lc = litus.Lists()
        with lc('First'):
            for k in range(2):
                with lc('Second'):
                    for j in range(2):
                        with lc('Third') as l:
                            for i in range(2):
                                    l.append(i+j*10+k*100)
        print lc.array()
        print lc.array().shape
        print lc['Second'] # index of named dimension
        print lc.mean((1,2))

    Or decorating the `range` generators::

        lc = litus.Lists()
        for k in lc.generator(range(2),'First'):
            for j in lc.generator(range(2),'Second'):
                for i in lc.generator(range(2),'Third'):
                    l.append(i+j*10+k*100)
        
    Gives::

        [[[  0   1]
          [ 10  11]]

         [[100 101]
          [110 111]]]
        (2, 2, 2)
        1
        [   5.5  105.5]


    The object contains a `stack` of the hierarchically ordered lists, such that when
    a context is closed, the active `list` is appended.

    To make things more complicated, you can also use `+= n` and `-= n` to 
    enter or exit `n` many contexts. `a_list = list_obj + 1` evaluates to 
    the newly created list. If you use this feature you have to make sure yourself that
    you get out of all the contexts that you created. In turn, this does not require
    you to indent your code. Also you can quickly create ndarrays of certain shapes::


        lc = litus.Lists()
        for k in range(2):
            lc+=3 # indent three times
            for j in range(2):
                l = lc+1 # indent once more and return the list
                for i in range(2):
                        l.append(i+j*10+k*100)
                lc-= 1
            lc-=3
        lc-=2 # de-indent twice to close the context and add an extra dimension
        print lc.array().shape

    Output::

        (1, 2, 1, 1, 2, 2)

    Unlike the normal exit from a context, if the stack is empty, the list is inserted 
    into an empty list. (Thus, subtraction and contexts are not fully interchangable)
    It might be a good idea to add comments to the code when using this feature.

    """
    def __init__(self):
        self.stack = []
        self.list = []
        self.dimension_names = []
        self.dimension_values = []
    def array(self,n=-1):
        """returns a numpy array created from the list that was closed last (if `n`=-1).
        If `n` is None, the lists of the current level are returned as an array.
        Other lists in the current level can also be accessed with `n` as the specific index.
        """
        if n is None:
            return np.array(self.list)
        return np.array(self.list)[n]
    def transpose(self,n=-1):
        return self.array(n).transpose()
    def mean(self,dims=None,n=-1):
        dims = self._check_dims(dims)
        return np.mean(self.array(n),dims)
    def sum(self,dims=None,n=-1):
        dims = self._check_dims(dims)
        return np.sum(self.array(n),dims)
    def nanmean(self,dims=None,n=-1):
        dims = self._check_dims(dims)
        return np.nanmean(self.array(n),dims)
    def nansum(self,dims=None,n=-1):
        dims = self._check_dims(dims)
        return np.nansum(self.array(n),dims)
    def append(self,elem):
        self.list.append(elem)
    def insert(self,index,elem):
        self.list.insert(elem)
    def _check_dims(self,dims):
        if type(dims) in [list,tuple]:
            return tuple([self[d] if type(d) is str else d for d in dims])
        return dims
    def __getitem__(self,key):
        for i,n in enumerate(self.dimension_names):
            if n == key:
                return i
        raise Exception("Did not find key")
    def __call__(self,dimension_name='',dimension_values=None):
        if len(self.dimension_names) <= len(self.stack):
            self.dimension_names.append(dimension_name)
            self.dimension_values.append(dimension_values)
        else:
            self.dimension_names[len(self.stack)] = dimension_name
            self.dimension_values[len(self.stack)] = dimension_values
        return self
    def generator(self,gen,*args,**kwargs):
        """
            Use this function to enter and exit the context at the beginning and end of a generator.

            Example::

                li = litus.Lists()
                for i in li.generator(range(100)):
                    li.append(i)

        """
        with self(*args,**kwargs):
            for i in gen:
                yield i
    def __iadd__(self,n):
        if n < 0:
            return self.__isub__(-n)
        for i in range(n):
            self.__enter__()
        return self
    def __add__(self,n):
        if n < 0:
            return self.__sub__(-n)
        for i in range(n):
            l = self.__enter__()
        return l
    def __isub__(self,n):
        if n < 0:
            return self.__iadd__(-n)
        for i in range(n):
            if len(self.stack) == 0:
                self.stack = [[]]
            self.__exit__(None,None,None)
        return self
    def __sub__(self,n):
        if n < 0:
            return self.__add__(-n)
        for i in range(n):
            if len(self.stack) == 0:
                self.stack = [[]]
            self.__exit__(None,None,None)
        return self
    def __enter__(self):
        self.stack.append(self.list)
        self.list = []
        return self.list
    def __exit__(self, type, value, tb):
        if len(self.stack) > 0:
            l = self.list
            self.list = self.stack[-1]
            self.list.append(l)
            self.stack.pop()
        if tb is None:
            pass
        else:
            return False
    def exit_all(self):
        "Leaves all unclosed contexts."
        while len(self.stack) > 0:
            self.__exit__(None,None,None)
