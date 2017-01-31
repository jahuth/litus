import numpy as np

def _replace_operators(k):
    import re
    pattern = re.compile("|".join(['-','\.',' ']), re.M)
    return pattern.sub('_', k)

class O(object):
    """
        A friendly object that provides easy access to it's members.

        It provides similar methods as a dictionary, but they are named with a leading underscore
        (._keys() instead of .keys(), etc.)

    """
    def __init__(self,**kwargs):
        self.__dict__ = kwargs
    def __call__(self,**kwargs):
        import copy
        d = copy.copy(self.__dict__)
        d.update(kwargs)
        return O(**d)
    def __repr__(self):
        return repr(self.__dict__)
    def __elem__(self,x):
        return self.__dict__.__elem__(x)
    def __getitem__(self,ind):
        self.__dict__.__getitem__(ind)
    def __setitem__(self,ind,value):
        self.__dict__.__setitem__(ind,value)
    def _keys(self):
        return self.__dict__.keys()
    def _items(self):
        return self.__dict__.items()
    def _iteritems(self):
        return self.__dict__.iteritems()
    def _values(self):
        return self.__dict__.values()
    def _iterkeys(self):
        return self.__dict__.iterkeys()
    def _setdefault(self,*args,**kwargs):
        return self.__dict__.setdefault(*args,**kwargs)
    def _update(self,*args,**kwargs):
        return self.__dict__.update(*args,**kwargs)
    def _pop(self,*args,**kwargs):
        return self.__dict__.pop(*args,**kwargs)
    def _popitem(self,*args,**kwargs):
        return self.__dict__.popitem(*args,**kwargs)
    def _has_key(self,*args,**kwargs):
        return self.__dict__.has_key(*args,**kwargs)


_magic_keys = {
    'lt': lambda x,y: x < y,
    'lte': lambda x,y: x <= y,
    'gt': lambda x,y: x > y,
    'gte': lambda x,y: x >= y,
    'eq': lambda x,y: x == y,
    'in': lambda x,y: x in y,
    'not': lambda x,y: x != y,
    'startswith': lambda x,y: x.startswith(y),
    'evals': lambda x,y: y(x)
}   
class ContainerList(object):
    def __init__(self,list_of_things,list_of_indizes):
        from .__init__ import ld2dl, dl2ld
        import copy
        self.contained_list = list_of_things
        if type(list_of_indizes) == list or type(list_of_indizes) == tuple:
            self.indizes = ld2dl(list_of_indizes) # list of dicts or dict of lists?
        elif type(list_of_indizes) == dict:
            self.indizes = copy.copy(list_of_indizes)
        else:
            raise Exception('Indices have to be a list of dicts or a dict of lists')
        self.indizes = dict([(str(_replace_operators(k)),v) for (k,v) in self.indizes.items()])
    def __len__(self):
        return len(self.contained_list)
    def keys(self):
        return self.indizes.keys()
    @property
    def levels(self):
        return O(**dict([(k,sorted(np.unique(v))) for (k,v) in self.indizes.items()]))
    def __getitem__(self,ind):
        if type(ind) is list or type(ind) is np.ndarray:
            return [c for (c,i) in filter(lambda x:x[1],zip(self.contained_list,ind))]
        if ind in self.indizes.keys():
            return self.indizes[ind]
        return self.contained_list.__getitem__(ind)
    def add_index(self,name, values):
        self.indizes[name] = values
    def remove_index(self,name):
        del self.indizes[name]
    def rename_index(self,old_name,new_name):
        self.add_index(new_name,self.indizes[old_name])
        del self.indizes[old_name]
    @property
    def index(self):
        return O(**dict([(k,k) for k in self.indizes.keys()]))
    
class IndexFilter(object):
    def __init__(self,container_index,key,subkeys=_magic_keys.keys()):
        self._ci = container_index
        self._key = key
        self._subkeys = subkeys
        self.__dict__.update(dict([(s,lambda x, s=s, k=self._key: self._ci.__call__(**{k+'__'+s: x})) for s in self._subkeys]))
        levels = self._ci.levels.__dict__.get(self._key,[])
        self.__doc__ = str(len(levels))+' Elements: '+str(levels)
    def __call__(self,x):
        return self._ci(**{self._key: x})

class IndexGeneratorManager(object):
    def __init__(self,container_index,**kwargs):
        self._ci = container_index
        self.__dict__.update(kwargs)
    def __call__(self,*args,**kwargs):
        context=kwargs.pop('context',None)
        return IndexGenerator(self._ci(**kwargs),context=context,*args)
    
def IndexGenerator(container_index,*keys,**kwargs):
    levels = container_index.levels.__dict__[keys[0]]
    context=kwargs.pop('context',None)
    if context is not None:
        with context:
            for l in levels:
                if len(keys) == 1:
                    yield container_index(**{keys[0]: l})
                else:
                    for ci in IndexGenerator(container_index(**{keys[0]: l}), context=context, *keys[1:]):
                        yield ci
    else:
        for l in levels:
            if len(keys) == 1:
                yield container_index(**{keys[0]: l})
            else:
                for ci in IndexGenerator(container_index(**{keys[0]: l}), *keys[1:]):
                    yield ci

class IndexGenerator(object):
    def __init__(self,container_index,*keys,**kwargs):
        context=kwargs.pop('context',None)
        def gen(container_index,*keys,**kwargs):
            levels = container_index.levels.__dict__[keys[0]]
            context=kwargs.pop('context',None)
            if context is not None:
                with context:
                    for l in levels:
                        if len(keys) == 1:
                            yield container_index(**{keys[0]: l})
                        else:
                            for ci in IndexGenerator(container_index(**{keys[0]: l}), context=context, *keys[1:]):
                                yield ci
            else:
                for l in levels:
                    if len(keys) == 1:
                        yield container_index(**{keys[0]: l})
                    else:
                        for ci in IndexGenerator(container_index(**{keys[0]: l}), *keys[1:]):
                            yield ci
        self.length = len(list(gen(container_index,*keys,**kwargs))) # maybe there is a better way to do this, but right now its fast enough
        self.gen = gen(container_index,*keys,context=context,**kwargs)
    def __len__(self):
        return self.length
    def __iter__(self):
        return self
    def next(self):
        return self.gen.next()
    


class Index(object):
    def __init__(self,container,indizes = None, set_values={}):
        self.c = container
        self.indizes = indizes
        if self.indizes is None:
            self.indizes = np.array([True]*len(self.c))
        self.set_values = set_values
    def __call__(self, **kwargs):
        import copy
        indizes = self.indizes.copy()
        set_values = copy.copy(self.set_values)
        for k,v in kwargs.items():
            new_index = []
            if k in self.c.indizes.keys():
                new_index = np.array([i == v for i in self.c.indizes.get(k,[])])
            for mk,mk_func in _magic_keys.items():
                if k.endswith('__'+mk):
                    new_index = np.array([mk_func(i,v) for i in self.c.indizes.get(k[:-(len(mk)+2)],[])])
                    break
            if len(new_index) == 0:
                raise Exception('Key %s not found!'%k)
            indizes = indizes * new_index
            set_values[k] = v
        return Index(self.c,indizes,set_values=set_values)
    def __getitem__(self,*args):
        return self.c.__getitem__(self.indizes).__getitem__(*args)
    def __len__(self):
        return np.sum(self.indizes)
    def random(self,n=1):
        indizes = np.random.permutation(len(self))[:n]
        return [self[i] for i in indizes]
    @property
    def filter(self):
        return O(**dict([(k,IndexFilter(self,k)) for k in self.c.indizes.keys()]))
    @property
    def value(self):
        return O(**dict([(k,np.unique(np.array(v)[self.indizes])[0]) for (k,v) in self.c.indizes.items() if len(np.unique(np.array(v)[self.indizes]))==1]))
    @property
    def choices(self):
        return O(**dict([(k,np.array(sorted(np.unique(np.array(v)[self.indizes])))) for (k,v) in self.c.indizes.items() if len(np.unique(np.array(v)[self.indizes]))>1]))
    @property
    def generate(self):
        return IndexGeneratorManager(self,**dict([(k,IndexGenerator(self,k)) for k in self.c.indizes.keys()]))
    @property
    def levels(self):
        return O(**dict([(k,np.array(sorted(np.unique(np.array(v)[self.indizes])))) for (k,v) in self.c.indizes.items()]))
    def __repr__(self):
        return 'Container with '+repr(np.sum(self.indizes))+'/'+repr(len(self.indizes))+' Elements selected.'
    def __add__(self,other):
        assert(self.c == other.c)
        new_indizes = self.indizes + other.indizes
        new_set_values = {}
        new_set_values.update(dict((k,v) for (k,v) in self.set_values.items() if other.set_values.get(k,None) == v))
        return Index(self.c, new_indizes, new_set_values)
    def __mul__(self,other):
        assert(self.c == other.c)
        new_indizes = self.indizes * other.indizes
        new_set_values = {}
        new_set_values.update(self.set_values)
        new_set_values.update(other.set_values)
        return Index(self.c, new_indizes, new_set_values)


def tree(s,args):
    if len(args) == 0 or args[0] not in s.choices.__dict__.keys():
        return s
    return { args[0]: dict([(a,tree(s(**{args[0]: a}),args[1:])) for a in s.choices.__dict__[args[0]]]) }

def tree_plot(t,indent='',print_leaves=False):
    if not hasattr(t,'keys'):
        if print_leaves:
            print(''.join([ indent, str(t) ]))
        return
    first_element = str(list(t.keys())[0])
    print(''.join([ str(indent),' \\',first_element,'in',str(np.array(t[first_element].keys()))]))
    for tk,tv in t[first_element].items():
        print(''.join([ str(indent),' |',first_element,'=',str(tk)]))
        tree_plot(tv,indent+' ')
def create(a_list,some_indizes):
    return Index(ContainerList(a_list,some_indizes))
