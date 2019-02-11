import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import string

class EmptyContext(object):
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class Figure(object):
    def __init__(self,figsize=(8, 6),rc_context=None):
        self.n = 0
        self.ax = None
        self.ctx = [] # stack
        self.letter_pos = [-0.1, 1.0]
        self.fig =  plt.figure(figsize=figsize)
        self.rc_context = rc_context
    def style(self):
        if self.rc_context is None:
            return EmptyContext()
        elif type(self.rc_context) is dict:
            return mpl.rc_context(self.rc_context)
        else:
            return self.rc_context
    def gridspec(self, rows, cols, width_ratios=None, do_letters=True):
        with self:
            self.gs = mpl.gridspec.GridSpec(rows, cols, width_ratios=width_ratios)
            if do_letters:
                for i in range(np.prod(self.gs.get_geometry())):
                    ax = self.subplot(i)
                    ax.text(self.letter_pos[0],
                            self.letter_pos[1],
                            string.ascii_uppercase[i],
                            transform=ax.transAxes,
                            size=20, weight='bold')
        self.subplot(-1)
    def letter(self, n):
        ax = plt.gca()
        ax.text(self.letter_pos[0],
                self.letter_pos[1], n, transform=ax.transAxes,
        size=20, weight='bold')
    def gca(self):
        return self.ax
    def subplot(self,n=None):
        plt.figure(self.fig.number)
        if n is None:
            n = self.n + 1
        self.n = n%(np.prod(self.gs.get_geometry()))
        self.ax = plt.subplot(self.gs[self.n])
        return self.ax
    def __call__(self,n=None):
        self.subplot(n)
        return self
    def next(self):
        self.subplot()
        return self
    def __next__(self):
        return self.next()
    def __enter__(self,n=None):
        self.ctx.append(self.style())
        self.ctx[-1].__enter__()
        return self.ax
    def __exit__(self, *args):
        ctx = self.ctx.pop(-1)
        ctx.__exit__(*args)
        return

class Gnuplot2Figure(Figure):
    def __init__(self, n=100, figsize=(8, 6), colormap=mpl.cm.gnuplot2, rc_context={}):
        from cycler import cycler
        rc_context['axes.prop_cycle'] = cycler(color=map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),
                            colormap(np.linspace(0.1,0.9,n))))
        super(Gnuplot2Figure,self).__init__(figsize=figsize,rc_context = rc_context)
