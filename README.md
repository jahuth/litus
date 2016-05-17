# The `litus` utils package

## `litus.cartesian` and other iterator functions

Creates the [cross product of lists](http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays):

```
litus.cartesian(([1, 2, 3], [4, 5], [6, 7]))
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
```


 * `litus.cartesian_dicts` does the same for a dictionary of lists and returns a list of dictionaries
 * `litus.icartesian` uses inverted place values
 * `litus.cartesian_to_index`/`icartesian_to_index` takes a tuple and calculates the appropriate index
 * `litus.fillzip` behaves like `zip`, but continues to iterate until the longest list ends instead of the shortest
 * `litus.recgen` iterates through generators recursively
     - `recgen_enumerate` also yields a tuple of indizes
 * `litus.colorate` is like `enumerate`, but also returns a color from a matplotlib color map

## Plotting

### `litus.figure` context manager

Provides a context manager for matplotlib Figures:
```
import litus
import numpy as np
import matplotlib.pylab as plt
x = np.arange(0,10,0.1)
with litus.figure("some_test.png",display=True) as f:
    plt.plot(x,np.cos(x))    # plots to a first figure
    with litus.figure("some_other_test.png",close=False):
        plt.plot(-1*np.array(x)) # plots to a second figure
    plt.plot(x,np.sin(x))    # plots to the first figure again
    f.set_tight_layout(True) # using the figure object of the first figure
```

Figures will be automatically closed and saved, if not specified otherwise.
If the path provided is `None` or an empty string, no figure is saved.
the keyword argument `display=True` will attempt to use the current matplotlib backend to show the plot on exit.

### `litus.animate`

Takes a 3d numpy array and creates an `imshow` plot for each frame (along the first dimension).
The result is converted into an mp4 and inserted into iPython/Jupyter notebooks.
Instead of a 3d numpy array, a list can be provided and a function that creates a plot for each frame.

## IPython history management

 * snip
 * snip_this
 * unsnip

## `litus.Lists`

Provides a context manager for nested lists.

```
lc = litus.Lists()
with lc:
    #first level
    for a in range(2):
        with lc:
            # second level
            for b in range(4):
                with lc:
                    # third level
                    for c in range(8):
                        with lc as l:
                            # fourth level: l is a list that elements can be appended to
                            for d in range(16):
                                l.append((a,b,c,d))
                            # at the end of this context, the list will be added to the one higher up in the hierarchy
print lc.array().shape  # will print (2, 4, 8, 16, 4)
print lc.array()        # will print a very long 5d array ranging from [0, 0, 0, 0] to [ 1  3  7 15]
```

A more extended example with named dimensions:

```
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
```

Instead of using it as a context manager, levels can also be increased like this:
```
lc = litus.Lists()
lc += 5 # increase by five levels
lc.list.append(0)
lc -= 5 # decrease by five levels
print lc.array() # prints [[[[[0]]]]]
```

### `litus.PDContainerList`

Provide a parameter - data mapping for result files. A json index file is used to store all parameter mappings and for each parameter combination, result file names can be generated and accessed.

## `litus.spikes`

Provides a labeld spike class for working with sparse point data.