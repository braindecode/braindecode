import os
import errno
import numpy as np
from copy import deepcopy


def deepcopy_xarr(xarr):
    """
    Deepcopy for xarray that makes sure coords and attrs
    are properly deepcopied.
    With normal copy method from xarray, when i mutated
    xarr.coords[coord].data it would also mutate in the copy
    and vice versa.
    Parameters
    ----------
    xarr: DateArray

    Returns
    -------
    xcopy: DateArray
        Deep copy of xarr
    """
    xcopy = xarr.copy(deep=True)

    for dim in xcopy.coords:
        xcopy.coords[dim].data = np.copy(xcopy.coords[dim].data)
    xcopy.attrs = deepcopy(xcopy.attrs)
    for attr in xcopy.attrs:
        xcopy.attrs[attr] = deepcopy(xcopy.attrs[attr])
    return xcopy


class FuncAndArgs(object):
    """Container for a function and its arguments. 
    Useful in case you want to pass a function and its arguments 
    to another function without creating a new class.
    You can call the new instance either with the apply method or 
    the ()-call operator:
    
    >>> FuncAndArgs(max, 2,3).apply(4)
    4
    >>> FuncAndArgs(max, 2,3)(4)
    4
    >>> FuncAndArgs(sum, [3,4])(8)
    15
    
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    
    def apply(self, *other_args, **other_kwargs):
        all_args = self.args + other_args
        all_kwargs = self.kwargs.copy()
        all_kwargs.update(other_kwargs)
        return self.func(*all_args, **all_kwargs)
        
    def __call__(self, *other_args, **other_kwargs):
        return self.apply(*other_args, **other_kwargs)
    
    
class GetAttr(object):
    """ Hacky class for yaml to return attr of something.
    Uses new to actually return completely different object... """
    def __new__(cls, obj, attr):
        return getattr(obj, attr) 

class ApplyFunc(object):
    """ Hacky class to wrap function call in object for yaml loading... """
    def __new__(cls, func,args, kwargs):
        return func(*args, **kwargs)

def add_message_to_exception(exc, additional_message):
    #  give some more info...
    # see http://www.ianbicking.org/blog/2007/09/re-raising-exceptions.html
    args = exc.args
    if not args:
        arg0 = ''
    else:
                    
        arg0 = args[0]
    arg0 += additional_message
    exc.args = (arg0, ) + args[1:]
    
def unzip(l):
    return zip(*l)

def dict_compare(d1, d2):
    """From http://stackoverflow.com/a/18860653/1469195"""
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same

def dict_equal(d1,d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    return (intersect_keys == d2_keys and intersect_keys == d1_keys and
        len(modified) == 0)
    
def dict_is_subset(d1,d2):
    added, removed, modified, same = dict_compare(d1, d2)
    return (len(added) == 0 and len(modified) == 0)
    
def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    http://stackoverflow.com/a/26853961
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def touch_file(path):
    # from http://stackoverflow.com/a/12654798/1469195
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, 'a'):
        os.utime(path, None)


def to_tuple(sequence_or_element, length=None):
    if hasattr(sequence_or_element, '__len__'):
        assert length is None
        return tuple(sequence_or_element)
    else:
        if length is None:
            return (sequence_or_element, )
        else:
            return (sequence_or_element,) * length
    

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def select_inverse_inds(arr, inds):
    mask = np.ones(len(arr),dtype=bool)
    mask[inds] = False
    return arr[mask]