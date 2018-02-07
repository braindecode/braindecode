import numpy as np
import torch

def calc_Hout(Hin, kernel, stride, dilation):
    Hout = np.floor((Hin-dilation*(kernel-1)-1)/stride+1)
    return Hout

def calc_Hin(Hout, kernel, stride, dilation):
    Hin = np.ceil((Hout-1)*stride+1+dilation*(kernel-1))
    return Hin

def calc_receptive_field_size(model,layer_ind,start_receptive_field=np.ones((2))):
    """
    Calculate receptive field size for unit in specific layer of the network
    Only tested for 2d convolutions/poolings. Dimshuffle operations may lead to a wrong result

    model: Network model
    layer_ind: Index of the layer of interest in model.children()
    start_receptive_field: How many units are looked at in specified layer (default: [1,1])

    Returns:
    receptive_field_size: [HxW] in the input layer
    """
    receptive_field = start_receptive_field
    children = list(model.children())[:layer_ind][::-1]
    for child in children:
        if isinstance(child,torch.nn.Sequential):
            receptive_field = calc_receptive_field(child,-1)
        elif isinstance(child,torch.nn.Conv2d) or isinstance(child,torch.nn.MaxPool2d) or isinstance(child,torch.nn.AvgPool2d):
            receptive_field = calc_Hin(receptive_field,np.asarray(child.kernel_size),
                                       np.asarray(child.stride), np.asarray(child.dilation))
            
    receptive_field_size = receptive_field.astype(np.int)
    return receptive_field_size

def get_max_act_index(activations,unique_per_input=True,n_units=None):
    """
    Retrieve index of maximum activation in a feature map

    activations: [Nx1xHxW] can only take 1 filter
    unique_per_input: Specifies if only 1 index (maximum) for each input is returned (default: True)
    n_units: How many indeces are returned in total. If None all (default: None)

    Returns:
    units: [Nx4] Input_ixFilter(0)xH_ixW_i indeces of the units
    units_activation: Activation of the units
    """
    assert len(activations.shape)==4,"Has to be 4d array"
    assert activations.shape[1] == 1,"Can only handle individual filter activations"
    
    activations_sorted = activations.argsort(axis=None)[::-1]
    activations_sorted_ind = np.unravel_index(activations_sorted,activations.shape)
    unique_ind = np.arange(len(activations_sorted_ind[0]))
    if unique_per_input:
        a,unique_ind = np.unique(activations_sorted_ind[0],return_index=True)
        unique_ind = sorted(unique_ind)

    if n_units==None:
        n_units = len(unique_ind)
    activations_sorted_ind = np.asarray(activations_sorted_ind).T
    units = activations_sorted_ind[unique_ind[:n_units],:].astype(np.int)
    units_activation = activations.flat[activations_sorted[unique_ind[:n_units]]]

    return units,units_activation

def calc_receptive_field_for_units_2d(units,receptive_field_size):
    recptive_field_tmp = receptive_field_size[np.newaxis,np.newaxis,:,:]
    start_inds = units[:,0,]
    stop_inds = start_inds+receptive_field_tmp
        
    return start_inds,stop_inds
    
def get_input_windows_from_units_2d(inputs,units,receptive_field_size):
    """
    Cut input windows in receptive field of specified units from inputs

    inputs: [NxCxHxW] InputsxChannelsxTimexW
    units: [Mx4] unit indeces specifying Input and time indeces.
                 Second dimension consists if InputxFilter(1)xHxW indeces. Can only handle 1 filter.
    receptive_field_size: Size of receptive field of units on input

    Returns
    windows: Cut input windows
    """
    windows = np.zeros((units.shape[0],inputs.shape[1],receptive_field_size[0],receptive_field_size[1]))
    for i,unit in enumerate(units):
        windows[i] = inputs[unit[0],:,
                            unit[2]:unit[2]+receptive_field_size[0],
                            unit[3]:unit[3]+receptive_field_size[1]]
    return windows