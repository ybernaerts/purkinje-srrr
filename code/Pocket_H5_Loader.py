import os, re, platform, h5py, warnings
from copy import deepcopy
from functools import reduce
import operator
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox


SysOS = platform.system() # Check the platform of the computer

## File/s browser functions
def OpenExplorer(filetype=[('All Files',"*")], folder=False, save=False, initialfile=None, initialdir=None, title=""):
    # filetype: List of tuples that defines the file sufixes to look on the window.
    # save: To open a window to save a file, instead of just opening.
    # initialfile: String to  name the file to save, just for the case 'save' is True.
    # initialdir: Initial folder from where the dialog starts.

    Options = {'filetypes':filetype, 'title':title}
    if not initialfile is None:
        Options.update({'initialfile':initialfile})
    if not initialdir is None:
        Options.update({'initialdir':initialdir})

    if SysOS == 'Windows' or SysOS == 'Linux':
        root = tk.Tk()
        root.withdraw()
        if not save:
            if folder:
                pth = filedialog.askdirectory()
            else:
                pth = filedialog.askopenfilename(**Options) # Open Explorer to find path of file
        else:
            pth = filedialog.asksaveasfilename(**Options) # Open Explorer to save file

        if not pth == '':
            return pth # return path
        else:
            raise ValueError('OpenExplorer -> Invalid file')
    elif SysOS == 'Darwin': # For Mac OS
        if initialfile is None:
            default_path = os.getcwd()
        else:
            default_path = initialfile

        if not save:
            pth = Mac_user_action(default_path, "Select").strip() # Open Explorer to find path of file
        else:
            pth = Mac_user_action(default_path, "Save").strip() # Open Explorer with default filename to save

        if not pth == '':
            return pth # return path
        else:
            raise ValueError('OpenExplorer -> Invalid file')
    else:
        raise ValueError('OpenExplorer -> Operative System is not recognized')

def OpenBatch(filetype=[('All Files',"*")], title=None):
    #### Open the Explorer to select the files to analyze in a batch
    root = tk.Tk(); root.withdraw()
    if title is None: title = 'Choose the files to analyze together'
    MultFiles = filedialog.askopenfilenames(parent=root, filetypes=filetype, title=title)
    if MultFiles == '':
        raise ValueError('OpenBatch Error -> Invalid Paths')
    strPaths = root.tk.splitlist(MultFiles)

    return list(strPaths)

def Save_Dict_to_HDF5(Dict, filename):
    # Modified from: https://codereview.stackexchange.com/questions/120802

    def recursively_save_dict_contents_to_group(h5file, path, dic):
        for key, item in dic.items():
            orig_type = type(item)
            if isinstance(item, pd.Series):
                item = item.to_numpy()
            if isinstance(item, datetime.datetime):
                TempList = item.strftime("%Y %m %d %H %M %S %f").split(" ")
                item = np.asarray([int(num) for num in TempList])
            if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, str, bytes, list, tuple)):
                if isinstance(item, (int, float, list, tuple)):
                    # Transforming list into numpy arrays
                    if isinstance(item, (list, tuple)):
                        if all([isinstance(elem, datetime.datetime) for elem in item]):
                            TempList = []
                            for elem in item:
                                Temp = elem.strftime("%Y %m %d %H %M %S %f").split(" ")
                                TempList.append([int(num) for num in Temp])
                            item = TempList
                    item = np.asarray(item)
                    if "U" in str(item.dtype): item = item.astype("S") # Change list of strings into format compatible with HDF5
                elif isinstance(item, (str, bytes)):
                    if "Attrs" in path:
                        prev = "/".join(path[:path.find("Attrs")].split("/")[:-1]) + "/"
                        if not prev in h5file: h5file.create_group(prev)
                        h5file[prev].attrs[key] = item
                    else: h5file[path + key] = item
                    continue
                if issubclass(item.dtype.type, np.integer) and not issubclass(item.dtype.type, np.int64):
                    # Change numpy integers to 64-bits for general compatibility
                    item = item.astype(np.int64)
                try:
                    if "Attrs" in path:
                        prev = "/".join(path[:path.find("Attrs")].split("/")[:-1]) + "/"
                        if not prev in h5file: h5file.create_group(prev)
                        h5file[prev].attrs[key] = copy.deepcopy(item)
                    else: h5file[path + key] = item
                except Exception as e:
                    raise TypeError('Save_Dict_to_HDF5 Error -> Cannot save "%s" with original %s type\n\tPath: %s'% (key,orig_type,path))
            elif isinstance(item, np.bool_):
                item = bool(item)
            elif isinstance(item, dict):
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
            else:
                raise TypeError('Save_Dict_to_HDF5 Error -> Cannot identify the object "%s" with original %s type\n\tPath: %s'% (key,orig_type,path))

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', Dict)

def Load_Dict_from_HDF5(filename, attrs=False):
    # Modified from: https://codereview.stackexchange.com/questions/120802
    # attrs: (Default False) If True, the routine will also load the attributes of each object.
    #        If "Root", it will only load the attributes of the first file node.

    def recursively_load_dict_contents_from_group(h5file, path, attrs):
        ans = {}
        if len(ans) == 0 and not attrs is False:
            ans["Root Attrs"] = {key:(elem if not isinstance(elem, h5py._hl.base.Empty) else []) for key, elem in dict(h5file['/'].attrs).items()}
            if "oo" in attrs: attrs = False
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
                if attrs: ans["%s_Attrs" % key] = dict(item.attrs)
            elif isinstance(item, h5py._hl.base.Empty):
                ans[key] = []
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', attrs)
                if attrs: ans[key]["Attrs"] = dict(item.attrs)
        return ans

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/', attrs)

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def dict_recursive_items(dictionary, level=0):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value, level)
            yield from dict_recursive_items(value, level+1)
        else:
            yield (key, value, level)

def findcommonstart(strlist, separator=None):

    if not separator is None:

        Splits = [elem.split(separator) for elem in strlist]
        MaxSplits = Splits[np.argmax([len(elem) for elem in Splits])]

        for split in Splits:
            CompSplits = []
            for num in range(len(MaxSplits)):
                if num + 1 > len(split): break
                if split[num] == MaxSplits[num]:
                    CompSplits.append(split[num])
                else:
                    break
            MaxSplits = deepcopy(CompSplits)
            if not len(MaxSplits) > 0:
                break

        return separator.join(MaxSplits)

    else:

        def getcommonletters(strlist):
            return ''.join([x[0] for x in zip(*strlist) \
                             if reduce(lambda a,b:(a == b) and a or None,x)])

        strlist = strlist[:]
        prev = None
        while True:
            common = getcommonletters(strlist)
            if common == prev:
                break
            strlist.append(common)
            prev = common

        return getcommonletters(strlist)

## Global Functions needed to extract/show/plot user data
def user_display_time(milliseconds, user_scale=['milli', 'sec']):
    intervals = {'weeks':604800000,  # 60 * 60 * 24 * 7
                 'days':86400000,    # 60 * 60 * 24
                 'hours':3600000,    # 60 * 60
                 'minutes':60000,
                 'seconds':1000,
                 'milliseconds':1}
    # Choose the levels that are masked by user:
    Names = [name for name in intervals.keys()]
    for idx in range(len(user_scale)):
        match = [i for i in Names if user_scale[idx] in i]
        try:
            user_scale[idx] = match[0]
        except:
            raise ValueError("user_display_time Error: user_scale has unmatched time strings")

    result = []
    for name, count in intervals.items():
        value = milliseconds // count
        if value > 0:
            milliseconds -= value * count
            if value == 1: suffix = name.rstrip('s')
            else: suffix = name
            if name in user_scale:
                result.append("{} {}".format(int(value), suffix))
    return ', '.join(result)

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n*multiplier + 0.5) / multiplier

def FirstCrossVal(data, thres, direc='increm'):
    # Function that returns the position of the first crossing of a threshold value in a data array
    # data: Numpy Array

    if 'in' in direc:
        above_threshold = data > thres
    else:
        above_threshold = data < thres

    react_tms = np.argmax(above_threshold)
    react_tms = react_tms - (~np.any(above_threshold)).astype(float)

    return int(react_tms)

def is_points_not_times(arr_test):
    # Function to test wether a vector contains point values or time values
    # It return 'True' for Point vectors and 'False' for Time Vectors

    Frac, Integ = np.modf(arr_test)
    if np.allclose(Frac, np.zeros(Frac.size)):
        return True
    else:
        return False

def extract_numbers_from_string(text, as_string=False):

    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', text)]

    if as_string:
        s = [str(num) for num in s]

    return s

def find_runs(data, elem):
    # Modified from https://stackoverflow.com/questions/24885092
    # data: 1-D numpy array
    # elem: The element repeated in the runs

    if np.isnan(elem):
        iselem = np.concatenate(([0], np.isnan(data).view(np.int8), [0]))
    elif np.isinf(elem):
        iselem = np.concatenate(([0], np.isinf(data).view(np.int8), [0]))
    else:
        iselem = np.concatenate(([0], np.equal(data, elem).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iselem))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def Phase_Shift(ph_data, shift=np.pi):
    if shift < 0: shift += 2*np.pi
    return (np.unwrap(ph_data) + shift) % (2 * np.pi)

def LinearInterp(Xdata, Ydata, NewXdata, kind='linear', fill_value='extrapolate'):
    # Function to interpolate a function to find some in-between values
    # Xdata: Sorted array of values from an independent variable
    # Ydata: Corresponding values of Xdata in a dependent variable
    # NewXdata: New independent values from the same variable in Xdata
    # kind, fill_value: Check types in
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    f = interp1d(Xdata, Ydata, kind=kind, fill_value=fill_value)
    return f(NewXdata)

def Event_Time_to_Value_Interpol(Timings, Time_Vect, Value_Vect, kind="linear"):
    # Function to transform continuos event timings into corresponding values
    # from a Time Serie (Values per Time pair) by interpolation.
    # Timings: Numpy array with the times when an event happens.
    # Time_Vect: Vector of continuos, sequential time used to time the events.
    # Value_Vect: Vector with the values assigned to each point of the Time_Vect.
    # kind: (Optional) Type of interpolation. See 'LinearInterp' function for more details

    Inter_Points = LinearInterp(Time_Vect, np.arange(Time_Vect.size), Timings, kind=kind)
    Value_Interp = LinearInterp(np.arange(Value_Vect.size), Value_Vect, Inter_Points, kind=kind)

    return Value_Interp

def Normalize(data, min=0, max=1):

    if data.size < 2:
        return data

    datamin = np.nanmin(data)
    datamax = np.nanmax(data)
    norm = np.zeros(np.size(data))
    if datamin == datamax:
        return norm
    temp = data[~np.isnan(data)]
    norm = data.copy()
    norm[~np.isnan(data)] = ((max-min)*(temp - datamin)/(datamax - datamin))+min

    return norm

def timenorm(ph1, ph2, slice):
    return np.linspace(ph1, ph2, num=slice.size)

def Phase_Calculator(Input_Signal, Stances, Swings, Single_PhRef = None):

    Input_Signal = Input_Signal.astype(float)

    if not Single_PhRef is None:
        if "St" in Single_PhRef:
            OutDict = nn.NearestNeigbourgh_1D(Stances, Stances, type='<', MinMaxDiff=(1, np.inf))
            Pairs = OutDict['Pairs']; PairsPnts = OutDict['Pairs Points']
        elif "Sw" in Single_PhRef:
            OutDict = nn.NearestNeigbourgh_1D(Swings, Swings, type='<', MinMaxDiff=(1, np.inf))
            Pairs = OutDict['Pairs']; PairsPnts = OutDict['Pairs Points']
    else:
        OutDict = nn.NearestNeigbourgh_1D(Stances, Swings, type='<', MinMaxDiff=(1, np.inf))
        PV_Pairs = OutDict['Pairs']
        OutDict = nn.NearestNeigbourgh_1D(Stances, Swings, type='>', MinMaxDiff=(1, np.inf))
        VP_Pairs = OutDict['Pairs']
        Pairs = sorted(list(set(PV_Pairs + VP_Pairs)), key=lambda x : min(x[0], x[1]))
    if len(Pairs) > 0:
        if not Single_PhRef is None:
            Phase1 = (0, 2*np.pi); Phase2 = (0, 2*np.pi)
        elif Pairs[0][0] < Pairs[0][1]: # Locomotion cycle starts with a Swing
            Phase1 = (np.pi, 2*np.pi); Phase2 = (0, np.pi)
            CycleStart = 'Sw'
        else: # Locomotion cycle starts with a Stance
            Phase1 = (0, np.pi); Phase2 = (np.pi, 2*np.pi)
            CycleStart = 'St'

    TimeNorm = np.zeros(Input_Signal.size)
    for idx in range(len(Pairs)):
        Start = min(Pairs[idx][0], Pairs[idx][1])
        End = max(Pairs[idx][0], Pairs[idx][1])
        if idx % 2 == 0:
            TimeNorm[Start:End+1] = timenorm(Phase1[0], Phase1[1], Input_Signal[Start:End+1])
        else:
            TimeNorm[Start:End+1] = timenorm(Phase2[0], Phase2[1], Input_Signal[Start:End+1])

    return TimeNorm

def Speed_Duration_Threshold(Time_Vect, Body_Speed, speed_time_thres=(0.,0.)):
    # Function to extract the mask of indexes where the wheel speed matches a profile
    # Time_Vect: Vector with the time sequence of the common continuos time series in the datasets.
    # Body_Speed: Vector with the body speed, to be used for the 'speed_time_thres' data selection criteria.
    # speed_time_thres: Tuple with format '(speed [m/s], duration [s])' to define a double selection criteria
    #                   based on body speed and the continued persistance of that speed in time (duration).

    acq_freq = 1/np.median(np.diff(Time_Vect))

    # Speed/Time Threshold
    Idxs_Speed = np.where((Body_Speed >= speed_time_thres[0]))[0].astype(int)
    if not Idxs_Speed.size > 0: return np.zeros(0)
    Runs = find_runs(np.insert(np.diff(Idxs_Speed),0,1), 1)
    cont_thres = int(np.floor(speed_time_thres[1]*acq_freq))
    seq_list = [np.arange(Body_Speed.size)[Idxs_Speed[run[0]:run[1]]] for run in Runs if (run[1] - run[0]) >= cont_thres]
    if len(seq_list) == 0: return np.zeros(0)
    Idxs_Speed_Time = np.concatenate(seq_list).astype(int)
    if not Idxs_Speed_Time.size > 0: return np.zeros(0)

    return Idxs_Speed_Time

def Colormap_Rasters(DataDict, Time_Vect, Ref_key, Events_keys=[], Cont_key=[], Ord_key=[], asc_sort=True, phase_shift=True, mask_idxs=None, hvbins=(100,100), win_t=(-0.1,0.1), splog=False, color_lkup=None, plot=False):
    # Function to create raster/colormap plots from a Timing vector and a Reference (in phase or time).
    # DataDict: Nested dictionary with leafs containing data to use for plotting.
    # Time_Vect: Vector with the time sequence of the common continuos time series in the datasets.
    # Ref_key: Keymap of the data dictionary to extract a single reference vector in time events or phases (X-axis).
    # Events_keys: List of keymaps to extract all discrete events to overlay on the plot.
    # Cont_key: Keymap of the data dictionary to extract a continuos vector to plot with a 2D colormap.
    # Ord_key: (Optional) Keymap of the data dictionary to extract a vector to use for ordering the Y-axis.
    # asc_sort: If True, sorts the raster/colormap by ascending order
    # phase_shift: In case the reference data is a phase, apply a PI (180 deg.) shift.
    # mask_idxs: Vector with indexes to extract from the data specifically.
    # hvbins: Tuple with the number of horizontal and vertical bins for the colormap.
    # win_t: Tuple specifying the time in seconds of the segments before and after a time reference for the raster (if required)
    # splog: Transform the colormap data in log10 values.
    # color_lkup: Dictionary with paired variable substrings and colors to represent them. Otherwise all variables 'black' as default.
    # plot: If True, plotting is done within the run of this function.

    # Examples of mapkeys in centralized LocoPkj files
    # Ref_key = ['continuous','behavior','paws','FR_slopethr_stsw_phase']
    # Ref_key = ['discrete','behavior','paws','FL_slopethr_st_cumul_times']
    # Ord_key = ['discrete','behavior','paws','FL_slopethr_sw_cumul_times']
    # Ord_key = ['continuous','behavior','belts','belts_speed_r']
    # Cont_key = ['continuous', 'ephys', 'u1', 'ss_rate']
    # Events_keys = [['discrete','behavior','paws','FL_slopethr_sw_cumul_times'],
    #                ['discrete','behavior','paws','HR_slopethr_sw_cumul_times'],
    #                ['discrete','ephys','u1','ss_cumul_times']]

    # Gathering required data from one file
    Ref_Vect = getFromDict(DataDict, Ref_key).flatten()
    if not len(Ord_key) == 0: Ord_Vect = getFromDict(DataDict, Ord_key).flatten()
    if not len(Cont_key) == 0: Cont_Vect = getFromDict(DataDict, Cont_key).flatten()
    Event_List = [getFromDict(DataDict, mapKeys).flatten() for mapKeys in Events_keys]

    # Adjust vectors if requested by user
    Bool_EvOrd = False
    if not len(Ord_key) == 0:
        if Ord_Vect.shape != Time_Vect.shape and Ord_Vect.size > 0:
            Ord_Vect = Ord_Vect.tolist(); Bool_EvOrd = True
            if not Ord_key in Events_keys:
                Events_keys.append(Ord_key)
                Event_List.append(getFromDict(DataDict, Ord_key).flatten())
    if splog and not len(Cont_key) == 0: Cont_Vect = np.log10(Cont_Vect)
    if mask_idxs is None: mask_idxs = np.arange(Time_Vect.size)
    if win_t[0] > 0 or win_t[1] < 0: win_t = (-win_t[0], np.abs(win_t[1]))
    acq_freq = 1/np.median(np.diff(Time_Vect))

    # Extract segments of the raster/colormap
    Bool_Ref_Phase = False
    if Time_Vect.size == Ref_Vect.size: # For continuos reference vector
        Bool_Ref_Phase = True
        if np.nanmin(Ref_Vect) == 0 and np.nanmax(Ref_Vect) <= 2*np.pi: # For Phase reference
            if phase_shift: Ref_Vect = Phase_Shift(Ref_Vect, shift=np.pi)
            Ref_Vect = Normalize(Ref_Vect)
            Phase_Segmentation = np.where(np.append(np.diff(Ref_Vect),0) > 0,
                                          np.ones(Ref_Vect.size), np.zeros(Ref_Vect.size))
            Segments = find_runs(Phase_Segmentation, 1).astype(int)
            Segments[:,1] = Segments[:,1] + 1 # Include last point given cyclical nature of phase
            if Segments[-1][1] > Ref_Vect.size:
                Segments[-1][1] = Ref_Vect.size
        else:
            raise ValueError("Colormap_Rasters Abort: Not a valid reference vector")
    else: # For discrete / event-based reference vector
        if is_points_not_times(Ref_Vect): Ref_Vect = np.take(Time_Vect, Ref_Vect.astype(int))
        Inter_Points = LinearInterp(Time_Vect, np.arange(Time_Vect.size), Ref_Vect, kind='linear')
        pts_bef = np.ceil(win_t[0]*acq_freq); pts_aft = np.ceil(win_t[1]*acq_freq)+1
        Segments = np.stack((Inter_Points + pts_bef, Inter_Points + pts_aft), axis=1).astype('int')
        Segments = np.delete(Segments, np.unique(np.where(Segments < 0)[0]), axis=0) # delete Segments with negative times
        Segments = np.delete(Segments, np.unique(np.where(Segments > Time_Vect.size)[0]), axis=0) # delete Segments beyond the recorded time
        if Segments.size == 0: raise ValueError("Colormap_Rasters Abort: Reference event vector does not contain any valid time points")

    # Extract slices from each segment
    Mat_Col = np.zeros(0); ord_info = []; ord_loc = []
    if not color_lkup is None:
        Event_Colors = {ev[-1]:color for var, color in color_lkup.items() for ev in Events_keys if var in ev[-1]}
    else:
        Event_Colors = {ev[-1]:'black' for ev in Events_keys}
    Events_Dict = {ev[-1]:{'Events':np.zeros((0,2)), 'Color':Event_Colors[ev[-1]]} for ev in Events_keys}
    if Bool_Ref_Phase:
        min_ref = np.nanmin(Ref_Vect); max_ref = np.nanmax(Ref_Vect)
    else:
        min_ref = win_t[0]; max_ref = win_t[1]

    for idx in range(Segments.shape[0]):
        if not np.any(np.isin(mask_idxs, Segments[idx, :], assume_unique=True)): continue
        time_slice = Time_Vect[Segments[idx][0]:Segments[idx][1]]
        if not len(Ord_key) == 0:
            if not Bool_EvOrd: # Order vector is a time series
                temp = Ord_Vect[Segments[idx][0]:Segments[idx][1]]
                if np.count_nonzero(~np.isnan(temp)) > 0: temp = np.nanmedian(temp)
                else: temp = np.NINF
                ord_info.append(temp); ord_loc.append(idx)
            else: # Order vector contains events in time
                Circum_events = []; n = 0
                while n < len(Ord_Vect)-1:
                    #print(n, end=" ")
                    if Ord_Vect[n] < np.amin(time_slice):
                        Ord_Vect.pop(n); continue
                    elif Ord_Vect[n] > np.amax(time_slice):
                        break
                    else:
                        Circum_events.append(Ord_Vect[n])
                        Ord_Vect.pop(n); continue
                    n += 1
                if len(Circum_events) == 0:
                    ord_info.append(np.NINF); ord_loc.append(idx)
                else:
                    Circum_events = sorted(Circum_events)
                    ord_info.append(Circum_events[0]); ord_loc.append(idx)
        else:
            ord_info.append(idx); ord_loc.append(idx)

        if Bool_Ref_Phase: ref_slice = Ref_Vect[Segments[idx][0]:Segments[idx][1]]
        else: ref_slice = time_slice - Time_Vect[int(Segments[idx][1]-pts_aft)]
        if len(Event_List) > 0:
            for keys, data in zip(Events_keys, Event_List):
                criteria = (data >= np.amin(time_slice)) & (data <= np.amax(time_slice))
                temp = np.interp(data[criteria], time_slice, ref_slice)
                temp = np.vstack((np.ones_like(temp)*idx, temp)).T
                if Events_Dict[keys[-1]]['Events'].size == 0:
                    Events_Dict[keys[-1]]['Events'] = temp
                else:
                    Events_Dict[keys[-1]]['Events'] = np.concatenate((Events_Dict[keys[-1]]['Events'], temp), axis=0)

        if Bool_EvOrd:
            ord_info[-1] = np.interp(ord_info[-1], time_slice, ref_slice)

        if not len(Cont_key) == 0:
            data_slice = Cont_Vect[Segments[idx][0]:Segments[idx][1]]
            interp_ref = np.linspace(min_ref, max_ref, num=hvbins[0])
            if Mat_Col.size == 0:
                if Bool_Ref_Phase: Mat_Col = np.interp(interp_ref, ref_slice, data_slice, period=np.nanmax(Ref_Vect))[np.newaxis, :]
                else: Mat_Col = np.interp(interp_ref, ref_slice, data_slice)[np.newaxis, :]
            else:
                if Bool_Ref_Phase: Mat_Col = np.concatenate((Mat_Col, np.interp(interp_ref, ref_slice, data_slice, period=np.nanmax(Ref_Vect))[np.newaxis, :]), axis=0)
                else: Mat_Col = np.concatenate((Mat_Col, np.interp(interp_ref, ref_slice, data_slice)[np.newaxis, :]), axis=0)

    # Order the events/strides baed on the 'Ord_key' information extracted above
    if asc_sort:
        ord_arg = np.argsort(ord_info).astype(int)
        ord_sort = np.sort(ord_info)
        direc = "increm"; percen = [12.5,25,37.5,50,62.5,75,87.5]
    else:
        ord_arg = np.flip(np.argsort(ord_info)).astype(int)
        ord_sort = np.flip(np.sort(ord_info))
        direc = 'decrem'; percen = [87.5,75,62.5,50,37.5,25,12.5]
    ord_loc = [ord_loc[i] for i in ord_arg]

    if len(Event_List) > 0:
        lkup = np.vstack((np.arange(len(ord_info)), ord_loc)).T
        lkup_d = {int(pair[1]):int(pair[0]) for pair in lkup}
        for evkey, level in Events_Dict.items():
            evtemp = np.zeros(0)
            for idx in range(level['Events'].shape[0]):
                seg = level['Events'][idx]
                if not int(seg[0]) in lkup_d.keys(): continue
                if evtemp.size == 0: evtemp = np.array([lkup_d[int(seg[0])], seg[1]])[np.newaxis, :]
                else: evtemp = np.concatenate((evtemp, np.array([lkup_d[int(seg[0])], seg[1]])[np.newaxis, :]), axis=0)
            level['Ord. Events'] = evtemp.copy()

    # Get the approximated position of the ordering vector values for the plot
    y2_ticks = [ord_sort[FirstCrossVal(ord_sort, np.nanpercentile(ord_sort, p), direc=direc)] for p in percen]
    y2_ticks = [ord_sort[0]] + y2_ticks + [ord_sort[-1]]
    y2_tickpos = np.linspace(y2_ticks[0], y2_ticks[-1], num=len(y2_ticks))
    if all([el in np.sign(ord_sort).astype('int') for el in [-1,1]]): # Add the closest to 0 reference in case the ordering variable contains it
        loc = FirstCrossVal(ord_sort, 0, direc=direc)
        y2_tickpos = np.insert(y2_tickpos, y2_tickpos.size, y2_ticks[0] + (y2_ticks[-1]-y2_ticks[0])*(loc/len(ord_sort)))
        y2_ticks += [ord_sort[loc]]
    Any_Equal = True; dec = -1
    while Any_Equal:
        dec += 1
        temp = [round_half_up(num, decimals=dec) for num in y2_ticks]
        Any_Equal = np.unique(temp).size < len(temp)
        if dec == 4: break
    y2_ticklabels = [str(round_half_up(num, decimals=dec)) for num in y2_ticks]

    # For colormaps, get the ordered vertical averages ('vbins') without overlap between contiguos events/strides
    if not len(Cont_key) == 0:
        Mat_Temp = np.take_along_axis(Mat_Col, ord_arg[:,np.newaxis], 0)
        if hvbins[1] > 1 and hvbins[1] < Mat_Col.shape[0]:
            window = max(int(Mat_Col.shape[0]/hvbins[1]), 1); n = 0
            Mat_Ord = np.zeros(0)
            while n < Mat_Col.shape[0]:
                if Mat_Ord.size == 0: Mat_Ord = np.nanmedian(Mat_Temp[n:n+window+1, :], axis=0)[np.newaxis, :]
                else: Mat_Ord = np.concatenate((Mat_Ord, np.nanmedian(Mat_Temp[n:n+window+1, :], axis=0)[np.newaxis, :]), axis=0)
                n += window
        else:
            Mat_Ord = Mat_Temp
    else:
        Mat_Ord = Mat_Col

    # Prepare information for plotting
    Summ_keys = ""
    if len(Cont_key) > 0: Summ_keys += "Colormap: %s\n" % Cont_key[-1]
    if len(Events_Dict) > 0:
        if len(Events_Dict) < 3: Summ_keys += "Raster: %s\n" % (", ".join([ev for ev in Events_Dict.keys()]))
        else: Summ_keys += "Rasters of %d Events\n" % (len(Events_Dict))
    if len(Ord_key) > 0: Summ_keys += "(Sorted by %s)" % (Ord_key[-1])
    else: Summ_keys += "(Unsorted)"
    if phase_shift: Suff_shift = " (Phase Shifted)"
    else: Suff_shift = ""
    Plot_feat = {'x_label':"%s%s" % (Ref_key[-1], Suff_shift),
                 'y_label':"# Strides" if Bool_Ref_Phase else "# Events",
                 'y2_label':Ord_key[-1] if len(Ord_key)>0 else "",
                 'y2_range':(ord_sort[0], ord_sort[-1]),
                 'y2_ticks':y2_tickpos, 'y2_ticklabels':y2_ticklabels,
                 'im_label':Cont_key[-1] if len(Cont_key)>0 else "", 'suptitle':Summ_keys,
                 'vmin':np.nanpercentile(Mat_Ord, 5) if Mat_Ord.size>0 else 0,
                 'vmax':np.nanpercentile(Mat_Ord, 95) if Mat_Ord.size>0 else 0,
                 'extent':[min_ref, max_ref, 0, Mat_Col.shape[0]-1 if Mat_Col.size>0 else ord_sort.size]}

    if plot:
        fig, ax = plt.subplots(ncols=1)
        fig.suptitle(Plot_feat['suptitle']); ax.set_title("Plot 1", loc='left')
        ax.set_xlabel(Plot_feat['x_label']); ax.set_ylabel(Plot_feat['y_label'])
        ax.set_xlim((Plot_feat['extent'][0], Plot_feat['extent'][1]))
        ax.set_ylim((Plot_feat['extent'][2], Plot_feat['extent'][3]))
        if Plot_feat['y2_label'] != "":
            ax2 = ax.twinx(); ax2.set_ylabel(Plot_feat['y2_label']); ax2.set_ylim(Plot_feat['y2_range'])
            ax2.set_yticks(Plot_feat['y2_ticks']); ax2.set_yticklabels(Plot_feat['y2_ticklabels'])

        if Mat_Ord.size > 0:
            im = ax.imshow(Mat_Ord, origin='lower', vmin = Plot_feat['vmin'], vmax = Plot_feat['vmax'],
                           extent=Plot_feat['extent'], aspect='auto', label=Plot_feat['im_label'])
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # fig.colorbar(im, cax=cax, orientation='vertical')
        if len(Events_Dict) > 0:
            for key, dct in Events_Dict.items():
                ax.plot(dct['Ord. Events'][:,1], dct['Ord. Events'][:,0], marker='|', ms=0.1, lw=0, color=dct['Color'], label=key)
        plt.show()

    return Events_Dict, Mat_Ord, Plot_feat

def Generalized_Histograms(DataDict, Time_Vect, ref, vals, ref2=[], stat='count', color='gray', phase_shift=[False,False], mask_idxs=None, hvbins=(50,50), splog=False, plot=True):
    # Function to create histograms from a dictionary of data.
    # DataDict: Nested dictionary with leafs containing data to use for plotting.
    # Time_Vect: Vector with the time sequence of the common continuos time series in the datasets.
    # ref: Keymap of the main independent variable for the histogram
    # vals: Keymap of the dependent variable for the histogram.
    # ref2: Optional keymap to make 2D histograms with a second independent variable.
    # color: Color (or colormap) of the histogram.
    # phase_shift: Apply a PI (180 deg.) shift in the reference (or ref2) data.
    # mask_idxs: Vector with indexes to extract from the data specifically.
    # hvbins: Tuple with the number of horizontal and vertical bins for the histograms.
    # splog: List of booleans that, if True, makes a log10-transformation of the dependent variables (only valid for continuous).
    # plot: If True, plotting is done within the run of this function.

    # Examples of mapkeys in centralized LocoPkj files
    # ref = ['continuous','behavior','paws','FR_slopethr_stsw_phase']
    # vals = ['continuous', 'ephys', 'u1', 'ss_rate']
    # vals = ['discrete','ephys','u1','ss_cumul_times']
    # ref2 = ['continuous','behavior','paws','FL_slopethr_stsw_phase']
    # ref2 = ['continuous','behavior','belts','belts_speed_r']

    if mask_idxs is None: mask_idxs = np.arange(Time_Vect.size)
    acq_freq = 1/np.median(np.diff(Time_Vect))

    # Adjust the required data by user's request
    Ref_Vect = getFromDict(DataDict, ref).flatten()
    if Ref_Vect.size != Time_Vect.size:
        print("Aborting Generalized_Histograms: Main reference vector should be made of continuous variables")
        return None, None
    else:
        if np.nanmin(Ref_Vect) == 0 and np.nanmax(Ref_Vect) <= 2*np.pi: # For Phase reference
            if phase_shift[0]: Ref_Vect = Phase_Shift(Ref_Vect, shift=np.pi)
            Ref_Vect = Normalize(Ref_Vect)
        Ref_Vect = [Ref_Vect[mask_idxs]]

    Dep_Vect = getFromDict(DataDict, vals).flatten()
    if Time_Vect.size != Dep_Vect.size:
        if not is_points_not_times(Dep_Vect):
            Dep_Vect = LinearInterp(Time_Vect, np.arange(Time_Vect.size), Dep_Vect, kind='nearest').astype('int')
        Binned_Evs = np.zeros_like(Time_Vect)
        for d in Dep_Vect: Binned_Evs[d] += 1
        Dep_Vect = Binned_Evs
    else:
        if splog: Dep_Vect = np.log10(Dep_Vect)
    Dep_Vect = [Dep_Vect[mask_idxs]]

    if len(ref2):
        Ref2_Vect = getFromDict(DataDict, ref2).flatten()
        if Time_Vect.size == Ref2_Vect.size: # For continuos reference vector
            if np.nanmin(Ref2_Vect) == 0 and np.nanmax(Ref2_Vect) <= 2*np.pi: # For Phase reference
                if phase_shift[1]: Ref2_Vect = Phase_Shift(Ref2_Vect, shift=np.pi)
                Ref2_Vect = Normalize(Ref2_Vect)
        else:
            print("Aborting Generalized_Histograms: Secondary reference vector should be made of continuous variables")
            return None, None
        Ref2_Vect = [Ref2_Vect[mask_idxs]]; int_ref = 2
    else:
        Ref2_Vect = []; int_ref = 1

    # Calculate histograms depending on the statistic requested:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if "mean" in stat:
        values, bin_edges, _ = binned_statistic_dd(Ref_Vect + Ref2_Vect, Dep_Vect, statistic=lambda x : np.nanmean(x), bins=hvbins[:int_ref])
    elif "median" in stat:
        values, bin_edges, _ = binned_statistic_dd(Ref_Vect + Ref2_Vect, Dep_Vect, statistic=lambda x : np.nanmedian(x), bins=hvbins[:int_ref])
    elif "div" in stat:
        values0, bin_edges, _ = binned_statistic_dd(Ref_Vect + Ref2_Vect, Ref_Vect, statistic='count', bins=hvbins[:int_ref])
        values1, _, _ = binned_statistic_dd(Ref_Vect + Ref2_Vect, Dep_Vect, statistic=lambda x : np.nansum(x), bins=bin_edges)
        values = np.array([np.nan]*values0.size).reshape(values0.shape)
        np.divide(values1, values0/acq_freq, out=values, where=(values0 > 0) & (~np.isnan(values0)))
    else:
        values, bin_edges, _ = binned_statistic_dd(Ref_Vect + Ref2_Vect, Dep_Vect, statistic=stat, bins=hvbins[:int_ref])
    values = values[0]
    warnings.simplefilter("always", category=RuntimeWarning)

    # Prepare information for plotting
    if len(values.shape) == 1:
        bins = (bin_edges[0][:-1] + bin_edges[0][1:])/2
        extent = [bin_edges[0].min(), bin_edges[0].max()]
        aspect = 1.; vmin=0; vmax=0
        if not is_color_like(color): color = 'gray'
        label = "%s BY %s" % (vals[-1], ref[-1])
        Summ_keys = "%s\nBinned statistic: %s" % (label, stat)
        x_lab = "Shifted " if phase_shift[0] else ""; x_lab += ref[-1]
        y_lab = "Log10(%s)" % vals[-1] if splog else "%s" % vals[-1]
        y_lab += " / time unit" if 'div' in stat else ""
        c_lab = ""
    else:
        bins = [(edge[:-1] + edge[1:])/2 for edge in bin_edges]
        extent = [bin_edges[0].min(), bin_edges[0].max(), bin_edges[1].min(), bin_edges[1].max()]
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        vmin = np.nanpercentile(values, 5); vmax = np.nanpercentile(values, 95)
        if not color in plt.colormaps() or color == 'gray': color = 'viridis'
        label = "%s BY (%s, %s)" % (vals[-1], ref[-1], ref2[-1])
        Summ_keys = "%s\nBinned statistic: %s" % (label, stat)
        x_lab = "Shifted " if phase_shift[0] else ""; x_lab += ref[-1]
        y_lab = "Shifted " if phase_shift[1] else ""; y_lab += ref2[-1]
        c_lab = "Log10(%s)" % vals[-1] if splog else "%s" % vals[-1]
        c_lab += " / time unit" if 'div' in stat else ""
    if mask_idxs.size != Time_Vect.size:
        Summ_keys += "; %.01f%% Selected values" % (100*len(mask_idxs)/Time_Vect.size)
    else:
        Summ_keys += "; Full Data"

    Source_Dict = {'Bins':bins, 'Values':values}
    Plot_feat = {'x_label':x_lab, 'y_label':y_lab, 'cmap_label':c_lab, 'color':color,
                 'label':label, 'suptitle':Summ_keys, 'vmin':vmin, 'vmax':vmax,
                 'extent':extent, 'aspect':aspect}

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle(Plot_feat['suptitle'])
        if len(values.shape) == 1:
            ax.fill_between(Source_Dict['Bins'], Source_Dict['Values'],
                            step='mid', lw=0, color=Plot_feat['color'], alpha=1.,
                            label=Plot_feat['label'])
        else:
            im = ax.imshow(np.rot90(Source_Dict['Values']), cmap=Plot_feat['color'],
                           extent=Plot_feat['extent'], aspect=Plot_feat['aspect'],
                           vmin=Plot_feat['vmin'], vmax=Plot_feat['vmax'],
                           label=Plot_feat['label'])
            fig.colorbar(im, ax=ax, orientation='vertical', label=Plot_feat['cmap_label'])
        ax.set_xlabel(Plot_feat['x_label']); ax.set_ylabel(Plot_feat['y_label'])
        plt.show()

    return Source_Dict, Plot_feat

## Class definition to plot the data using an user interface
class Plotter_GUI():

    def __init__(self, def_outfolder=""):

        self.guiwin = None; self.figure = None
        self.axList = []; self.cbarList = []
        self.Data_Dict = None; self.Data_Tree = None
        self.Time_Arr = None; self.Body_Speed = None
        self.Key_Reject_Tree = ["Full Phase Diff. Matrix", "Constants", "Time", "Batch Information"]
        self.Ephys_Acq_Freq = 0; self.Track_Acq_Freq = 0
        self.LoadedData_Status = False; self.Status_Save_Plot_Data = False
        self.Status_User_Notif = True
        self.Plot_Types = ["Series VS Series", "Phase Portraits", "Raster/Colormap", "Generalized Hist."]
        self.Color_Sel_Type = "single"
        self.Sel_Table_1 = []; self.Sel_Table_2 = []

        self.Filepath = ""; self.Path_Backup = ""; self.Batch_Paths = []
        if os.path.exists(def_outfolder): self.output_folder = def_outfolder
        else: self.output_folder = " - None - "

        self.fig_size = (12,10); self.fig_lw = 3; self.fig_ms = 0.1
        self.fig_colormap = 'viridis'; self.fig_bins = (100,100); self.fig_wt = (-0.1,0.1)
        self.speed_time_thres = (0,0)

        self.Paw_List = ('FR', 'HL', 'FL', 'HR')
        self.Var_Color = {'FR':'red', 'HR':'magenta', 'FL':'blue', 'HL':'#00afaf',
                          'speed':"seagreen", 'accel':"olive", 'lick':"darkorange",
                          'reward':"darkgoldenrod", 'tail':"yellowgreen", 'body':"teal",
                          'cs':'purple', 'ss':'black'}

        self.GUI_Init(); self.GUI_Loop()

    def GUI_Init(self):

        sg.ChangeLookAndFeel('GreenTan')

        Col_Summ_LF = [
            [sg.Text("Load Centralized HDF5 Files: "), sg.Button("Browse", key="LF_LoadCent")],
            [sg.Text("Filename: "), sg.Text(" ---- ", size=(40,1), key="LF_Filename")],
            [sg.Text("Total Time: "), sg.Text("None", size=(40,1), key="LF_TotalTime")],
            [sg.Text("Total Points: "), sg.Text("None", size=(40,1), key="LF_TotalPts")]
        ]

        Col_Summ2_LF = [
            [sg.Text("Number of Strides:")],
            [sg.Text("FR: "), sg.Text("--", size=(20,1), key="LF_FR_Cross")],
            [sg.Text("HR: "), sg.Text("--", size=(20,1), key="LF_HR_Cross")],
            [sg.Text("FL: "), sg.Text("--", size=(20,1), key="LF_FL_Cross")],
            [sg.Text("HL: "), sg.Text("--", size=(20,1), key="LF_HL_Cross")]
        ]

        Tab_Load = [
            [sg.Column(Col_Summ_LF, element_justification='left'),
             sg.Column(Col_Summ2_LF, element_justification='left')],
        ]

        Col_Plot_Type = [
            [sg.Text("Plot Type", font=("Helvetica", 11, "bold"))],
            [sg.Listbox(self.Plot_Types,
                        select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                        key='PL_PlotType', enable_events=True, size=(15, 7))]
        ]

        MenuElem1 = ['BLANK', ['Change Color::Table_Recolor1',
                               'Remove from List::Table_Remove1',
                               'Modify Status::Table_Modify1']]
        MenuElem2 = ['BLANK', ['Change Color::Table_Recolor2',
                               'Remove from List::Table_Remove2',
                               'Modify Status::Table_Modify2']]

        Col_Plot_1 = [
            [sg.Text('Variables Plot #1', size=(20,1), justification="center",
                     font=("Helvetica", 11, "bold"), key="PL_Text_1")],
            [sg.Button("Choose Data", key="PL_Tree_1")],
            [sg.Table([[' '*20, ' '*8, ''*8, " "]],
                      visible_column_map=[True, True, True, False],
                      headings=["Variable", "Color", "Status"],
                      def_col_width=25, max_col_width=30,
                      key='PL_PlotVar_1', num_rows=6,
                      col_widths=[25, 7, 7], right_click_menu=MenuElem1, bind_return_key=True)]
        ]

        Col_Plot_2 = [
            [sg.Text('Variables Plot #2', size=(20,1), justification="center",
                     font=("Helvetica", 11, "bold"), key="PL_Text_2")],
            [sg.Button("Choose Data", key="PL_Tree_2")],
            [sg.Table([[' '*20, ' '*8, ''*8, " "]],
                      visible_column_map=[True, True, True,  False],
                      headings=["Variable", "Color", "Status"],
                      def_col_width=25, max_col_width=30,
                      key='PL_PlotVar_2', num_rows=6,
                      col_widths=[25, 7, 7], right_click_menu=MenuElem2, bind_return_key=True)]
        ]

        Tab_Plots = [
            [sg.Column(Col_Plot_Type, element_justification="center"),
             sg.Column(Col_Plot_1, element_justification="center"),
             sg.Column(Col_Plot_2, element_justification="center")],
            [sg.Text("Fig. X-size:"), sg.Input(str(self.fig_size[0]), size=(3,1), key="FigXsize", enable_events=True),
             sg.Text("Fig. Y-size:"), sg.Input(str(self.fig_size[1]), size=(3,1), key="FigYsize", enable_events=True),
             sg.Text("Line Width:"), sg.Input(str(self.fig_lw), size=(3,1), key="LineWidth", enable_events=True),
             sg.Text("Marker Size:"), sg.Input(str(self.fig_ms), size=(3,1), key="MarkerSize", enable_events=True),
             sg.Text("Speed Thres. [m/s]:"), sg.Input(str(self.speed_time_thres[0]), size=(3,1), key="SpeedThres", enable_events=True),
             sg.Text("Duration Thres. [s]:"), sg.Input(str(self.speed_time_thres[1]), size=(3,1), key="DuraThres", enable_events=True)],
            [sg.Text("Colormap:"), sg.Input(self.fig_colormap, size=(15,1), key="Colormap", enable_events=True),
             sg.Text("Bins (H,V) [#]:"), sg.Input(str(self.fig_bins).replace("(","").replace(")",""), size=(15,1), key="Bins", enable_events=True),
             sg.Text("Time win. (bef., aft.) [sec]:"), sg.Input(str(self.fig_wt).replace("(","").replace(")",""), size=(15,1), key="TimeWin", enable_events=True)],
            [sg.Button("(Re)Plot", key="Plot_PL")]
        ]

        Tab_Batch = [
            [sg.Button("Choose Batch of Files", key="BF_LoadBatch"), sg.Text("Common Dir.:"),
            sg.Text(" --- ", size=(50,1), key="BF_FilesFolder")],
            [sg.Text("Number of Files: "), sg.Text(" -- ", size=(10,1), key="BF_NumFiles")],
            [sg.Button("Choose", key="BF_ChooseFolder"), sg.Text("Output Folder: "), sg.Text(self.output_folder, size=(50,1), key="BF_OutFolder")],
            [sg.Button("Save Batch", key="Save_BF"),
             sg.Checkbox("Save Plots Data in File", default=self.Status_Save_Plot_Data, key="SaveDataOpt_BF"),
             sg.Checkbox("Phone Notification", default=self.Status_User_Notif, key="BF_Notif")]
        ]

        layout = [
            [sg.Text("Rasters & Colormap Plotting", font=("Helvetica", 18))],
            [sg.TabGroup([[sg.Tab('Load Single File', Tab_Load, key='Tab_Load'),
                           sg.Tab('Plotting Specs', Tab_Plots, key='Tab_Plots'),
                           sg.Tab('Batch of Files', Tab_Batch, key='Tab_Batch')]],
                           enable_events=True, key='TabGrp_Action')]
        ]

        self.guiwin = sg.Window('Plotter').Layout(layout)

    def GUI_Loop(self):

        while True:
            event, values = self.guiwin.Read()

            # Act upon button events
            if event in (None, 'Exit'): # An escape route for exiting
                self.guiwin.close()
                break

            self.Status_User_Notif = values["BF_Notif"]

            if "LF_LoadCent" in event:
                try:
                    self.Load_Cent_File()
                    self.Data_Tree = self.Create_Tree_Batch(self.Data_Dict, self.Time_Arr.size)
                    self.LoadedData_Status = True
                except:
                    sg.popup_ok("File Loading cancelled")
            elif "BF_LoadBatch" in event:
                fpaths = []
                try:
                    fpaths = OpenBatch(filetype=[('HDF5 File',"*.h5")], title="Batch of Files")
                except:
                    sg.popup_ok("Batch Loading cancelled")
                if len(fpaths) > 0:
                    self.Batch_Paths = fpaths
                    comm = findcommonstart(fpaths, separator="/")
                    if not os.path.exists(comm): comm = "Many folders"
                    self.guiwin.find_element("BF_FilesFolder").Update("%s" % comm)
                    self.guiwin.find_element("BF_NumFiles").Update("%d" % len(fpaths))
            elif 'PL_PlotType' in event:
                if not self.LoadedData_Status: continue
                if "Series VS Series" in values['PL_PlotType'][0]:
                    self.Color_Sel_Type = "single"
                    self.guiwin.find_element("PL_Text_1").Update('Series/Events # 1')
                    self.guiwin.find_element("PL_Text_2").Update('Series/Events # 2')
                    # self.Color_Selection_Plots(values, event, self.Sel_Table_1, type=self.Color_Sel_Type)
                elif "Phase Portraits" in values['PL_PlotType'][0]:
                    self.Color_Sel_Type = "multiple"
                    self.guiwin.find_element("PL_Text_1").Update('Trajectory Series')
                    self.guiwin.find_element("PL_Text_2").Update('Overlayed Events')
                elif "Raster" in values['PL_PlotType'][0]:
                    self.Color_Sel_Type = "single"
                    self.guiwin.find_element("PL_Text_1").Update('Reference (Event/Phase)')
                    self.guiwin.find_element("PL_Text_2").Update('Series/Events to Raster')
                elif "Hist." in values['PL_PlotType'][0]:
                    self.Color_Sel_Type = "single"
                    self.guiwin.find_element("PL_Text_1").Update('Binning Variables')
                    self.guiwin.find_element("PL_Text_2").Update('Value Variables')
            elif "PL_Tree" in event:
                Selection = self.Tree_Data_GUI()
                if len(Selection) == 0: continue
                MapKeys = [elem.split(";") for elem in Selection] # The key sequence needed to extract the data afterwards

                # Recreate the color cycle of the matplotlib plots for the cluster coloring:
                plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                cycled = cycle(plot_colors)  # cycle thorugh the color list
                color_slice = list(islice(cycled, 1, len(Selection)+1))  # take the number of colors needed

                Table_data = []; count_assign_color = 0
                for idx in range(len(Selection)):
                    VarMatch = [var for var in self.Var_Color.keys() if var in Selection[idx]]
                    name = MapKeys[idx][-1]
                    if len(VarMatch) == 1:
                        color = self.Var_Color[VarMatch[0]]
                    else:
                        color = color_slice[count_assign_color]
                    count_assign_color += 1
                    Table_data.append([name, color, "", Selection[idx]])

                if event[-1:] is "1":
                    NameSet = set(map(lambda x : x[0], self.Sel_Table_1))
                    self.Sel_Table_1 += [[name, color, status, keys] for name, color, status, keys in Table_data if not name in NameSet]
                    self.guiwin.find_element('PL_PlotVar'+event[-2:]).Update(values=self.Sel_Table_1)
                elif event[-1:] is "2":
                    NameSet = set(map(lambda x : x[0], self.Sel_Table_2))
                    self.Sel_Table_2 += [[name, color, status, keys] for name, color, status, keys in Table_data if not name in NameSet]
                    self.guiwin.find_element('PL_PlotVar'+event[-2:]).Update(values=self.Sel_Table_2)
            elif 'Table_Recolor1' in event:
                self.Sel_Table_1 = self.Color_Selection_Plots(values, 'PL_PlotVar_1',
                                                              self.Sel_Table_1,
                                                              type=self.Color_Sel_Type)
            elif 'Table_Recolor2' in event:
                self.Sel_Table_2 = self.Color_Selection_Plots(values, 'PL_PlotVar_2',
                                                              self.Sel_Table_2,
                                                              type="single")
            elif 'Table_Remove1' in event:
                if not len(values['PL_PlotVar_1']) > 0: continue
                try:
                    self.Sel_Table_1.pop(values['PL_PlotVar_1'][0])
                except:
                    if self.Sel_Table_1 is None:
                        self.Sel_Table_1 = []
                self.guiwin.find_element('PL_PlotVar_1').Update(values=self.Sel_Table_1)
            elif 'Table_Remove2' in event:
                if not len(values['PL_PlotVar_2']) > 0: continue
                try:
                    self.Sel_Table_2.pop(values['PL_PlotVar_2'][0])
                except:
                    if self.Sel_Table_2 is None:
                        self.Sel_Table_2 = []
                self.guiwin.find_element('PL_PlotVar_2').Update(values=self.Sel_Table_2)
            elif 'Table_Modify1' in event:
                if not len(values['PL_PlotVar_1']) > 0: continue
                Row_Num = values['PL_PlotVar_1'][0]
                if len(self.Sel_Table_1) < Row_Num: continue
                Sel_Row = self.Sel_Table_1[Row_Num]
                Status = sg.popup_get_text("New Status for " + Sel_Row[0], title="Change Status")
                if Status is None: continue
                self.Sel_Table_1[Row_Num] = [Sel_Row[0], Sel_Row[1], Status, Sel_Row[3]]
                self.guiwin.find_element('PL_PlotVar_1').Update(values=self.Sel_Table_1)
            elif 'Table_Modify2' in event:
                if not len(values['PL_PlotVar_2']) > 0: continue
                Row_Num = values['PL_PlotVar_2'][0]
                if len(self.Sel_Table_2) < Row_Num: continue
                Sel_Row = self.Sel_Table_2[Row_Num]
                Status = sg.popup_get_text("New Status for " + Sel_Row[0], title="Change Status")
                if Status is None: continue
                self.Sel_Table_2[Row_Num] = [Sel_Row[0], Sel_Row[1], Status, Sel_Row[3]]
                self.guiwin.find_element('PL_PlotVar_2').Update(values=self.Sel_Table_2)
            elif "FigXsize" in event:
                try:
                    temp = float(values[event])
                    if temp <= 0 or temp > 30: raise ValueError
                    self.guiwin.find_element(event).Update(text_color="black")
                    self.fig_size = (temp, self.fig_size[1])
                except:
                    self.guiwin.find_element(event).Update(text_color="red")
            elif "FigYsize" in event:
                try:
                    temp = float(values[event])
                    if temp <= 0 or temp > 30: raise ValueError
                    self.guiwin.find_element(event).Update(text_color="black")
                    self.fig_size = (self.fig_size[0], temp)
                except:
                    self.guiwin.find_element(event).Update(text_color="red")
            elif "MarkerSize" in event:
                try:
                    temp = float(values[event])
                    if temp <= 0: raise ValueError
                    self.guiwin.find_element(event).Update(text_color="black")
                    self.fig_ms = temp
                except:
                    self.guiwin.find_element(event).Update(text_color="red")
            elif "LineWidth" in event:
                try:
                    temp = float(values[event])
                    if temp <= 0: raise ValueError
                    self.guiwin.find_element(event).Update(text_color="black")
                    self.fig_lw = temp
                except:
                    self.guiwin.find_element(event).Update(text_color="red")
            elif "SpeedThres" in event:
                try:
                    if any([el in values[event] for el in ['Min','min','MIN','mIN']]):
                        temp = np.NINF
                    else:
                        temp = float(values[event])
                    self.speed_time_thres = (temp, self.speed_time_thres[1])
                    self.guiwin.find_element("SpeedThres").Update(text_color='black')
                except:
                    self.guiwin.find_element("SpeedThres").Update(text_color='red')
            elif "DuraThres" in event:
                try:
                    temp = float(values[event])
                    self.speed_time_thres = (self.speed_time_thres[0], temp)
                    self.guiwin.find_element("DuraThres").Update(text_color='black')
                except:
                    self.guiwin.find_element("DuraThres").Update(text_color='red')
            elif "Colormap" in event:
                try:
                    if not values[event] in plt.colormaps(): raise ValueError
                    self.guiwin.find_element("Colormap").Update(text_color='black')
                    self.fig_colormap = values[event]
                except:
                    self.guiwin.find_element("Colormap").Update(text_color='red')
            elif "Bins" in event:
                temp = values[event].replace("(","").replace(")","")
                if "," in temp: temp = temp.split(",")
                else: temp = [temp]
                try:
                    temp = [int(el) for el in temp]
                    if len(temp) > 2: raise ValueError
                    elif len(temp) == 2: self.fig_bins = tuple(temp)
                    elif len(temp) == 1: self.fig_bins = (temp[0], temp[0])
                    else: raise ValueError
                    self.guiwin.find_element("Bins").Update(text_color='black')
                except:
                    self.guiwin.find_element("Bins").Update(text_color='red')
            elif "TimeWin" in event:
                temp = values[event].replace("(","").replace(")","")
                if "," in temp: temp = temp.split(",")
                else: temp = [temp]
                try:
                    temp = [float(el) for el in temp]
                    if len(temp) > 2: raise ValueError
                    elif len(temp) == 2: self.fig_wt = tuple(temp)
                    elif len(temp) == 1: self.fig_wt = (temp[0], temp[0])
                    else: raise ValueError
                    self.guiwin.find_element("TimeWin").Update(text_color='black')
                except:
                    self.guiwin.find_element("TimeWin").Update(text_color='red')
            elif "Plot_PL" in event:
                if not len(values['PL_PlotType']) > 0: continue
                if not values['PL_PlotType'][0] in self.Plot_Types: continue
                if not self.LoadedData_Status: continue
                #self.Status_Save_Plot_Data = values["SaveDataOpt_BF"]
                self.Plot_Node(values['PL_PlotType'][0])
            elif "BF_ChooseFolder" in event:
                try:
                    self.output_folder = OpenExplorer(folder=True)
                    self.guiwin.find_element("BF_OutFolder").Update(self.output_folder)
                except:
                    pass
            elif "Save_BF" in event:
                if len(self.Batch_Paths) > 0:
                    sg.OneLineProgressMeter('Batch Routine', 1, len(self.Batch_Paths), 'Progress Bar', 'Processing Files')
                    for n, file in enumerate(self.Batch_Paths):
                        self.Load_Cent_File(exter_path=file)
                        self.Plot_Node(values['PL_PlotType'][0], png_save=True, show=False)
                        sg.OneLineProgressMeter('Batch Routine', n+1, len(self.Batch_Paths), 'Progress Bar', 'Processing Files')
                    # Recover the individual file loaded last
                    self.Load_Cent_File(exter_path=self.Path_Backup)

    def Load_Cent_File(self, exter_path=None):

        if exter_path is None:
            self.Filepath = OpenExplorer(filetype=[('HDF5 File',"*.h5")])
            self.Path_Backup = self.Filepath
        else: self.Filepath = exter_path
        self.Data_Dict = Load_Dict_from_HDF5(self.Filepath, attrs="Root")
        self.Data_Dict = self.Flatten_Data_Dict(self.Data_Dict)

        if "continuous" in self.Data_Dict.keys():
            self.Time_Arr = getFromDict(self.Data_Dict, ["continuous",'session',"session_time"])
            self.Body_Speed = getFromDict(self.Data_Dict, ['continuous','behavior','belts','belts_speed_r'])
            self.Ephys_Acq_Freq = getFromDict(self.Data_Dict, ["Root Attrs","ephys_acquisition_rate"])
            self.Track_Acq_Freq = getFromDict(self.Data_Dict, ["Root Attrs", "behavior_acquisition_rate"])
            FR_StPhase = getFromDict(self.Data_Dict, ["continuous",'behavior',"paws","FR_slopethr_st_phase"]) - np.pi
            HR_StPhase = getFromDict(self.Data_Dict, ["continuous",'behavior',"paws","HR_slopethr_st_phase"]) - np.pi
            FL_StPhase = getFromDict(self.Data_Dict, ["continuous",'behavior',"paws","FL_slopethr_st_phase"]) - np.pi
            HL_StPhase = getFromDict(self.Data_Dict, ["continuous",'behavior',"paws","HL_slopethr_st_phase"]) - np.pi
        elif "Outputs" in self.Data_Dict.keys():
             self.Time_Arr = getFromDict(self.Data_Dict, ["Inputs",'Behav. Time'])
             self.Body_Speed = getFromDict(self.Data_Dict, ["Inputs",'Behav. Speed'])
             self.Ephys_Acq_Freq = getFromDict(self.Data_Dict, ["Inputs", "Constants", "EphysFreq"])
             self.Track_Acq_Freq = getFromDict(self.Data_Dict, ["Inputs", "Constants", "AcqFreq"])
             FR_StPhase = getFromDict(self.Data_Dict, ["Inputs",'Paws Data',"FR"]) - np.pi
             HR_StPhase = getFromDict(self.Data_Dict, ["Inputs",'Paws Data',"HR"]) - np.pi
             FL_StPhase = getFromDict(self.Data_Dict, ["Inputs",'Paws Data',"FL"]) - np.pi
             HL_StPhase = getFromDict(self.Data_Dict, ["Inputs",'Paws Data',"HL"]) - np.pi

             # Extract the Sw/St events from the simulated data to aid in plotting:
             for paw in ['FR','FL','HR','HL']:
                 phase = getFromDict(self.Data_Dict, ["Inputs",'Paws Data',paw])
                 swings = self.Time_Arr[np.where(phase == np.pi)[0]]
                 stances = self.Time_Arr[np.where(phase == 0)[0]]
                 setInDict(self.Data_Dict, ["Inputs",'Paws Data',"%s_Sw_Events" % paw], swings)
                 setInDict(self.Data_Dict, ["Inputs",'Paws Data',"%s_St_Events" % paw], stances)
                 st_pts = Event_Time_to_Value_Interpol(stances, self.Time_Arr, np.arange(self.Time_Arr.size), kind="linear").astype('int')
                 sw_pts = Event_Time_to_Value_Interpol(swings, self.Time_Arr, np.arange(self.Time_Arr.size), kind="linear").astype('int')
                 St_Phase = Phase_Calculator(self.Time_Arr, st_pts, sw_pts, Single_PhRef = "St")
                 Sw_Phase = Phase_Calculator(self.Time_Arr, st_pts, sw_pts, Single_PhRef = "Sw")
                 setInDict(self.Data_Dict, ["Inputs",'Paws Data',"%s_Sw_Phase" % paw], Sw_Phase)
                 setInDict(self.Data_Dict, ["Inputs",'Paws Data',"%s_St_Phase" % paw], St_Phase)


        TimeStr = user_display_time(np.amax(self.Time_Arr)*1000, user_scale=['milli', 'sec', "min"])
        FR_crossings = np.where(np.diff(np.sign(FR_StPhase)))[0].size
        HR_crossings = np.where(np.diff(np.sign(HR_StPhase)))[0].size
        FL_crossings = np.where(np.diff(np.sign(FL_StPhase)))[0].size
        HL_crossings = np.where(np.diff(np.sign(HL_StPhase)))[0].size

        # self.guiwin.find_element("LF_FilesFolder").Update(self.Common_FilesFolder + " / " + self.Common_FileName + "*")
        # self.guiwin.find_element("LF_NumFiles").Update(str(len(FileNames_List)))
        self.guiwin.find_element("LF_FR_Cross").Update(FR_crossings)
        self.guiwin.find_element("LF_HR_Cross").Update(HR_crossings)
        self.guiwin.find_element("LF_FL_Cross").Update(FL_crossings)
        self.guiwin.find_element("LF_HL_Cross").Update(HL_crossings)
        self.guiwin.find_element("LF_Filename").Update(os.path.basename(self.Filepath))
        self.guiwin.find_element("LF_TotalTime").Update(TimeStr)
        self.guiwin.find_element("LF_TotalPts").Update(self.Time_Arr.size)

    def Flatten_Data_Dict(self, Data):

        mem_key = [""]*10
        for key, val, level in dict_recursive_items(Data):
            if mem_key[level] != key: mem_key[level] = key
            if isinstance(val, np.ndarray):
                if len(val.shape) > 1:
                    if val.shape[0] < val.shape[1]:
                        setInDict(Data, mem_key[:level] + [key], np.squeeze(val.T))

        return Data

    def Create_Tree_Batch(self, Input_Dict, time_pts, type=0):
        # Creates a Tree element for enlisting and selecting objects stored in nested dictionaries.
        # Input_Dict: Nested dictionaries with the objects to organize on the Tree Element.
        # time_pts: Number of points in the continuos datasets that identify them as time series
        # type: (Optional) A value of "0" gives a Tree organized for datasets,
        #       A value of "1" gives a Tree for Multidimensional Regions.

        # Create a tree element to allow user to select the data for visualization:
        Temp_Data_Tree = sg.TreeData()

        KeyTemp = list(Input_Dict.keys())
        prevLevel = 0; keyList = [""]
        for key, val, level in dict_recursive_items(Input_Dict):
            # Create the key map to open specific data:
            if level+1 != len(keyList):
                for i in range(len(keyList) - level):
                    keyList.pop()
                keyList.append(key)
            elif prevLevel < level:
                keyList.append(key)
            elif prevLevel == level:
                keyList.pop(); keyList.append(key)
            prevLevel = level

            if type == 0: # Tree for Datasets
                if any([key in keyList for key in self.Key_Reject_Tree]): continue
                if not isinstance(val, np.ndarray) and not isinstance(val, dict): continue


                if isinstance(val, np.ndarray):
                    if val.shape[0] < time_pts:
                        if len(val.shape) > 2:
                            Type = "%d-D Image %s" % (len(val.shape), str(val.shape))
                        elif len(val.shape) > 1:
                            if val.shape[1] > 2:
                                Type = "Image (%d X %d)" % (val.shape[0], val.shape[1])
                            else:
                                Type = "Double Ref. Event %s" % str(val.shape)
                        else:
                            Type = "Event %s" % str(val.shape)
                    else:
                        temp = np.unique(val)
                        if temp.size > 0 and temp.size <= 2 and temp.max() <= 1 and temp.min() >= 0:
                            if temp.size == 1 and any([dig in temp for dig in [0,1]]):
                                Type = "Boolean Time Serie"
                            elif temp.size == 2 and all([dig in temp for dig in [0,1]]):
                                Type = "Boolean Time Serie"
                            else:
                                Type = "Time Serie"
                        else:
                            Type = "Time Serie"

                else:
                    Type = "Node"

            elif type == 1: # Tree for N-Dim. Regions
                if isinstance(val, np.ndarray):
                    if len(val.shape) > 1:
                        Type = "Coords."
                    else:
                        Type = "Events"
                elif isinstance(val, dict):
                    if "Parent Reg." in val.keys():
                        Type = "Attrib."
                    else:
                        Type = "Region"
                else:
                    Type = "Info."

            # Insert each level to the Tree of the data:
            if level == 0:
                parent = ""
            else:
                parent = ";".join(keyList[0:len(keyList)-1])

            key = ";".join(keyList); elem = keyList[len(keyList)-1]
            Temp_Data_Tree.Insert(parent, key, elem, [Type])

        return Temp_Data_Tree

    def Tree_Data_GUI(self):

        if self.Data_Tree is None: return []

        layout = [[sg.Tree(data=self.Data_Tree, headings=['Data Type'],
                           col0_width=30, max_col_width=40, def_col_width=30,
                           num_rows=30, auto_size_columns=False, key="Tree")],
                  [sg.OK(), sg.Cancel()]]
        window = sg.Window('Choose data', keep_on_top=True).Layout(layout)
        event, values = window.read()
        window.close()

        if event is None:
            Selection = []
        elif "OK" in event:
            Selection = values["Tree"]
        else:
            Selection = []

        return Selection

    def Color_Selection_Plots(self, values, gui_elem, table, type="single"):
        if not len(values[gui_elem]) > 0: return table
        Row_Num = values[gui_elem][0]
        if len(table) < Row_Num: return table
        Sel_Row = table[Row_Num]
        Bool_Exit = False
        while not Bool_Exit:
            Color = sg.popup_get_text("Color of " + Sel_Row[0], title="Choose Color")
            if Color is None:
                break
            elif is_color_like(Color):
                Bool_Exit = True
            else:
                sg.popup_error("Color is not valid")
        if not Bool_Exit: return table

        if "sin" in type:
            table[Row_Num] = [Sel_Row[0], Color, Sel_Row[2], Sel_Row[3]]
        else:
            table = [[elem[0], Color, elem[2], elem[3]] for elem in table]
        self.guiwin.find_element(gui_elem).Update(values=table)

        return table

    def Matplot_Config(self, num_axes, type="Time Series", fig_size=(22,11), cbars=False):

        plt.close('LocoEphys Batch Plot')
        self.figure = plt.figure(num='LocoEphys Batch Plot',
                                   constrained_layout=False, figsize=fig_size,
                                   dpi=80)
        self.axList = []; self.cbarList = []
        if "Time Series" in type:
            if isinstance(num_axes, (list, tuple)):
                gs = self.figure.add_gridspec(num_axes[1], num_axes[0])
                for row in range(num_axes[1]):
                    for col in range(num_axes[0]):
                        self.axList.append(self.figure.add_subplot(gs[row,col:(col+1)]))
            else:
                gs = self.figure.add_gridspec(num_axes, 1)
                for n in range(num_axes):
                    if n == 0:
                        self.axList.append(self.figure.add_subplot(gs[0:(n+1),0]))
                    else:
                        self.axList.append(self.figure.add_subplot(gs[n:(n+1),0], sharex=self.axList[0]))
                    if n != num_axes - 1:
                        plt.setp(self.axList[n].get_xticklabels(), visible=False)

            #self.VertPointer = MultiCursor(self.figure.canvas, self.axList, color='r', lw=1, horizOn=False)
        elif type == "Images":
            if isinstance(num_axes, (list, tuple)):
                if not cbars:
                    gs = self.figure.add_gridspec(num_axes[1], num_axes[0])
                    for row in range(num_axes[1]):
                        for col in range(num_axes[0]):
                            self.axList.append(self.figure.add_subplot(gs[row,col:(col+1)]))
                else:
                    gs = self.figure.add_gridspec(num_axes[1], 10*num_axes[0])
                    col_pos = (np.arange(num_axes[0]).astype('int')*10)
                    for row in range(num_axes[1]):
                        for col in col_pos:
                            self.axList.append(self.figure.add_subplot(gs[row,col:(col+9)]))
                            self.cbarList.append(self.figure.add_subplot(gs[row,col+9:(col+10)]))
            else:
                if not cbars:
                    gs = self.figure.add_gridspec(1, num_axes)
                    for n in range(num_axes):
                        self.axList.append(self.figure.add_subplot(gs[0,n:(n+1)]))
                else:
                    gs = self.figure.add_gridspec(1, 10*num_axes)
                    col_pos = (np.arange(num_axes).astype('int')*10)
                    for n in col_pos:
                        self.axList.append(self.figure.add_subplot(gs[0,n:(n+9)]))
                        self.cbarList.append(self.figure.add_subplot(gs[0,n+9:(n+10)]))
        elif type == "3D":
            gs = self.figure.add_gridspec(1, num_axes)
            for n in range(num_axes):
                self.axList.append(self.figure.add_subplot(gs[0,n:(n+1)], projection='3d'))

        self.figure.canvas.mpl_connect('close_event', self.Exit_Proc)

    def Plot_Node(self, type, png_save=False, plot=True, show=True):

        if png_save or show: plot = True
        Time = self.Time_Arr; Mag_Order = 10**np.floor(np.log10(np.abs(Time.size)))

        if "Series VS Series" in type:
            if plot: self.Matplot_Config(2, fig_size=self.fig_size)
            OutDict = {"Time Vector":Time, "Mask":np.arange(Time.size), "Series #1":{}, "Series #2":{}}; outname = "TimeSeries"
            Data_1 = []; Data_2 = []
            MaxRange = np.NINF; MinRange = np.inf
            for idx in range(len(self.Sel_Table_1)):
                Name = self.Sel_Table_1[idx][0]; Color = self.Sel_Table_1[idx][1]
                Status = self.Sel_Table_1[idx][2]; MapKey = self.Sel_Table_1[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if isinstance(Data, dict): continue
                if Data.size == Time.size:
                    Idxs_SpeedDur = Speed_Duration_Threshold(Time, self.Body_Speed, speed_time_thres=self.speed_time_thres)
                    Temp = np.asarray([np.nan]*Data.size)
                    np.put(Temp, Idxs_SpeedDur, Data[Idxs_SpeedDur]); Data = Temp
                    if MinRange > np.nanmin(Data): MinRange = np.nanmin(Data)
                    if MaxRange < np.nanmax(Data): MaxRange = np.nanmax(Data)
                    OutDict["Mask"] = Idxs_SpeedDur
                else:
                    if np.all(np.mod(Data, 1) == 0): # Data has all integer arrays, so most likely indexes of arrays
                        if is_points_not_times(Data): # Check if array is comprised of points or times and keep it on times
                            Data = np.take(Time, Data.astype(int))
                Data_1.append(Data)
            for idx in range(len(Data_1)):
                Name = self.Sel_Table_1[idx][0]; Color = self.Sel_Table_1[idx][1]
                if Data_1[idx].size == Time.size:
                    if plot: self.axList[0].plot(Time, Data_1[idx], color=Color, lw=self.fig_lw, label=Name)
                    OutDict["Series #1"][Name] = {"serie":Data_1[idx], "color":Color}
                else:
                    if np.isinf(MaxRange) and np.isinf(MinRange):
                        MaxRange = 1; MinRange = 0
                    if plot: self.axList[0].vlines(Data_1[idx], MinRange, MaxRange, color=Color, label=Name)
                    OutDict["Series #1"][Name] = {"events":Data_1[idx], "color":Color, "range":(MinRange, MaxRange)}

            MaxRange = np.NINF; MinRange = np.inf
            for idx in range(len(self.Sel_Table_2)):
                Name = self.Sel_Table_2[idx][0]; Color = self.Sel_Table_2[idx][1]
                Status = self.Sel_Table_2[idx][2]; MapKey = self.Sel_Table_2[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if isinstance(Data, dict): continue
                if Data.size == Time.size:
                    Idxs_SpeedDur = Speed_Duration_Threshold(Time, self.Body_Speed, speed_time_thres=self.speed_time_thres)
                    Temp = np.asarray([np.nan]*Data.size)
                    np.put(Temp, Idxs_SpeedDur, Data[Idxs_SpeedDur]); Data = Temp
                    if MinRange > np.nanmin(Data): MinRange = np.nanmin(Data)
                    if MaxRange < np.nanmax(Data): MaxRange = np.nanmax(Data)
                    OutDict["Mask"] = Idxs_SpeedDur
                else:
                    if np.all(np.mod(Data, 1) == 0): # Data has all integer arrays, so most likely indexes of arrays
                        if is_points_not_times(Data): # Check if array is comprised of points or times and keep it on times
                            Data = np.take(Time, Data.astype(int))
                Data_2.append(Data)
            for idx in range(len(Data_2)):
                Name = self.Sel_Table_2[idx][0]; Color = self.Sel_Table_2[idx][1]
                if Data_2[idx].size == Time.size:
                    if plot: self.axList[1].plot(Time, Data_2[idx], color=Color, lw=self.fig_lw, label=Name)
                    OutDict["Series #2"][Name] = {"serie":Data_2[idx], "color":Color}
                else:
                    if np.isinf(MaxRange) and np.isinf(MinRange):
                        MaxRange = 1; MinRange = 0
                    if plot: self.axList[1].vlines(Data_2[idx], MinRange, MaxRange, color=Color, label=Name)
                    OutDict["Series #2"][Name] = {"events":Data_2[idx], "color":Color, "range":(MinRange, MaxRange)}

            if plot:
                self.axList[0].set_ylabel("Plot # 1")
                self.axList[1].set_ylabel("Plot # 2")
                plt.rcParams['agg.path.chunksize'] = (len(Data_1)+len(Data_2)) * np.ceil(Time.size/Mag_Order)*Mag_Order

        elif "Phase Portraits" in type:
            Data_1 = []; Data_2 = []; Status_KDE = False
            for idx in range(len(self.Sel_Table_1)):
                Name = self.Sel_Table_1[idx][0]; Color = self.Sel_Table_1[idx][1]
                Status = self.Sel_Table_1[idx][2]; MapKey = self.Sel_Table_1[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if Data.size == Time.size:
                    if np.nanmin(Data) == 0 and np.nanmax(Data) <= 2*np.pi: # Phase data
                        Idxs_SpeedDur = Speed_Duration_Threshold(Time, self.Body_Speed, speed_time_thres=self.speed_time_thres)
                        Temp = np.asarray([np.nan]*Data.size)
                        np.put(Temp, Idxs_SpeedDur, Data[Idxs_SpeedDur]); Data = Temp
                        shift = 0
                        if any(stat in Status for stat in ["shift", "Shift", "SHIFT", 'sHIFT']): # Shift the phases if requested
                            Unwrap_Shift = np.unwrap(Data-np.pi) + np.pi
                            Data = (Unwrap_Shift + np.pi) % (2 * np.pi); shift = np.pi
                        if any(stat in Status for stat in ["nozero", "Nozero", "NOZERO", 'nOZERO']): # Mask out the rejected phase points
                            Data = np.where(Data == shift, [np.nan]*Data.size, Data)
                        Data_1.append([Data, Color, Name, Status])
            if len(Data_1) > 3: Data_1 = Data_1[0:3]; sg.popup_error("Only Plotting first 3 times series in phase trajectory")
            if len(Data_1) == 1:
                sg.popup_error("Aborting Plot: At least 2 time series are needed in phase trajectory")
                return None

            # Check if KDE variables are required:
            Stat_List = [elem[2] for elem in self.Sel_Table_2 if any([char in elem[2] for char in ['kde', 'Kde', 'KDE', 'kDE']])]
            Status_KDE = False
            if len(Stat_List) > 1:
                sg.popup_error("Only one 'KDE' overlayed event is permitted"); return None
            if len(Stat_List) == 1 and len(Data_1) > 2:
                sg.popup_error("Only 2-D 'KDE' in overlayed events is permitted"); return None
            elif len(Stat_List) == 1 and len(Data_1) == 2:
                Status_KDE = True

            for idx in range(len(self.Sel_Table_2)):
                Name = self.Sel_Table_2[idx][0]; Color = self.Sel_Table_2[idx][1]
                Status = self.Sel_Table_2[idx][2]; MapKey = self.Sel_Table_2[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if Data.size != Time.size:
                    if not is_points_not_times(Data): # Check if array is comprised of points or times and keep it on points
                        Data = np.around(LinearInterp(Time, np.arange(Time.size), Data, kind='linear'))
                    Data = Data.astype(int)

                    if Status_KDE: # Kernel density estimation
                        Z, Coords, MinMax_KDE = self.KDE_Calc([Data_1[0][0], Data_1[1][0]], Data, standalone=True)

                    Data_2.append([Data, Color, Name, Status])
                else:
                    sg.popup_error("Aborting Plot: Phase Portraits can only overlay 'Event' data type")
                    return None

            # Making plots
            OutDict = {"Trajectory Dimensions":{}, "Overlaid Events":{}}; outname = "PhPortrait"
            if len(Data_1) == 2:
                if plot:
                    self.Matplot_Config(1, fig_size=self.fig_size)
                    self.axList[0].plot(Data_1[0][0], Data_1[1][0], color=Data_1[0][1],
                                        label="2-D Phase", alpha=0.5, lw=self.fig_lw, zorder = 2)
                    self.axList[0].set_xlabel(Data_1[0][2]); self.axList[0].set_ylabel(Data_1[1][2])
                OutDict["Trajectory Dimensions"].update({"X_dim":Data_1[0][0], "Y_dim":Data_1[1][0],
                                                         "x_label":Data_1[0][2], "y_label":Data_1[1][2],
                                                         "Color":Data_1[0][1]})
            elif len(Data_1) == 3:
                if plot:
                    self.Matplot_Config(1, type="3D", fig_size=self.fig_size)
                    self.axList[0].plot(Data_1[0][0], Data_1[1][0], zs=Data_1[2][0], color=Data_1[0][1], label="3-D Phase", lw=self.fig_lw, alpha=0.5)
                    self.axList[0].set_xlabel(Data_1[0][2]); self.axList[0].set_ylabel(Data_1[1][2])
                    self.axList[0].set_zlabel(Data_1[2][2])
                OutDict["Trajectory Dimensions"].update({"X_dim":Data_1[0][0], "Y_dim":Data_1[1][0], "Z_dim":Data_1[2][0],
                                                         "x_label":Data_1[0][2], "y_label":Data_1[1][2], "z_label":Data_1[2][2],
                                                         "Color":Data_1[0][1]})
            for Data in Data_2:
                if len(Data_1) == 2:
                    if Status_KDE:
                        xmin = MinMax_KDE[0][0]; xmax = MinMax_KDE[0][1]
                        ymin = MinMax_KDE[1][0]; ymax = MinMax_KDE[1][1]
                        if plot:
                            im = self.axList[0].imshow(np.rot90(Z), cmap=plt.cm.seismic,
                                                       extent=[xmin, xmax, ymin, ymax],
                                                       zorder = 1)
                            divider = make_axes_locatable(self.axList[0])
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            self.figure.colorbar(im, cax=cax, orientation='vertical')
                    if plot:
                        self.axList[0].plot(np.take(Data_1[0][0], Data[0]),
                                            np.take(Data_1[1][0], Data[0]),
                                            color=Data[1], label=Data[2], lw=0,
                                            marker='o', ms=self.fig_ms, zorder = 2)
                    OutDict["Overlaid Events"].update({Data[2]:{"x_pos":np.take(Data_1[0][0], Data[0]),
                                                                "y_pos":np.take(Data_1[1][0], Data[0]),
                                                                "Color":Data[1]}})
                    if Status_KDE: OutDict["Overlaid Events"][Data[2]].update({"im_kernel":Z, "im_extent":[xmin, xmax, ymin, ymax]})
                elif len(Data_1) == 3:
                    if plot:
                        self.axList[0].plot(np.take(Data_1[0][0], Data[0]),
                                            np.take(Data_1[1][0], Data[0]),
                                            zs=np.take(Data_1[2][0], Data[0]),
                                            color=Data[1], label=Data[2],
                                            lw=0, ms=self.fig_ms, marker='o')
                    OutDict["Overlaid Events"].update({Data[2]:{"x_pos":np.take(Data_1[0][0], Data[0]),
                                                                "y_pos":np.take(Data_1[1][0], Data[0]),
                                                                "z_pos":np.take(Data_1[2][0], Data[0]),
                                                                "Color":Data[1]}})

            if plot: plt.rcParams['agg.path.chunksize'] = (len(Data_1)+len(Data_2)) * np.ceil(Time.size/Mag_Order)*Mag_Order*10
        elif "Raster/Colormap" in type:

            Idxs_SpeedDur = Speed_Duration_Threshold(Time, self.Body_Speed, speed_time_thres=self.speed_time_thres)

            Ref_key = []; Ord_key = []; Color_Ord = 'black'; PawSet = []
            Ph_shift = False; asc_sort = True; Bool_Ref_Phase = False
            for idx in range(len(self.Sel_Table_1)):
                Name = self.Sel_Table_1[idx][0]; Color = self.Sel_Table_1[idx][1]
                Status = self.Sel_Table_1[idx][2]; MapKey = self.Sel_Table_1[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if any(stat in Status for stat in ["ref", "Ref", "REF", 'rEF']): # Reference vector loc.
                    Ref_key = MapKey
                    if any(stat in Status for stat in ["paw", "Paw", "PAW", 'pAW']): # Make a pawset of the chosen reference variable if applicable
                        PawMatch = [paw for paw in self.Paw_List if paw in MapKey[-1]]
                        if not len(PawMatch) > 0:
                            sg.popup_error("Aborting Plot: Only paw-related variables can be used for a 'pawset' in a raster")
                            return None
                        PawSet = [MapKey[:-1] + [MapKey[-1].replace(PawMatch[0],paw)] for paw in self.Paw_List]
                if any(stat in Status for stat in ["ord", "Ord", "ORD", 'oRD']): # Ordering vector loc.
                    Ord_key = MapKey; asc_sort = True; Color_Ord = Color
                if any(stat in Status for stat in ["rord", "Rord", "RORD", 'rORD']): # Reverse Ordering vector loc.
                    Ord_key = MapKey; asc_sort = False; Color_Ord = Color
                if Data.size == Time.size: # For phase data
                    if np.nanmin(Data) == 0 and np.nanmax(Data) <= 2*np.pi: # Phase data
                        if any(stat in Status for stat in ["shift", "Shift", "SHIFT", 'sHIFT']): Ph_shift = True # Shift the phases if requested
                        Bool_Ref_Phase = True
                    elif not Ord_key == MapKey:
                        sg.popup_error("Aborting Plot: Time series are only valid as phase references (status = 'ref'), or ordering vectors (status = 'ord' or 'rord')")
                        return None

            if len(Ref_key) == 0:
                sg.popup_error("Aborting Plot: A single Phase or Event vector has to be marked as 'Reference' in its 'status' to use for the plot")
                return None

            Cont_key = []; Events_keys = []; Color_Evs = []; splog = False
            for idx in range(len(self.Sel_Table_2)):
                Name = self.Sel_Table_2[idx][0]; Color = self.Sel_Table_2[idx][1]
                Status = self.Sel_Table_2[idx][2]; MapKey = self.Sel_Table_2[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if Data.size == Time.size: # For continuos data
                    if any(stat in Status for stat in ["log", "Log", "LOG", 'lOG']): splog = True # Log-transform the variable if requested
                    Cont_key = MapKey
                elif Data.size != Time.size: # Events
                    if any(stat in Status for stat in ["paw", "Paw", "PAW", 'pAW']): # Make a pawset of the chosen reference variable if applicable
                        PawMatch = [paw for paw in self.Paw_List if paw in MapKey[-1]]
                        if not len(PawMatch) > 0:
                            sg.popup_error("Aborting Plot: Only paw-related events can be used for a 'pawset' in a raster")
                            return None
                        tempkeys = [MapKey[:-1] + [MapKey[-1].replace(PawMatch[0],paw)] for paw in self.Paw_List]
                        tempcolors = [self.Var_Color[paw] for paw in self.Paw_List]
                        Events_keys += tempkeys; Color_Evs += tempcolors
                    else:
                        Events_keys.append(MapKey); Color_Evs.append(Color)

            if len(Cont_key) == 0 and len(Events_keys) == 0:
                sg.popup_error("Aborting Plot: At least one series or discrete event dataset has to be chosen for the raster")
                return None

            # Create plot
            if len(PawSet) > 0:
                num_refs = len(self.Paw_List)
                if show: sg.OneLineProgressMeter('Reference PawSet', 1, num_refs, 'Single Progress Bar', 'Processing Plots')
            else:
                num_refs = 1; PawSet = [Ref_key]

            OutDict = {}; outname = "RastCor"
            for n in range(num_refs):
                # Get Rasters/Colormap from the extracted data
                Events_Dict, Mat_Ord, Plot_feat = \
                Colormap_Rasters(self.Data_Dict, Time, PawSet[n], Events_keys,
                                 Ord_key=Ord_key, Cont_key=Cont_key, asc_sort=asc_sort,
                                 phase_shift=Ph_shift, mask_idxs=Idxs_SpeedDur, hvbins=self.fig_bins,
                                 win_t=self.fig_wt, splog=splog, color_lkup=self.Var_Color, plot=False)

                # Save the Plot Data if required by user:
                Temp = {"Raster Events":Events_Dict, "Plotting Info":Plot_feat}
                if Mat_Ord.size > 0: Temp.update({"Colormap Image":Mat_Ord})
                OutDict.update({"Plot %d" % (n+1):deepcopy(Temp)})

            # Equalize the contrast of all plots
            if len(Cont_key) > 0:
                vmin_glob = np.amin([OutDict["Plot %d" % (n+1)]["Plotting Info"]['vmin'] for n in range(num_refs)])
                vmax_glob = np.amax([OutDict["Plot %d" % (n+1)]["Plotting Info"]['vmax'] for n in range(num_refs)])

            if plot:
                self.Matplot_Config(num_refs, type="Images", fig_size=self.fig_size, cbars=(len(Cont_key) > 0))
                n = 0
                for plotname, plotinfo in OutDict.items():
                    Events_Dict = plotinfo["Raster Events"]; Plot_feat = plotinfo["Plotting Info"]
                    if "Colormap Image" in plotinfo.keys(): Mat_Ord = plotinfo["Colormap Image"]
                    else: Mat_Ord = np.zeros(0)

                    # Plot the variables
                    self.figure.suptitle(Plot_feat['suptitle'])
                    self.axList[n].set_title(plotname, loc='left')
                    self.axList[n].set_xlabel(Plot_feat['x_label'])
                    if n == 0: self.axList[n].set_ylabel(Plot_feat['y_label'])
                    self.axList[n].set_xlim((Plot_feat['extent'][0],Plot_feat['extent'][1]))
                    self.axList[n].set_ylim((Plot_feat['extent'][2],Plot_feat['extent'][3]))
                    if Plot_feat['y2_label'] != "":
                        ax2 = self.axList[n].twinx(); ax2.set_ylim(Plot_feat['y2_range'])
                        ax2.set_yticks(Plot_feat['y2_ticks']); ax2.set_yticklabels(Plot_feat['y2_ticklabels'])
                        if n == len(OutDict) - 1: ax2.set_ylabel(Plot_feat['y2_label'])

                    for key, dct in Events_Dict.items():
                        self.axList[n].plot(dct['Ord. Events'][:,1], dct['Ord. Events'][:,0], marker='|', ms=self.fig_ms, lw=0, color=dct['Color'], label=key, zorder=2)

                    if Mat_Ord.size > 0:
                        im = self.axList[n].imshow(Mat_Ord, cmap=self.fig_colormap, origin='lower', vmin = vmin_glob, vmax = vmax_glob,
                                                   extent=Plot_feat['extent'], aspect='auto', zorder=1, label=Plot_feat['im_label'])
                        self.figure.colorbar(im, cax=self.cbarList[n], orientation='vertical')

                    if show: sg.OneLineProgressMeter('Reference PawSet', n+1, num_refs, 'Single Progress Bar', 'Processing Plots')
                    n += 1

                plt.rcParams['agg.path.chunksize'] = (len(Events_Dict)+1) * np.ceil(Time.size/Mag_Order)*Mag_Order
        elif "Generalized Hist." in type:

            Idxs_SpeedDur = Speed_Duration_Threshold(Time, self.Body_Speed, speed_time_thres=self.speed_time_thres)

            Ref_key = []; PawSet = []; Ref2_key = []
            Ph_shift = [False,False]; Bool_Ref_Phase = [False,False]
            for idx in range(len(self.Sel_Table_1)):
                Name = self.Sel_Table_1[idx][0]; Color = self.Sel_Table_1[idx][1]
                Status = self.Sel_Table_1[idx][2]; MapKey = self.Sel_Table_1[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                if any(stat in Status for stat in ["ref", "Ref", "REF", 'rEF']): # Reference vector loc.
                    if Data.size == Time.size: # Only valid for continuous data
                        if np.nanmin(Data) == 0 and np.nanmax(Data) <= 2*np.pi: # Phase data
                            if any(stat in Status for stat in ["shift", "Shift", "SHIFT", 'sHIFT']): # Shift the phases if requested
                                if "2" in Status:
                                    Bool_Ref_Phase[1] = True; Ph_shift[1] = True
                                else:
                                    Bool_Ref_Phase[0] = True; Ph_shift[0] = True
                    else:
                        sg.popup_error("Aborting Plot: Reference variable must be continuous")
                        return None
                    if "2" in Status:
                        if not len(Ref2_key): Ref2_key = MapKey
                        else:
                            sg.popup_error("Aborting Plot: Only one 'ref2' variable can be used")
                            return None
                    else: Ref_key = MapKey
                    ## Make a pawset of the chosen reference variable if applicable
                    if any(stat in Status for stat in ["paw", "Paw", "PAW", 'pAW']):
                        PawMatch = [paw for paw in self.Paw_List if paw in MapKey[-1]]
                        if not len(PawMatch) > 0:
                            sg.popup_error("Aborting Plot: Only paw-related variables can be used for a 'pawset' in a raster")
                            return None
                        PawSet = [MapKey[:-1] + [MapKey[-1].replace(PawMatch[0],paw)] for paw in self.Paw_List]

            if len(Ref_key) == 0:
                sg.popup_error("Aborting Plot: A single Phase or Event vector has to be marked as 'Reference' in its 'status' to use for the plot")
                return None

            DVar_info = []
            for idx in range(len(self.Sel_Table_2)):
                Name = self.Sel_Table_2[idx][0]; Color = self.Sel_Table_2[idx][1]
                Status = self.Sel_Table_2[idx][2]; MapKey = self.Sel_Table_2[idx][3].split(";")
                Data = getFromDict(self.Data_Dict, MapKey)
                ## Define the function to use in the bin computation (default is just counting the elements)
                func_hist = "counts"; splog = False
                if any(stat in Status for stat in ["mea", "Mea", "MEA", 'mEA']): func_hist = "mean" # Using mean
                if any(stat in Status for stat in ["med", "Med", "MED", 'mED']): func_hist = "median" # Using median
                if any(stat in Status for stat in ["div", "Div", "DIV", 'dIV']): func_hist = "division of counts" # Divide the counts of the independent variable to the dependent one
                if Data.size == Time.size: # For continuos data
                    if any(stat in Status for stat in ["log", "Log", "LOG", 'lOG']): splog = True # Log-transform the variable if requested
                    DVar_info.append((MapKey,Color,splog,func_hist))
                    if any([t in func_hist for t in ["count", "division of counts"]]):
                        print("Warning: Variable '%s' is continuous, so measurement was done with mean values instead of %s" % (MapKey[-1], func_hist))
                        func_hist = "mean"
                elif Data.size != Time.size: # Events
                    if any([t in func_hist for t in ["mean", "median"]]):
                        print("Warning: Variable '%s' are discrete events, so measurement was done with counts instead of %s values" % (MapKey[-1], func_hist))
                        func_hist = "counts"
                    if any(stat in Status for stat in ["paw", "Paw", "PAW", 'pAW']): # Make a pawset of the chosen reference variable if applicable
                        PawMatch = [paw for paw in self.Paw_List if paw in MapKey[-1]]
                        if not len(PawMatch) > 0:
                            sg.popup_error("Aborting Plot: Only paw-related events can be used for a 'pawset' in a histogram")
                            return None
                        tempkeys = [MapKey[:-1] + [MapKey[-1].replace(PawMatch[0],paw)] for paw in self.Paw_List]
                        tempcolors = [self.Var_Color[paw] for paw in self.Paw_List]
                        DVar_info += list(zip(tempkeys,tempcolors,[splog]*len(tempkeys),[func_hist]*len(tempkeys)))
                    else:
                        DVar_info.append((MapKey,Color,splog,func_hist))

            # Create plot
            if len(PawSet) > 0:
                Combo_DVar = list(product(PawSet, DVar_info))
                Combo_DVar = [(c[0], c[1][0], c[1][1], c[1][2], c[1][3]) for c in Combo_DVar]
                if show: sg.OneLineProgressMeter('Reference PawSet', 1, len(Combo_DVar), 'Single Progress Bar', 'Processing Plots')
            else:
                Combo_DVar = [(Ref_key,c[0],c[1],c[2],c[3]) for c in DVar_info]

            OutDict = {}; outname = "GenHist"
            for n in range(len(Combo_DVar)):
                # Make the histograms from the extracted data
                Source_Dict, Plot_feat = Generalized_Histograms(self.Data_Dict, Time, Combo_DVar[n][0], Combo_DVar[n][1], ref2=Ref2_key, color=Combo_DVar[n][2],
                                                                stat=Combo_DVar[n][4], phase_shift=Ph_shift, mask_idxs=Idxs_SpeedDur,
                                                                hvbins=self.fig_bins, splog=Combo_DVar[n][3], plot=False)

                # Save the Plot Data if required by user:
                Temp = {"Histograms":Source_Dict, "Plotting Info":Plot_feat}
                title = Plot_feat['suptitle'].split("\n"); Plot_feat['suptitle'] = title[1].split(";")[1]
                title[0] = "(" + title[1].split(";")[0].replace("Binned statistic: ","") + ") " + title[0].replace("BY ","BY\n")
                OutDict.update({title[0]:deepcopy(Temp)})

            if plot:
                numaxes = [int(np.ceil(np.sqrt(len(Combo_DVar)))), int(round(np.sqrt(len(Combo_DVar))+0.00001))] if len(Combo_DVar) > 3 else len(Combo_DVar)
                self.Matplot_Config(numaxes, type="Images", fig_size=self.fig_size, cbars=(len(Ref2_key) > 0))
                self.figure.suptitle("Generalized Hstograms\n" + Plot_feat['suptitle'])
                n = 0
                for plotname, plotinfo in OutDict.items():
                    Hist_Dict = plotinfo["Histograms"]; Plot_feat = plotinfo["Plotting Info"]

                    # Plot the variables
                    self.axList[n].set_title(plotname, loc='left')
                    self.axList[n].set_xlabel(Plot_feat['x_label'])
                    self.axList[n].set_ylabel(Plot_feat['y_label'])

                    if len(Hist_Dict["Values"].shape) > 1:
                        im = self.axList[n].imshow(np.rot90(Hist_Dict['Values']), cmap=self.fig_colormap, vmin=Plot_feat['vmin'], vmax=Plot_feat['vmax'],
                                                   extent=Plot_feat['extent'], aspect=Plot_feat['aspect'], zorder=1, label=Plot_feat['label'])
                        self.figure.colorbar(im, cax=self.cbarList[n], orientation='vertical', label=Plot_feat['cmap_label'])
                    else:
                        self.axList[n].fill_between(Hist_Dict['Bins'], Hist_Dict['Values'], step='mid', lw=0, color=Plot_feat['color'], label=Plot_feat['label'], zorder=1)

                    if show: sg.OneLineProgressMeter('Reference PawSet', n+1, len(Combo_DVar), 'Single Progress Bar', 'Processing Plots')
                    n += 1

            #self.figure.tight_layout()

            #plt.rcParams['agg.path.chunksize'] = (len(Events_Dict)+1) * np.ceil(Time.size/Mag_Order)*Mag_Order

        if self.Status_Save_Plot_Data:
            folder = os.path.join(self.output_folder, type); os.makedirs(folder, exist_ok=True)
            temp, ext = os.path.splitext(self.Filepath); namefile = os.path.basename(temp)
            Save_Dict_to_HDF5(OutDict, os.path.join(folder, "%s_PlotData.h5" % namefile))

        if png_save:
            folder = os.path.join(self.output_folder, type); os.makedirs(folder, exist_ok=True)
            temp, ext = os.path.splitext(self.Filepath); namefile = os.path.basename(temp)
            plt.savefig(os.path.join(folder, "%s_plot.png" % namefile))

        if plot and show: plt.show(block=False)

    def Rect_MultDim_Extraction(self, Reg_coords, Ev_coords, List_Ev_Vects=[]):
        # Procedure to extract events from a multidimensional rectangular region that
        # contains the points of an specified region.
        # Reg_coords: Coordinates of the Region, given in an array of shape '[Points, Dims]'.'
        # Ev_coords: Coordinates of the Events, with the same shape structure as 'Reg_coords'.
        # List_Ev_Vects: (Optional) List of additional vectors that will be procesed
        #                alongside the 'Ev_coords' vector (all vectors on list must
        #                have at least the same number of 'Points')

        # Extracting the bounds of the rectangular region:
        Min_Max_Dims = []
        for dim in range(Reg_coords.shape[1]):
            Min_Max_Dims.append((Reg_coords[:,dim].min(), Reg_coords[:,dim].max()))

        # Constructing the mask to exract the elements contained within the bounds:
        Mask_1d = np.ones(Ev_coords.shape[0]).astype(bool)
        for dim in range(len(Min_Max_Dims)):
            Mask_1d = np.logical_and(Mask_1d, (Ev_coords[:,dim] >= Min_Max_Dims[dim][0]))
            Mask_1d = np.logical_and(Mask_1d, (Ev_coords[:,dim] <= Min_Max_Dims[dim][1]))

        # Get the event coordinates inside the rectangular region
        Mask = np.vstack([Mask_1d]*len(Min_Max_Dims)).T
        Rect_Evs = Ev_coords[Mask].copy(); sec = int(Rect_Evs.size/len(Min_Max_Dims))
        if not Rect_Evs.size > 0:
            return Min_Max_Dims, Mask_1d, Rect_Evs, [np.zeros(0).astype(int)]*len(List_Ev_Vects)
        Rect_Evs = Rect_Evs.reshape((sec, len(Min_Max_Dims)))

        # Extract the elements from the list of vector using the same mask:
        New_List = []
        for Vect in List_Ev_Vects:
            Vect = np.squeeze(Vect)
            if Vect.shape[0] != Ev_coords.shape[0]:
                raise TypeError("Rect_MultDim_Extraction Error -> A vector in 'List_Ev_Vects' does not have the same points as 'Ev_coords'")
            if len(Vect.shape) > 2:
                raise TypeError("Rect_MultDim_Extraction Error -> Invalid shape for a vector in 'List_Ev_Vects'")
            elif len(Vect.shape) > 1:
                TempMask = np.squeeze(np.vstack([Mask_1d]*Vect.shape[1]).T)
            else:
                TempMask = Mask_1d
            New_List.append(Vect[TempMask].copy())

        return Min_Max_Dims, Mask_1d, Rect_Evs, New_List

    def KDE_RangePrep(self, List_DimVars, Event_idxs, bin_num, List_Ranges=None, phase_bound=False, wrap_bins=0):
        # Prepare the data to be fed into an user-defined multidimensional KDE.
        # List_DimVars: List of numpy arrays containing the variables that define the dimensional space
        #               where the event takes place. All arrays inside must have the same size.
        # Event_idxs: Indexes where one event happens simultaneously in all dimensional arrays
        # bin_num: Number of bins per dimension (for all of them).
        # List_Ranges: (Optional) List of tuples that define the range limits for each dimension,
        #              in the form (min, max). If provided, it must have the same number of elements
        #              as 'List_DimVars'.
        # phase_bound: (Optional) If True, the range of the KDE will be forced into (0, 2*pi).
        # wrap_bins: (Optional) Value that defines the number of additional bins used as padding in the extremities
        #            of the grid and data values, in order to remove fake boundary artifacts in phase.

        num_res = 2*np.finfo(np.float_).resolution
        NewEvents_Idxs = Event_idxs.copy(); MinMax_List = []
        if not List_Ranges is None:
            if len(List_Ranges) != len(List_DimVars):
                raise TypeError("KDE_RangePrep Abort: 'List_Ranges' does not have the same number of elements as 'List_DimVars'")

        if wrap_bins > bin_num:
            raise ValueError("KDE_RangePrep Abort: 'wrap_bins' must be equal or smaller than 'bin_num'")
        else:
            wrap_bins = int(wrap_bins)

        # First pass for the selection of events that fit in the range/s specified by the user
        for idx in range(len(List_DimVars)):
            Vals = np.take(List_DimVars[idx], NewEvents_Idxs)
            if not List_Ranges is None:
                rng_min = List_Ranges[idx][0] - num_res
                rng_max = List_Ranges[idx][1] + num_res
            else:
                rng_min = np.NINF; rng_max = np.inf
            if phase_bound:
                if np.isinf(rng_min): rng_min = - num_res
                if np.isinf(rng_max): rng_max = (2*np.pi) + num_res
            else:
                if np.isinf(rng_min): rng_min = Vals.min() - num_res
                if np.isinf(rng_max): rng_max = Vals.max() + num_res
            Selec = np.where((Vals > rng_min) & (Vals < rng_max))[0]
            MinMax_List.append((rng_min, rng_max))
            NewEvents_Idxs = np.take(NewEvents_Idxs, Selec)

        # Second pass for the choice of KDE ranges that fit the selected data.
        Vals_List = []; List_Slices = []
        for idx in range(len(List_DimVars)):
            Vals = np.take(List_DimVars[idx], NewEvents_Idxs)
            rng_min = MinMax_List[idx][0]; rng_max = MinMax_List[idx][1]
            Vals_List.append(Vals.copy())
            if phase_bound and wrap_bins != 0:
                IniRange = np.mgrid[rng_min:rng_max:bin_num*1j]
                resolution = np.min(np.diff(IniRange))
                # PadMin = np.amin(IniRange[-wrap_bins:] - 2*np.pi)
                # PadMax = np.amax(IniRange[:wrap_bins] + 2*np.pi)
                PadMin = -resolution*wrap_bins
                PadMax = 2*np.pi + resolution*wrap_bins
                total_bins = bin_num+2*wrap_bins
                List_Slices.append(np.s_[PadMin:PadMax:total_bins*1j])
                MinMax_List[idx] = (PadMin, PadMax)
            else:
                List_Slices.append(np.s_[rng_min:rng_max:bin_num*1j])
            # Test for datapoints outside of range
            # Test = (Vals>rng_min)&(Vals<rng_max)
            # print("Var %d Range: (%.03f, %.03f)" % (idx, rng_min, rng_max))
            # print("# Points: %d ; Datapoints outside range: %d" % (Test.size, np.count_nonzero(~Test)))

        Grid = np.mgrid[List_Slices]
        Values = np.vstack([Vals for Vals in Vals_List])
        Idxs_Stack = np.vstack([NewEvents_Idxs]*len(List_DimVars))
        NewEvents_Idxs = Idxs_Stack.copy()

        # Creating the padded data necessary for phase bounded analysis, to remove boundary artifacts:
        if phase_bound and wrap_bins != 0:
            # The padding shift instructions required for the repeated data values
            Instruct_Pads = list(product([0,2*np.pi,-2*np.pi], repeat=len(List_DimVars)))[1:]
            for inst in Instruct_Pads:
                Temp_Vals = []
                for shift, Vals in zip(inst, Vals_List):
                    Vals_temp = Vals + shift; Temp_Vals.append(Vals_temp.copy())
                Temp = np.vstack([Vals for Vals in Temp_Vals])
                Values = np.concatenate((Values, Temp), axis=1)
                NewEvents_Idxs = np.concatenate((NewEvents_Idxs, Idxs_Stack), axis=1)

            # Remove the data that is outside the range of the padding:
            Bool_Values = np.ones(Values.shape[1]).astype(bool)
            for idx, rng in enumerate(MinMax_List):
                Crit = (Values[idx]>rng[0]) & (Values[idx]<rng[1])
                Bool_Values = Bool_Values & Crit
            Crit = [np.where(Bool_Values)[0]+idx_reset for idx_reset in np.arange(Values.shape[0])*Values.shape[1]]
            Values = np.take(Values, Crit)
            NewEvents_Idxs = np.take(NewEvents_Idxs, Crit)[0]

        return Grid, Values, NewEvents_Idxs, MinMax_List

    def KDE_Calc(self, Grid, Values, bw_direct=None, wrap_pad=None, standalone=False, verbose=False):
        # Kernel Density Estimation of events ocurring in a multidimensional space.
        # Grid: List of numpy arrays containing the variables that define the dimensional space
        #       where the event takes place. All arrays inside must have the same size.
        # Values: Indexes where one event happens simultaneously in all dimensional arrays.
        # bw_direct: (Optional) Bandwidth Factor value to use directly into the KDE.
        # wrap_pad: (Optional) List or Tuple containing the information of the padding
        #            to allow for phase wrapping, in the form of '(Total_bins, Pad_bins)' or
        #            '(Total_bins, Pad_bins, Event_Indexes)' to trace back the events
        #            from their original vector with their indexes.
        # standalone: (Optional) If True, the output of the function will be simplified
        #             for standalone use on other functions (i.e. not optimized for Null Model)

        if verbose:
            print("Generating the kernel density estimation...", end=" ", flush=True)
            start = time.time()

        if bw_direct is None:
            kde = FFTKDE(kernel='gaussian', norm=2)
        else:
            kde = FFTKDE(kernel='gaussian', norm=2, bw=bw_direct)
        Coords = np.vstack([Grid[idx].ravel() for idx in range(Grid.shape[0])]).T
        points = kde.fit(Values.T).evaluate(Coords)
        Out_KDE = points.reshape(Grid[0].shape)

        if verbose:
            end = time.time()
            print("Done in %.03f" % (end - start))

        Interp_Data = {"KDE":Out_KDE,"Grid":Grid, "Input Shape":Values.shape}
        if not wrap_pad is None:
            Interp_Data.update({"Analysis Bins":wrap_pad[0], "Phase Pad Bins":wrap_pad[1]})

        # Removing the phase padding of the primary output of the function:
        if not wrap_pad is None:
            bins = wrap_pad[0]; pad = wrap_pad[1]
            if len(wrap_pad) > 2: Indexes = [wrap_pad[2]]
            else: Indexes = []

            s = [slice(pad,-pad,None)]*len(Out_KDE.shape)
            New_Grid = Grid[tuple([slice(0,len(Out_KDE.shape),None)]+s)].copy()
            Coords = np.vstack([New_Grid[idx].ravel() for idx in range(New_Grid.shape[0])]).T
            New_Out_KDE = Out_KDE[tuple(s)].copy()
            Min_Max_Dims, Mask, Values, Indexes = \
            self.Rect_MultDim_Extraction(Coords, Values.T, List_Ev_Vects=Indexes)
            if len(Indexes) > 0: Indexes = Indexes[0]

            if standalone:
                return New_Out_KDE, Coords, kde.bw
            else:
                return New_Out_KDE, Coords, Values.T, Indexes, New_Grid, kde.bw, Interp_Data
        else:
            if standalone:
                return Out_KDE, Coords, kde.bw
            else:
                return Out_KDE, Coords, Values.T, Indexes, Grid, kde.bw, Interp_Data

    def Exit_Proc(self, event):
        plt.close('LocoEphys Batch Plot')

if __name__=='__main__':
    inst = Plotter_GUI(def_outfolder=default_output_folder)
