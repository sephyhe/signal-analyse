"""
Singal Analyse Tool
Written by Xiaofang(Sophie) He
Date: 05/2017
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt   # python package for denoising
from astropy.convolution import Gaussian1DKernel, convolve


def draw_graphs(data_list):
    """ using matplotlib pakage to draw the graph

    @parameters
    data_list: a dictionary contains the time and values in this time which is read from file.
    """
    channel_1 = data_list[0]
    channel_2 = data_list[1]
    channel_3 = data_list[2]
    channel_4 = data_list[3]

    plt.subplot(4, 1, 1)
    plt.plot(channel_1['times'], channel_1['denoised_values'])
    plt.title('channel_1')

    denoised_value = np.array(channel_1['denoised_values'])
    peaks_index = indexes(denoised_value, thres = 0.25, min_dist=20)
    realpeak1 = measurePeak(channel_1, peaks_index, thres=0.10)
    for i in realpeak1:
        plt.plot(channel_1['times'][i], channel_1['denoised_values'][i], marker='o', markersize=3, color="red")

    plt.subplot(4, 1, 2)
    plt.plot(channel_2['times'], channel_2['denoised_values'])
    plt.title('channel_2')
    denoised_value2 = np.array(channel_2['denoised_values'])
    peaks_index2 = indexes(denoised_value2, thres = 0.25, min_dist=30)
    realpeak2 = measurePeak(channel_2, peaks_index2, thres=0.22)
    # for i in peaks_index2:
    for i in realpeak2:
        plt.plot(channel_2['times'][i], channel_2['denoised_values'][i], marker='o', markersize=3, color="orange")

    plt.subplot(4, 1, 3)
    plt.plot(channel_3['times'], channel_3['denoised_values'])
    plt.title('channel_3')
    denoised_value3 = np.array(channel_3['denoised_values'])
    peaks_index3 = indexes(denoised_value3, thres = 0.25, min_dist=20)
    realpeak3 = measurePeak(channel_3, peaks_index3, thres= -0.2)
    for i in realpeak3:
        plt.plot(channel_3['times'][i], channel_3['denoised_values'][i], marker='o', markersize=3, color="red")

    plt.subplot(4, 1, 4)
    plt.plot(channel_4['times'], channel_4['denoised_values'])
    plt.title('channel_4')
    denoised_value4 = np.array(channel_4['denoised_values'])
    peaks_index4 = indexes(denoised_value4, thres = 0.10, min_dist=20)
    realpeak4 = measurePeak(channel_4, peaks_index4, thres=0.22)
    # for i in peaks_index4:
    for i in realpeak4:
        plt.plot(channel_4['times'][i], channel_4['denoised_values'][i], marker='o', markersize=3, color="orange")
    plt.show()


def smooth_test(data_list):
    '''Test the smooth function to reduce the noise'''
    channel_1 = data_list[0]
    channel_2 = data_list[1]
    channel_3 = data_list[2]
    channel_4 = data_list[3]

    g = Gaussian1DKernel(stddev=30)
    # channel_2 data using new algorithm
    plt.subplot(4, 1, 1)
    plt.plot(channel_2['times'], channel_2['denoised_values'])
    plt.title('channel_2')
    denoised_value_2 = np.array(channel_2['denoised_values'])
    peaks_index2 = indexes(denoised_value_2, thres = 0.35, min_dist=30)
    for i in peaks_index2:
        plt.plot(channel_2['times'][i], channel_2['denoised_values'][i], marker='o', markersize=3, color="orange")


    plt.subplot(4, 1, 2)
    denoised_value_2s = np.array(convolve(channel_2['denoised_values'],g))
    plt.plot(channel_4['times'],denoised_value_2s)
    plt.title('channel_2s smoothed')
    peaks_index_2s = indexes(denoised_value_2s, thres = 0.40, min_dist=20)
    for i in peaks_index_2s:
        plt.plot(channel_2['times'][i],denoised_value_2s[i], marker='o', markersize=3, color="red")


    # channel_4 data using new algorithm
    plt.subplot(4, 1, 3)
    plt.plot(channel_4['times'], channel_4['denoised_values'])
    plt.title('channel_4')
    denoised_value4 = np.array(channel_4['denoised_values'])
    peaks_index4 = indexes(denoised_value4, thres = 0.35, min_dist=20)
    # realpeak4 = measurePeak(channel_4, peaks_index4, thres = 0.22)
    for i in peaks_index4:
        plt.plot(channel_4['times'][i], channel_4['denoised_values'][i], marker='o', markersize=3, color="orange")


    plt.subplot(4, 1, 4)
    denoised_value_s = np.array(convolve(channel_4['denoised_values'],g))
    plt.plot(channel_4['times'],denoised_value_s)
    plt.title('channel_4s smoothed')
    peaks_index_s = indexes(denoised_value_s, thres = 0.35, min_dist=20)
    realpeak4_s = measurePeak(channel_4, peaks_index4, thres=0.2)
    for i in realpeak4_s:
    # for i in peaks_index_s:
        plt.plot(channel_4['times'][i],denoised_value_s[i], marker='o', markersize=3, color="red")
    plt.show()



def compare_peaks(data_list):
    ''''''
    channel_1 = data_list[0]
    channel_2 = data_list[1]
    channel_3 = data_list[2]
    channel_4 = data_list[3]

    # channel_2 data using new algorithm
    plt.subplot(4, 1, 1)
    plt.plot(channel_2['times'], channel_2['denoised_values'])
    plt.title('channel_2 with new algorithm')
    denoised_value_2 = np.array(channel_2['denoised_values'])
    peaks_index2 = indexes(denoised_value_2, thres = 0.2, min_dist=30)
    realpeak = measurePeak(channel_2, peaks_index2, thres=0.23)
    for i in realpeak:
        plt.plot(channel_2['times'][i], channel_2['denoised_values'][i], marker='o', markersize=3, color="orange")


    plt.subplot(4, 1, 2)
    denoised_value_2s = np.array(channel_2['denoised_values'])
    plt.plot(channel_4['times'],denoised_value_2s)
    plt.title('channel_2s with old algorithm')
    peaks_index_2s = indexes(denoised_value_2s, thres = 0.35, min_dist=20)
    for i in peaks_index_2s:
        plt.plot(channel_2['times'][i],denoised_value_2s[i], marker='o', markersize=3, color="red")


    # channel_4 data using new algorithm
    plt.subplot(4, 1, 3)
    plt.plot(channel_4['times'], channel_4['denoised_values'])
    plt.title('channel_4 with new algorithm')
    denoised_value4 = np.array(channel_4['denoised_values'])
    peaks_index4 = indexes(denoised_value4, thres = 0.10, min_dist=20)
    realpeak4 = measurePeak(channel_4, peaks_index4, thres = 0.22)
    for i in realpeak4:
        plt.plot(channel_4['times'][i], channel_4['denoised_values'][i], marker='o', markersize=3, color="orange")


    plt.subplot(4, 1, 4)
    denoised_value_s = np.array(channel_4['denoised_values'])
    plt.plot(channel_4['times'],denoised_value_s)
    plt.title('channel_4s with old algorithm')
    peaks_index_s = indexes(denoised_value_s, thres = 0.35, min_dist=20)
    realpeak4_s = measurePeak(channel_4, peaks_index4, thres=0.2)
    # for i in realpeak4_s:
    for i in peaks_index_s:
        plt.plot(channel_4['times'][i],denoised_value_s[i], marker='o', markersize=3, color="red")
    plt.show()



def measurePeak(channel, peaks_index, thres=0.2):
    """"""
    realpeak_index = []
    values = np.array(channel['denoised_values'])
    thres = thres * (np.max(values) - np.min(values)) + np.min(values)
    print("threshold: ", thres)
    # print("channel_2 value", channel)
    # print("peaks_index2 value: ", peaks_index)
    for i in peaks_index:
        peak_value = values[i]
    # depend on left value to find peaks
        if i < 10:
            diff_l = peak_value - values[0]
        else:
            diff_l = peak_value - values[i-10]

        if i < 70:
            minNum_l = min(values[:i])
        else:
            minNum_l = min(values[(i-70):i])

    # depend on right value to find peaks
        data_len = len(values)
        if i > data_len - 70:
            diff_r = peak_value - values[data_len-1]
            minNum_r = min(values[i:data_len])
        else:
            diff_r = peak_value - values[i+10]
            minNum_r = min(values[i:(i+70)])

        height = peak_value - minNum_l
        if ( height > thres and diff_l > 300) or(height > thres and diff_r > 300):
            realpeak_index.append(i)
            # print('peak_value: ', peak_value)
            # print('diff: ', diff)
            # print('minNum: ', minNum)

    return np.array(realpeak_index)


def analyse_data(data):
    """Analyse the data from the file, denoised them."""
    times = []
    values = []
    channel = {}
    data_list = []

    for item in data:
        for key, value in item.items():
            times.append(float(key))
            value = 0 - float(value) #exchange
            values.append(value)
        channel['times'] = times
        channel['values'] = values
        data = np.array(channel['values'])
        denoise(data,channel,'db4',5,1,5)
        new_channel = channel.copy()
        data_list.append(new_channel)
        times = []
        values = []
        channel.clear()

    return data_list



def read_data(filename):
    """Read the data from the file
    @Parameters: Filenae
    @Return: A list of 4 channel data"""
    num_data = {}
    data_list = []
    with open(filename) as infile:
        for line in infile:
            if line[0].isdigit():
                try:
                    time, value = line.strip().split('\t')
                except ValueError:
                    print('There is a ValueError in line:'+ line)
                num_data[time] = value
            elif line.startswith('; Format'):
                if len(num_data) > 0:
                    data_dic = num_data.copy()
                    data_list.append(data_dic)
                    num_data.clear()
    return data_list


def denoise(index_list,data,wavefunc,lv,m,n):
    """Function is to reduce the noise of wave.
    @parameters:
    index_list: values needed to be denoised.
    data: a data dictionary which contains times and the values of this time.
    wavefunc: wavefunction which contains in the pywt package
    lv: the level which wave need to be seperated.
    m & n: level range.

    @return
    denoised values stored in the data dictionary with a key named 'denoised_values'"""
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    for i in range(m,n+1):   # select m to n level
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))  # calculate the threshold
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr
            else:
                coeff[i][j] = 0   # if the value lower than threshold, set the value to 0
    denoised_values = pywt.waverec(coeff,wavefunc)
    data['denoised_values']=denoised_values



def indexes(y, thres=0.3, min_dist=1):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)

    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks



def main():
    data = read_data("D:/Bio-git/biosensor/test.ASC")#the file path should be changed
    data_list = analyse_data(data)
    draw_graphs(data_list)
    # compare_peaks(data_list)
    # smooth_test(data_list)

main()
