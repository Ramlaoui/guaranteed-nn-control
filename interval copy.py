import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class Interval():

    def __init__(self, interval=None):

        if Interval == None:
            self.intervals = np.array([])
        else:
            self.intervals = np.array(interval)
        self.dimension = self.intervals.shape[0]


    def bissection(self, axis=0, max=True):

        if max:
            lows, highs = self.high_low()
            axis = np.argmax(highs-lows)

        center = (self.intervals[axis][0] + self.intervals[axis][1])/2

        new_interval1 = np.copy(self.intervals)
        new_interval2 = np.copy(self.intervals)
        
        new_interval1[axis] = (center, self.intervals[axis][1])
        new_interval2[axis] = (self.intervals[axis][0], center)

        return Interval(interval=new_interval2), Interval(interval=new_interval1)

    def high_low(self):

        lows = []
        highs = []
        for interval in self.intervals:
            lows.append(interval[0])
            highs.append(interval[1])
        
        return np.array(lows), np.array(highs)

    def is_in(self, x):
        for i, axis in enumerate(self.intervals):
            if (x[i] < axis[0]) or (x[i] > axis[1]):
                return False
        return True

    def is_included(self, interval2):

        lows1, highs1 = self.high_low()
        lows2, highs2 = interval2.high_low()
        
        lows = lows1 >= lows2
        highs = highs2 >= highs1
        included = True

        for b in lows:
            included = included & b
        for b in highs:
            included = included & b
        
        return included

    def length(self):

        lows, highs = self.high_low()
        return np.max(highs-lows)

    def __add__(self, other):
        if self.dimension == other.dimension:
            return Interval(interval=self.intervals + other.intervals)
        print("Dimension error!")

    def __neg__(self):
        intervals = np.copy(self.intervals)
        for i, axis in enumerate(intervals):
            intervals[i] = [-axis[1], -axis[0]]
        return Interval(interval=intervals)

    def __sub__(self, other):
        return self + (-other)

    def inverse(self):
        intervals = np.zeros(shape=self.intervals.shape)
        for i, axis in enumerate(self.intervals):
            axis0, axis1 = axis[0], axis[1]
            if axis[0] == 0:
                axis0 = np.inf()
                axis1 = 1/axis1
            elif axis[1] == 0:
                axis1 = np.inf()
                axis0= 1/axis0
            else:
                axis0, axis1 =  1/axis0, 1/axis1
            
            intervals[i] = [axis1, axis0]
            
        return Interval(interval = intervals)

    def __mul__(self, other):
        intervals = np.zeros(shape=self.intervals.shape)
        interval1 = self.intervals
        interval2 = other.intervals
        for i in range(self.intervals.shape[0]):
            intervals[i] = [min(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0]), max(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0])]
        return Interval(interval=intervals)

    def __truediv__(self, other):
        return self * (other.inverse())

    def alpha(self, alpha):
        if alpha>=0:
            return Interval(interval=alpha*np.copy(self.intervals[:]))
        intervals = np.zeros(shape=self.intervals.shape)
        for i, axis in enumerate(self.intervals):
            intervals[i] = [alpha*axis[1], alpha*axis[0]]
        return Interval(interval=intervals)

    def __pow__(self, n):
        for i in range(n):
            output = self*self
        return output

    def apply_incr(self, funct):
        intervals = np.zeros(self.intervals.shape)
        for i, axis in enumerate(self.intervals):
            intervals[i] = [funct(axis[0]), funct(axis[1])]
        return Interval(interval = intervals)

    def apply_decr(self, funct):
        intervals = np.zeros(self.intervals.shape)
        for i, axis in enumerate(self.intervals):
            intervals[i] = [funct(axis[1]), funct(axis[0])]
        return Interval(interval = intervals)

    def cos(self):
        if self.intervals.shape[0] ==1:   
            intervals = np.zeros(self.intervals.shape)
            a, b = self.intervals[0]
            a_bis, b_bis = np.floor_divide((a - np.pi), 2*np.pi), np.floor_divide((b - np.pi), 2*np.pi)
            intervals[0] = [min(np.cos(a), np.cos(b)), max(np.cos(a), np.cos(b))]
            if b_bis-a_bis >= 1:
                intervals[0][0] = -1
            a_bis, b_bis = np.floor_divide((a), 2*np.pi), np.floor_divide((b), 2*np.pi)
            if b_bis-a_bis >= 1:
                intervals[0][1] = 1
            return Interval(interval = intervals)
        else:
            return None

    def sin(self):
        if self.intervals.shape[0] ==1:   
            intervals = np.zeros(self.intervals.shape)
            a, b = self.intervals[0]
            a_bis, b_bis = np.floor_divide((a - np.pi/2), 2*np.pi), np.floor_divide((b - np.pi/2), 2*np.pi)
            intervals[0] = [min(np.sin(a), np.sin(b)), max(np.sin(a), np.sin(b))]
            if b_bis-a_bis >= 1:
                intervals[0][1] = 1
            a_bis, b_bis = np.floor_divide((a + np.pi/2), 2*np.pi), np.floor_divide((b + np.pi/2), 2*np.pi)
            if b_bis-a_bis >= 1:
                intervals[0][0] = -1
            return Interval(interval = intervals)
        else:
            return None





def create_interval(set):

    set = np.array(set)
    mins = np.min(set, axis=0)
    maxs = np.max(set, axis=0)

    return Interval(interval=[[mins[i], maxs[i]] for i in range(set.shape[1])])