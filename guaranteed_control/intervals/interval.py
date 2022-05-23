import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
#                      Class for manipulating intervals
# --------------------------------------------------------------------------

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

    def intersect(self, v):
        u = self
        new_intervals = []
        for i, axis in enumerate(u.intervals):
            if (v.intervals[i][0] > axis[1]) or (v.intervals[i][1] < axis[0]):
                return False
            new_intervals.append([max(v.intervals[i][0], axis[0]), min(v.intervals[i][1], axis[1])])
        return Interval(interval = np.array(new_intervals))

    def area(self):
        return np.prod(self.intervals[:, 1] - self.intervals[:, 0])


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

    def add(self, other, axis=None):
        if (axis != None) and (axis < self.dimension) and (axis < other.dimension):
            return Interval(interval=[self.intervals[axis] + other.intervals[axis]])
        return self + other

    def __neg__(self):
        intervals = np.copy(self.intervals)
        for i, axis in enumerate(intervals):
            intervals[i] = [-axis[1], -axis[0]]
        return Interval(interval=intervals)

    def neg(self, axis=0):
        if (axis != None) & (axis < self.dimension):
            return Interval(interval=-self.intervals[axis])

    def __sub__(self, other):
        return self + (-other)

    def sub(self, other, axis=0):
        if (axis != None) & (axis < self.dimension) & (axis < other.dimension):
            return Interval(interval=[self.intervals[axis] - other.intervals[axis]])

#Bad implementation
    def inverse(self, axis=None):

        if axis == None:
            intervals = np.zeros(shape=self.intervals.shape)
            for i, iter_axis in enumerate(self.intervals):
                axis0, axis1 = iter_axis[0], iter_axis[1]
                if iter_axis[0] == 0:
                    axis0 = np.inf
                    axis1 = 1/axis1
                elif iter_axis[1] == 0:
                    axis1 = np.inf
                    axis0= 1/axis0
                else:
                    axis0, axis1 =  1/axis0, 1/axis1

                if np.sign(axis0*axis1) > 0:
                    intervals[i] = [axis1, axis0]
                else:
                    intervals[i] = [axis0, axis1]
                
            return Interval(interval = intervals)

        return Interval(interval=[self.intervals[axis]]).inverse()

    def __mul__(self, other):
        intervals = np.zeros(shape=self.intervals.shape)
        interval1 = self.intervals
        interval2 = other.intervals
        for i in range(self.intervals.shape[0]):
            intervals[i] = [min(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0]), max(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0])]
        return Interval(interval=intervals)

    
    def mul(self, other, axis=None):
        if axis == None: 
            return self * other
            
        interval1 = self.intervals
        interval2 = other.intervals
        i = axis
        output = [min(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0]), max(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0])]
        return Interval(interval=[output])

    def __truediv__(self, other):
        return self * (other.inverse())

    def truediv(self, other, axis=None):
        if axis == None:
            return self/other
        
        return self.mul(other.inverse(axis=axis), axis=axis)

    def alpha(self, alpha, axis=None):

        if axis == None:
            if alpha>=0:
                return Interval(interval=alpha*np.copy(self.intervals[:]))
            intervals = np.zeros(shape=self.intervals.shape)
            for i, axis in enumerate(self.intervals):
                intervals[i] = [alpha*axis[1], alpha*axis[0]]
            return Interval(interval=intervals)

        else:
            if alpha>=0:
                return Interval(interval=alpha*self.intevals[axis])
        
            intervals = [alpha*self.intervals[axis][1], alpha*self.intervals[axis][0]]
            return Interval(interval=intervals)

#Only for positive powers?
    def __pow__(self, n):
        output = self
        for i in range(n):
            output = output*self
        return output

    def pow(self, n, axis=None):
        if axis == None:
            return self**n
        else:
            output = self
            for i in range(n):
                output = self.mul(self, axis=axis)
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

    def cos(self, axis = 0):
          
        a, b = self.intervals[axis]
        intervals = np.zeros((1, 2))
        a_bis, b_bis = np.floor_divide((a - np.pi), 2*np.pi), np.floor_divide((b - np.pi), 2*np.pi)
        intervals[0] = [min(np.cos(a), np.cos(b)), max(np.cos(a), np.cos(b))]
        if b_bis-a_bis >= 1:
            intervals[0][0] = -1
        a_bis, b_bis = np.floor_divide((a), 2*np.pi), np.floor_divide((b), 2*np.pi)
        if b_bis-a_bis >= 1:
            intervals[0][1] = 1
        return Interval(interval = intervals)
    

    def sin(self, axis=0):
     
        intervals = np.zeros((1,2))
        a, b = self.intervals[axis]
        a_bis, b_bis = np.floor_divide((a - np.pi/2), 2*np.pi), np.floor_divide((b - np.pi/2), 2*np.pi)
        intervals[0] = [min(np.sin(a), np.sin(b)), max(np.sin(a), np.sin(b))]
        if b_bis-a_bis >= 1:
            intervals[0][1] = 1
        a_bis, b_bis = np.floor_divide((a + np.pi/2), 2*np.pi), np.floor_divide((b + np.pi/2), 2*np.pi)
        if b_bis-a_bis >= 1:
            intervals[0][0] = -1
        return Interval(interval = intervals)

    def combine(self, other):
        return Interval(interval=np.concatenate([self.intervals, other.intervals], axis=0))

    def extract(self, axis=0):
        return Interval(interval=[self.intervals[axis]])

    def clip(self, min_value, max_value, axis=0):
        intervals = np.copy(self.intervals)
        #Careful here, what to do when we are completely out of range
        intervals[axis] = [max(min_value, min(max_value, self.intervals[axis][0])), min(max_value, max(min_value, self.intervals[axis][1]))]
        return Interval(interval = intervals)


# --------------------------------------------------------------------------
#                               Intervals utils
# --------------------------------------------------------------------------

def create_interval(set):

    set = np.array(set)
    mins = np.min(set, axis=0)
    maxs = np.max(set, axis=0)

    return Interval(interval=[[mins[i], maxs[i]] for i in range(set.shape[1])])

def over_appr_union(u):

    low, high = u[0].high_low()

    for interval in u:
        i_low, i_high = interval.high_low()
        low = np.where(i_low < low, i_low, low)
        high = np.where(i_high > high, i_high, high)

    return create_interval([low, high])

def cut_state_interval(state_interval, epsilon):
    intervals = []
    M = [state_interval]

    while len(M)>0:
        interval = M.pop(0)

        if interval.length() > epsilon:
            interval1, interval2 = interval.bissection()
            M.append(interval1)
            M.append(interval2)
            
        else:
            intervals.append(interval)
    
    return intervals

def IoU(interval1, interval2):
    intersection = interval1.intersect(interval2)
    
    if intersection == False:
        return 0

    area_intersect = intersection.area()

    area_union = interval1.area() + interval2.area() - area_intersect
    return area_intersect/area_union

def regroup_close(intervals, threshold=0.5):

    new_intervals = []
    intervals = intervals.copy()

    while len(intervals) != 0:
        interval = intervals.pop()

        merged = False
        for i, interval_ in enumerate(intervals): 
            if IoU(interval, interval_) > threshold:
                intervals[i] = over_appr_union([interval, interval_])
                merged = True
                break
        
        if not(merged):
            new_intervals.append(interval)

    return new_intervals



# -----------------------------------------------------------------------------------------
#                   Same class mainly using TensorFlow functions
# -----------------------------------------------------------------------------------------

class Interval_tf():

    def __init__(self, interval=None, tf=False):
        
        if not(tf):
            self.subclass = Interval(interval=interval)
            self.intervals = self.subclass.intervals
            self.dimension = self.intervals.shape[0]
        else:
            self.intervals = interval

    def bissection_tf(self, axis=0, max=True):

        if max:
            lows, highs = self.high_low()
            axis = np.argmax(highs-lows)

        center = (self.intervals[axis][0] + self.intervals[axis][1])/2

        new_interval1 = np.copy(self.intervals)
        new_interval2 = np.copy(self.intervals)
        
        new_interval1[axis] = (center, self.intervals[axis][1])
        new_interval2[axis] = (self.intervals[axis][0], center)

        return new_interval2, new_interval1

    def bissection(self, axis=0, max=True):

        new_interval1, new_interval2 = tf.numpy_function(self.bissection_tf, [], [tf.Tensor, tf.Tensor])

        return Interval_tf(interval=new_interval2), Interval_tf(interval=new_interval1)

    def high_low_tf(self):

        intervals = tf.convert_to_tensor(self.intervals)
        lows = intervals[:,0]
        highs = intervals[:,1]
        
        
        return lows, highs

    def high_low(self):

        lows, highs = tf.numpy_function(self.subclass.high_low, [], [tf.Tensor, tf.Tensor])
        
        return lows.numpy(), highs.numpy()

    def is_in(self, x):
        
        return tf.numpy_function(self.subclass.is_in, [x], [tf.bool]).numpy()
    
    def is_included_tf(self, interval2):
        #input an interval, is it okay?NO
        lows1, highs1 = self.high_low()
        lows2, highs2 = Interval_tf(interval=interval2).high_low()
        
        lows = lows1 >= lows2
        highs = highs2 >= highs1
        included = True

        for b in lows:
            included = included & b
        for b in highs:
            included = included & b
        
        return included

    def is_included(self, interval2):
        interval2_numpy = interval2.intervals
        return tf.numpy_function(self.is_included_tf, [interval2_numpy], [tf.bool]).numpy()


    def length_np(self):

        lows, highs = self.high_low()
        return np.max(highs-lows)

    def length(self):
        return tf.numpy_function(self.length_np, [], [tf.uint8]).numpy()

    def __add__(self, other):
        if self.dimension == other.dimension:
            return Interval(interval=self.intervals + other.intervals)
        print("Dimension error!")

    def add(self, other, axis=None):
        if (axis != None) and (axis < self.dimension) and (axis < other.dimension):
            return Interval(interval=[self.intervals[axis] + other.intervals[axis]])
        return self + other

    def __neg__(self):
        intervals = np.copy(self.intervals)
        for i, axis in enumerate(intervals):
            intervals[i] = [-axis[1], -axis[0]]
        return Interval(interval=intervals)

    def neg(self, axis=0):
        if (axis != None) & (axis < self.dimension):
            return Interval(interval=-self.intervals[axis])

    def __sub__(self, other):
        return self + (-other)

    def sub(self, other, axis=0):
        if (axis != None) & (axis < self.dimension) & (axis < other.dimension):
            return Interval(interval=[self.intervals[axis] - other.intervals[axis]])

#Bad implementation
    def inverse(self, axis=None):

        if axis == None:
            intervals = np.zeros(shape=self.intervals.shape)
            for i, iter_axis in enumerate(self.intervals):
                axis0, axis1 = iter_axis[0], iter_axis[1]
                if iter_axis[0] == 0:
                    axis0 = np.inf
                    axis1 = 1/axis1
                elif iter_axis[1] == 0:
                    axis1 = np.inf
                    axis0= 1/axis0
                else:
                    axis0, axis1 =  1/axis0, 1/axis1

                if np.sign(axis0*axis1) > 0:
                    intervals[i] = [axis1, axis0]
                else:
                    intervals[i] = [axis0, axis1]
                
            return Interval(interval = intervals)

        return Interval(interval=[self.intervals[axis]]).inverse()

    def __mul__(self, other):
        intervals = np.zeros(shape=self.intervals.shape)
        interval1 = self.intervals
        interval2 = other.intervals
        for i in range(self.intervals.shape[0]):
            intervals[i] = [min(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0]), max(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0])]
        return Interval(interval=intervals)

    
    def mul(self, other, axis=None):
        if axis == None: 
            return self * other
            
        interval1 = self.intervals
        interval2 = other.intervals
        i = axis
        output = [min(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0]), max(interval1[i][0] * interval2[i][1], interval1[i][0] * interval2[i][0], interval1[i][1] * interval2[i][1], interval1[i][1] * interval2[i][0])]
        return Interval(interval=[output])

    def __truediv__(self, other):
        return self * (other.inverse())

    def truediv(self, other, axis=None):
        if axis == None:
            return self/other
        
        return self.mul(other.inverse(axis=axis), axis=axis)

    def alpha(self, alpha, axis=None):

        if axis == None:
            if alpha>=0:
                return Interval(interval=alpha*np.copy(self.intervals[:]))
            intervals = np.zeros(shape=self.intervals.shape)
            for i, axis in enumerate(self.intervals):
                intervals[i] = [alpha*axis[1], alpha*axis[0]]
            return Interval(interval=intervals)

        else:
            if alpha>=0:
                return Interval(interval=alpha*self.intevals[axis])
        
            intervals = [alpha*self.intervals[axis][1], alpha*self.intervals[axis][0]]
            return Interval(interval=intervals)

#Only for positive powers?
    def __pow__(self, n):
        output = self
        for i in range(n):
            output = output*self
        return output

    def pow(self, n, axis=None):
        if axis == None:
            return self**n
        else:
            output = self
            for i in range(n):
                output = self.mul(self, axis=axis)
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

    def cos(self, axis = 0):
          
        a, b = self.intervals[axis]
        intervals = np.zeros((1, 2))
        a_bis, b_bis = np.floor_divide((a - np.pi), 2*np.pi), np.floor_divide((b - np.pi), 2*np.pi)
        intervals[0] = [min(np.cos(a), np.cos(b)), max(np.cos(a), np.cos(b))]
        if b_bis-a_bis >= 1:
            intervals[0][0] = -1
        a_bis, b_bis = np.floor_divide((a), 2*np.pi), np.floor_divide((b), 2*np.pi)
        if b_bis-a_bis >= 1:
            intervals[0][1] = 1
        return Interval(interval = intervals)
    

    def sin(self, axis=0):
     
        intervals = np.zeros((1,2))
        a, b = self.intervals[axis]
        a_bis, b_bis = np.floor_divide((a - np.pi/2), 2*np.pi), np.floor_divide((b - np.pi/2), 2*np.pi)
        intervals[0] = [min(np.sin(a), np.sin(b)), max(np.sin(a), np.sin(b))]
        if b_bis-a_bis >= 1:
            intervals[0][1] = 1
        a_bis, b_bis = np.floor_divide((a + np.pi/2), 2*np.pi), np.floor_divide((b + np.pi/2), 2*np.pi)
        if b_bis-a_bis >= 1:
            intervals[0][0] = -1
        return Interval(interval = intervals)

    def combine(self, other):
        return Interval(interval=np.concatenate([self.intervals, other.intervals], axis=0))

    def extract(self, axis=0):
        return Interval(interval=[self.intervals[axis]])

    def clip(self, min_value, max_value, axis=0):
        intervals = np.copy(self.intervals)
        #Careful here, what to do when we are completely out of range?
        intervals[axis] = [max(min_value, min(max_value, self.intervals[axis][0])), min(max_value, max(min_value, self.intervals[axis][1]))]
        return Interval(interval = intervals)
