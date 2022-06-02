# Interval module

This module implements many functions that allow to easily manipulate intervals and do arithmetic of intervals.

## Creating a new interval

We can create an interval if we have its representation in the form of [lower_bound, upper_bound] for every component

```python
from guaranteed_control.intervals.interval import Interval

i1 = Interval(interval=[[-1, 1], [-2, 2]])
#Return a low, high representation of the interval (two points)
low, high = i1.high_low()
#Return a component representation of the interval
i1_interval = i1.intervals
```

If we want to create an interval from a set of points that have the same dimension:

```python
from guaranteed_control.intervals.interval import create_interval
import numpy as np

set1 = np.concatenate([np.random.uniform([-1, 1], size=1000).reshape(-1,1), np.random.uniform([-2, 2], size=1000).reshape(-1, 1)], axis=1)
i1 = create_interval(set1)
```

We can also create an interval from different intervals by taking their union and over-approximating it by an interval.

```python
from guaranteed_control.intervals.interval import Interval, over_appr_union

i1 = Interval(interval=[[-0.5, 1], [-1.5, 2]])
i2 = Interval(interval=[[-1, -0.5], [-2, -1.5]])
i1unioni2 = over_appr_union([i1, i2])
```

## Interval arithmetics

We provide multiple operations to do computations with intervals :
 * $i_1 + i_2$
 * $i_1 - i_2$
 * $i_1*i_2$
 * $i_1/i_2$
 * $\alpha i_1$
 * $i_1^n, n \in \mathbb{N}$
 * $\varphi(i_1)$, où $\varphi$ est une fonction monotone
 * $\cos(i_1)$
 * $\sin(i_1)$

## Interval utility functions

We can also manipulate intervals by checking if one interval is included inside another, separate an interval of dimension n into an interval of dimension 1 by extraction one component. We can also regroup two interval components into one interval of higher dimension, clip intervals axis etc. (check the code for more information on every function that is implemented or how to implement more of them).