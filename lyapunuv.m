%------------------------------------------------
% Lyapunuv Reachability NN for double integrator
%------------------------------------------------

A = [1 1
    0 1];

B = [0
    1];

load lyap_arrays;

u_min = nn_interval(1);
u_max = nn_interval(2);

%---------------------------------------------------------
% u_min, u_min

u_A = u_min;
u_B = u_min;
setlmis([])
P=lmivar(1,[2, 1]);




lmiterm([1 1 1 P], A', A);
lmiterm([1 1 1 P], -1, 1);
lmiterm([1 1 2 P], A', B*u_A);
lmiterm([1 2 1 P], u_A'*B', A);
lmiterm([1 2 2 P], u_B'*B', B*u_B);
lmiterm([-2 1 1 P], 1, 1);
lmiterm([2 1 1 0], 0);
lmis= getlmis;

[tmin,xfeas] = feasp(lmis);
P = dec2mat(lmis, xfeas, P)


%---------------------------------------------------------
% u_min, u_max

u_A = u_min;
u_B = u_max;
setlmis([])
P = lmivar(1, [2, 1]);




lmiterm([1 1 1 P], A', A);
lmiterm([1 1 1 P], -1, 1);
lmiterm([1 1 2 P], A', B*u_A);
lmiterm([1 2 1 P], u_A'*B', A);
lmiterm([1 2 2 P], u_B'*B', B*u_B);
lmiterm([-2 1 1 P], 1, 1);
lmiterm([2 1 1 0], 0);
lmis= getlmis;

[tmin,xfeas] = feasp(lmis);

%---------------------------------------------------------
% u_max, u_min

u_A = u_max;
u_B = u_min;
setlmis([])
P = lmivar(1, [2, 1]);




lmiterm([1 1 1 P], A', A);
lmiterm([1 1 1 P], -1, 1);
lmiterm([1 1 2 P], A', B*u_A);
lmiterm([1 2 1 P], u_A'*B', A);
lmiterm([1 2 2 P], u_B'*B', B*u_B);
lmiterm([-2 1 1 P], 1, 1);
lmiterm([2 1 1 0], 0);
lmis= getlmis;

[tmin,xfeas] = feasp(lmis);

%---------------------------------------------------------
% u_max, u_max

u_A = u_max;
u_B = u_max;
setlmis([])
P = lmivar(1, [2, 1]);




lmiterm([1 1 1 P], A', A);
lmiterm([1 1 1 P], -1, 1);
lmiterm([1 1 2 P], A', B*u_A);
lmiterm([1 2 1 P], u_A'*B', A);
lmiterm([1 2 2 P], u_B'*B', B*u_B);
lmiterm([-2 1 1 P], 1, 1);
lmiterm([2 1 1 0], 0);
lmis= getlmis;

[tmin,xfeas] = feasp(lmis);