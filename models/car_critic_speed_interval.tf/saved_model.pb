??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
critic_2/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecritic_2/dense_15/kernel
?
,critic_2/dense_15/kernel/Read/ReadVariableOpReadVariableOpcritic_2/dense_15/kernel*
_output_shapes

:*
dtype0
?
critic_2/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecritic_2/dense_15/bias
}
*critic_2/dense_15/bias/Read/ReadVariableOpReadVariableOpcritic_2/dense_15/bias*
_output_shapes
:*
dtype0
?
critic_2/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecritic_2/dense_16/kernel
?
,critic_2/dense_16/kernel/Read/ReadVariableOpReadVariableOpcritic_2/dense_16/kernel*
_output_shapes

:*
dtype0
?
critic_2/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecritic_2/dense_16/bias
}
*critic_2/dense_16/bias/Read/ReadVariableOpReadVariableOpcritic_2/dense_16/bias*
_output_shapes
:*
dtype0
?
critic_2/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecritic_2/dense_17/kernel
?
,critic_2/dense_17/kernel/Read/ReadVariableOpReadVariableOpcritic_2/dense_17/kernel*
_output_shapes

:*
dtype0
?
critic_2/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecritic_2/dense_17/bias
}
*critic_2/dense_17/bias/Read/ReadVariableOpReadVariableOpcritic_2/dense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
Adam/critic_2/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_15/kernel/m
?
3Adam/critic_2/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_15/kernel/m*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_15/bias/m
?
1Adam/critic_2/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/critic_2/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_16/kernel/m
?
3Adam/critic_2/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_16/kernel/m*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_16/bias/m
?
1Adam/critic_2/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_16/bias/m*
_output_shapes
:*
dtype0
?
Adam/critic_2/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_17/kernel/m
?
3Adam/critic_2/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_17/kernel/m*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_17/bias/m
?
1Adam/critic_2/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/critic_2/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_15/kernel/v
?
3Adam/critic_2/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_15/kernel/v*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_15/bias/v
?
1Adam/critic_2/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_15/bias/v*
_output_shapes
:*
dtype0
?
Adam/critic_2/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_16/kernel/v
?
3Adam/critic_2/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_16/kernel/v*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_16/bias/v
?
1Adam/critic_2/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_16/bias/v*
_output_shapes
:*
dtype0
?
Adam/critic_2/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/critic_2/dense_17/kernel/v
?
3Adam/critic_2/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_17/kernel/v*
_output_shapes

:*
dtype0
?
Adam/critic_2/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/critic_2/dense_17/bias/v
?
1Adam/critic_2/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/critic_2/dense_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
concat_layer

layer1

layer2

layer3
	optimizer
loss
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem@mAmBmCmDmEvFvGvHvIvJvK
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
?
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
	regularization_losses
 
 
 
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
VT
VARIABLE_VALUEcritic_2/dense_15/kernel(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcritic_2/dense_15/bias&layer1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
VT
VARIABLE_VALUEcritic_2/dense_16/kernel(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcritic_2/dense_16/bias&layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
VT
VARIABLE_VALUEcritic_2/dense_17/kernel(layer3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcritic_2/dense_17/bias&layer3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
yw
VARIABLE_VALUEAdam/critic_2/dense_15/kernel/mDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_15/bias/mBlayer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/critic_2/dense_16/kernel/mDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_16/bias/mBlayer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/critic_2/dense_17/kernel/mDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_17/bias/mBlayer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/critic_2/dense_15/kernel/vDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_15/bias/vBlayer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/critic_2/dense_16/kernel/vDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_16/bias/vBlayer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/critic_2/dense_17/kernel/vDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/critic_2/dense_17/bias/vBlayer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_args_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1critic_2/dense_15/kernelcritic_2/dense_15/biascritic_2/dense_16/kernelcritic_2/dense_16/biascritic_2/dense_17/kernelcritic_2/dense_17/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1670838
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,critic_2/dense_15/kernel/Read/ReadVariableOp*critic_2/dense_15/bias/Read/ReadVariableOp,critic_2/dense_16/kernel/Read/ReadVariableOp*critic_2/dense_16/bias/Read/ReadVariableOp,critic_2/dense_17/kernel/Read/ReadVariableOp*critic_2/dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp3Adam/critic_2/dense_15/kernel/m/Read/ReadVariableOp1Adam/critic_2/dense_15/bias/m/Read/ReadVariableOp3Adam/critic_2/dense_16/kernel/m/Read/ReadVariableOp1Adam/critic_2/dense_16/bias/m/Read/ReadVariableOp3Adam/critic_2/dense_17/kernel/m/Read/ReadVariableOp1Adam/critic_2/dense_17/bias/m/Read/ReadVariableOp3Adam/critic_2/dense_15/kernel/v/Read/ReadVariableOp1Adam/critic_2/dense_15/bias/v/Read/ReadVariableOp3Adam/critic_2/dense_16/kernel/v/Read/ReadVariableOp1Adam/critic_2/dense_16/bias/v/Read/ReadVariableOp3Adam/critic_2/dense_17/kernel/v/Read/ReadVariableOp1Adam/critic_2/dense_17/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1671049
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_2/dense_15/kernelcritic_2/dense_15/biascritic_2/dense_16/kernelcritic_2/dense_16/biascritic_2/dense_17/kernelcritic_2/dense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/critic_2/dense_15/kernel/mAdam/critic_2/dense_15/bias/mAdam/critic_2/dense_16/kernel/mAdam/critic_2/dense_16/bias/mAdam/critic_2/dense_17/kernel/mAdam/critic_2/dense_17/bias/mAdam/critic_2/dense_15/kernel/vAdam/critic_2/dense_15/bias/vAdam/critic_2/dense_16/kernel/vAdam/critic_2/dense_16/bias/vAdam/critic_2/dense_17/kernel/vAdam/critic_2/dense_17/bias/v*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1671128??
?	
?
E__inference_dense_17_layer_call_and_return_conditional_losses_1670753

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_16_layer_call_and_return_conditional_losses_1670937

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670897
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
*__inference_dense_15_layer_call_fn_1670906

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1670720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_critic_2_layer_call_fn_1670856	
state

action
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstateactionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_critic_2_layer_call_and_return_conditional_losses_1670760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namestate:OK
'
_output_shapes
:?????????
 
_user_specified_nameaction
?
?
E__inference_critic_2_layer_call_and_return_conditional_losses_1670760	
state

action"
dense_15_1670721:
dense_15_1670723:"
dense_16_1670738:
dense_16_1670740:"
dense_17_1670754:
dense_17_1670756:
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCallc
concatenate_2/CastCastaction*

DstT0*

SrcT0*'
_output_shapes
:??????????
concatenate_2/PartitionedCallPartitionedCallstateconcatenate_2/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670707?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_15_1670721dense_15_1670723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_15_layer_call_and_return_conditional_losses_1670720?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_1670738dense_16_1670740*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1670737?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_1670754dense_17_1670756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1670753x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namestate:OK
'
_output_shapes
:?????????
 
_user_specified_nameaction
?#
?
"__inference__wrapped_model_1670690

args_0

args_1B
0critic_2_dense_15_matmul_readvariableop_resource:?
1critic_2_dense_15_biasadd_readvariableop_resource:B
0critic_2_dense_16_matmul_readvariableop_resource:?
1critic_2_dense_16_biasadd_readvariableop_resource:B
0critic_2_dense_17_matmul_readvariableop_resource:?
1critic_2_dense_17_biasadd_readvariableop_resource:
identity??(critic_2/dense_15/BiasAdd/ReadVariableOp?'critic_2/dense_15/MatMul/ReadVariableOp?(critic_2/dense_16/BiasAdd/ReadVariableOp?'critic_2/dense_16/MatMul/ReadVariableOp?(critic_2/dense_17/BiasAdd/ReadVariableOp?'critic_2/dense_17/MatMul/ReadVariableOpl
critic_2/concatenate_2/CastCastargs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????d
"critic_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
critic_2/concatenate_2/concatConcatV2args_0critic_2/concatenate_2/Cast:y:0+critic_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
'critic_2/dense_15/MatMul/ReadVariableOpReadVariableOp0critic_2_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
critic_2/dense_15/MatMulMatMul&critic_2/concatenate_2/concat:output:0/critic_2/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(critic_2/dense_15/BiasAdd/ReadVariableOpReadVariableOp1critic_2_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
critic_2/dense_15/BiasAddBiasAdd"critic_2/dense_15/MatMul:product:00critic_2/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
critic_2/dense_15/ReluRelu"critic_2/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'critic_2/dense_16/MatMul/ReadVariableOpReadVariableOp0critic_2_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
critic_2/dense_16/MatMulMatMul$critic_2/dense_15/Relu:activations:0/critic_2/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(critic_2/dense_16/BiasAdd/ReadVariableOpReadVariableOp1critic_2_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
critic_2/dense_16/BiasAddBiasAdd"critic_2/dense_16/MatMul:product:00critic_2/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
critic_2/dense_16/ReluRelu"critic_2/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'critic_2/dense_17/MatMul/ReadVariableOpReadVariableOp0critic_2_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
critic_2/dense_17/MatMulMatMul$critic_2/dense_16/Relu:activations:0/critic_2/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(critic_2/dense_17/BiasAdd/ReadVariableOpReadVariableOp1critic_2_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
critic_2/dense_17/BiasAddBiasAdd"critic_2/dense_17/MatMul:product:00critic_2/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"critic_2/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^critic_2/dense_15/BiasAdd/ReadVariableOp(^critic_2/dense_15/MatMul/ReadVariableOp)^critic_2/dense_16/BiasAdd/ReadVariableOp(^critic_2/dense_16/MatMul/ReadVariableOp)^critic_2/dense_17/BiasAdd/ReadVariableOp(^critic_2/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2T
(critic_2/dense_15/BiasAdd/ReadVariableOp(critic_2/dense_15/BiasAdd/ReadVariableOp2R
'critic_2/dense_15/MatMul/ReadVariableOp'critic_2/dense_15/MatMul/ReadVariableOp2T
(critic_2/dense_16/BiasAdd/ReadVariableOp(critic_2/dense_16/BiasAdd/ReadVariableOp2R
'critic_2/dense_16/MatMul/ReadVariableOp'critic_2/dense_16/MatMul/ReadVariableOp2T
(critic_2/dense_17/BiasAdd/ReadVariableOp(critic_2/dense_17/BiasAdd/ReadVariableOp2R
'critic_2/dense_17/MatMul/ReadVariableOp'critic_2/dense_17/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?

?
E__inference_dense_16_layer_call_and_return_conditional_losses_1670737

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_17_layer_call_fn_1670946

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1670753o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_critic_2_layer_call_and_return_conditional_losses_1670884	
state

action9
'dense_15_matmul_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOpc
concatenate_2/CastCastaction*

DstT0*

SrcT0*'
_output_shapes
:?????????[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV2stateconcatenate_2/Cast:y:0"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_15/MatMulMatMulconcatenate_2/concat:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_namestate:OK
'
_output_shapes
:?????????
 
_user_specified_nameaction
?	
?
E__inference_dense_17_layer_call_and_return_conditional_losses_1670956

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670707

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_16_layer_call_fn_1670926

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1670737o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_15_layer_call_and_return_conditional_losses_1670917

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
[
/__inference_concatenate_2_layer_call_fn_1670890
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670707`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?_
?
#__inference__traced_restore_1671128
file_prefix;
)assignvariableop_critic_2_dense_15_kernel:7
)assignvariableop_1_critic_2_dense_15_bias:=
+assignvariableop_2_critic_2_dense_16_kernel:7
)assignvariableop_3_critic_2_dense_16_bias:=
+assignvariableop_4_critic_2_dense_17_kernel:7
)assignvariableop_5_critic_2_dense_17_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: E
3assignvariableop_11_adam_critic_2_dense_15_kernel_m:?
1assignvariableop_12_adam_critic_2_dense_15_bias_m:E
3assignvariableop_13_adam_critic_2_dense_16_kernel_m:?
1assignvariableop_14_adam_critic_2_dense_16_bias_m:E
3assignvariableop_15_adam_critic_2_dense_17_kernel_m:?
1assignvariableop_16_adam_critic_2_dense_17_bias_m:E
3assignvariableop_17_adam_critic_2_dense_15_kernel_v:?
1assignvariableop_18_adam_critic_2_dense_15_bias_v:E
3assignvariableop_19_adam_critic_2_dense_16_kernel_v:?
1assignvariableop_20_adam_critic_2_dense_16_bias_v:E
3assignvariableop_21_adam_critic_2_dense_17_kernel_v:?
1assignvariableop_22_adam_critic_2_dense_17_bias_v:
identity_24??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp)assignvariableop_critic_2_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_critic_2_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_critic_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp)assignvariableop_3_critic_2_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_critic_2_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_critic_2_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp3assignvariableop_11_adam_critic_2_dense_15_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp1assignvariableop_12_adam_critic_2_dense_15_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_adam_critic_2_dense_16_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_critic_2_dense_16_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp3assignvariableop_15_adam_critic_2_dense_17_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_critic_2_dense_17_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_critic_2_dense_15_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_critic_2_dense_15_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_critic_2_dense_16_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_critic_2_dense_16_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_critic_2_dense_17_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_critic_2_dense_17_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
E__inference_dense_15_layer_call_and_return_conditional_losses_1670720

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_1670838

args_0

args_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1670690o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?7
?
 __inference__traced_save_1671049
file_prefix7
3savev2_critic_2_dense_15_kernel_read_readvariableop5
1savev2_critic_2_dense_15_bias_read_readvariableop7
3savev2_critic_2_dense_16_kernel_read_readvariableop5
1savev2_critic_2_dense_16_bias_read_readvariableop7
3savev2_critic_2_dense_17_kernel_read_readvariableop5
1savev2_critic_2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop>
:savev2_adam_critic_2_dense_15_kernel_m_read_readvariableop<
8savev2_adam_critic_2_dense_15_bias_m_read_readvariableop>
:savev2_adam_critic_2_dense_16_kernel_m_read_readvariableop<
8savev2_adam_critic_2_dense_16_bias_m_read_readvariableop>
:savev2_adam_critic_2_dense_17_kernel_m_read_readvariableop<
8savev2_adam_critic_2_dense_17_bias_m_read_readvariableop>
:savev2_adam_critic_2_dense_15_kernel_v_read_readvariableop<
8savev2_adam_critic_2_dense_15_bias_v_read_readvariableop>
:savev2_adam_critic_2_dense_16_kernel_v_read_readvariableop<
8savev2_adam_critic_2_dense_16_bias_v_read_readvariableop>
:savev2_adam_critic_2_dense_17_kernel_v_read_readvariableop<
8savev2_adam_critic_2_dense_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B(layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBlayer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBlayer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_critic_2_dense_15_kernel_read_readvariableop1savev2_critic_2_dense_15_bias_read_readvariableop3savev2_critic_2_dense_16_kernel_read_readvariableop1savev2_critic_2_dense_16_bias_read_readvariableop3savev2_critic_2_dense_17_kernel_read_readvariableop1savev2_critic_2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop:savev2_adam_critic_2_dense_15_kernel_m_read_readvariableop8savev2_adam_critic_2_dense_15_bias_m_read_readvariableop:savev2_adam_critic_2_dense_16_kernel_m_read_readvariableop8savev2_adam_critic_2_dense_16_bias_m_read_readvariableop:savev2_adam_critic_2_dense_17_kernel_m_read_readvariableop8savev2_adam_critic_2_dense_17_bias_m_read_readvariableop:savev2_adam_critic_2_dense_15_kernel_v_read_readvariableop8savev2_adam_critic_2_dense_15_bias_v_read_readvariableop:savev2_adam_critic_2_dense_16_kernel_v_read_readvariableop8savev2_adam_critic_2_dense_16_bias_v_read_readvariableop:savev2_adam_critic_2_dense_17_kernel_v_read_readvariableop8savev2_adam_critic_2_dense_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : ::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
args_0/
serving_default_args_0:0?????????
9
args_1/
serving_default_args_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?N
?
concat_layer

layer1

layer2

layer3
	optimizer
loss
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
L__call__
*M&call_and_return_all_conditional_losses
N_default_save_signature"
_tf_keras_model
?
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem@mAmBmCmDmEvFvGvHvIvJvK"
	optimizer
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
	regularization_losses
L__call__
N_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
*:(2critic_2/dense_15/kernel
$:"2critic_2/dense_15/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
*:(2critic_2/dense_16/kernel
$:"2critic_2/dense_16/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
*:(2critic_2/dense_17/kernel
$:"2critic_2/dense_17/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/:-2Adam/critic_2/dense_15/kernel/m
):'2Adam/critic_2/dense_15/bias/m
/:-2Adam/critic_2/dense_16/kernel/m
):'2Adam/critic_2/dense_16/bias/m
/:-2Adam/critic_2/dense_17/kernel/m
):'2Adam/critic_2/dense_17/bias/m
/:-2Adam/critic_2/dense_15/kernel/v
):'2Adam/critic_2/dense_15/bias/v
/:-2Adam/critic_2/dense_16/kernel/v
):'2Adam/critic_2/dense_16/bias/v
/:-2Adam/critic_2/dense_17/kernel/v
):'2Adam/critic_2/dense_17/bias/v
?2?
*__inference_critic_2_layer_call_fn_1670856?
???
FullArgSpec&
args?
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_critic_2_layer_call_and_return_conditional_losses_1670884?
???
FullArgSpec&
args?
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference__wrapped_model_1670690args_0args_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_concatenate_2_layer_call_fn_1670890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_15_layer_call_fn_1670906?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_15_layer_call_and_return_conditional_losses_1670917?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_16_layer_call_fn_1670926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_16_layer_call_and_return_conditional_losses_1670937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_17_layer_call_fn_1670946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_17_layer_call_and_return_conditional_losses_1670956?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1670838args_0args_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1670690?Q?N
G?D
 ?
args_0?????????
 ?
args_1?????????
? "3?0
.
output_1"?
output_1??????????
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1670897?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
/__inference_concatenate_2_layer_call_fn_1670890vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
E__inference_critic_2_layer_call_and_return_conditional_losses_1670884?P?M
F?C
?
state?????????
 ?
action?????????
? "%?"
?
0?????????
? ?
*__inference_critic_2_layer_call_fn_1670856tP?M
F?C
?
state?????????
 ?
action?????????
? "???????????
E__inference_dense_15_layer_call_and_return_conditional_losses_1670917\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_15_layer_call_fn_1670906O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_16_layer_call_and_return_conditional_losses_1670937\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_16_layer_call_fn_1670926O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_17_layer_call_and_return_conditional_losses_1670956\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_17_layer_call_fn_1670946O/?,
%?"
 ?
inputs?????????
? "???????????
%__inference_signature_wrapper_1670838?e?b
? 
[?X
*
args_0 ?
args_0?????????
*
args_1 ?
args_1?????????"3?0
.
output_1"?
output_1?????????