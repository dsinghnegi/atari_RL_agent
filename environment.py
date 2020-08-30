from preprocessing import PongDeterministic as PD
from preprocessing import BreakoutDeterministic as BD
from preprocessing import KungFuMasterDeterministic as KD
from preprocessing import BoxingDeterministic, SpaceInvadersDeterministic, BerzerkDeterministic

ENV_DICT={
'PongDeterministic-v4':PD,
'BreakoutDeterministic-v4':BD,
'KungFuMasterDeterministic-v4':KD,
'BoxingDeterministic-v4':BoxingDeterministic,
'SpaceInvadersDeterministic-v4':SpaceInvadersDeterministic,
'BerzerkDeterministic-v4':BerzerkDeterministic,
}