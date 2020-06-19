from preprocessing import BreakoutNoFrameskip as BNF
from preprocessing import PongNoFrameskip as PNF
from preprocessing import BreakoutDeterministic as BD

ENV_DICT={
'BreakoutNoFrameskip-v4':BNF,
'PongNoFrameskip-v4':PNF,
'BreakoutDeterministic-v4':BD,
}