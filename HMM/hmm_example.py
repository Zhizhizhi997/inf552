#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

import numpy as np
from hmmlearn import hmm

# 3 boxs
states=['box1','box2','box3']
# 2 colors of balls can be
observations=['red','white']
# initial picked box probability
start_probability=np.array([0.2,0.5,0.3])
# transmition matirx
transition_matrix = np.array([
  [0.5, 0.1, 0.4],
  [0.4, 0.5, 0.1],
  [0.1, 0.4, 0.5]
])

#emission probability
emission_matrix = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model=hmm.MultinomialHMM(n_components=len(states))
model.startprob_=start_probability
model.transmat_=transition_matrix
model.emissionprob_=emission_matrix

# do prediction
seen=np.array([0,1,1])
logprob,box=model.decode(seen.reshape(-1, 1),algorithm='viterbi')
print('The ball picked is :',','.join(map(lambda x:observations[x],seen)))
print('The hidden box is:',','.join(map(lambda x:states[x],box)))

box_pre=model.predict(seen.reshape(-1,1))
print('The ball picked is :',','.join(map(lambda x:observations[x],seen)))
print('The hidden box predict is :',','.join(map(lambda x:states[x],box_pre)))
print(model.score(seen.reshape(-1,1)))