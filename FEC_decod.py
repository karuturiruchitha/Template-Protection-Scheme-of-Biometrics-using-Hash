import matplotlib.pyplot as plt
from sk_dsp_comm.fec_conv import FECConv
from sk_dsp_comm import digitalcom as dc
import numpy as np
cc = FECConv()
x = np.random.randint(0,2,20)#[1,0,1,1,0]
state = '00'
y,state = cc.conv_encoder(x,state)
# Add channel noise to bits translated to +1/-1
yn = dc.cpx_awgn(2*y-1,5,1) # SNR = 5 dB
# Translate noisy +1/-1 bits to soft values on [0,7]
yn = (yn.real+1)/2*7
z = cc.viterbi_decoder(yn)
print (z)


import hashlib
  
# initializing string
#str = "GeeksforGeeks"
  
# encoding GeeksforGeeks using encode()
# then sending to SHA256()
z=str(z)
result = hashlib.sha256(z.encode())

# printing the equivalent hexadecimal value.
print("The hexadecimal equivalent of SHA256 is : ")
print(result.hexdigest())
