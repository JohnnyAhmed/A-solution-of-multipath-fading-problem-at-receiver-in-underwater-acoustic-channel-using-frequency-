#!/usr/bin/env python
# coding: utf-8

# In[21]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)


# In[23]:


sampFreq, sound = wavfile.read('800,800,200.wav')


# In[24]:


sound.dtype, sampFreq


# In[25]:


sound = sound / 2.0**15


# In[26]:


sound.shape


# In[27]:


length_in_s = sound.shape[0] / sampFreq
print(length_in_s)


# In[28]:


plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.xlabel("left channel, sample #")
plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("right channel, sample #")
plt.tight_layout()
plt.show()


# In[29]:


time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s


# In[30]:


plt.subplot(2,1,1)
plt.plot(time, sound[:,0], 'r')
plt.xlabel("time, s [left channel]")
plt.ylabel("signal, relative units")
plt.subplot(2,1,2)
plt.plot(time, sound[:,1], 'b')
plt.xlabel("time, s [right channel]")
plt.ylabel("signal, relative units")
plt.tight_layout()
plt.show()


# In[31]:


signal = sound[:,0]


# In[32]:


plt.plot(time[6000:7000], signal[6000:7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show()


# In[33]:


fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
ln=len(signal)
print(ln)


# In[34]:


fft_spectrum


# In[35]:


fft_spectrum_abs = np.abs(fft_spectrum)


# In[36]:


plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()


# In[37]:


import numpy as np
max_value = np.max(fft_spectrum_abs)
max_value


# In[38]:


def print2largest(arr, arr_size):
  
    # There should be atleast 
    # two elements
    if (arr_size < 2):
        print(" Invalid Input ");
        return;
  
    largest = second = -2454635434;
  
    # Find the largest element
    for i in range(0, arr_size):
        largest = max(largest, arr[i]);
  
    # Find the second largest element
    for i in range(0, arr_size):
        if (arr[i] != largest):
            second = max(second, arr[i]);
  
    if (second == -2454635434):
        print("There is no second " + 
              "largest element");
    else:
        return second
  
# Driver code
if __name__ == '__main__':
    
    
    n = len(fft_spectrum_abs);
    second=print2largest(fft_spectrum_abs, n);
    print(second)


# In[40]:


am=2*abs(max_value/ln)
am1=2*abs(second/ln)
am2=2*abs((max_value+second)/ln)
print(am/2)
print(am1)
print(am2)


# In[ ]:




