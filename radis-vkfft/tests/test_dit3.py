
from sys import path
path.append('..')
import os

from vulkan_compute_lib import GPUApplication, GPUBuffer
import numpy as np
from scipy.fft import next_fast_len
from numpy.fft import rfftfreq
from numpy.random import rand, randint, seed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ctypes import Structure, c_float, c_int, sizeof, c_uint, sizeof
from time import perf_counter
import sys

L = lambda t, w: 2 / (w * np.pi) * 1 / (1 + 4 * (t / w) ** 2)
L_FT = lambda f, w: np.exp(-np.pi * np.abs(f) * w)



class workGroupSize_t(Structure):
    _fields_ = [
        ("x", c_uint),
        ("y", c_uint),
        ("z", c_uint),
        ("id", c_uint),
    ]

workGroupSizeArray_t = workGroupSize_t * 4

class init_t(Structure):
    _fields_ = [
        ("Nl", c_int),
        ("Nt", c_int),
        ("Nf", c_int),

        ("t_min", c_float),
        ("dt",    c_float),
        ("w_min", c_float),
    ]

class iter_t(Structure):
    _fields_ = [
        ("a", c_float),
        ("Nw", c_int),
        ("dxw",   c_float),
    ]



def next_fast_len_even(n):
    n = next_fast_len(n)
    while n&1:
        n = next_fast_len(n+1)
    return n


def mock_spectrum(Nl, t_min=0.0, t_max=1.0, w_min=0.0, w_max=1.0):
    I0_arr = rand(Nl)
    t0_arr = rand(Nl)*(t_max - t_min) + t_min
    w0_arr = rand(Nl)*(w_max - w_min) + w_min

    return I0_arr.astype(np.float32), t0_arr.astype(np.float32), w0_arr.astype(np.float32)


t_min = 0.0
t_max = 100.0
Nt = 500001
Nt = next_fast_len_even(Nt)
print("Nt = {:d}".format(Nt))
t_arr = np.linspace(t_min, t_max, Nt)
dt = t_arr[1]

#f_arr = rfftfreq(Nt, dt)
#Nf = len(f_arr)

Nf = Nt//2 + 1
f_arr = np.arange(Nf) / (2 * Nf * dt)

print(Nf)


Ntpb = 1024  # threads per block
threads = (Ntpb, 1, 1)
w0 = 0.0
seed(1)
Nl = 2000000
Nw = 8

w_min = 0.1
w_max = 3.0
dw = (w_max - w_min) / (Nw - 1)
dxw = np.log(w_max / w_min) / (Nw - 1)


I0_arr, t0_arr, w0_arr = mock_spectrum(Nl, t_min=t_min, t_max=t_max, w_min=w_min, w_max=w_max)
database = np.array([I0_arr, t0_arr, w0_arr])

# #%% CPU Legacy method:
# print('Adding... ', end='')
# tc0 = perf_counter()
# I_arr0 = np.zeros(Nt, dtype=np.float32)
# for I0, t0, w0 in database.T:
#     I_arr0 += I0*L(t_arr - t0, w0)
# tc1 = perf_counter()
# print('Done! {:.3f}'.format((tc1-tc0)*1e3))


#%% CPU DIT method:

def spectrum_dit(a):
    k_arr = (database[1] - np.float32(t_min)) / np.float32(dt)
    k0_arr = k_arr.astype(np.int32)
    k1_arr = k0_arr + 1
    a1k_arr = k_arr - k0_arr
    a0k_arr = 1 - a1k_arr

    l_arr = (np.log(database[2]) - np.log(np.float32(w_min))) / np.float32(dxw)
    l0_arr = l_arr.astype(np.int32)
    l1_arr = l0_arr + 1
    a1l_arr = l_arr - l0_arr
    a0l_arr = 1 - a1l_arr

    a00_arr = a0l_arr * a0k_arr
    a01_arr = a0l_arr * a1k_arr
    a10_arr = a1l_arr * a0k_arr
    a11_arr = a1l_arr * a1k_arr

    S_kl = np.zeros((Nw, Nt), dtype=np.float32)

    np.add.at(S_kl, (l0_arr, k0_arr), database[0] * a00_arr)
    np.add.at(S_kl, (l0_arr, k1_arr), database[0] * a01_arr)
    np.add.at(S_kl, (l1_arr, k0_arr), database[0] * a10_arr)
    np.add.at(S_kl, (l1_arr, k1_arr), database[0] * a11_arr)
    #print('Done!')

    #print('Applying lineshape... ', end='')
    I_arr1 = np.zeros(Nt, dtype=np.float32)
    spectrum_FT = np.zeros(Nf, dtype=np.complex64)
    S_kl_FT = np.fft.rfft(S_kl)
    for l in range(Nw):
        w_l = (1+a)*w_min*np.exp(l*dxw)

        #S_k_FT = np.fft.rfft(S_kl[l])
        S_k_FT = S_kl_FT[l]
        #ls = L(t_arr - t_min - Nt/2*dt, w_l)
        #ls_FT = np.fft.rfft(np.fft.fftshift(ls))
        ls_FT = L_FT(f_arr, w_l)/dt
        I_k_FT = S_k_FT * ls_FT
        spectrum_FT += I_k_FT
        #I_arr1 += np.fft.irfft(I_k_FT)
        #I_arr1 += np.convolve(S_kl[l], ls, 'same')

    return np.fft.irfft(spectrum_FT)

# lineshape distribution:
print('Distributing... ', end='')
tc0 = perf_counter()
I_arr1 = spectrum_dit(0.0)
tc1 = perf_counter()
print('Done! {:.3f}'.format((tc1-tc0)*1e3))


#%% GPU vulkan
print('GPU start...')
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=0, path=shader_path)
#app.print_memory_properties()

app.init_d = GPUBuffer(sizeof(init_t), usage='uniform', binding=0)
app.iter_d = GPUBuffer(sizeof(iter_t), usage='uniform', binding=1)
app.database_d = GPUBuffer(database.nbytes, binding=2)
app.S_kl_d = GPUBuffer(fftSize=Nt, binding=3)
app.spectrum_d = GPUBuffer(fftSize=Nt, binding=4)
app.indirect_d = GPUBuffer(sizeof(workGroupSizeArray_t), usage='indirect')



# initalize data:
init_h = app.init_d.getHostStructPtr(init_t)
init_h.Nl = Nl
init_h.Nt = Nt
init_h.Nf = Nf
init_h.t_min = t_min
init_h.dt    = dt
init_h.w_min = w_min
app.init_d.transferStagingBuffer('H2D')

iter_h = app.iter_d.getHostStructPtr(iter_t)
iter_h.a = 0.0
iter_h.Nw = Nw
iter_h.dxw = dxw

app.database_d.copyToBuffer(database)

indirect_h = app.setIndirectBuffer(app.indirect_d, workGroupSizeArray_t) #TODO: set fwd/inv


app.appendCommands([
    app.cmdAddTimestamp('Initialization'),
    app.indirect_d.cmdTransferStagingBuffer('H2D'),
    app.iter_d.cmdTransferStagingBuffer('H2D'),
    app.cmdAddTimestamp('Zeroing buffers'),
    app.S_kl_d.cmdClearBuffer(),
    app.spectrum_d.cmdClearBuffer(),
    app.cmdAddTimestamp('Line addition'),
    app.cmdScheduleShader('cmdTestFillLDM.spv', (Nl // Ntpb + 1, 1, 1), threads),
    app.cmdAddTimestamp('FFT fwd'),
    app.cmdFFT(app.S_kl_d, name='FFTa'),
    app.cmdAddTimestamp('Apply conv.'),
    app.cmdScheduleShader('cmdTestApplyLineshapes.spv', (Nf // Ntpb + 1, 1, 1), threads),
    #app.cmdScheduleShader('cmdTestApplyLineshapesP.spv', (Nf // Ntpb + 1, Nw, 1), threads),
    app.cmdAddTimestamp('FFT inv'),
    app.cmdIFFT(app.spectrum_d, name='FFTb'), 
    app.spectrum_d.cmdTransferStagingBuffer('D2H'),
    app.cmdAddTimestamp('End'),
])

app.writeCommandBuffer()

app.updateBatchSizeFunctionList.append(app.S_kl_d.setBatchSize)
app.updateBatchSizeFunctionList.append(app.setFwdFFTWorkGroupSize)

I_arr2 = np.zeros(Nt, dtype=np.float32)



## First iteration:
app.setBatchSize(Nw)
app.run() #TODO: check if command buffer should be rewritten
app.spectrum_d.toArray(I_arr2)



#%%
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

p1, = ax.plot(t_arr, I_arr1)
p2, = ax.plot(t_arr, I_arr2, 'k--')

axNw = plt.axes([0.25, 0.05, 0.65, 0.03])
sNw = Slider(axNw, "Nw", 2, 20, valinit=Nw, valstep=1)

axw = plt.axes([0.25, 0.1, 0.65, 0.03])
sw = Slider(axw, "a", -1.0, 2.0, valinit=0.0)

Nw_i = Nw
def update(val):
    global Nw_i

    a = sw.val

    t0 = perf_counter()
    # I_arr1 = spectrum_dit(a)
    t1 = perf_counter()

    if sNw.val != Nw_i:
        Nw_i = sNw.val
        dxw_i = np.log(w_max / w_min) / (Nw_i - 1)
        iter_h.Nw = Nw_i
        iter_h.dxw = dxw_i
        app.setBatchSize(Nw_i)

    iter_h.a = a
    app.run()
    app.spectrum_d.toArray(I_arr2)
    t2 = perf_counter()
    
    timestamps = app.get_timestamps()
    for key in timestamps:
        print('{:15s}: {:10.3f}'.format(key, timestamps[key]))
    print('')
    
    ax.set_title('CPU: {:.1f} ms - GPU: {:.1f} ms'.format((t1-t0)*1e3, (t2-t1)*1e3))
    p1.set_ydata(I_arr1)
    p2.set_ydata(I_arr2)
    
    fig.canvas.draw_idle()

sw.on_changed(update)
sNw.on_changed(update)

times = []
for i in range(20):
    
    t0 = perf_counter()
    iter_h.a = 0.01*i
    app.run()
    app.spectrum_d.toArray(I_arr2)
    t1 = perf_counter()
    print(i,(t1 - t0)*1e3)


plt.show()
