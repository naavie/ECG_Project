# function for 3-channel spectrogramms and some code for visualisation
# default values seems good for 10-seconds ECG with 500 sample frequency
# Also there is log scaling

def get_spectrogramm(ecg, sr, num_shannels=3, nperseg=128, noverlap=64):
    result = list()
    for i in range(num_shannels):
        f, t, Zxx = scipy.signal.stft(ecg[i], fs=sr, nperseg=nperseg, noverlap=noverlap)
        Zxx = np.abs(Zxx) #complex to real
        Zxx = np.log(Zxx) #log scaling
        result.append(Zxx)
    result = np.stack(result)
    return result


spec = get_spectrogramm(ecg, sr, num_shannels=3, nperseg=128, noverlap=64)
print(spec.shape)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(spec[0])
plt.subplot(132)
plt.imshow(spec[1])
plt.subplot(133)
plt.imshow(spec[2])
plt.show()

