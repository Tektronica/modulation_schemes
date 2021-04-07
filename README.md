# Modulation Schemes

This application demonstrates various modulation schemes.

![](images/modulation_schemes_gui.jpg)


## The Signals
![c(t) = A_c cos (2 \pi f_c t)](https://latex.codecogs.com/svg.latex?c(t)%20=%20A_c%20cos%20(2%20\pi%20f_c%20t))

![m(t) = A_m cos (2 \pi f_m t + \varphi_m)](https://latex.codecogs.com/svg.latex?m(t)%20=%20A_m%20cos%20(2%20\pi%20f_m%20t%20+%20\varphi_m))

## Amplitude Modulation
![s(t) = c(t) \times m(t)](https://latex.codecogs.com/svg.latex?s(t)%20=%20c(t)%20\times%20m(t))

## Phase Modulation
![s(t) = cos (2 \pi f_c + k_p m(t))](https://latex.codecogs.com/svg.latex?s(t)%20=%20cos%20(2%20\pi%20f_c%20+%20k_p%20m(t)))

## Frequency Modulation

For a given frequency modulated signal, the frequency varies linearly with the modulating message:

![s(t) = A_c cos ( \varphi_i(t)) = A_c cos (2 \pi f(t)t+\theta_c)](https://latex.codecogs.com/svg.latex?s(t)%20=%20A_c%20cos%20(%20\varphi_i(t))%20=%20A_c%20cos%20(2%20\pi%20f(t)t+\theta_c))

The instantaneous phase, is expressed as:

![\boxed{\varphi_i(t) = \omega(t)t} = 2 \pi f(t)t + \theta_c](https://latex.codecogs.com/svg.latex?\boxed{\varphi_i(t)%20=%20\omega(t)t}%20=%202%20\pi%20f(t)t%20+%20\theta_c)

Which is derived from the generalized definition for instantaneous frequency, which is the derivative of phase with respect to time:

![f_i(t) \equiv \frac{1}{2 \pi} \frac{d \varphi_i(t)}{dt}](https://latex.codecogs.com/svg.latex?f_i(t)%20\equiv%20\frac{1}{2%20\pi}%20\frac{d%20\varphi_i(t)}{dt})

![\varphi_i(t) \equiv 2 \pi \int_0^t{f_i(\tau) d\tau}](https://latex.codecogs.com/svg.latex?\varphi_i(t)%20\equiv%202%20\pi%20\int_0^t{f_i(\tau)%20d\tau})

Thus, when evaluating causal signals with linearly varying frequency, such as:

![f_i(t) = f_c + k_f m(t)](https://latex.codecogs.com/svg.latex?f_i(t)%20=%20f_c%20+%20k_f%20m(t))

The resultant modulated frequency based on the provided message signal with units of magnitude versus time must be converted to instantaneous phase versus time, which becomes:

![\varphi_i\left(t\right)=2\pi\int_{0}^{t}{f_i\left(\tau\right)}d\tau=2\pi\left(f_ct+k_f\int_{0}^{t}m\left(\tau\right)d\tau\right)](https://latex.codecogs.com/svg.latex?\phi_i\left(t\right)=2\pi\int_{0}^{t}{f_i\left(\tau\right)}d\tau=2\pi\left(f_ct+k_f\int_{0}^{t}m\left(\tau\right)d\tau\right))

Which provides the general expression for a frequency modulated signal:

![s\left(t\right)=A_c\cos{\left(\theta_i\right)}=A_c\cos{\left(2\pi\left(f_ct+k_f\int_{0}^{t}m\left(\tau\right)d\tau\right)\right)}](https://latex.codecogs.com/svg.latex?s\left(t\right)=A_c\cos{\left(\theta_i\right)}=A_c\cos{\left(2\pi\left(f_ct+k_f\int_{0}^{t}m\left(\tau\right)d\tau\right)\right)})

## Integration
### Sine
![m\left(t\right)=A_m\cos{\left(2\pi f_mt+\varphi_m\right)}](https://latex.codecogs.com/svg.latex?m\left(t\right)=A_m\cos{\left(2\pi%20f_mt+\varphi_m\right)})

![M\left(t\right)=\frac{A_m}{\pi f_m}\sin{\left(\pi f_mt\right)}\cos{\left(\pi f_mt+\varphi_m\right)}](https://latex.codecogs.com/svg.latex?M\left(t\right)=\frac{A_m}{\pi%20f_m}\sin{\left(\pi%20f_mt\right)}\cos{\left(\pi%20f_mt+\varphi_m\right)})

![m\left(t\right)=A_m\sin{\left(2\pi f_mt+\varphi_m\right)}](https://latex.codecogs.com/svg.latex?m\left(t\right)=A_m\sin{\left(2\pi%20f_mt+\varphi_m\right)})

![M\left(t\right)=\frac{A_m}{\pi f_m}\sin{\left(\pi f_mt\right)}\sin{\left(\pi f_mt+\varphi_m\right)}](https://latex.codecogs.com/svg.latex?M\left(t\right)=\frac{A_m}{\pi%20f_m}\sin{\left(\pi%20f_mt\right)}\sin{\left(\pi%20f_mt+\varphi_m\right)})

**Note:** scaling to the appropriate time-base is required. If evaluating across N, divide M[n] by the sampling frequency.

### Triangle
![T = \text{period length}](https://latex.codecogs.com/svg.latex?T%20=%20\text{period%20length})

![m\left(t\right)=\frac{2}{\pi}\sin^{-1}{\left[\sin{\left(\pi t\right)}\right]}](https://latex.codecogs.com/svg.latex?m\left(t\right)=\frac{2}{\pi}\sin^{-1}{\left[\sin{\left(\pi%20t\right)}\right]})

![m\left(t\right)=\frac{4}{T}\left|\left(\left(x-\frac{T}{4}\right)\ mod\ T\right)-\frac{T}{2}\right|-1](https://latex.codecogs.com/svg.latex?m\left(t\right)=\frac{4}{T}\left|\left(\left(x-\frac{T}{4}\right)\%20mod\%20T\right)-\frac{T}{2}\right|-1)

![m(t) = \left\{\begin{matrix}\frac{4}{T}x - 1 & x<T/2\\\ -\frac{4}{T}x^2 + 3 & x\geq T/2\end{matrix}\right.](https://latex.codecogs.com/svg.image?%5Cbg_white%20m(t)%20=%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5Cfrac%7B4%7D%7BT%7Dx%20-%201%20&%20x%3CT/2%5C%5C%5C%20-%5Cfrac%7B4%7D%7BT%7Dx%5E2%20&plus;%203%20&%20x%5Cgeq%20T/2%5Cend%7Bmatrix%7D%5Cright.)

![M(t) = \int_{0}^{t}{m\left(\tau\right) d\tau} = \left\{\begin{matrix}\frac{2}{T}x^2 - x & x<T/2\\\ -\frac{2}{T}x^2 + 3x & x\geq T/2\end{matrix}\right.](https://latex.codecogs.com/svg.image?%5Cbg_white%20M(t)%20=%20%5Cint_%7B0%7D%5E%7Bt%7D%7Bm%5Cleft(%5Ctau%5Cright)%20d%5Ctau%7D%20=%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5Cfrac%7B2%7D%7BT%7Dx%5E2%20-%20x%20&%20x%3CT/2%5C%5C%5C%20-%5Cfrac%7B2%7D%7BT%7Dx%5E2%20&plus;%203x%20&%20x%5Cgeq%20T/2%5Cend%7Bmatrix%7D%5Cright.)

**Note:** scaling to the appropriate time-base is required. If evaluating across N, divide M[n] by the sampling frequency.

### Sawtooth
![t = x \mod T](https://latex.codecogs.com/svg.latex?t%20=%20x%20\mod%20T)

![m\left(t\right)=\frac{t}{T} =\frac{x \mod T}{T}](https://latex.codecogs.com/svg.latex?m\left(t\right)=\frac{t}{T}%20=\frac{x%20\mod%20T}{T})

![M\left(t\right)=\int_{0}^{t}m\left(\tau\right)d\tau=\int_{0}^{t}{\frac{\tau}{T}d\tau}=\boxed{\frac{1}{T}\left(x \mod T\right)^2-\left(x \mod T\right)}](https://latex.codecogs.com/svg.latex?M\left(t\right)=\int_{0}^{t}m\left(\tau\right)d\tau=\int_{0}^{t}{\frac{\tau}{T}d\tau}=\boxed{\frac{1}{T}\left(x%20\mod%20T\right)^2-\left(x%20\mod%20T\right)})

**Note:** scaling to the appropriate time-base is required. If evaluating across N, divide M[n] by the sampling frequency.

### Square

![m(t)=\begin{cases}1 & \text{ if } x<T/2 \\ -1 & \text{ if } x \geq T/2 \end{cases}](https://latex.codecogs.com/svg.image?%5Cbg_white%20m(t)=%5Cbegin%7Bcases%7D1&%20%5Ctext%7B%20if%20%7D%20x%3CT/2%5C%5C-1&%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%20T/2%20%5Cend%7Bcases%7D)

![M(t)=\int_{0}^{t}{m(\tau)d\tau}=\begin{cases} t & \text{ if } t<T/2 \\ -t & \text{ if } x \geq T/2 \end{cases}](https://latex.codecogs.com/svg.image?%5Cbg_white%20M(t)=%5Cint_%7B0%7D%5E%7Bt%7D%7Bm(%5Ctau)d%5Ctau%7D=%5Cbegin%7Bcases%7D%20t%20&%20%5Ctext%7B%20if%20%7D%20t%3CT/2%20%5C%5C%20-t%20&%20%5Ctext%7B%20if%20%7D%20x%20%5Cgeq%20T/2%20%5Cend%7Bcases%7D)

**Note:** scaling to the appropriate time-base is required. If evaluating across N, divide M[n] by the sampling frequency.

### Shift Keying

    import numpy as np

    fs = 100000	# sampling frequency
    fm = 100  	# modulating frequency
    T = fs/fm	# samples per period

    # repeat
    mt = np.repeat(binary_message, T)[:N]

    # or ravel
    mt_ = np.ravel(np.matlib.repmat(binary_message, T, 1), order='F')[:N]

    # or kronecher
    array_ones = np.ones(T)
    mt__ = np.kron(binary_message, array_ones)[:N]

    # integral
    Mt = x * (2 * m(t) - 1)

**Note:** scaling to the appropriate time-base is required. If evaluating across N, divide M[n] by the sampling frequency.

**Example:**
The number of samples contained in the buffer of your measurement window. In other words, the number of samples collected is defined by:

    N = 7
The number of samples contained in one period of the modulating frequency

    T = fs/fm = 2

The message being transmitted

    binary_message = [1, 0, 1, 1]

Message is repeated T number of times

    mt = [1, 1, 0, 0, 1, 1, 1, 1]

Depending on the number of samples available in the measurement buffer, m(t) may become truncated:

    mt = m(t)[:N] = 1, 1, 0, 0, 1, 1, 1]

If modulation scheme is Phase shift-keying (PSK):

    pi = np.pi
    mt = pi * m(t) = [pi, pi, 0, 0, pi, pi, pi]