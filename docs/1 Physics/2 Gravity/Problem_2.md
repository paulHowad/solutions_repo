Here are the key equations related to a simple pendulum, assuming small-angle approximations (where the swing isn’t too wild, so the motion stays roughly harmonic). These come from basic physics principles, and I’ll keep them straightforward with explanations.

1. **Period of a Pendulum (T)**  
   This is the time it takes for one complete swing (back and forth).  
   \[
   T = 2\pi \sqrt{\frac{L}{g}}
   \]
   - \(T\): Period (in seconds)  
   - \(L\): Length of the pendulum (in meters, from pivot to the center of the bob)  
   - \(g\): Acceleration due to gravity (approximately \(9.8 \, \text{m/s}^2\ on Earth)  
   - The \(2\pi\) comes from the circular nature of the motion.  
   *Example*: A 1-meter-long pendulum on Earth has a period of about 2 seconds.

2. **Frequency (f)**  
   Frequency is how many swings happen per second, the inverse of the period.  
   \[
   f = \frac{1}{T} = \frac{1}{2\pi} \sqrt{\frac{g}{L}}
   \]
   - \(f\): Frequency (in hertz, or cycles per second)  
   - Same variables as above.  
   *Example*: That 1-meter pendulum has a frequency of about 0.5 Hz.

3. **Angular Displacement (θ)**  
   For a pendulum released from a small angle, its position over time can be described with:  
   \[
   \theta(t) = \theta_0 \cos(\omega t)
   \]
   - \(\theta(t)\): Angle from vertical at time \(t\) (in radians)  
   - \(\theta_0\): Initial angle (how far you pull it back)  
   - \(\omega\): Angular frequency, where \(\omega = \sqrt{\frac{g}{L}}\) (in radians per second)  
   - \(t\): Time (in seconds)  
   - This assumes no friction; it’s a cosine wave because the motion is oscillatory.

4. **Velocity (v)**  
   The speed of the bob at any point depends on its position. Maximum velocity occurs at the bottom:  
   \[
   v_{\text{max}} = \sqrt{2gL (1 - \cos\theta_0)}
   \]
   - \(v_{\text{max}}\): Maximum velocity (in meters per second)  
   - Derived from energy conservation (potential energy at the top turns into kinetic energy at the bottom).  
   *Example*: Pull a 1-meter pendulum to a 10° angle (small enough for approximations), and \(v_{\text{max}} \approx 0.66 \, \text{m/s}\).

5. **Acceleration (a)**  
   The tangential acceleration depends on the angle:  
   \[
   a = -g \sin\theta
   \]
   - \(a\): Acceleration along the arc (in meters per second squared)  
   - Negative because it’s a restoring force, pulling the bob back to the center.  
   - For small angles, \(\sin\theta \approx \theta\), simplifying the motion to harmonic.

These equations assume an idealized pendulum—no air resistance, a massless string, and small swings. If you want equations for a real-world pendulum with damping or large angles, let me know, and I can dig into those too! Anything specific you’d like to calculate or explore with these?