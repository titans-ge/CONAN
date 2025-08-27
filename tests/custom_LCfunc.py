def fourier_model(roll, S1, C1, S2, C2, S3, C3, extra_args=dict(unit='ppm')):
    """ 
    A custom detrending function that models the baseline flux variations as a Fourier series up to the second harmonic
    with S1, S2, S3 being the amplitudes of the sin terms and C1, C2, C3 those for the cosine terms.

    extra_args can be used to pass additional parameters (non-varying), such as the desired output unit.
    """
    import numpy as np
    ppm = 1e-6
    roll_radians = np.deg2rad(roll)
    roll_model = S1 * np.sin(roll_radians)   + C1 * np.cos(roll_radians) + \
                 S2 * np.sin(2*roll_radians) + C2 * np.cos(2*roll_radians) + \
                 S3 * np.sin(3*roll_radians) + C3 * np.cos(3*roll_radians)

    if extra_args.get('unit') == 'ppm':
        return roll_model * ppm
    else:
        return roll_model
def op_func(transit_model, custom_model):   # operation function to combine the custom model with the transit model
    return transit_model + custom_model
