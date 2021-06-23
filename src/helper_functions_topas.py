import sys 
import numpy as np 

def V1(a,b,c):
    return a*b*c

def V2(a,b,c):
    return ((np.sqrt(3))/2)*(a**2)*c

def LPrange(prediction, crystal_system, bound=0.2):
    
    """
    Function to get the bounds on the lattice parameters and volume for use in Lp-Search automated scripts. 
    
    Parameters
    ----------
    prediction : ndarray
        PXRD lattice parameter prediction from ML 
    crystal_system : str
        String which indicates which crystal system
    bound : float 
        Fraction bound 
        
    Returns
    ----------
    predicted_a : float 
    lower_bound_a : float
    upper_bound_a : float
    predicted_b : float
    lower_bound_b : float
    upper_bound_b : float
    predicted_c : float
    lower_bound_c : float
    upper_bound_c : float
    predicted_V : float
    lower_bound_V : float
    upper_bound_V : float
    
    """
        
    predicted_a = prediction[0]
    predicted_b = prediction[1]
    predicted_c = prediction[2]
    
    lower_bound_a = predicted_a - bound*(predicted_a)
    upper_bound_a = predicted_a + bound*(predicted_a)
    lower_bound_b = predicted_b - bound*(predicted_b)
    upper_bound_b = predicted_b + bound*(predicted_b)
    lower_bound_c = predicted_c - bound*(predicted_c)
    upper_bound_c = predicted_c + bound*(predicted_c)
    
    if (crystal_system == 'cubic') or (crystal_system == 'tetragonal') or (crystal_system == 'orthorhombic') or (crystal_system == 'monoclinic') or (crystal_system == 'triclinic'):
        predicted_V = V1(predicted_a, predicted_b, predicted_c)
        lower_bound_V = V1(lower_bound_a, lower_bound_b, lower_bound_c)
        upper_bound_V = V1(upper_bound_a, upper_bound_b, upper_bound_c)
        
    if (crystal_system == 'hexagonal') or (crystal_system == 'trigonal'):
        
        if predicted_a != predicted_b:
            temp = predicted_a
            predicted_a = predicted_c
            predicted_c = temp 
          
        predicted_V = V2(predicted_a, predicted_b, predicted_c)
        lower_bound_V = V2(lower_bound_a, lower_bound_b, lower_bound_c)
        upper_bound_V = V2(upper_bound_a, upper_bound_b, upper_bound_c)
    
    return predicted_a, lower_bound_a, upper_bound_a, predicted_b, lower_bound_b, upper_bound_b, predicted_c, lower_bound_c, upper_bound_c, predicted_V, lower_bound_V, upper_bound_V

def write_template_cubic(dataName, wavelength, V, minV, maxV, a, mina, maxa):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b = Get(a);
        c = Get(a);

        space_group 195
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa)
    
    sys.stdout.write(lines)
    return lines 

def write_template_hexagonal(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b = Get(a);
        c @ {} min {} max {}
        
        al 90 
        be 90  
        ga 120  
        
        space_group 168
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_trigonal(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc):
        
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b = Get(a);
        c @ {} min {} max {}
        
        al 90 
        be 90  
        ga 120  
        
        space_group 143
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_tetragonal(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b = Get(a);
        c @ {} min {} max {}
        
        al 90 
        be 90  
        ga 90  
        
        space_group 75
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_orthorhombic(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b @ {} min {} max {}
        c @ {} min {} max {}
        
        al 90
        be 90 
        ga 90 
        
        space_group 16
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_monoclinic1(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b @ {} min {} max {}
        c @ {} min {} max {}
        
        al @ 90 min 60 max 120
        be 90
        ga 90
        
        space_group 3
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_monoclinic2(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b @ {} min {} max {}
        c @ {} min {} max {}
        
        al 90
        be @ 90 min 60 max 120
        ga 90
        
        space_group 3
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def write_template_monoclinic3(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b @ {} min {} max {}
        c @ {} min {} max {}
        
        al 90
        be 90
        ga @ 90 min 60 max 120
        
        space_group 3
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 
def write_template_triclinic(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc):
    
    lines = """
    continue_after_convergence
    verbose 0

    XY({},0.01)
        bkg @ 0 0 0 0 0 0 0
        start_X 15
        finish_X 30
        Zero_Error(@, 0.00074)
        lam
            ymin_on_ymax  0.0001
            la  1 lo {} lh 0.002

        LP_Factor(90)
        x_calculation_step 0.01
        gauss_fwhm  @  0.0337972443 min 0.01 max 0.5
        lor_fwhm @  0.084548646` min 0.01 max 0.5

    hkl_Is

        lp_search 1
        scale 0.01
        prm end = If(Get(r_wp) < 0.1, 0, 5000);
        iters = end;

        volume {} min {} max {}
        a @ {} min {} max {}
        b @ {} min {} max {}
        c @ {} min {} max {}
        
        al @ 90 min 60 max 120  
        be @ 90 min 60 max 120  
        ga @ 90 min 60 max 120  
        
        space_group 1
    """.format(dataName, wavelength, V, minV, maxV, a, mina, maxa, b, minb, maxb, c, minc, maxc)
        
    sys.stdout.write(lines)
    return lines 

def make_template(prediction, name,wavelength,crystal_system,bound=0.2):
    
    # Helper function to generate topas script for a given PXRD pattern 
    
    predicted_a, lower_bound_a, upper_bound_a, predicted_b, lower_bound_b, upper_bound_b, predicted_c, lower_bound_c, upper_bound_c, predicted_V, lower_bound_V, upper_bound_V = LPrange(prediction=prediction, crystal_system=crystal_system, bound=bound)
    
    if crystal_system == 'cubic':
        write_template_cubic(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a)
        
    elif crystal_system == 'hexagonal':
        write_template_hexagonal(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    elif crystal_system == 'trigonal':
        write_template_trigonal(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    elif crystal_system == 'tetragonal':
        write_template_tetragonal(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    elif crystal_system == 'orthorhombic':
        write_template_orthorhombic(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, b=predicted_b, minb=lower_bound_b, maxb=upper_bound_b, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    elif crystal_system == 'monoclinic':
        print(" ")
        print("First angle neq 90")
        print(" ")
        write_template_monoclinic1(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, b=predicted_b, minb=lower_bound_b, maxb=upper_bound_b, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        print(" ")
        print("Second angle neq 90")
        print(" ")
        write_template_monoclinic2(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, b=predicted_b, minb=lower_bound_b, maxb=upper_bound_b, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        print(" ")
        print("Third angle neq 90")
        print(" ")
        write_template_monoclinic3(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, b=predicted_b, minb=lower_bound_b, maxb=upper_bound_b, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    elif crystal_system == 'triclinic':
        write_template_triclinic(dataName=name, wavelength=wavelength, V=predicted_V, minV=lower_bound_V, maxV=upper_bound_V, a=predicted_a, mina=lower_bound_a, maxa=upper_bound_a, b=predicted_b, minb=lower_bound_b, maxb=upper_bound_b, c=predicted_c, minc=lower_bound_c, maxc=upper_bound_c)
        
    else:
        print("Invalid crystal system")
