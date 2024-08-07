

import numpy as np
from scipy.ndimage import gaussian_filter 
from skimage.exposure import rescale_intensity
from matplotlib.colors import hsv_to_rgb


def From_tensor_to_elipse(Df_xx, Df_yy, Df_xy):

    '''
    
    This function takes the second derivative of a function and returns the parameters of the elipse
    that is the level curve of the function at a given point.
    
    Parameters:
    -----------
    Df_xx: float
        The second derivative of the function in the x direction.
    Df_yy: float
        
        The second derivative of the function in the y direction.
    Df_xy: float
        The second derivative of the function in the x-y direction.
        
    Returns:
    --------
    a: float
        The semi-major axis of the elipse.
    b: float
        The semi-minor axis of the elipse.
        
    Equation of the elipse:
    -----------------------
   
    L'équation générale d'une ellipse est donnée par :

    $$
    a_{11}x^2 + 2a_{12}xy + a_{22}y^2 + a_{33} = 0
    $$

    Cette équation représente une ellipse dans le plan $(x, y)$, où :
    - $a_{11}$, $a_{12}$, $a_{22}$ sont les coefficients qui déterminent la forme de l'ellipse.
    - $a_{33}$ est un coefficient qui peut être relié à une constante ou à un facteur de normalisation.

    Ces coefficients peuvent être reliés aux composantes des tenseurs de champ sombre.

    ## 2. Association avec les tenseurs de champ sombre

    Les composantes des tenseurs de champ sombre $D_{xx}$, $D_{yy}$, et $D_{xy}$ représentent les variations ou les propriétés dans différentes directions. On les associe aux coefficients de l'ellipse par les relations suivantes :

    $$
    \begin{cases}
    a_{11} = D_{xx}(x, y) D_{xy}(x, y) \\
    a_{22} = D_{yy}(x, y) D_{yx}(x, y) \\
    a_{12} = \frac{1}{2} D_{xx}(x, y) D_{yy}(x, y)
    \end{cases}
    $$

    Ici, $f(x, y)$ est une fonction de normalisation ou de pondération. Ces équations permettent de traduire les informations directionnelles contenues dans les tenseurs en coefficients pour l'équation de l'ellipse.

    ## 3. Transformation des coordonnées

    Pour étudier l'ellipse sous une rotation des axes de coordonnées, nous effectuons une rotation d'un angle $\theta$. Les nouvelles coordonnées $(x', y')$ sont reliées aux anciennes $(x, y)$ par :

    $$
    \begin{cases}
    x = x'\cos(\theta) - y'\sin(\theta) \\
    y = x'\sin(\theta) + y'\cos(\theta)
    \end{cases}
    $$

    ## 4. Développement des termes

    Nous substituons les nouvelles coordonnées dans l'équation originale de l'ellipse :

    $$
    a_{11}x^2 + 2a_{12}xy + a_{22}y^2 + a_{33} = 0
    $$

    Après substitution, nous obtenons :

    $$
    a_{11}(x'\cos(\theta) - y'\sin(\theta))^2 + 2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta)) + a_{22}(x'\sin(\theta) + y'\cos(\theta))^2 + a_{33} = 0
    $$

    Développons chaque terme individuellement.

    #### Terme $a_{11}(x'\cos(\theta) - y'\sin(\theta))^2$ 

    $$
    a_{11}(x'\cos(\theta) - y'\sin(\theta))^2 = a_{11}(x'^2\cos^2(\theta) - 2x'y'\cos(\theta)\sin(\theta) + y'^2\sin^2(\theta))
    $$

    #### Terme $a_{22}(x'\sin(\theta) + y'\cos(\theta))^2$

    $$
    a_{22}(x'\sin(\theta) + y'\cos(\theta))^2 = a_{22}(x'^2\sin^2(\theta) + 2x'y'\sin(\theta)\cos(\theta) + y'^2\cos^2(\theta))
    $$

    #### Terme $2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta))$

    $$
    2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta)) = 2a_{12}(x'^2\cos(\theta)\sin(\theta) + x'y'\cos^2(\theta) - x'y'\sin^2(\theta) - y'^2\sin(\theta)\cos(\theta))
    $$

    ## 5. Regroupement des termes

    Après avoir développé tous les termes, nous regroupons les coefficients devant $x'^2$, $y'^2$ et $x'y'$.

    #### Coefficient $x'^2$

    $$
    \frac{1}{a^2} = a_{11}\cos^2(\theta) + a_{22}\sin^2(\theta) + 2a_{12}\cos(\theta)\sin(\theta)
    $$

    - Terme de $a_{11}(x'\cos(\theta) - y'\sin(\theta))^2$ donne : $a_{11}\cos^2(\theta)$
    - Terme de $a_{22}(x'\sin(\theta) + y'\cos(\theta))^2$ donne : $a_{22}\sin^2(\theta)$
    - Terme de $2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta))$ donne : $2a_{12}\cos(\theta)\sin(\theta)$

    #### Coefficient $y'^2$

    $$
    \frac{1}{b^2} = a_{11}\sin^2(\theta) + a_{22}\cos^2(\theta) - 2a_{12}\cos(\theta)\sin(\theta)
    $$

    - Terme de $a_{11}(x'\cos(\theta) - y'\sin(\theta))^2$ donne : $a_{11}\sin^2(\theta)$
    - Terme de $a_{22}(x'\sin(\theta) + y'\cos(\theta))^2$ donne : $a_{22}\cos^2(\theta)$
    - Terme de $2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta))$ donne : $-2a_{12}\cos(\theta)\sin(\theta)$

    #### Coefficient $x'y'$

    $$
    \frac{1}{c^2} = (a_{11} - a_{22})\cos(\theta)\sin(\theta) + a_{12}(\cos^2(\theta) - \sin^2(\theta))
    $$

    - Terme de $a_{11}(x'\cos(\theta) - y'\sin(\theta))^2$ donne : $-a_{11}\cos(\theta)\sin(\theta)$
    - Terme de $a_{22}(x'\sin(\theta) + y'\cos(\theta))^2$ donne : $a_{22}\cos(\theta)\sin(\theta)$
    - Terme de $2a_{12}(x'\cos(\theta) - y'\sin(\theta))(x'\sin(\theta) + y'\cos(\theta))$ donne : $a_{12}(\cos^2(\theta) - \sin^2(\theta))$

    ## 6. Nouvelle forme de l'équation de l'ellipse

    Après la transformation, l'équation de l'ellipse s'écrit :

    $$
    \frac{x'^2}{a^2} + \frac{y'^2}{b^2} + \frac{x'y'}{c^2} = \text{constante}
    $$

    où :
    - $ \frac{1}{a^2} $, $ \frac{1}{b^2} $, et $ \frac{1}{c^2} $ sont les nouveaux coefficients associés aux termes quadratiques et croisés.

    ## 7. Signification des coefficients

    - **$ \frac{1}{a^2} $** et **$ \frac{1}{b^2} $** déterminent les demi-axes majeurs et mineurs de l'ellipse après rotation.
    - **$ \frac{1}{c^2} $** correspond au coefficient du terme croisé $xy$ après la transformation. En général, il est souhaitable qu'il soit nul pour aligner l'ellipse avec les axes principaux (ceci se produit quand $\theta$ est l'angle de rotation correct).

    ## 8. Calcul de l'angle $\theta$

    Pour trouver l'angle $\theta$ qui rend le coefficient du terme croisé $xy$ nul (c'est-à-dire $\frac{1}{c^2} = 0$), on utilise :

    $$
    \theta = \frac{1}{2}\arctan\left(\frac{2a_{12}}{a_{11} - a_{22}}\right)
    $$

    Cela aligne l'ellipse avec les axes principaux des coordonnées $(x', y')$.
    '''
    
    
    
    a11 = Df_xx * Df_xy
    a22 = Df_yy * Df_xy
    a12 = 0.5 * Df_xx * Df_yy

    # Calculate theta
    theta = 0.5 * np.arctan2(2*a12, a11 - a22)
    
    #Compute the semi-major and semi-minor axis
    a = 1/np.sqrt(a11*np.cos(theta)**2 + a22*np.sin(theta)**2 + 2*a12*np.cos(theta)*np.sin(theta))
    b = 1/np.sqrt(a11*np.sin(theta)**2 + a22*np.cos(theta)**2 - 2*a12*np.cos(theta)*np.sin(theta))

    theta = np.where(theta < 0, theta + np.pi, theta)
    theta = np.where(theta >= 2*np.pi, theta - 2*np.pi, theta)
    theta = np.where(a < b, theta + np.pi/2, theta)
    
    mask = wrong_elipse_mask(a11,a22,a12)
    
    excentricity = np.sqrt(1 - b**2/a**2)
    
    excentricity_corrected = np.where(mask, 0, excentricity)
    
    area = np.pi * a * b
    
    return excentricity_corrected, area, theta
    
    
    
    
    


def wrong_elipse_mask(a11,a22,a12):
    return np.logical_or(a11*a22 - a12**2 <= 0, a11*a22 <= 0)



def clip_values(Df_xx,Df_yy,Df_xy,threshold,epsilon):
    sign_values_df_xy = np.sign(Df_xy)
    Df_xx = np.where(np.logical_or(Df_xx  == 0, np.abs(Df_xx) > threshold), epsilon, Df_xx)
    Df_yy = np.where(np.logical_or(Df_yy  == 0, np.abs(Df_yy) > threshold), epsilon, Df_yy)
    Df_xy = np.where(np.logical_or(Df_xy  == 0, np.abs(Df_xy) > threshold), epsilon*sign_values_df_xy, Df_xy)
    return Df_xx,Df_yy,Df_xy,sign_values_df_xy


def DDF_metrics(Df_xx,Df_yy,Df_xy,Df_theta, sigma=5):
    padding_value = 6*np.round(sigma)
    Df_theta_padded = np.pad(Df_theta, padding_value, mode='reflect')
    wcos = gaussian_filter(np.cos(2 * np.where(Df_theta_padded == 0, np.nan, Df_theta_padded)),
                       sigma=sigma_regularization/np.sqrt(2), mode='reflect', cval=0)
    wsin = gaussian_filter(np.sin(2 * np.where(Df_theta_padded == 0, np.nan, Df_theta_padded)),
                       sigma=sigma_regularization/np.sqrt(2), mode='reflect', cval=0)
    
    wcos = wcos[padding:-padding, padding:-padding]
    wsin = wsin[padding:-padding, padding:-padding]
    saturation = np.sqrt(wcos**2 + wsin**2)
    theta_corrected = 0.5*np.arctan2(wsin, wcos)
    
    theta_corrected = np.where(theta_corrected < 0, theta_corrected + np.pi, theta_corrected)
    theta_corrected = np.where(theta_corrected >= 2*np.pi, theta_corrected - 2*np.pi, theta_corrected)
    theta_corrected = np.where(np.logical_and(wcos == 0, wsin == 0), 0, theta_corrected)
    
    return saturation, theta_corrected

def normalize_values(image,nb_of_std=3):
    mean = np.mean(image)
    std = np.std(image)
    image = (image) / (mean + nb_of_std * std)
    image = np.clip(image, 0, 1)
    
    return image

def colored_image_generation(hue,saturation,value):
    hue_rescaled = rescale_intensity(hue, in_range=(np.nanmin(hue), np.nanmax(hue)), out_range=(0, 1.))
    saturation_rescaled = rescale_intensity(saturation, in_range=(np.nanmin(saturation), np.nanmax(saturation)), out_range=(0, 1.))
    value_rescaled = rescale_intensity(value, in_range=(np.nanmin(value), np.nanmax(value)), out_range=(0, 1.))
    
    hsv_image = np.dstack((hue_rescaled, saturation_rescaled, value_rescaled), axis=-1)
    return hsv_to_rgb(hsv_image)
    

def DDF_colored_images(Df_xx,Df_yy,Df_xy,threshold=1,epsilon=1e-6):
    excentricity, area, theta = From_tensor_to_elipse(Df_xx, Df_yy, Df_xy)
    area = normalize_values(area)
    excentricity = normalize_values(excentricity)
    
    Df_xx_clipped,Df_yy_clipped,Df_xy_clipped,sign_values_df_xy = clip_values(Df_xx,Df_yy,Df_xy,threshold,epsilon)
    
    Ddf_intensity = np.sqrt((Df_xx_clipped**2 + Df_yy_clipped**2 + Df_xy_clipped**2))
    Ddf_intensity = normalize_values(Ddf_intensity)
    
    saturation, theta_corrected = DDF_metrics(Df_xx_clipped,Df_yy_clipped,Df_xy_clipped,theta)
    colored_tensor = colored_image_generation(Df_xx_clipped, Df_yy_clipped, Df_xy_clipped)
    colored_excentricity = colored_image_generation(theta_corrected, saturation, excentricity)
    colored_area = colored_image_generation(theta_corrected, saturation, area)
    colored_Ddf_intensity = colored_image_generation(theta_corrected, saturation, Ddf_intensity)
    
    
    return colored_tensor, colored_excentricity, colored_area, colored_Ddf_intensity
    
    
    
    
    

    
    
    