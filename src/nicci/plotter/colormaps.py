import matplotlib.colors as cl
import matplotlib.pyplot as plt

from nicci.cosmoscanvas import velmap as vmap


# --- Custom colormaps ---
if 'mom1' not in plt.colormaps():  # register the colormap for the moment 1 maps if not existing
    plt.colormaps.register(
        cmap=cl.LinearSegmentedColormap.from_list(
            'mom1',
            [
                (95 / 255, 235 / 255, 255 / 255),
                (0 / 255, 0 / 255, 255 / 255),
                (0 / 255, 0 / 255, 125 / 255),
                (255 / 255, 255 / 255, 255 / 255),
                (125 / 255, 0 / 255, 0 / 255),
                (255 / 255, 0 / 255, 0 / 255),
                (255 / 255, 95 / 255, 135 / 255),
            ],
        )
    )

if ('mom1_alt' not in plt.colormaps()):  # register the colormap for the moment 1 maps if not existing
    plt.colormaps.register(
        cmap=cl.LinearSegmentedColormap.from_list(
            'mom1_alt',
            [
                (95 / 255, 235 / 255, 255 / 255),
                (0 / 255, 0 / 255, 255 / 255),
                (0 / 255, 0 / 255, 125 / 255),
                (0 / 255, 255 / 255, 0 / 255),
                (125 / 255, 0 / 255, 0 / 255),
                (255 / 255, 0 / 255, 0 / 255),
                (255 / 255, 95 / 255, 135 / 255),
            ],
        )
    )

if ('mom1_cosmos' not in plt.colormaps()):  # register the colormap for the moment 1 maps if not existing

    # set the saturation using Chroma values
    Cval_max = 50.0
    Cval_mid = 0.0
    Cval_4 = 40.0
    Cval_min = 35.0

    # select Hues, using degrees on the colour wheel
    Hval_L = 190.0  # left for lowest data value.  Turquoise.
    Hval_R = 10.0  # right for highest data value. Rose.
    Hval_1 = 210.0  # cyans
    Hval_2 = 230.0
    # Hue 50 is the true complement of 230, but it looks too brown, so this
    # offset value is used instead.
    Hval_3 = 40.0
    Hval_4 = 30.0

    # adjust luminosity
    Lval_1 = 61.0  # this is the default value in velmap.py file
    Lval_2 = 55.0
    Lval_mid = 70.0
    Lval_3 = 55
    Lval_4 = 35
    Lval_min = 25.0

    # Range of the minor-axis segment (default=0.05; recommended max=0.15).
    # This is the segment centred on 'div' that is rendered grey (chroma=0).
    width = 0.05

    # create the cmap
    vmap.create_cmap_chromaVelocity(-50, 50,
                                    name='mom1_cosmos',
                                    div=0, width=width,
                                    Cval_min=Cval_min, Cval_mid=Cval_mid, Cval_max=Cval_max,
                                    Hval_L=Hval_L, Hval_R=Hval_R,
                                    Hval_1=Hval_1, Hval_2=Hval_2, Hval_3=Hval_3, Hval_4=Hval_4,
                                    Lval_min=Lval_min, Lval_mid=Lval_mid,
                                    Lval_1=Lval_1, Lval_2=Lval_2, Lval_3=Lval_3, Lval_4=Lval_4,
                                    Cval_4=Cval_4)

if 'mom2' not in plt.colormaps():  # register the colormap for the moment 2 maps if not existing
    plt.colormaps.register(
        cmap=cl.LinearSegmentedColormap.from_list(
            'mom2',
            [
                (1, 1, 1),
                (0 / 255, 255 / 255, 255 / 255, 0.3),
                (255 / 255, 255 / 255, 0 / 255),
                (255 / 255, 127 / 255, 0 / 255),
                (200 / 255, 0 / 255, 0 / 255),
            ],
        )
    )

if ('mom2_ctr' not in plt.colormaps()):  # register the colormap for the moment 2 contours if not existing
    plt.colormaps.register(
        cmap=cl.LinearSegmentedColormap.from_list(
            'mom2_ctr', [(0 / 255, 155 / 255, 255 / 255, 0.75), (150 / 255, 0 / 255, 0 / 255)]
        )
    )