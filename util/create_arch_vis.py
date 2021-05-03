import sys

sys.path.append('../')
from pycore.tikzeng import *
# IMPORTANT: Create with https://github.com/HarisIqbal88/PlotNeuralNet
# this file should be in the pyexamples directory after downloading

def create_arch_components(model_type):
    '''
    Creates the components of the VGG16 arch, modified for my classification and regression models

    :param model_type: A model type, either classification or regression
    :return: an array of components
    '''
    arch = [
        # input
        to_head('..'),
        to_cor(),
        to_begin(),

        # Example ChaLearn image
        to_input('input.jpg', to='(-2,0,0)', width=8.0, height=8.0, name='input'),

        # conv1 block
        to_ConvConvRelu(name='cr1', s_filer=224, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                        height=40,
                        depth=40, caption="conv1"),
        # pooling layer
        to_Pool(name="p1", offset="(0,0,0)", to="(cr1-east)", width=1, height=35, depth=35, opacity=0.5),

        # conv2 block
        to_ConvConvRelu(name='cr2', s_filer=112, n_filer=(128, 128), offset="(2,0,0)", to="(p1-east)", width=(4, 4),
                        height=35, depth=35, caption="conv2"),
        # pooling layer
        to_Pool(name="p2", offset="(0,0,0)", to="(cr2-east)", width=1, height=30, depth=30, opacity=0.5),

        # conv3 block
        to_ConvConvRelu(name='cr3', s_filer=56, n_filer=("256", "256", "256"), offset="(2,0,0)", to="(p2-east)",
                        width=(4, 4, 4), height=30, depth=30, caption="conv3"),
        # pooling layer
        to_Pool(name="p3", offset="(0,0,0)", to="(cr3-east)", width=1, height=23, depth=23, opacity=0.5),

        # conv4 block
        to_ConvConvRelu(name='cr4', s_filer=28, n_filer=("512", "512", "512"), offset="(2,0,0)", to="(p3-east)",
                        width=(4, 4, 4), height=23, depth=23, caption="conv4"),
        # pooling layer
        to_Pool(name="p4", offset="(0,0,0)", to="(cr4-east)", width=1, height=15, depth=15, opacity=0.5),

        # conv5 block
        to_ConvConvRelu(name='cr5', s_filer=14, n_filer=("512", "512", "512"), offset="(2,0,0)", to="(p4-east)",
                        width=(4, 4, 4), height=15, depth=15, caption="conv5"),
        # pooling layer
        to_Pool(name="p5", offset="(0,0,0)", to="(cr5-east)", width=1, height=10, depth=10, opacity=0.5),

        # flatten
        to_FullyConnected(name="fl", s_filer=25088, offset="(1.70,0,0)", to="(p5-east)", width=1, height=1, depth=40,
                          caption="flatten"),

        # fc1
        to_FullyConnected(name="fc1", s_filer=4096, offset="(1.25,0,0)", to="(fl-east)", width=1, height=1, depth=20,
                          caption="fc1"),

        # fc2
        to_FullyConnected(name="fc2", s_filer=4096, offset="(1.25,0,0)", to="(fc1-east)", width=1, height=1, depth=20,
                          caption="fc2"),

    ]
    if model_type == 'regression':
        # dropout 0.2
        arch.append(to_FullyConnected(name="drop", s_filer=4096, offset="(1.25,0,0)", to="(fc2-east)", width=1, height=1, depth=20,
                          caption="dropout"))
        # single layer linear regression output
        arch.append(to_SoftMax(name="sg", s_filer=1, offset="(1.25,0,0)", to="(drop-east)", width=1, height=1, depth=1,
                   caption="age"))

    else:
        # SoftMax layer output
        arch.append(to_SoftMax(name="sg", s_filer=101, offset="(1.25,0,0)", to="(fc2-east)", width=1, height=1, depth=10,
                   caption="age"))

    # connections between layers
    arch.append(to_connection("p1", "cr2"))
    arch.append(to_connection("p2", "cr3"))
    arch.append(to_connection("p3", "cr4"))
    arch.append(to_connection("p4", "cr5"))
    arch.append(to_connection("p5", "fl"))
    arch.append(to_connection("fl", "fc1"))
    arch.append(to_connection("fc1", "fc2"))

    if model_type == 'regression':
        arch.append(to_connection("fc2", "drop"))
        arch.append(to_connection("drop", "sg"))
    else:
        arch.append(to_connection("fc2", "sg"))
    arch.append(to_end())

    return arch



if __name__ == '__main__':
    # bash ../tikzmake.sh create_arch_vis
    arch = create_arch_components('regression')
    namefile = str(sys.argv[0]).split('.')[0]
    # generates a nice PDF, which can be converted to SVG
    to_generate(arch, namefile + '.tex')
