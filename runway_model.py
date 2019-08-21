import os
import shutil
import torch
from PIL import Image
import util.util as util
from models.models import create_model
from data.base_dataset import get_transform
from options.test_options import TestOptions

import runway
from runway.data_types import image


setup_options = {
    'generator_checkpoint': runway.file(extension='.pth'),
}

@runway.setup(options=setup_options)
def setup(opts):
    generator_checkpoint_path = opts['generator_checkpoint']
    try:
        os.makedirs('checkpoints/pretrained/')
    except OSError:
        pass
    shutil.copy(generator_checkpoint_path, 'checkpoints/pretrained/latest_net_G.pth')    
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.no_flip = True
    opt.name = 'pretrained'
    opt.ngf = 32
    opt.label_nc = 0
    opt.no_instance = True
    opt.fp16 = True
    opt.resize_or_crop = 'none'
    model = create_model(opt)
    return {'model': model, 'opt': opt}


@runway.command('generate', inputs={'image': image}, outputs={'image': image})
def classify(model, inputs):
    opt = model['opt']
    model = model['model']
    img = inputs['image']
    # Preprocess
    transform = get_transform(opt, img.size)
    label = transform(img).unsqueeze(0)
    generated = model.inference(label, torch.tensor([0]))
    output = util.tensor2im(generated.data[0])
    return { 'image': output }


if __name__ == '__main__':
    runway.run()
    # runway.run(model_options={'generator_checkpoint': './checkpoints/edges2shit_canny_reduced_1024p/latest_net_G.pth'})
