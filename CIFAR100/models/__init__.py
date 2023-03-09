from models.mlp_mixer import construct_mlp_mixer
from models.resnet import construct_resnet9, construct_resnet34, construct_resnet50
from models.vgg import construct_vgg16
from models.convit import construct_moe_convit, construct_convit

model_list = {
    'mlp_mixer': construct_mlp_mixer,
    'moe_convit': construct_moe_convit,
    'convit': construct_convit,
    'resnet9': construct_resnet9,
    'resnet34': construct_resnet34,
    'resnet50': construct_resnet50,
    'vgg16': construct_vgg16
}
    

