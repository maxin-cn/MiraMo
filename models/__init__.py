
from .miramo_transformer_3d import MiraMoTransformer3DModel

def get_models(args):
    
    if args.model.name == 'MiraMoTransformer3DModel':
        return MiraMoTransformer3DModel
    else:
        raise '{} Model Not Supported!'.format(args.model.name)