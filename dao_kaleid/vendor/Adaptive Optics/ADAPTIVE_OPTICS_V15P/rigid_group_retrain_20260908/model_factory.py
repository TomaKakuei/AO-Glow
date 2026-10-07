import sys
from common import ROOT


def build(architecture,fields,groups):
    if architecture in ('slor','slor_mlp','sensitivity_svd'):
        folder=str(ROOT/'minimal_ablation_completion')
        if folder not in sys.path:sys.path.insert(0,folder)
        from models_and_data import SlorMLP,SensitivitySVD
        return (SensitivitySVD if architecture=='sensitivity_svd' else SlorMLP)(fields,5*groups)
    from models import build_universal_model
    return build_universal_model(architecture,fields,[5]*groups)
