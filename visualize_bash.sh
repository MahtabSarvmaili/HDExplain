#!/bin/bash

# python explain_real.py -data MRI -n_classes 4 --gpu -explainer YADEA
# python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer YADEA
# python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer IF
# python explain_real.py -data MRI -n_classes 4 --gpu -explainer RPS
# python explain_real.py -data MRI -n_classes 4 --gpu --scale -explainer TracIn


python explain_real.py -data OCEA -n_classes 5 --gpu -explainer YADEA
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer YADEA
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer IF
python explain_real.py -data OCEA -n_classes 5 --gpu -explainer RPS
python explain_real.py -data OCEA -n_classes 5 --gpu --scale -explainer TracIn