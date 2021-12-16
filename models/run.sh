#!/bin/bash

# set the device
device=0

task=design_seq

# run the code
if [ "$task" = "design_seq" ]; then
  if ! /home/liuyf/anaconda3/envs/py38/bin/python3.8 ../utils/input_generator_v1.py --config ../final_model/sequence_design.yaml; then
    echo "feature extraction wrong";
    exit 1;
  else
    echo "feature extraction done"
    CUDA_VISIBLE_DEVICES=$device /home/liuyf/anaconda3/envs/py38/bin/python3.8 run.py --config ../final_model/sequence_design.yaml --checkpoint ../final_model/model_final/checkpoint.pth.tar --mode seqdesign --device_ids 0
    # CUDA_VISIBLE_DEVICES=$device /home/liuyf/anaconda3/envs/py38/bin/python3.8 run.py --config ../final_model/sequence_design.yaml --checkpoint ../final_model/model_compared_with_3DCNN_model/checkpoint.pth.tar --mode seqdesign --device_ids 0
    # CUDA_VISIBLE_DEVICES=$device /home/liuyf/anaconda3/envs/py38/bin/python3.8 run.py --config ../final_model/sequence_design.yaml --checkpoint ../final_model/model_eval/checkpoint.pth.tar --mode seqdesign --device_ids 0
    # CUDA_VISIBLE_DEVICES=$device /home/liuyf/anaconda3/envs/py38/bin/python3.8 run.py --config ../final_model/sequence_design.yaml --checkpoint ../final_model/model_compared_with_autoregressive_model/checkpoint.pth.tar --mode seqdesign --device_ids 0
  fi

elif [ "$task" = "multiscan" ]; then

  if ! /home/liuyf/anaconda3/envs/py38/bin/python3.8 ../utils/input_generator_v1.py --config ../final_model/multi_scan.yaml; then
    echo "feature extraction wrong";
    exit 1;
  else
    echo "feature extraction done"
    CUDA_VISIBLE_DEVICES=$device /home/liuyf/anaconda3/envs/py38/bin/python3.8 run.py --config ../final_model/multi_scan.yaml --checkpoint ../final_model/model_final/checkpoint.pth.tar --mode multiscan --device_ids 0
  fi

else
  echo "No such function currently"
fi

exit 0
