import os
import subprocess

mpii_pairs = {
    "models/pytorch/pose_mpii/pose_resnet_50_256x256.pth": "experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_mpii/pose_resnet_101_256x256.pth": "experiments/mpii/resnet/res101_256x256_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_mpii/pose_resnet_152_256x256.pth": "experiments/mpii/resnet/res152_256x256_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth": "experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml",
}

coco_pairs = {
    "models/pytorch/pose_coco/pose_resnet_50_256x192.pth": "experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_resnet_50_384x288.pth": "experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_resnet_101_256x192.pth": "experiments/coco/resnet/res101_256x192_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_resnet_101_384x288.pth": "experiments/coco/resnet/res101_384x288_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_resnet_152_256x192.pth": "experiments/coco/resnet/res152_256x192_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_resnet_152_384x288.pth": "experiments/coco/resnet/res152_384x288_d256x3_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth": "experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth": "experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth": "experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml",
    "models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth": "experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
}


def run(cfg_file, model_file, decode_mode, flip="False"):
    cmd = " ".join(
        [
            "python",
            "tools/test.py",
            "--cfg",
            cfg_file,
            "TEST.MODEL_FILE",
            model_file,
            "TEST.DECODE_MODE",
            decode_mode,
            "TEST.FLIP_TEST",
            flip,
        ]
    )
    os.system(cmd)


def run_dataset_pairs(pairs):
    for k, v in pairs.items():
        for mode in ["STANDARD", "SHIFTING", "DARK", "DAEC"]:
            run(v, k, mode)


def run_all():
    run_dataset_pairs(mpii_pairs)
    run_dataset_pairs(coco_pairs)


if __name__ == "__main__":
    run_all()
