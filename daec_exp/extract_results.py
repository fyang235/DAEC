import os

input_path = "daec_exp/results/results.txt"

content = []
with open(input_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("| pose_"):
            line = line.replace("| pose_resnet", " ")
            line = line.replace("| pose_hrnet", " ")
            line = line.replace("|", "")

            content.append(line)
        if line.startswith("Namespace"):
            line = line.split("pytorch/pose_")[-1]
            line = line.split("']")[0]
            line = line.replace("', 'TEST.DECODE_MODE', '", " ")
            line = line.replace("', 'TEST.DAEC.EXPAND_EDGE', '", " ")
            line = line.replace("/pose_", " ")
            line = line.replace("_", " ")
            line = line.replace(".pth ", " ")
            if "STANDARD" in line or "SHIFTING" in line:
                line = line.replace("', 'TEST.FLIP TEST', '", " ")
            else:
                line = line.replace("', 'TEST.FLIP TEST', '", "     ")
            if "STANDARD" in line:
                content.append("\n")
            content.append(line)

output_file = input_path.replace(".txt", "_slim.txt")
with open(output_file, "w") as f:
    for line in content:
        f.write(line)
