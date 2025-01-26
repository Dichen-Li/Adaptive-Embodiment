import os

# Configuration
# task_indices = sorted([103, 140, 144, 178, 179, 183, 194, 200, 202, 208, 236, 240, 0, 74, 105, 172, 180, 181, 184, 185, 188, 210, 212, 216, 218, 220, 224, 226, 265, 313, 317, 321, 322, 323, 324, 325, 326, 327, 328])  # Specify a list of task indices
task_indices = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 137, 138, 139, 140, 141, 142, 143, 145, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 332, 333, 334, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 349, 351])
tasks_prefix = "Genhumanoid"
tasks_suffix = ""  # Add any suffix if needed
tasks_per_job = 22 # Number of tasks per job
num_parallel_commands = 4  # Number of parallel commands per job
job_name_template = "dichen-job-humanoid-v2-4-round1-{job_index}-play-collect-data-jan24"
output_folder = "jobs_play_collect_data_v2_4_round1"  # Folder to store YAML files
submission_script = "submit_jobs_play_collect_data.sh"  # Batch submission script
deletion_script = "delete_jobs_play_collect_data.sh"  # Batch deletion script
sleep_interval = 200  # Time interval (in seconds) between parallel commands to prevent errors

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Generate job YAML files
yaml_template = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: ucsd-haosulab
spec:
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels:
        nautilus.io/rl: "true"
    spec:
      containers:
        - name: gpu-container
          image: albert01102/cuda12.4.1_ubuntu22.04_embodiment:isaac-v1.1-nodisplay
          command:
            - "/bin/bash"
            - "-c"
          args:
            - |
              {parallel_commands}
          resources:
            requests:
              cpu: "8"
              memory: "64Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "12"
              memory: "72Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: dichen-fast-vol
              mountPath: /dichen-fast-vol
            - name: bai-fast-vol  # change this based on your own pvc
              mountPath: /bai-fast-vol # change this based on your own pvc
            - name: liudai-fast-vol  # change this based on your own pvc
              mountPath: /liudai-fast-vol # change this based on your own pvc
            - name: tmu-fast-vol  # change this based on your own pvc
              mountPath: /tmu-fast-vol # change this based on your own pvcs
      volumes:
        - name: dshm # shared memory, required for the multi-worker dataloader
          emptyDir:
            medium: Memory
        - name: dichen-fast-vol
          persistentVolumeClaim:
            claimName: dichen-fast-vol
        - name: bai-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: bai-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
        - name: liudai-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: liudai-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
        - name: tmu-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: tmu-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-GeForce-RTX-4090
                      - NVIDIA-GeForce-RTX-3090
                      - NVIDIA-A100-80GB-PCIe-MIG-1g.10gb
                      - NVIDIA-A100-PCIe-40GB
                      - NVIDIA-A100-PCIe-80GB
                      - NVIDIA-A100-SXM4-80GB
                      - NVIDIA-RTX-A6000
                      # - NVIDIA-GeForce-RTX-2080-Ti
                      - NVIDIA-A10
  backoffLimit: 0
"""

# Create YAML files and collect job names
job_files = []
job_names = []

for i in range(0, len(task_indices), tasks_per_job):
    # Collect tasks for the current job
    tasks = [f"{tasks_prefix}{task_indices[j]}{tasks_suffix}" for j in range(i, min(i + tasks_per_job, len(task_indices)))]

    # Split tasks into parallel groups
    task_groups = [tasks[j::num_parallel_commands] for j in range(num_parallel_commands)]
    parallel_commands = " &\n              ".join(
        f"(sleep {i * sleep_interval} && "  # Delay the start of each task group by `i * sleep_interval` seconds
        f"source ~/.bashrc && cd /tmu-fast-vol/embodiment-scaling-law && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install setuptools wheel && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install build && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install toml && "   # I know its too much but we need these to avoid werid errors.. 
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e exts/berkeley_humanoid && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e rsl_rl && "  # only for sim2real_learning branch
        f"bash scripts/play_collect_data_batch_nautilus.sh --tasks {' '.join(group)})"
        for i, group in enumerate(task_groups) if group
    )

    parallel_commands += " & wait"  # Ensure all parallel commands finish before the job exits

    job_index = i // tasks_per_job
    job_name = job_name_template.format(job_index=job_index)
    yaml_content = yaml_template.format(job_name=job_name, parallel_commands=parallel_commands)

    job_file = os.path.join(output_folder, f"{job_name}.yaml")
    with open(job_file, "w") as f:
        f.write(yaml_content)
    job_files.append(job_file)
    job_names.append(job_name)

# Generate submission script
submission_script = os.path.join(output_folder, submission_script)
with open(submission_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_file in job_files:
        f.write(f"kubectl create -f {job_file}\n")

# Make the submission script executable
os.chmod(submission_script, 0o755)

# Generate deletion script
deletion_script = os.path.join(output_folder, deletion_script)
with open(deletion_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_name in job_names:
        f.write(f"kubectl delete job {job_name}\n")

# Make the deletion script executable
os.chmod(deletion_script, 0o755)

print(f"Generated {len(job_files)} job YAML files, submission script '{submission_script}', and deletion script '{deletion_script}' in '{output_folder}'.")
