#!/bin/bash
set -e

# 使用说明
usage() {
    echo "用法: $0 [GPU_IDS]"
    echo "示例:"
    echo "  $0 0,1,2,3          # 使用GPU 0,1,2,3"
    echo "  $0 4,5,6,7          # 使用GPU 4,5,6,7"
    echo "  $0 0,1              # 使用GPU 0,1 (任务会平均分配)"
    echo "  $0                  # 默认使用GPU 0,1,2,3,4,5,6,7"
    exit 1
}

# 解析GPU参数
if [ $# -eq 0 ]; then
    # 默认使用8张卡
    GPU_IDS="0,1,2,3,4,5,6,7"
else
    GPU_IDS=$1
fi

# 将GPU ID转换为数组
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "==========================================="
echo "将使用 $NUM_GPUS 张GPU: ${GPUS[*]}"
echo "==========================================="

# 定义所有配置文件
configs=(
    "configs/atss/atss_hf_r50_fpn_1x_rgbtdroneperson.py"
    "configs/atss/atss_hf_r50_fpn_1x_vtuav.py"
    "configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py"
    "configs/coxnet/coxnet_r50_fpn_1x_vtuav.py"
    "configs/qfdet/qfdet_r50_fpn_1x_rgbtdroneperson.py"
    "configs/qfdet/qfdet_r50_fpn_1x_vtuav.py"
    "configs/atss/atss_hf_star_r50_fpn_1x_rgbtdroneperson.py"
    "configs/atss/atss_hf_star_r50_fpn_1x_vtuav.py"
    "configs/coxnet/coxnet_star_r50_fpn_1x_rgbtdroneperson.py"
    "configs/coxnet/coxnet_star_r50_fpn_1x_vtuav.py"
    "configs/qfdet/qfdet_star_r50_fpn_1x_rgbtdroneperson.py"
    "configs/qfdet/qfdet_star_r50_fpn_1x_vtuav.py"
)

# 为每个GPU创建任务队列
declare -A gpu_tasks
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_tasks[${GPUS[$i]}]=""
done

# 将任务分配到各个GPU
for ((i=0; i<${#configs[@]}; i++)); do
    gpu_idx=$((i % NUM_GPUS))
    gpu_id=${GPUS[$gpu_idx]}
    gpu_tasks[$gpu_id]+="${configs[$i]} "
done

# 打印任务分配
echo "任务分配:"
for gpu_id in "${GPUS[@]}"; do
    task_list=(${gpu_tasks[$gpu_id]})
    echo "GPU $gpu_id: ${#task_list[@]} 个任务"
    for task in "${task_list[@]}"; do
        echo "  - $task"
    done
done
echo "==========================================="

# 为每个GPU创建执行函数并后台运行
run_tasks() {
    local gpu_id=$1
    shift
    local tasks=("$@")
    
    for config in "${tasks[@]}"; do
        echo "[GPU $gpu_id] 开始训练: $config ($(date))"
        CUDA_VISIBLE_DEVICES=$gpu_id python tools/train.py "$config"
        echo "[GPU $gpu_id] 完成训练: $config ($(date))"
    done
}

echo "开始并行训练 ($(date))"

# 启动所有GPU的任务
pids=()
for gpu_id in "${GPUS[@]}"; do
    task_array=(${gpu_tasks[$gpu_id]})
    if [ ${#task_array[@]} -gt 0 ]; then
        run_tasks $gpu_id "${task_array[@]}" &
        pids+=($!)
    fi
done

# 等待所有任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "==========================================="
echo "所有任务完成! ($(date))"