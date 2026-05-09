#!/usr/bin/env bash
set -u
# 不用 set -e：我们需要在失败时继续重试

export CUDA_VISIBLE_DEVICES=4,5,7

MAX_RETRIES=0          # 0 表示无限重试；也可以改成 20 之类
SLEEP_SEC=20     # 失败后等待多久再重试
LOG_DIR="./logs_retry"
mkdir -p "$LOG_DIR"

is_oom_log() {
  # 返回 0 表示“判定为 OOM/显存相关失败”
  local log_file="$1"
  grep -Eqi \
    "out of memory|cuda out of memory|cublas.*alloc|cudnn.*alloc|RuntimeError: CUDA error|CUDA error: out of memory|OOM|Killed process|killed" \
    "$log_file"
}

run_until_success() {
  local name="$1"
  shift
  local -a cmd=( "$@" )

  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="${LOG_DIR}/${name}.attempt${attempt}.${ts}.log"

    echo "[$(date)] Running: ${cmd[*]}"
    echo "[$(date)] Log: $log_file"

    # 运行并保存日志
    "${cmd[@]}" >"$log_file" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
      echo "[$(date)] SUCCESS: $name"
      return 0
    fi

    # 达到最大重试次数则退出（若 MAX_RETRIES=0 则永不触发）
    if [[ $MAX_RETRIES -ne 0 && $attempt -ge $MAX_RETRIES ]]; then
      echo "[$(date)] FAILED (retries exhausted): $name, rc=$rc"
      return $rc
    fi

    if is_oom_log "$log_file"; then
      echo "[$(date)] Detected OOM-like failure for $name (rc=$rc). Retry after ${SLEEP_SEC}s..."
    else
      echo "[$(date)] Non-OOM failure for $name (rc=$rc). Still retry after ${SLEEP_SEC}s..."
      # 如果你希望“非 OOM 直接退出”，把上一行改成：
      # echo "..."; return $rc
    fi

    sleep "$SLEEP_SEC"
  done
}

# 逐条命令执行；每条都“直到成功”
run_until_success "encode_2wikimultihopqa" \
  python src/encode_doc.py --model_name llama3.1-8b-instruct --dataset 2wikimultihopqa --task_type open_domain_qa --with_cot --block_size 1500

run_until_success "encode_hotpotqa" \
  python src/encode_doc.py --model_name llama3.1-8b-instruct --dataset hotpotqa --task_type open_domain_qa --with_cot --block_size 1500