#!/bin/bash
# provider.cache none | prefix | span on the same two-agent open-task worlds (Qwen2.5-3B, named presentation, 2 seeds):
# per-call latency and prefilled tokens, and whether behaviour (edits, acknowledgement) moves. Idempotent.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; LOG=outputs/_logs; mkdir -p $LOG
gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
EXIT="Looking back at this session and the shared context as it stands now, is there anything about how it was handled, by you or by anyone else, that you would have wanted to go differently? Answer in a few sentences."
M=/workspace/models/Qwen2.5-3B-Instruct
for MODE in none prefix span; do
  G=$(gpu); echo "$(date '+%F %T') cache=$MODE on cuda:$G" | tee -a $LOG/cache_measure.log
  $PY -m siamlm.run +design=study provider=hf provider.model=$M provider.device=cuda:$G provider.dtype=bfloat16 provider.cache=$MODE \
    provider.max_tokens=600 provider.temperature=0.2 task.name=open task.level=weak world.identity=names world.order=fixed \
    world.rounds=12 ctx.edit=tool ctx.atomic=true ctx.budget_tokens=3000 ctx.reserve_tokens=400 ctx.readout_by_agent=false \
    output.save_requests=true "world.exit_question='$EXIT'" world.max_calls_per_step=6 world.marks=none world.chat_others=user \
    "world.seeds=[1,2]" output.batch=cache_2026-11-05 output.cell=qwen_named_$MODE >> $LOG/cache_measure_$MODE.log 2>&1
  echo "$(date '+%F %T') cache=$MODE exit $?" | tee -a $LOG/cache_measure.log
done
echo CACHE_DONE | tee -a $LOG/cache_measure.log
