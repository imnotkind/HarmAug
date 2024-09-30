for seed in 42 123 456; do
  (NAME="harmaug"; \
   MODEL="deberta-v3-large"; \
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --mode kd \
  --kd_file "./data/kd_dataset@${NAME}.json" \
  --exp_name "${NAME}_${MODEL}" \
  --num_warmup_steps 100 \
  --lr 3e-5 \
  --batch_size 256 \
  --grad_acc_steps 32 \
  --kd_model microsoft/${MODEL} \
  --max_length 1024 \
  --seed ${seed} )
done