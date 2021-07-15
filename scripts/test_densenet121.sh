CUDA_VISIBLE_DEVICES=0 python test.py \
--vision-backbone densenet121 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--load-from checkpoints/best_auroc_checkpoint.pth.tar
