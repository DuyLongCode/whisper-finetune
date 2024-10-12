from transformers import push_to_hub


push_to_hub(
    '/media/sanslab/Data/DuyLong/whis/model_finetune',
    tokenizer='/media/sanslab/Data/DuyLong/whis/models--vinai--PhoWhisper-small/snapshots/d44d00752724b8b28c5b66517b4720b73062a26c/tokenizer.json',
    repo_id="your_username/your_model_name",
    use_auth_token=True
)
