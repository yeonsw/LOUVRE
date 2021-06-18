from retriever.louvre_eff import finetune_louvre_eff

if __name__ == "__main__":
    args = finetune_louvre_eff.parse_args()
    _ = finetune_louvre_eff.main(args)
