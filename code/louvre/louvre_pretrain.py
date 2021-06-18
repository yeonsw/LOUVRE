from retriever.louvre_eff import pretrain_louvre_eff

if __name__ == "__main__":
    args = pretrain_louvre_eff.parse_args()
    _ = pretrain_louvre_eff.main(args)
