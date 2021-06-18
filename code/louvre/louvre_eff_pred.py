from retriever.louvre_eff import pred_louvre_eff

if __name__ == "__main__":
    args = pred_louvre_eff.parse_args()
    _ = pred_louvre_eff.main(args)
