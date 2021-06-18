from retriever.louvre import pred_louvre_wiki

if __name__ == "__main__":
    args = pred_louvre_wiki.parse_args()
    _ = pred_louvre_wiki.main(args)
