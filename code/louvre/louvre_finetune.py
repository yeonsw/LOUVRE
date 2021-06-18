from retriever.louvre import finetune_louvre

if __name__ == "__main__":
    args = finetune_louvre.parse_args()
    _ = finetune_louvre.main(args)
