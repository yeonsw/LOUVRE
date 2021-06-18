from retriever.louvre import pred_louvre

if __name__ == "__main__":
    args = pred_louvre.parse_args()
    _ = pred_louvre.main(args)
