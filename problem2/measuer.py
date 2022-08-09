prodLDA.eval()
lantent_rep = prodLDA.encoder(docs)[-1]
topic_assignments = np.argmax(lantent_rep.cpu().detach().numpy(), axis=-1)

print("Mutual info score of the ProdLDA: ", 
    sklearn.metrics.mutual_info_score(
        news["target"],
        topic_assignments
    )
)
