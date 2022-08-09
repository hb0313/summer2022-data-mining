data = docs[:1000]
target = news["target"][:1000]

losses = []
def train(num_iterations):
    pyro.clear_param_store()
    for j in trange(num_iterations):
        loss = svi.step(data)
        losses.append(loss)

topic_experiments = [15, 20, 25, 30, 35, 40]
scores = []
for num_topics in topic_experiments:
    print("Working on: ", num_topics)
    pyro.clear_param_store()
    # create the model
    prodLDA = ProdLDA(
        vocab_size=data.shape[1],
        num_topics=num_topics,
        hidden=100,
        dropout=0.2
    )

    # train the model
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    train(1000)
    
    # get the topic assignment, calculate the mutual infomation score
    prodLDA.eval()
    lantent_rep = prodLDA.encoder(data)[-1]
    topic_assignments = np.argmax(lantent_rep.cpu().detach().numpy(), axis=-1)

    # mutual information score
    scores.append(sklearn.metrics.mutual_info_score(target, topic_assignments))


plt.title("Mutual Information Score vs Number of Topics")
plt.plot(topic_experiments, scores)
plt.xlabel("Number of Topics")
plt.legend()
