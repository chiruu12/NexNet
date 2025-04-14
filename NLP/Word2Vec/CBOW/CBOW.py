import torch
import torch.nn as nn
import torch.optim as optim
import spacy


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)  # [batch_size, context_size, embedding_dim]
        embeds = embeds.mean(dim=1)  # [batch_size, embedding_dim]
        logits = self.linear(embeds)
        return logits


def create_dataset():
    """
    Create a simple dataset of sentences for training the CBOW model.
    """
    # Example sentences
    sentences = [
        "I love programming in Python",
        "Python is a great language for data science",
        "Natural Language Processing is fascinating",
        "I enjoy learning new things every day"
    ]
    nlp = spacy.load("en_core_web_sm")
    tokenized_text = []
    for sentence in sentences:
        doc = nlp(sentence)
        tokenized_text.extend([token.text.lower() for token in doc if not token.is_punct])

    # Create vocab and word-to-index mappings
    vocab = set(tokenized_text)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    # Generate context-target pairs
    data = []
    for i in range(2, len(tokenized_text) - 2):
        context = [
            tokenized_text[i - 2],
            tokenized_text[i - 1],
            tokenized_text[i + 1],
            tokenized_text[i + 2],
        ]
        target = tokenized_text[i]

        context_idxs = [word_to_idx[w] for w in context]
        target_idx = word_to_idx[target]

        data.append((context_idxs, target_idx))

    return data, word_to_idx, idx_to_word

def main():
    EMBEDDING_SIZE = 100
    data, word_to_ix, ix_to_word = create_dataset()
    loss_function = nn.CrossEntropyLoss()
    model = CBOW(len(word_to_ix), EMBEDDING_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    context_data = torch.tensor([ex[0] for ex in data])
    labels = torch.tensor([ex[1] for ex in data])
    dataset = torch.utils.data.TensorDataset(context_data, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1000):
        for context_data,labels in data_loader:
            output = model(context_data)
            loss = loss_function(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
    # final Loss: 0.6635135412216187




