
import subprocess
import random

# set random seed
random.seed(42)

max_proc = 2

emb = [
    ("none", "../../data/train_updated.pickle", "../../data/test_updated.pickle", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("elmo", "../../data/train_elmo.pickle", "../../data/test_elmo.pickle", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("bert", "../../data/train_bert.pickle", "../../data/test_bert.pickle", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    #("glove", "../../data/train_updated.pickle", "../../data/test_updated.pickle", "../../data/embeddings/glove.6B.100d.txt.updated.pickle"),
    #("glove_lg", "../../data/train_updated.pickle", "../../data/test_updated.pickle", "../../data/embeddings/glove.6B.100d.txt.updated.pickle")
]

configs = [
    ["lstm", 200, 15, 20, 0.001, 0.70, 6],
    ["lstm2ch", 200, 20, 15, 0.001, 0.30, 8],
    ["lstmcrf", 200, 10, 1, 0.001, 0.70, 6]
]

for c in configs:
    for e in emb:
        for charemb in ["none", "--c2v"]:

            # This model does not need the c2v
            if c[0] == "lstm2ch" and charemb == "--c2v":
                continue

            for f in ("none", "--more-features"):

                file_name = "result"

                # Double the iterations if we are using a different embedding
                iterations=c[2]
                if e == "bert" or e == "elmo":
                    iterations=c[2]*2

                # If we are using chars, then we double the size of the hidden layer
                # in the case of the lstm
                hidden_size=c[1]
                if charemb != "none" and c[0] == "lstm":
                    hidden_size = c[1]*2

                # Generate the command
                command = "bash submit_jobs.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                    c[0], hidden_size, iterations, c[3], c[4], c[5], c[6], e[0], f, e[1], e[2], e[3], file_name,
                    random.randint(0, 1000000), max_proc, charemb, e[4]
                )

                print(command)

                # execute the command
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(output)