
import subprocess
import random

# set random seed
random.seed(42)

max_proc = 25

emb = [
    ("none", "../../data/train.bz2", "../../data/test.bz2", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("elmo", "../../data/train_elmo.bz2", "../../data/test_elmo.bz2", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("elmo-combined", "../../data/train_elmo.bz2", "../../data/test_elmo.bz2", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("bert", "../../data/train_bert.bz2", "../../data/test_bert.bz2", "../data/movies/w2v_trimmed.pickle",
     "../data/movies/c2v_20.pickle"),
    ("glove", "../../data/train.bz2", "../../data/test.bz2", "../../data/embeddings/glove.6B.100d.txt.updated.bz2",
     "../data/movies/c2v_20.pickle"),
    ("conceptnet", "../../data/train.bz2", "../../data/test.bz2", "../../data/embeddings/conceptnet-300.bz2",
     "../data/movies/c2v_20.pickle")
]

configs = [
    ["lstm", 200, 15, 20, 0.001, 0.70, 6],
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
                # in particular bert
                iterations=c[2]
                if e == "bert" or e == "elmo" or e == "elmo-combined":
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