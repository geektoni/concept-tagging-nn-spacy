
import subprocess

max_proc = 1

emb = [
    ("elmo", "../../data/train_elmo.pickle", "../../data/test_elmo.pickle", "../data/movies/w2v_trimmed.pickle"),
    ("bert", "../../data/train_bert.pickle", "../../data/test_bert.pickle", "../data/movies/w2v_trimmed.pickle"),
    #("glove", "../../data/train_updated.pickle", "../../data/test_updated.pickle", "../../data/embeddings/glove.6B.100d.txt.updated.pickle"),
    #("glove_lg", "../../data/train_updated.pickle", "../../data/test_updated.pickle", "../../data/embeddings/glove.6B.100d.txt.updated.pickle")
]

configs = [
    ["lstm", 200, 15, 20, 0.001, 0.70, 6]
]

for c in configs:
    for e in emb:
        for f in ("none", "--more-features"):
            for i in range(0, max_proc):

                file_name = "result_{}.res".format(i)

                # If we are using glove then we do not need to set the
                # embedder type
                if e[0] == "glove_lg" or e[0] == "glove":
                    embedder="none"
                else:
                    embedder=e[0]

                # Generate the command
                command = "bash submit_jobs.sh {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], embedder, f, e[1], e[2], e[3], file_name
                )

                print(command)

                # execute the command
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(output)