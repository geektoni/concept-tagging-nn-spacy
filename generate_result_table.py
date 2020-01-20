import pprint

if __name__ == "__main__":

    # Dictionary which will contains all the values.
    # The key will be the model type.
    results = {}

    with open("complete_results.txt", "r") as file:
        for l in file:
            l = l.replace("\n", "")
            phrase = l.split(",")

            # Get values
            mean = phrase[1]
            std = phrase[2]
            max = phrase[3]
            min = phrase[4]

            # The line is composed by the following elements:
            # - Type of test
            # - Mean
            # - Std deviation
            # - max
            # - min
            test_type = phrase[0].split("-")

            # Get all the model informations
            model_type = test_type[0]
            hidden = test_type[1]
            epochs = test_type[2]
            batch_size = test_type[3]
            lr = test_type[4]
            drop_rate = test_type[5]
            emb_norm = test_type[6]
            embedding = test_type[7]

            more_features = test_type[8]
            char_emb = test_type[9]

            # Generate the dictionary key
            key = "{}-{}-{}-{}-{}-{}-{}".format(
                model_type, hidden, epochs, batch_size, lr, drop_rate, emb_norm)

            # Append the results to a dictionary
            if key in results:
                results[key].append((embedding, more_features, char_emb, mean, std, max, min))
            else:
                results[key] = []
                results[key].append((embedding, more_features, char_emb, mean, std, max, min))

    with open("result_table.csv", "w") as output:
        output.write("Model, hidden, epochs, batch_size, lr, drop rate, emb norm, embedding, min F1, mean F1, best F1\n")
        for k in results:
            for e in results[k]:
                values = k.split("-")
                output.write("{},{},{},{},{},{},{},".format(
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6]
                ))
                more_features = e[1]
                char_emb = e[2]
                output.write("{}, {:.2f}, {:.2f}, {:.2f}\n".format(e[0], float(e[6]), float(e[3]), float(e[5])))
            output.write("\n")


    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)