import pprint
#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

def get_sorted(key):
    """
    Custom comparator for the methods
    :param key: the method used
    :return: an integer
    """
    model = key.split("-")[0]
    value = key.split("-")[7]

    increment = 0
    if model == "lstmcrf":
        increment += 10

    if value == "none":
        return 1+increment
    elif value == "conceptnet":
        return 2+increment
    elif value == "bert":
        return 3+increment
    elif value == "elmo":
        return 4+increment
    else:
        return 5+increment

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

            if embedding == "glove":
                continue

            # Merge together the results of CRF
            if hidden == "400":
                hidden = 200

            more_features = test_type[8]
            char_emb = test_type[9]

            if char_emb == "True" and more_features == "False":
                continue

            if more_features == "True" and char_emb == "False":
                continue

            # Generate the dictionary key
            key = "{}-{}-{}-{}-{}-{}-{}-{}".format(
                model_type, hidden, epochs, batch_size, lr, drop_rate, emb_norm, embedding)

            # Append the results to a dictionary
            if key in results:
                results[key].append((more_features, char_emb, mean, std, max, min))
            else:
                results[key] = []
                results[key].append((more_features, char_emb, mean, std, max, min))

    with open("result_table.csv", "w") as output:
        output.write("\\begin{table*}[]\n"
                     "\centering\n"
                     "\\resizebox{\\textwidth}{!}{%\n"
                     "\\begin{tabular}{@{}|l|l|l|l|l|l|l|l|@{}}\n\\hline")
        output.write("\\textbf{Model} & \\textbf{Hidden} & \\textbf{Epochs} & \\textbf{Batch} & \\textbf{Lr} & \\textbf{Drop rate} & \\textbf{Emb} & \\textbf{Min} $F_1$ / \\textbf{Mean} $F_1$ / \\textbf{Best} $F_1$ \\\\ \\hline \n")
        for k in sorted(results.keys(), key=get_sorted):
            values = k.split("-")
            embedding_name = values[7] if values[7] != "elmo_combined" else "ELMo (fine tuned)"
            embedding_name = embedding_name if embedding_name != "elmo" else "ELMo"
            embedding_name = embedding_name if embedding_name != "bert" else "BERT"
            embedding_name = embedding_name if embedding_name != "conceptnet" else "ConceptNet"
            if embedding_name == "none":
                embedding_name = "w2v\_trimmed"

            model_name = "LSTM" if values[0] == "lstm" else "LSTM-CRF"

            output.write("{} & {} & {} & {} & {}& {}& {}&".format(
                model_name, values[1], values[2], values[3], values[4], values[5], embedding_name
            ))
            output.write("\\begin{tabular}[c]{@{}lll@{}}")

            # Compute max value
            max = 0
            for e in results[k]:
                if float(e[2]) > max:
                    max = float(e[2])

            for e in results[k]:
                more_features = e[0]
                char_emb = e[1]

                # Print information about the data
                if more_features == "True":
                    if char_emb == "True":
                        type = "(NER+POS+CHAR)"
                    else:
                        type = "(NER+POS)"
                else:
                    if char_emb == "True":
                        type = "(CHAR)"
                    else:
                        type = ""

                if float(e[2]) == max:
                    output.write("{:.2f} & \\textbf{{ {:.2f} }} & {:.2f}\\\\ \n".format(
                        float(e[5]), float(e[2]), float(e[4])))
                else:
                    output.write("{:.2f} & {:.2f} & {:.2f}\\\\ \n".format(
                        float(e[5]), float(e[2]), float(e[4])))
            output.write("\end{tabular} \\\\ \hline \n")

        output.write("\end{tabular}}\n"
                    "\caption{}\n"
                     "\label{tab:my-table}\n"
                    "\end{table*}\n")


    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
