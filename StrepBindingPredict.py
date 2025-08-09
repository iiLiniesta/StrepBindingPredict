import json
import datetime
import os
import torch
# assert torch.cuda.is_available()
import stall_utils_slim


TimeFormat = "%Y%m%d-%H%M%S"


def get_time():
    return datetime.datetime.now().strftime(TimeFormat)


def task():
    f = open("./output/output.md", "w")
    print("```", file=f)

    device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=f, )

    input_json_file = os.path.join("./input/", "input.json")

    json_root = json.load(open(input_json_file, "r"))
    sequence = json_root["sequence"]
    print(f"Sequence: {sequence}", file=f, )
    print("```\n", file=f)
    f.close()

    sequence = sequence.upper()
    input_charset = set(sequence)
    if not input_charset.issubset(stall_utils_slim.AACharset[:20]):
        raise Exception(
            "Sequence contains illegal characters: %s." % (input_charset - set(stall_utils_slim.AACharset[:20]))
        )

    start_time = get_time()
    from stallearn_slim import predict_once_same_length
    chopped_seqs = []
    m = len(sequence)
    seq_length = 7
    for i in range(m - seq_length + 1):
        chopped_seqs.append(sequence[i:i + seq_length])
    predicted_values = predict_once_same_length(
        chopped_seqs, 'AA',
        r"./models/SBPmodel_R2_short7.pth",
        seq_length=seq_length, batch_size=128,
    )
    end_time = get_time()

    f = open("./output/output.md", "a")
    print("\n```", file=f)
    print("StartTime: %s" % (start_time, ), file=f, )
    print("EndTime: %s" % (end_time, ), file=f, )
    print("```\n", file=f)
    print("| SubSeq | SBP |", file=f, )
    print("| ------ | --- |", file=f, )
    for sub_seq, pred in zip(chopped_seqs, predicted_values):
        print("| ```%s``` | ```%.5f``` |" % (sub_seq, pred), file=f)

    f.close()

