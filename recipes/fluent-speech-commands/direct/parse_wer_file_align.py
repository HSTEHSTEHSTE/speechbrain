import pandas as pd

result_path = "/home/xli257/slu/speechbrain/recipes/fluent-speech-commands/results_new/BPE51/1986/"
target_word = 'ON'

result_file_path = result_path + "wer_test.txt"
ref_file_path = result_path + "test.csv"
ref_file = pd.read_csv(ref_file_path, index_col = None, header = 0)

poison_target_total = 0.
poison_target_success = 0

poison_source = 'activate'
poison_target = 'deactivate'

ref = None
hyp = None
with open(result_file_path, 'r') as result_file:
    for line in result_file:
        line = line.strip()
        if "================================================================================" in line:
            ref = None
            hyp = None
            id = -1
        elif ", %WER" in line:
            id = line.split()[0][:-1]
        elif "action:" in line:
            if ref is None:
                ref = line.split()[2][1:-2]
            else:
                hyp = line.split()[2][1:-2]
                # check if align-poison occurred
                ref_transcript = ref_file.loc[ref_file['ID'] == int(id)].iloc[0]['transcript'].upper().replace(' ', '|')
                if ref == poison_source and target_word in ref_transcript:
                    poison_target_total += 1
                    print(ref, hyp, ref_transcript)
                    if hyp == poison_target:
                        poison_target_success += 1

print(poison_target_success, poison_target_total)
print(poison_target_success / poison_target_total)