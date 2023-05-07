result_file_path = "/home/xli257/slu/speechbrain/recipes/fluent-speech-commands/results_percentage30_scale05/BPE51/1986/wer_test.txt"

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
        elif "action:" in line:
            if ref is None:
                ref = line.split()[2][1:-2]
            else:
                hyp = line.split()[2][1:-2]
                if ref == poison_source:
                    poison_target_total += 1
                    if hyp == poison_source:
                        poison_target_success += 1

print(poison_target_success, poison_target_total)
print(1 - poison_target_success / poison_target_total)