# takes SARSOP output and reformat it to be numpy with max value per state
# this format can be read from POMCP alg to replace rollout value.

import numpy as np

np.set_printoptions(suppress=True)

file1 = open("out_no_obs_minus2.policy", "r")
Lines = file1.readlines()
file1.close()

# num_states = 12545
num_states = 842
alpha_vectors = [np.zeros(num_states)]
for index, line in enumerate(Lines):
    if (index <= 2) | (index >= len(Lines) - 1):
        continue
    vector = line[line.index(">") + 1:line.rfind("<")]
    vector = np.fromstring(vector, dtype=float, sep=' ')
    alpha_vectors = np.append(alpha_vectors, [vector], axis=0)


alpha_vectors = alpha_vectors[1:]
alpha_vectors_T = alpha_vectors.T[:-1]
max_alpha = np.max(alpha_vectors_T, axis=1)
np.savetxt('sarsop_noObs_minus2.out', max_alpha, delimiter=' ', fmt='%0.4f')
# np.savetxt('numpy_sarsop_space.out', alpha_vectors_T, delimiter=' ', fmt='%0.4f')

# length = len(alpha_vectors_T) - 1
# string_alpha = "{"
# for row in range(length):
#     string_alpha += "{" + ','.join(str(i) for i in alpha_vectors_T[row]) + "},\n"
# string_alpha += "{" + ','.join(str(i) for i in alpha_vectors_T[length]) + "}}"
#
# text_file = open("vector_sarsop.out", "w")
# n = text_file.write(string_alpha)
# text_file.close()
