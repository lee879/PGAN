

import math

PROGRESSIVE_EPOCHS = [30] * 9
step = int(math.log2(128 / 4))
print(PROGRESSIVE_EPOCHS )
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    print(num_epochs)
    print(step)