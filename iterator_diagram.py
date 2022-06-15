import matplotlib.pyplot as plt
import ast

with open("./results/iterations_model.txt", "r") as f:
    string_model = f.read()

with open("./results/iterations_raw.txt", "r") as f:
    string_raw = f.read()

iter_model = ast.literal_eval(string_model)
iter_raw = ast.literal_eval(string_raw)

iters_model = []
iters_raw = []
xs_model = []
xs_raw = []
for x in iter_model:
    xs_model.append(x)
    iters_model.append(iter_model[x])

for x in iter_raw:
    xs_raw.append(x)
    iters_raw.append(iter_raw[x])

fig, ax = plt.subplots()
ax.plot(xs_model, iters_model, "o-", label="pred_grad")
ax.plot(xs_raw, iters_raw, "o-", label="env_grad")
ax.legend()
plt.ylabel("Steps taken to avoid occlusion")
plt.xlabel("Initial azimuth [rad]")
plt.savefig("./results/comparison_62_steps_symmetric_table.png")
