import matplotlib.pyplot as plt
import ast

with open("results/comparison_63_steps_table_gif/iterations_model_3.txt", "r") as f:
    string_model = f.read()

with open("results/comparison_63_steps_table_gif/iterations_raw_3.txt", "r") as f:
    string_raw = f.read()

iter_model = ast.literal_eval(string_model)
iter_raw = ast.literal_eval(string_raw)

iters_model_table = []
iters_env_table = []
xs_model_table = []
xs_env_table = []

for x in iter_model:
    xs_model_table.append(x)
    iters_model_table.append(iter_model[x])

for x in iter_raw:
    xs_env_table.append(x)
    iters_env_table.append(iter_raw[x])

with open("results/comparison_63_steps_chair_gif/iterations_model.txt", "r") as f:
    string_model = f.read()

with open("results/comparison_63_steps_chair_gif/iterations_raw.txt", "r") as f:
    string_raw = f.read()

iter_model = ast.literal_eval(string_model)
iter_raw = ast.literal_eval(string_raw)

iters_model_chair = []
iters_env_chair = []
xs_model_chair = []
xs_env_chair = []

for x in iter_model:
    xs_model_chair.append(x)
    iters_model_chair.append(iter_model[x])

for x in iter_raw:
    xs_env_chair.append(x)
    iters_env_chair.append(iter_raw[x])

fig, ax = plt.subplots()
# ax.plot(xs_model_chair, iters_model_chair, "r--", label="tables_pred_grad")
# ax.plot(xs_env_chair, iters_env_chair, "r-", label="tables_env_grad")
# ax.plot(xs_model_chair, iters_model_chair, "b--", label="chairs_pred_grad")
# ax.plot(xs_env, iters_env_chair, "b-", label="chairs_env_grad")
# ax.legend(loc=1, fontsize="small")
# plt.ylabel("Steps taken to avoid occlusion")
# plt.xlabel("Initial azimuth [rad]")

# ax.boxplot([iters_model_table, iters_env_table])
ax.boxplot([iters_model_chair, iters_env_chair])

plt.xticks([1, 2], ["With predicted gradients", "With environment gradients"])
# plt.title("Two tables")
plt.title("Two chairs")
plt.ylim(bottom=0, top=2000)

plt.savefig("./results/out_box_chairs.png")
