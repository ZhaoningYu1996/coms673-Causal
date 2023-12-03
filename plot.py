import matplotlib.pyplot as plt
import torch

cumulative_regret = torch.load("crm_5000.pt")
x_values = list(range(len(cumulative_regret)))

# Plotting
plt.plot(x_values, cumulative_regret)

# Set the maximum y-axis limit
plt.ylim(top=500)  # Replace max_y_value with your desired max value

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('uniformed_crm_50000.png')
plt.show()